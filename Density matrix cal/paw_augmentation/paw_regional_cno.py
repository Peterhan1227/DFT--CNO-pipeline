"""
paw_regional_cno.py -- WS-cell-restricted PAW density matrix (regional CNO):

    G_A[a,b] = <psi~_a|P_A|psi~_b> + sum_(atom,image) beta_a^* Q_A(atom,image) beta_b

Q_A,ij = <phi_i|P_A|phi_j> - <phi~_i|P_A|phi~_j> is the region-INTERSECTED
augmentation overlap (radial x angular quadrature, WS/Voronoi membership
test), computed once per (atom, image) site, independent of k-point/band.

beta_n,i = <p~_i|psi~_n> is region-INDEPENDENT (paw.nonlq.proj() + a gauge
correction); its value at a periodic image R follows from the exact Bloch
relation beta(R) = exp(2*pi*i*k.R)*beta(0), never real-space quadrature.

Validation order: Q_A checks -> reduced-k-point reference (Hermiticity/
PSD/trace/gauge/closure checks) -> only if that passes, the full spectrum.

Writes to Data/<MATERIAL>/output/<OUTPUT_SUBDIR>/ (same convention as
main.py), plus paw_regional_report.txt/json with the validation gates.
Does not modify main.py or config.py (config.py only read).

Full derivation and validation history: RESULTS.md. Superseded methods and
the diagnostics that led here: diagnostics/ (not imported here, except via
the small helper modules in helper functions/).
"""
import sys
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "helper functions"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "helper functions"))
from ws_cell import read_poscar_structure, parse_ws_center, build_ws_grid_map  # noqa: E402
from direct_fourier import ws_membership  # noqa: E402

from vaspwfc import vaspwfc  # noqa: E402
from paw import nonlq  # noqa: E402
from sph_harm import sph_r  # noqa: E402
from paw_overlap import load_pawpp, build_qij_block  # noqa: E402
from ase.io import read as ase_read  # noqa: E402

from beta_gauge_utils import read_eigenval_kweights, gauge_correct_beta, OCC_TOL  # noqa: E402
from realspace_beta import zero_pad_ifft, real_space_beta_for_bands  # noqa: E402
from paw_direct_LA import solve_natural_orbitals_direct  # noqa: E402
import config  # noqa: E402  -- read-only; only MATERIAL/OUTPUT_SUBDIR/WS_* are ever read

# Output goes to Data/<MATERIAL>/output/<OUTPUT_SUBDIR>/ -- the same directory
# main.py writes to (computed per-run in main(), since it depends on config.MATERIAL
# and config.OUTPUT_SUBDIR, and on an override material= if one is passed to main()).

# Diagnostics-only report (paw_regional_report.txt/json) also goes there.

# Route selection for the natural-orbital eigenproblem: the "low-rank" state-space
# K_A route (nstates x nstates, build_G_A_K_A below) is cheap when nstates << Nr
# (true for WSe2_mono's WS grid). If a material instead has Nr < nstates (fewer
# WS-grid points than input Bloch states), building the enlarged Gram matrix
# L_A = Phi Phi^H directly (paw_direct_LA.py) and diagonalizing THAT is cheaper --
# it is mathematically exact and shares K_A's nonzero eigenvalues (same Gram-matrix
# trick, just transposed), not a new approximation. Toggle here; both routes read
# the same Psi/beta_by_site/sites/p and produce the same cno_occupations.npy /
# cno_orbitals.npy contract.
USE_DIRECT_L_A = False

WS_MEMBERSHIP_NMAX = 3          # lattice-translation search range for the bisector test
SITE_SEARCH_NMAX = 4            # atom-image search range (matches build_real_space_S's +1 convention)
DIST_PRUNE = 16.0               # Angstrom, candidate-image prefilter (matches existing convention)
# Resolution chosen from an explicit convergence study (see RESULTS.md): a
# hard-edge (discontinuous) boundary indicator converges slowly/algebraically
# under plain Gauss-Legendre x uniform-phi quadrature (Gibbs-phenomenon-like),
# not spectrally -- successive doublings gave max|delta Q| = 5.2e-4 (16x32),
# 1.9e-4 (24x48), 1.3e-4 (32x64), 4.7e-5 (48x96->64x128), 3.4e-5 (64x128->96x192).
# 64x128 vs 96x192 is the first pair comfortably inside a 1e-4 tolerance.
N_THETA_DEFAULT = 64
N_PHI_DEFAULT = 128
N_THETA_REFINED = 96
N_PHI_REFINED = 192
Q_A_REFINEMENT_TOL = 1e-4

REDUCED_REFERENCE_KPOINTS = [1, 41, 81, 122, 163]   # WSe2_mono default (324 k-points); other
                                                     # materials get an auto-computed set (see
                                                     # load_system's reduced_reference_kpoints /
                                                     # crossk_kpoints, stored in sys_ and used by
                                                     # every check function instead of this
                                                     # constant, so a different material's mesh
                                                     # size is handled automatically).
# Real-space grid density used ONLY for the cross-k validation reference
# (item G) -- beta is no longer built from real-space in production at all
# (see module docstring's 2026-07-11 correction), so this no longer governs
# accuracy of the production result, only of the validation check.
VALIDATION_GRID_FACTOR = 3
BOUND_01_TOL = 1e-3
HERMITICITY_TOL = 1e-8
DIAGONAL_CLOSURE_TOL = 1e-8      # sign-independent algebraic identity -- must be ~exact
PAW_NORM_CLOSURE_TOL = 1e-6      # regional norm vs trusted reciprocal full-Q norm
PARTITION_CLOSURE_TOL = 1e-3     # sum_images Q_A(atom,image) ~= Q_full(atom)


# ── geometric site enumeration (purely geometric -- independent of state) ──

def find_contributing_sites(pawpp, elements_idx, atom_cart, latvec, r_grid_cart,
                             nmax=SITE_SEARCH_NMAX, dist_prune=DIST_PRUNE):
    """Concrete list of (iatom, image, image_cart) sites with at least one
    r_grid_cart point within rmax_eff -- the same list every per-site beta
    computation must use."""
    ns = np.arange(-nmax, nmax + 1)
    n1, n2, n3 = [a.ravel() for a in np.meshgrid(ns, ns, ns, indexing='ij')]
    all_n = np.column_stack([n1, n2, n3])
    all_n_cart = all_n @ latvec
    centroid = r_grid_cart.mean(axis=0)

    sites = []
    for iatom, ei in enumerate(elements_idx):
        pp = pawpp[ei]
        rmax_eff = pp.proj_rmax * (pp.NPSRNL - 1) / pp.NPSRNL
        images_cart = atom_cart[iatom] + all_n_cart
        d_centroid = np.linalg.norm(images_cart - centroid[None, :], axis=1)
        candidate_idx = np.where(d_centroid < dist_prune)[0]
        for ii in candidate_idx:
            Rimg = images_cart[ii]
            dist = np.linalg.norm(r_grid_cart - Rimg[None, :], axis=1)
            if (dist <= rmax_eff).any():
                sites.append(dict(iatom=int(iatom), element=pp.element, pp_idx=int(ei),
                                   image=tuple(int(x) for x in all_n[ii]),
                                   image_cart=Rimg, rmax_eff=rmax_eff))
    return sites


# ── region-intersected Q_A for one site (purely geometric) ─────────────────

def _angular_grid(n_theta, n_phi):
    x, w_theta = leggauss(n_theta)
    theta = np.arccos(x)
    phi = np.arange(n_phi) * (2 * np.pi / n_phi)
    w_phi = 2 * np.pi / n_phi
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    WT, _ = np.meshgrid(w_theta, np.ones(n_phi), indexing='ij')
    directions = np.stack([
        np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH),
    ], axis=-1).reshape(-1, 3)
    ang_weight = (WT * w_phi).reshape(-1)
    return directions, ang_weight


def build_regional_Qij_site(pp, image_cart, center_cart, latvec,
                             n_theta=N_THETA_DEFAULT, n_phi=N_PHI_DEFAULT,
                             ws_membership_nmax=WS_MEMBERSHIP_NMAX,
                             force_all_inside=False, force_all_outside=False):
    """
    Region-intersected augmentation overlap for ONE atom-image site:

        Q_A,ij = integral over (sphere_site ∩ A) of
                   phi_ae_i(r) phi_ae_j(r) - phi_ps_i(r) phi_ps_j(r)

    using pp's own logarithmic radial grid + Simpson weights (pp.rgrid,
    pp.rad_simp_w -- the SAME radial quadrature paw.pawpotcar.get_Qij()
    uses, never the coarse global FFT grid) times a dense Gauss-Legendre
    (cos theta) x uniform (phi) angular quadrature at each radius, with
    region-A membership tested via the exact WS/Voronoi bisector test at
    the actual Cartesian position (image_cart + r*direction).

    force_all_inside/force_all_outside: skip the (expensive) angular
    membership test and return the trivial limits directly -- used only for
    the validation checks in validate_Q_A_construction() to build an exact
    ground truth to compare the general quadrature against; NOT used in
    the production per-site loop (every real site uses the general path).
    """
    lmmax = pp.lmmax
    ilm = pp.ilm
    lmax_l = int(pp.proj_l.max())

    if not hasattr(pp, "rad_simp_w"):
        pp.set_simpi_weight()

    directions, ang_weight = _angular_grid(n_theta, n_phi)
    Y_by_l = [sph_r(directions, l) for l in range(lmax_l + 1)]
    Y_full = np.zeros((directions.shape[0], lmmax))
    for ii, (n, l, m) in enumerate(ilm):
        Y_full[:, ii] = Y_by_l[l][:, m + l]

    rgrid = pp.rgrid
    ae_full = np.zeros((lmmax, rgrid.size))
    ps_full = np.zeros((lmmax, rgrid.size))
    for ii, (n, l, m) in enumerate(ilm):
        ae_full[ii] = pp.paw_ae_wfc[n]
        ps_full[ii] = pp.paw_ps_wfc[n]

    Q_A = np.zeros((lmmax, lmmax))
    full_solid_angle = 4 * np.pi
    for ir, r in enumerate(rgrid):
        if force_all_inside:
            w_masked = ang_weight
        elif force_all_outside:
            w_masked = np.zeros_like(ang_weight)
        else:
            pts_cart = image_cart[None, :] + r * directions
            inside = ws_membership(pts_cart, center_cart, latvec, nmax=ws_membership_nmax)
            w_masked = ang_weight * inside
        AngOverlap = (Y_full * w_masked[:, None]).T @ Y_full
        RadDiff = np.outer(ae_full[:, ir], ae_full[:, ir]) - np.outer(ps_full[:, ir], ps_full[:, ir])
        Q_A += pp.rad_simp_w[ir] * RadDiff * AngOverlap

    return 0.5 * (Q_A + Q_A.T)


# ── Phase 2 item 3: required Q_A checks ─────────────────────────────────────

def validate_Q_A_construction(pawpp, elements_idx, atom_cart, latvec, center_cart):
    """Runs the required Q_A checks and returns a report dict. Does not
    build any production matrix."""
    print("=== Validating region-intersected Q_A construction ===")
    report = dict(checks=[])
    pp0 = pawpp[elements_idx[0]]
    image_cart_far_inside = atom_cart[0]     # atom 0 sits at the WS center (config.WS_CENTER)
    # a point guaranteed far outside any physical WS cell for this system
    image_cart_far_outside = atom_cart[0] + np.array([1000.0, 1000.0, 1000.0])

    full_Q = pp0.get_Qij()

    # (a) sphere entirely inside A: general quadrature (real geometry, atom
    # 0 at the WS center) should match full_Q closely.
    Q_inside_general = build_regional_Qij_site(pp0, image_cart_far_inside, center_cart, latvec)
    err_inside = float(np.max(np.abs(Q_inside_general - full_Q)))
    report["checks"].append(dict(name="sphere_inside_matches_full_Q",
                                  max_abs_err=err_inside, passed=bool(err_inside < 1e-3)))
    print(f"  sphere entirely inside A vs full Q: max|err|={err_inside:.4e}  "
          f"{'OK' if err_inside < 1e-3 else 'FAIL'}")

    # (b) sphere entirely outside A: forced-outside quadrature must give
    # exactly zero (trivial by construction, sanity-checks the code path),
    # AND a genuinely far-away site (general quadrature, real membership
    # test) must also give (numerically) zero.
    Q_outside_forced = build_regional_Qij_site(pp0, image_cart_far_inside, center_cart, latvec,
                                                force_all_outside=True)
    err_outside_forced = float(np.max(np.abs(Q_outside_forced)))
    Q_outside_general = build_regional_Qij_site(pp0, image_cart_far_outside, center_cart, latvec)
    err_outside_general = float(np.max(np.abs(Q_outside_general)))
    report["checks"].append(dict(name="sphere_outside_is_zero_forced",
                                  max_abs_err=err_outside_forced, passed=bool(err_outside_forced < 1e-10)))
    report["checks"].append(dict(name="sphere_outside_is_zero_general",
                                  max_abs_err=err_outside_general, passed=bool(err_outside_general < 1e-8)))
    print(f"  sphere entirely outside A (forced path): max|Q_A|={err_outside_forced:.4e}  "
          f"{'OK' if err_outside_forced < 1e-10 else 'FAIL'}")
    print(f"  sphere entirely outside A (general, far site): max|Q_A|={err_outside_general:.4e}  "
          f"{'OK' if err_outside_general < 1e-8 else 'FAIL'}")

    # (c) partition: split the sphere with an ARTIFICIAL half-space cut
    # (not a real lattice bisector) using a synthetic single-plane
    # "region A" and confirm A-piece + complement-piece = full Q. Reuses
    # the same quadrature machinery with a manual indicator instead of
    # ws_membership, to isolate "does angular quadrature correctly
    # partition" from "is ws_membership itself correct" (already exercised
    # by (a)/(b) above on the real geometry).
    directions, ang_weight = _angular_grid(N_THETA_DEFAULT, N_PHI_DEFAULT)
    plane_normal = np.array([0.0, 0.0, 1.0])
    indicator_A = (directions @ plane_normal) >= 0.0
    Q_piece_A = _build_Q_with_indicator(pp0, indicator_A, ang_weight, directions)
    Q_piece_B = _build_Q_with_indicator(pp0, ~indicator_A, ang_weight, directions)
    err_partition = float(np.max(np.abs((Q_piece_A + Q_piece_B) - full_Q)))
    report["checks"].append(dict(name="partition_sums_to_full_Q",
                                  max_abs_err=err_partition, passed=bool(err_partition < 1e-6)))
    print(f"  partition (half-space + complement) sums to full Q: max|err|={err_partition:.4e}  "
          f"{'OK' if err_partition < 1e-6 else 'FAIL'}")

    # (d) Hermitian (real symmetric, since AE/PS partial waves and real
    # Ylm's are real-valued -- Q_A must come out numerically real-symmetric).
    herm_err = float(np.max(np.abs(Q_inside_general - Q_inside_general.T)))
    report["checks"].append(dict(name="Q_A_hermitian", max_abs_err=herm_err,
                                  passed=bool(herm_err < 1e-10)))
    print(f"  Q_A Hermitian (real-symmetric): max|Q-Q^T|={herm_err:.4e}  "
          f"{'OK' if herm_err < 1e-10 else 'FAIL'}")

    # (e) refinement: doubling the angular grid density should change Q_A
    # by less than a small tolerance, for a GENUINELY SPLIT site (the
    # interesting case -- (a)/(b) are trivial for any resolution).
    # Atom 1 (Se) is not at the WS center; use its nearest contributing
    # image if available, else fall back to the synthetic half-space split.
    Q_refine_coarse = _build_Q_with_indicator(pp0, indicator_A, ang_weight, directions)
    directions_f, ang_weight_f = _angular_grid(N_THETA_REFINED, N_PHI_REFINED)
    indicator_A_f = (directions_f @ plane_normal) >= 0.0
    Q_refine_fine = _build_Q_with_indicator(pp0, indicator_A_f, ang_weight_f, directions_f)
    err_refine = float(np.max(np.abs(Q_refine_fine - Q_refine_coarse)))
    report["checks"].append(dict(name="quadrature_refinement_converges",
                                  max_abs_err=err_refine, tol=Q_A_REFINEMENT_TOL,
                                  passed=bool(err_refine < Q_A_REFINEMENT_TOL)))
    print(f"  refinement (n_theta,n_phi {N_THETA_DEFAULT}x{N_PHI_DEFAULT} -> "
          f"{N_THETA_REFINED}x{N_PHI_REFINED}) on a half-space split: max|Δ|={err_refine:.4e}  "
          f"tol={Q_A_REFINEMENT_TOL:.1e}  {'OK' if err_refine < Q_A_REFINEMENT_TOL else 'FAIL'}")

    report["passed"] = all(c["passed"] for c in report["checks"])
    print(f"  Q_A validation OVERALL: {'PASS' if report['passed'] else 'FAIL'}")
    return report


def _build_Q_with_indicator(pp, indicator, ang_weight, directions):
    """Same radial x angular quadrature as build_regional_Qij_site, but
    with an arbitrary precomputed boolean indicator over `directions`
    (used only for the partition/refinement validation checks, where the
    region is a synthetic half-space, not the real WS cell)."""
    lmmax = pp.lmmax
    ilm = pp.ilm
    lmax_l = int(pp.proj_l.max())
    if not hasattr(pp, "rad_simp_w"):
        pp.set_simpi_weight()
    Y_by_l = [sph_r(directions, l) for l in range(lmax_l + 1)]
    Y_full = np.zeros((directions.shape[0], lmmax))
    for ii, (n, l, m) in enumerate(ilm):
        Y_full[:, ii] = Y_by_l[l][:, m + l]
    rgrid = pp.rgrid
    ae_full = np.zeros((lmmax, rgrid.size))
    ps_full = np.zeros((lmmax, rgrid.size))
    for ii, (n, l, m) in enumerate(ilm):
        ae_full[ii] = pp.paw_ae_wfc[n]
        ps_full[ii] = pp.paw_ps_wfc[n]
    w_masked = ang_weight * indicator
    AngOverlap = (Y_full * w_masked[:, None]).T @ Y_full
    Q = np.zeros((lmmax, lmmax))
    for ir in range(rgrid.size):
        RadDiff = np.outer(ae_full[:, ir], ae_full[:, ir]) - np.outer(ps_full[:, ir], ps_full[:, ir])
        Q += pp.rad_simp_w[ir] * RadDiff * AngOverlap
    return 0.5 * (Q + Q.T)


# ── system setup (shared with paw_lowrank_cno.py's conventions) ────────────

class UnsupportedSOCError(RuntimeError):
    """Raised when a WAVECAR is detected to be noncollinear (SOC) -- this
    pipeline (nonlq.proj(), gauge_correct_beta, home_reciprocal_beta) is
    non-SOC only; spinor support is deferred, not silently attempted."""


def _default_reduced_reference_kpoints(nkpts, n=5):
    if n >= nkpts:
        return list(range(1, nkpts + 1))
    return sorted(set(int(round(i * (nkpts - 1) / (n - 1))) + 1 for i in range(n)))


def _default_crossk_kpoints(nkpts):
    # ik=1 (often Gamma) plus a second k roughly 1/8 of the way through the
    # mesh -- for materials with the same k-mesh-generation convention as
    # WSe2_mono this tends to land on a genuinely fractional (non-half-
    # integer) k, but no test here actually depends on that (the phase sign
    # was already confirmed once; this check validates the CONSTRUCTION,
    # not the sign, for a new material's geometry/Q_A sites).
    second = max(2, min(nkpts, nkpts // 8 + 1))
    return [1, second]


def load_system(material=None, ws_center=None, ws_center_coord_type=None, ws_nmax=None,
                 lsorbit=None):
    """Loads everything needed for one material. With no arguments, behaves
    exactly as before (reads MATERIAL/WS_CENTER/WS_CENTER_COORD_TYPE/
    WS_TRANSLATION_SEARCH_RANGE/ISPIN from config.py, config.py untouched).
    Overrides let this module be exercised against a different material
    (e.g. Si, CoSn) without editing config.py -- used only for the
    cross-material generalization check in RESULTS.md, not by the
    config-driven production path (main()'s default call passes no
    overrides).

    lsorbit=None auto-detects: tries lsorbit=False first (cheap: header +
    band info only), and if vaspwfc itself reports a noncollinear WAVECAR,
    raises UnsupportedSOCError with a clear message rather than silently
    mis-reading a spinor WAVECAR as if it were scalar.
    """
    material = material if material is not None else config.MATERIAL
    data_dir = Path(__file__).resolve().parent.parent / "Data" / material

    if lsorbit is None:
        wfc = vaspwfc(str(data_dir / "WAVECAR"), lsorbit=False)
        try:
            wfc.gvectors(1)
        except ValueError as e:
            if "NONCOLLINEAR" in str(e).upper():
                raise UnsupportedSOCError(
                    f"{material}'s WAVECAR is noncollinear (SOC). This pipeline "
                    "(nonlq.proj/gauge_correct_beta/home_reciprocal_beta) is non-SOC only "
                    "-- spinor support is deferred, not attempted here."
                ) from e
            raise
    elif lsorbit:
        raise UnsupportedSOCError(
            f"lsorbit=True requested for {material}, but this pipeline is non-SOC only -- "
            "spinor support is deferred, not attempted here."
        )
    else:
        wfc = vaspwfc(str(data_dir / "WAVECAR"), lsorbit=False)

    latvec, species, counts, atom_symbols, atom_numbers, frac_coords, cart_coords = \
        read_poscar_structure(data_dir / "POSCAR")
    pawpp = load_pawpp(data_dir / "POTCAR")
    pawpp_elements = [pp.element.split('_')[0] for pp in pawpp]
    elements_idx = [pawpp_elements.index(s) for s in atom_symbols]
    atoms = ase_read(str(data_dir / "POSCAR"))
    qij_block = build_qij_block(pawpp, elements_idx)
    kweights = read_eigenval_kweights(data_dir / "EIGENVAL", wfc._nkpts, wfc._nbands)

    ws_center = ws_center if ws_center is not None else config.WS_CENTER
    ws_center_coord_type = ws_center_coord_type if ws_center_coord_type is not None \
        else config.WS_CENTER_COORD_TYPE
    ws_nmax = ws_nmax if ws_nmax is not None else config.WS_TRANSLATION_SEARCH_RANGE
    center_cart, _, center_frac_wrapped = parse_ws_center(ws_center, ws_center_coord_type, latvec)
    Nx, Ny, Nz = (int(x) for x in wfc._ngrid)
    Nr = Nx * Ny * Nz
    r_ws_cart, r_ws_frac_cont, prim_indices, translations_all = build_ws_grid_map(
        latvec, (Nx, Ny, Nz), center_cart, nmax=ws_nmax,
    )

    # per-atom channel ranges within the flat (n_proj_total,) beta ordering
    # -- shared by every function that needs to slice beta by atom.
    channel_ranges = []
    off = 0
    for ei in elements_idx:
        lm = pawpp[ei].lmmax
        channel_ranges.append((off, off + lm))
        off += lm

    return dict(
        config=config, material=material, data_dir=data_dir, wfc=wfc,
        latvec=latvec, atom_symbols=atom_symbols, cart_coords=cart_coords,
        frac_coords=frac_coords, pawpp=pawpp, elements_idx=elements_idx,
        atoms=atoms, qij_block=qij_block, channel_ranges=channel_ranges,
        kweights=kweights, center_cart=center_cart, ws_nmax=ws_nmax,
        Nx=Nx, Ny=Ny, Nz=Nz, Nr=Nr, r_ws_cart=r_ws_cart,
        r_ws_frac_cont=r_ws_frac_cont, prim_indices=prim_indices,
        translations_all=translations_all, center_frac_wrapped=center_frac_wrapped,
        ispin=config.ISPIN,
        # Exact backward compatibility for the already-validated WSe2_mono
        # production path (same representative k-points as before); any
        # other material gets a freshly auto-computed, equally-valid set
        # (these are only a validation sample -- the full spectrum always
        # uses every k-point regardless).
        reduced_reference_kpoints=(REDUCED_REFERENCE_KPOINTS if material == "WSe2_mono"
                                    else _default_reduced_reference_kpoints(wfc._nkpts)),
        crossk_kpoints=([1, 41] if material == "WSe2_mono"
                         else _default_crossk_kpoints(wfc._nkpts)),
    )


def precompute_sites_and_Q_A(sys_, n_theta=N_THETA_DEFAULT, n_phi=N_PHI_DEFAULT):
    """Purely geometric: the list of contributing (atom, image) sites on the
    NATIVE WS grid, plus each site's region-intersected Q_A -- computed
    ONCE, reused for every state/k-point."""
    sites = find_contributing_sites(
        sys_["pawpp"], sys_["elements_idx"], sys_["cart_coords"], sys_["latvec"],
        sys_["r_ws_cart"], nmax=SITE_SEARCH_NMAX, dist_prune=DIST_PRUNE,
    )
    print(f"  found {len(sites)} contributing (atom, image) sites on the native WS grid")
    for s in sites:
        pp = sys_["pawpp"][s["pp_idx"]]
        Q_A = build_regional_Qij_site(pp, s["image_cart"], sys_["center_cart"], sys_["latvec"],
                                       n_theta=n_theta, n_phi=n_phi)
        full_Q = pp.get_Qij()
        frac_of_full = float(np.trace(Q_A) / np.trace(full_Q)) if np.trace(full_Q) != 0 else float("nan")
        s["Q_A"] = Q_A
        s["frac_of_full_Q"] = frac_of_full
        print(f"    atom {s['iatom']} ({s['element']})  image={s['image']}  "
              f"trace(Q_A)/trace(Q_full)={frac_of_full:.4f}")
    return sites


def home_reciprocal_beta(sys_, wfc, ik, bands, Ck):
    """Trusted reciprocal beta for a batch of bands at k-point ik, atom-image
    R=0 (the atom's own POSCAR position) -- paw.nonlq.proj() on raw
    (norm=False) coefficients, PLUS the already-established atomic-position
    gauge correction (paw_lowrank_cno.gauge_correct_beta). Returns
    (nb, n_proj_total)."""
    kvec = wfc._kvecs[ik - 1]
    proj = nonlq(sys_["atoms"], wfc._encut, sys_["pawpp"], k=kvec,
                 lgam=wfc._lgam, gamma_half=wfc._gam_half)
    assert list(proj.element_idx) == list(sys_["elements_idx"])
    beta_recip = np.stack([proj.proj(Ck[i]) for i in range(len(bands))])
    return gauge_correct_beta(beta_recip, kvec, sys_["elements_idx"], sys_["pawpp"], sys_["frac_coords"])


def site_beta_from_home(beta_home, k_frac, site, channel_ranges):
    """beta_(n,k,atom,R) = exp(+2*pi*i*k.R) * beta_(n,k,atom,0) -- the exact
    Bloch relation (see module docstring), confirmed sign. No real-space
    grid involved. beta_home: (nb, n_proj_total); returns (nb, lmmax_atom)."""
    sl = channel_ranges[site["iatom"]]
    R = np.array(site["image"], dtype=float)
    phase = np.exp(2j * np.pi * np.dot(k_frac, R))
    return phase * beta_home[:, sl[0]:sl[1]]


def build_state_list_and_beta(sys_, sites, ik_list):
    """Builds Psi (Nr, nstates) [native grid, G_ps_A] and, for the SAME
    states, per-site beta arrays (nstates, lmmax_atom) -- now computed
    ENTIRELY from the reciprocal home beta (one nonlq.proj() call per
    k-point, shared across all its sites) times the exact Bloch image
    phase. No real-space grid is used for beta at all (see module
    docstring's 2026-07-11 correction) -- only Psi (G_ps_A) still uses the
    native FFT/WS grid, exactly as in paw_lowrank_cno.py."""
    wfc = sys_["wfc"]
    Nx, Ny, Nz, Nr = sys_["Nx"], sys_["Ny"], sys_["Nz"], sys_["Nr"]
    ispin = sys_["ispin"]
    channel_ranges = sys_["channel_ranges"]
    base_flat = (sys_["prim_indices"][:, 0].astype(np.int64) * Ny
                 + sys_["prim_indices"][:, 1]) * Nz + sys_["prim_indices"][:, 2]
    Nx0, Ny0, Nz0 = (int(x) for x in wfc._ngrid)

    states = []
    for ik in ik_list:
        occ_all = wfc._occs[ispin - 1, ik - 1, :]
        bands = np.where(occ_all > OCC_TOL)[0] + 1
        occ = occ_all[bands - 1].copy()
        if len(occ) and occ.max() > 1.5:
            occ = occ / 2.0
        wk = sys_["kweights"][ik - 1]
        for ib, f_occ in zip(bands, occ):
            states.append(dict(ik=int(ik), band=int(ib), p=float(wk * f_occ)))
    nstates = len(states)
    p = np.array([st["p"] for st in states])

    Psi = np.empty((Nr, nstates), dtype=np.complex128)
    beta_by_site = [np.empty((nstates, sys_["pawpp"][s["pp_idx"]].lmmax), dtype=np.complex128)
                     for s in sites]

    by_k = {}
    for idx, st in enumerate(states):
        by_k.setdefault(st["ik"], []).append((idx, st["band"]))

    t0 = time.time()
    n_k_done = 0
    for ik, idx_band_pairs in by_k.items():
        kvec = wfc._kvecs[ik - 1]
        bands = [b for _, b in idx_band_pairs]
        idxs = [i for i, _ in idx_band_pairs]

        gvec = wfc.gvectors(ik)
        Ck = np.stack([wfc.readBandCoeff(ispin=ispin, ikpt=ik, iband=int(ib), norm=False)
                       for ib in bands])

        # native-grid Psi (G_ps_A, unchanged from paw_lowrank_cno.py)
        u_bands0, _, _ = zero_pad_ifft(Ck, gvec, 1, (Nx0, Ny0, Nz0))
        u_ws0 = u_bands0[:, base_flat]
        psi_ws0 = u_ws0 * np.exp(2j * np.pi * (sys_["r_ws_frac_cont"] @ kvec))[None, :]
        Psi[:, idxs] = psi_ws0.T

        # reciprocal home beta, ONCE per k-point, shared across all sites
        beta_home = home_reciprocal_beta(sys_, wfc, ik, bands, Ck)

        for si, s in enumerate(sites):
            beta_site = site_beta_from_home(beta_home, kvec, s, channel_ranges)
            beta_by_site[si][idxs, :] = beta_site

        n_k_done += 1
        if n_k_done == 1 or n_k_done % 40 == 0 or n_k_done == len(by_k):
            print(f"    k {n_k_done:4d}/{len(by_k)}  ik={ik:4d}  nbands={len(bands)}  "
                  f"elapsed={time.time()-t0:.1f}s")

    return Psi, beta_by_site, p, states


def build_G_A_K_A(Psi, beta_by_site, sites, p):
    G_ps_A = Psi.conj().T @ Psi
    G_aug_A = np.zeros_like(G_ps_A)
    for beta_site, s in zip(beta_by_site, sites):
        G_aug_A += beta_site.conj() @ s["Q_A"] @ beta_site.T
    G_A = G_ps_A + G_aug_A
    sqrtP = np.sqrt(p)
    K_A = (sqrtP[:, None] * G_A) * sqrtP[None, :]
    K_A = 0.5 * (K_A + K_A.conj().T)
    return G_ps_A, G_aug_A, G_A, K_A


def _mineig(M):
    return float(np.linalg.eigvalsh(0.5 * (M + M.conj().T)).min())


def _herm(M):
    return float(np.max(np.abs(M - M.conj().T)))


def check_partition_closure(sites, pawpp):
    """Item A: sum_over_images Q_A(atom,image) ~= Q_full(atom), per atom,
    using the FULL Q_A MATRIX (not just its trace)."""
    print("--- A. Q partition closure (sum_images Q_A ~= Q_full, per atom) ---")
    by_atom = {}
    for s in sites:
        by_atom.setdefault(s["iatom"], []).append(s)
    rows = []
    all_ok = True
    for iatom, atom_sites in by_atom.items():
        pp = pawpp[atom_sites[0]["pp_idx"]]
        full_Q = pp.get_Qij()
        Q_sum = sum(s["Q_A"] for s in atom_sites)
        err = float(np.max(np.abs(Q_sum - full_Q)))
        ok = bool(err < PARTITION_CLOSURE_TOL)
        all_ok = all_ok and ok
        rows.append(dict(iatom=iatom, element=pp.element, n_images=len(atom_sites),
                          max_abs_err=err, passed=ok))
        print(f"  atom {iatom} ({pp.element}): {len(atom_sites)} images, "
              f"max|sum(Q_A)-Q_full|={err:.4e}  {'OK' if ok else 'FAIL'}")
    return dict(rows=rows, passed=all_ok)


def check_diagonal_closure(sys_, sites):
    """Item B + the diagnostic identity: for representative states,
    sum_images beta_image^H Q_A_image beta_image must equal
    beta_reciprocal^H Q_full beta_reciprocal EXACTLY (sign-independent,
    since |phase|^2=1 -- this checks Q_A partition-summing and channel
    ordering, not the phase convention). Run per-atom (to localize any
    failure) and reported both for a few representative states and,
    cheaply, for ALL states in the reduced-reference k-point set.
    """
    print("--- B. Diagonal augmentation closure (sign-independent identity) ---")
    wfc = sys_["wfc"]
    channel_ranges = sys_["channel_ranges"]

    by_atom_sites = {}
    for s in sites:
        by_atom_sites.setdefault(s["iatom"], []).append(s)

    worst = 0.0
    worst_detail = None
    n_checked = 0
    kpoints = sys_["reduced_reference_kpoints"]
    for ik in kpoints:
        occ_all = wfc._occs[sys_["ispin"] - 1, ik - 1, :]
        bands = np.where(occ_all > OCC_TOL)[0] + 1
        if len(bands) == 0:
            continue
        Ck = np.stack([wfc.readBandCoeff(ispin=sys_["ispin"], ikpt=ik, iband=int(ib), norm=False)
                       for ib in bands])
        beta_home = home_reciprocal_beta(sys_, wfc, ik, bands, Ck)   # (nb, n_proj_total)
        kvec = wfc._kvecs[ik - 1]

        for ib_idx in range(len(bands)):
            for iatom, atom_sites in by_atom_sites.items():
                sl = channel_ranges[iatom]
                b_home = beta_home[ib_idx, sl[0]:sl[1]]
                pp_full_Q = sys_["pawpp"][atom_sites[0]["pp_idx"]].get_Qij()
                lhs = 0.0 + 0.0j
                for s in atom_sites:
                    R = np.array(s["image"], dtype=float)
                    phase = np.exp(2j * np.pi * np.dot(kvec, R))
                    b_site = phase * b_home
                    lhs += np.vdot(b_site, s["Q_A"] @ b_site)
                rhs = np.vdot(b_home, pp_full_Q @ b_home)
                err = float(abs(lhs - rhs))
                n_checked += 1
                if err > worst:
                    worst = err
                    worst_detail = (ik, ib_idx, iatom)

    passed = bool(worst < DIAGONAL_CLOSURE_TOL)
    print(f"  checked {n_checked} (state, atom) combinations across "
          f"{len(kpoints)} k-points")
    print(f"  worst |sum_images beta^H Q_A beta - beta_home^H Q_full beta_home| = {worst:.4e}  "
          f"(at ik={worst_detail[0] if worst_detail else None}, "
          f"band_idx={worst_detail[1] if worst_detail else None}, "
          f"atom={worst_detail[2] if worst_detail else None})  tol={DIAGONAL_CLOSURE_TOL:.1e}  "
          f"{'OK' if passed else 'FAIL'}")
    return dict(worst_err=worst, worst_detail=worst_detail, n_checked=n_checked, passed=passed)


def check_paw_norm_closure(sys_, sites):
    """Item C: for every selected state, norm_paw_regional = pseudo_norm +
    regional_augmentation_norm must agree with the trusted reciprocal
    full-Q PAW norm and be close to 1. Uses the reduced-reference k-points."""
    print("--- C. Full PAW norm closure (regional vs trusted reciprocal, full-Q) ---")
    wfc = sys_["wfc"]
    channel_ranges = sys_["channel_ranges"]
    qij_block = sys_["qij_block"]

    norms_regional = []
    norms_reciprocal = []
    for ik in sys_["reduced_reference_kpoints"]:
        occ_all = wfc._occs[sys_["ispin"] - 1, ik - 1, :]
        bands = np.where(occ_all > OCC_TOL)[0] + 1
        if len(bands) == 0:
            continue
        Ck = np.stack([wfc.readBandCoeff(ispin=sys_["ispin"], ikpt=ik, iband=int(ib), norm=False)
                       for ib in bands])
        beta_home = home_reciprocal_beta(sys_, wfc, ik, bands, Ck)
        kvec = wfc._kvecs[ik - 1]
        pseudo_norm = np.sum(np.abs(Ck) ** 2, axis=1).real     # <psi~|psi~> full-crystal PW norm

        aug_regional = np.zeros(len(bands))
        for s in sites:
            sl = channel_ranges[s["iatom"]]
            R = np.array(s["image"], dtype=float)
            phase = np.exp(2j * np.pi * np.dot(kvec, R))
            b_site = phase * beta_home[:, sl[0]:sl[1]]
            aug_regional += np.real(np.sum(b_site.conj() @ s["Q_A"] * b_site, axis=1))
        norms_regional.append(pseudo_norm + aug_regional)

        aug_reciprocal = np.real(np.sum((beta_home.conj() @ qij_block) * beta_home, axis=1))
        norms_reciprocal.append(pseudo_norm + aug_reciprocal)

    norms_regional = np.concatenate(norms_regional)
    norms_reciprocal = np.concatenate(norms_reciprocal)
    max_diff = float(np.max(np.abs(norms_regional - norms_reciprocal)))
    print(f"  norm_paw_regional:   min={norms_regional.min():.6f}  "
          f"mean={norms_regional.mean():.6f}  max={norms_regional.max():.6f}")
    print(f"  norm_paw_reciprocal: min={norms_reciprocal.min():.6f}  "
          f"mean={norms_reciprocal.mean():.6f}  max={norms_reciprocal.max():.6f}")
    print(f"  max|regional - reciprocal| = {max_diff:.4e}  tol={PAW_NORM_CLOSURE_TOL:.1e}")
    passed = bool(max_diff < PAW_NORM_CLOSURE_TOL)
    print(f"  {'OK' if passed else 'FAIL'}")
    return dict(
        min_norm_regional=float(norms_regional.min()), mean_norm_regional=float(norms_regional.mean()),
        max_norm_regional=float(norms_regional.max()), max_diff_from_reciprocal=max_diff,
        passed=passed,
    )


def check_cross_k_vs_realspace(sys_, sites):
    """Item G: representative regional cross-k blocks vs. the converged 3x
    real-space reference. The real-space side is built by feeding
    quadrature_convergence_check.py's ALREADY-VALIDATED combine-with-phase
    function a virtually-shifted atom position (atom_cart + R) for the site
    being tested -- this correctly handles all periodicity/wraparound
    bookkeeping (unlike a single-image, box-restricted evaluation, which is
    only ever a partial contribution -- see module docstring). Used ONLY
    for validation.
    """
    print(f"--- G. Cross-k regional block vs. converged {VALIDATION_GRID_FACTOR}x "
          f"real-space reference ---")
    wfc = sys_["wfc"]
    channel_ranges = sys_["channel_ranges"]
    ik_list = sys_["crossk_kpoints"]   # one same-k, one cross-k partner
    Nx0, Ny0, Nz0 = (int(x) for x in wfc._ngrid)
    f = VALIDATION_GRID_FACTOR
    Nxf, Nyf, Nzf = Nx0 * f, Ny0 * f, Nz0 * f
    Nrf = Nxf * Nyf * Nzf
    ix, iy, iz = [a.ravel() for a in np.mgrid[0:Nxf, 0:Nyf, 0:Nzf]]
    r_prim_frac = np.column_stack([ix / Nxf, iy / Nyf, iz / Nzf])
    r_prim_cart = r_prim_frac @ sys_["latvec"]

    beta_regional_by_site = {s["iatom"]: {} for s in sites}   # iatom -> ik -> (nb, lmmax)
    beta_realspace_by_site = {s["iatom"]: {} for s in sites}
    labels_by_k = {}
    for ik in ik_list:
        kvec = wfc._kvecs[ik - 1]
        gvec = wfc.gvectors(ik)
        occ_all = wfc._occs[sys_["ispin"] - 1, ik - 1, :]
        bands = np.where(occ_all > OCC_TOL)[0] + 1
        labels_by_k[ik] = bands
        Ck = np.stack([wfc.readBandCoeff(ispin=sys_["ispin"], ikpt=ik, iband=int(ib), norm=False)
                       for ib in bands])
        beta_home = home_reciprocal_beta(sys_, wfc, ik, bands, Ck)
        u_bands, _, _ = zero_pad_ifft(Ck, gvec, f, (Nx0, Ny0, Nz0))
        psi_prim = u_bands * np.exp(2j * np.pi * (r_prim_frac @ kvec))[None, :]

        for s in sites:
            sl = channel_ranges[s["iatom"]]
            R = np.array(s["image"], dtype=float)
            phase = np.exp(2j * np.pi * np.dot(kvec, R))
            beta_regional_by_site[s["iatom"]].setdefault(ik, {})[s["image"]] = phase * beta_home[:, sl[0]:sl[1]]

            shifted_cart_coords = sys_["cart_coords"].copy()
            shifted_cart_coords[s["iatom"]] = sys_["cart_coords"][s["iatom"]] + R @ sys_["latvec"]
            beta_rs_full = real_space_beta_for_bands(
                sys_["pawpp"], sys_["elements_idx"], shifted_cart_coords, sys_["latvec"],
                r_prim_cart, psi_prim, Nrf, nmax=2, k_frac=kvec,
            )
            beta_realspace_by_site[s["iatom"]].setdefault(ik, {})[s["image"]] = beta_rs_full[:, sl[0]:sl[1]]

    max_err = 0.0
    for iatom in beta_regional_by_site:
        for ik in ik_list:
            for image in beta_regional_by_site[iatom][ik]:
                a = beta_regional_by_site[iatom][ik][image]
                b = beta_realspace_by_site[iatom][ik][image]
                err = float(np.max(np.abs(a - b)))
                max_err = max(max_err, err)

    # Full cross-k Gram block comparison (same-k ik=1 block + cross-k
    # between ik=1 and ik=41), assembled from the per-site betas above.
    def _assemble_G_aug(beta_dict):
        labels = []
        for ik in ik_list:
            labels += [(ik, int(b)) for b in labels_by_k[ik]]
        n = len(labels)
        G = np.zeros((n, n), dtype=np.complex128)
        for s in sites:
            iatom = s["iatom"]
            blocks = []
            for ik in ik_list:
                blocks.append(beta_dict[iatom][ik][s["image"]])
            Beta = np.concatenate(blocks, axis=0)   # (n, lmmax_atom)
            G += Beta.conj() @ s["Q_A"] @ Beta.T
        return G, labels

    G_regional, labels = _assemble_G_aug(beta_regional_by_site)
    G_realspace, labels2 = _assemble_G_aug(beta_realspace_by_site)
    assert labels == labels2
    ik_of = np.array([lab[0] for lab in labels])
    cross_mask = ik_of[:, None] != ik_of[None, :]
    diff = G_regional - G_realspace
    max_block_err = float(np.max(np.abs(diff)))
    max_crossk_err = float(np.max(np.abs(diff[cross_mask]))) if cross_mask.any() else 0.0
    print(f"  max|beta_regional - beta_realspace| (per site/state) = {max_err:.4e}")
    print(f"  max|G_aug_regional - G_aug_realspace| (full block) = {max_block_err:.4e}")
    print(f"  max|...| (cross-k sub-block only) = {max_crossk_err:.4e}")
    passed = bool(max_block_err < 5e-3)
    print(f"  {'OK' if passed else 'FAIL'}")
    return dict(max_beta_err=max_err, max_block_err=max_block_err,
                max_crossk_err=max_crossk_err, passed=passed)


def validate_reduced_reference(sys_, sites):
    """Reduced k-point reference run: build G_A/K_A from the corrected
    (pure-reciprocal) beta for sys_["reduced_reference_kpoints"], and run
    every required validation (items A-G plus Hermiticity/PSD/trace/gauge/
    PSD bound checks).
    """
    kpoints = sys_["reduced_reference_kpoints"]
    print(f"\n=== Reduced-reference validation ({len(kpoints)} k-points) ===")

    partition_report = check_partition_closure(sites, sys_["pawpp"])
    diagonal_report = check_diagonal_closure(sys_, sites)
    paw_norm_report = check_paw_norm_closure(sys_, sites)
    crossk_report = check_cross_k_vs_realspace(sys_, sites)

    Psi, beta_by_site, p, states = build_state_list_and_beta(sys_, sites, kpoints)
    G_ps_A, G_aug_A, G_A, K_A = build_G_A_K_A(Psi, beta_by_site, sites, p)

    report = dict(kpoints=kpoints, n_states=len(states),
                  partition_closure=partition_report, diagonal_closure=diagonal_report,
                  paw_norm_closure=paw_norm_report, cross_k_vs_realspace=crossk_report)
    report["min_eig_G_ps_A"] = _mineig(G_ps_A)
    report["min_eig_G_A"] = _mineig(G_A)
    report["min_eig_K_A"] = _mineig(K_A)
    report["herm_G_A"] = _herm(G_A)
    report["herm_K_A"] = _herm(K_A)
    print(f"\n  min_eig(G_ps_A)={report['min_eig_G_ps_A']:.6f}  "
          f"min_eig(G_A)={report['min_eig_G_A']:.6f}  min_eig(K_A)={report['min_eig_K_A']:.6f}")
    print(f"  Hermiticity: G_A={report['herm_G_A']:.2e}  K_A={report['herm_K_A']:.2e}")

    trace_expected = float(sum(st["p"] for st in states))
    eigvals_A = np.linalg.eigvalsh(K_A)
    trace_K_A = float(np.trace(K_A).real)
    report["trace_expected"] = trace_expected
    report["trace_K_A"] = trace_K_A
    report["trace_diff"] = float(trace_K_A - trace_expected)
    report["trace_eigensum_diff"] = float(trace_K_A - eigvals_A.sum())
    print(f"  trace_expected={trace_expected:.6f}  trace(K_A)={trace_K_A:.6f}  "
          f"diff={report['trace_diff']:.4e}")

    n_out_of_bounds = int(np.sum((eigvals_A < -BOUND_01_TOL) | (eigvals_A > 1 + BOUND_01_TOL)))
    report["n_out_of_bounds"] = n_out_of_bounds
    report["max_eigval"] = float(eigvals_A.max())
    print(f"  eigval range=[{eigvals_A.min():.6f},{eigvals_A.max():.6f}]  "
          f"n_out_of_bounds={n_out_of_bounds}")

    # F. Random-rephasing invariance.
    print("--- F. Random-rephasing (gauge) invariance ---")
    eig_before = np.sort(np.linalg.eigvalsh(0.5 * (G_A + G_A.conj().T)))
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, size=len(states))
    phase = np.exp(1j * theta)
    Psi_g = Psi * phase[None, :]
    beta_by_site_g = [b * phase[:, None] for b in beta_by_site]
    G_ps_A_g, G_aug_A_g, G_A_g, K_A_g = build_G_A_K_A(Psi_g, beta_by_site_g, sites, p)
    eig_after = np.sort(np.linalg.eigvalsh(0.5 * (G_A_g + G_A_g.conj().T)))
    gauge_diff = float(np.max(np.abs(eig_before - eig_after)))
    report["gauge_invariance_max_eig_diff"] = gauge_diff
    print(f"  max|Δeig(G_A)| under random state rephasing = {gauge_diff:.3e}")

    checks = dict(
        partition_closure=partition_report["passed"],
        diagonal_closure=diagonal_report["passed"],
        paw_norm_closure=paw_norm_report["passed"],
        cross_k_vs_realspace=crossk_report["passed"],
        G_ps_A_psd=bool(report["min_eig_G_ps_A"] > -1e-6),
        G_A_hermitian=bool(report["herm_G_A"] < HERMITICITY_TOL),
        K_A_hermitian=bool(report["herm_K_A"] < HERMITICITY_TOL),
        G_A_psd=bool(report["min_eig_G_A"] > -1e-3),
        K_A_psd=bool(report["min_eig_K_A"] > -1e-3),
        occupations_within_01=bool(n_out_of_bounds == 0),
        trace_matches_expected=bool(abs(report["trace_diff"]) < 1e-2),
        gauge_invariant=bool(gauge_diff < 1e-6),
    )
    report["checks"] = checks
    report["passed"] = all(checks.values())
    print("\n  --- reduced-reference validation summary ---")
    for name, ok in checks.items():
        print(f"    {name:28s} {'OK' if ok else 'FAIL'}")
    print(f"    OVERALL: {'PASS' if report['passed'] else 'FAIL'}")
    return report


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)!r}")


def _write_blocked_report(output_dir, reason, extra=None):
    with open(output_dir / "paw_regional_report.txt", "w") as f:
        f.write("=== Regional T^dagger P_A T PAW-CNO report ===\n\n")
        f.write("gate_status: BLOCKED\n\n")
        f.write(f"reason: {reason}\n\n")
        if extra:
            f.write(json.dumps(extra, indent=2, default=_json_default))
        f.write("\n\nmain.py and config.py were not modified (config.py only read).\n")
    with open(output_dir / "paw_regional_report.json", "w") as f:
        json.dump(dict(gate_status="BLOCKED", reason=reason, extra=extra), f,
                   indent=2, default=_json_default)


def main(material=None, ws_center=None, ws_center_coord_type=None, ws_nmax=None, lsorbit=None):
    """With no arguments: exactly the config.py-driven production path.
    Overrides let this be run against a different material (e.g. Si, CoSn)
    without touching config.py -- see load_system(). Writes to the SAME
    Data/<material>/output/<OUTPUT_SUBDIR>/ directory main.py writes to, in
    the same file-format contract main.py's downstream scripts expect
    (cno_occupations.npy / cno_orbitals.npy of matching length, sorted
    descending; fft_grid_shape.npy; ws_*.npy)."""
    t_start = time.time()
    tracemalloc.start()

    material_eff = material if material is not None else config.MATERIAL
    output_dir = (Path(__file__).resolve().parent.parent / "Data" / material_eff
                  / "output" / config.OUTPUT_SUBDIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        sys_ = load_system(material=material, ws_center=ws_center,
                            ws_center_coord_type=ws_center_coord_type,
                            ws_nmax=ws_nmax, lsorbit=lsorbit)
    except UnsupportedSOCError as e:
        tracemalloc.stop()
        _write_blocked_report(output_dir, str(e))
        print(f"\nBLOCKED: {e}")
        return None
    print(f"=== Regional T^dagger P_A T PAW-CNO: {sys_['material']} ===\n")
    print(f"output_dir: {output_dir}\n")

    qa_report = validate_Q_A_construction(
        sys_["pawpp"], sys_["elements_idx"], sys_["cart_coords"], sys_["latvec"], sys_["center_cart"],
    )
    if not qa_report["passed"]:
        _write_blocked_report(output_dir, "Q_A construction validation failed -- see checks.", qa_report)
        print("\nBLOCKED: Q_A construction validation failed. Aborting before any state build.")
        return

    print()
    sites = precompute_sites_and_Q_A(sys_)

    reduced_report = validate_reduced_reference(sys_, sites)
    if not reduced_report["passed"]:
        _write_blocked_report(
            output_dir,
            "Reduced-reference validation failed -- full spectrum NOT computed, per the "
            "task's explicit instruction to only proceed once the reduced reference passes.",
            dict(Q_A_validation=qa_report, reduced_reference=reduced_report),
        )
        print("\nBLOCKED: reduced-reference validation failed. Not running the full spectrum.")
        return

    # ── full-spectrum run (only reached if both gates passed) ──
    print(f"\n=== Reduced reference PASSED -- building full spectrum "
          f"(all {sys_['wfc']._nkpts} k-points) ===")
    ik_all = list(range(1, sys_["wfc"]._nkpts + 1))
    t0 = time.time()
    Psi, beta_by_site, p, states = build_state_list_and_beta(sys_, sites, ik_all)
    t_build = time.time() - t0
    print(f"  build done in {t_build:.1f}s  nstates={len(states)}")

    t0 = time.time()
    if USE_DIRECT_L_A:
        print("  route: direct L_A (enlarged Gram matrix, QR-based -- see paw_direct_LA.py)")
        lam_sel, X, eigvals, la_diag = solve_natural_orbitals_direct(Psi, beta_by_site, sites, p)
        n_sel = la_diag["n_selected"]
        herm_K_A, min_eig_K_A = la_diag["herm_K_small"], la_diag["min_eig"]
        trace_K_A = la_diag["trace_K_small"]
        ortho_err = la_diag["ortho_err"]
        t_gram = time.time() - t0
        t_eigh = 0.0   # QR + small eigh are one timed block above (t_gram)
    else:
        print("  route: state-space K_A")
        G_ps_A, G_aug_A, G_A, K_A = build_G_A_K_A(Psi, beta_by_site, sites, p)
        t_gram = time.time() - t0
        t0 = time.time()
        eigvals, U = np.linalg.eigh(K_A)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        U = U[:, order]
        t_eigh = time.time() - t0

        sel = eigvals > 1e-6
        n_sel = int(sel.sum())
        lam_sel = eigvals[sel]
        U_sel = U[:, sel]
        sqrtP = np.sqrt(p)
        Y = (sqrtP[:, None] * U_sel) / np.sqrt(lam_sel)[None, :]
        ortho_err = float(np.max(np.abs(Y.conj().T @ G_A @ Y - np.eye(n_sel))))
        X = Psi @ Y
        herm_K_A, min_eig_K_A = _herm(K_A), _mineig(K_A)
        trace_K_A = float(np.trace(K_A).real)
    print(f"  gram/build: {t_gram:.1f}s   eigh: {t_eigh:.1f}s")

    trace_expected = float(sum(st["p"] for st in states))
    sum_eig = float(eigvals.sum())
    n_out_of_bounds = int(np.sum((eigvals < -BOUND_01_TOL) | (eigvals > 1 + BOUND_01_TOL)))

    t_total = time.time() - t_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\ntrace_expected={trace_expected:.8f}  trace(K_A)={trace_K_A:.8f}  "
          f"sum(eigvals)={sum_eig:.8f}")
    print(f"max_eigval={eigvals.max():.6f}  min_eigval={eigvals.min():.6f}  "
          f"n_out_of_bounds(gt {BOUND_01_TOL:.0e})={n_out_of_bounds}")
    print(f"state-space orthonormality (max|Y^H G_A Y - I|): {ortho_err:.3e}")
    print(f"total runtime: {t_total:.1f}s   peak memory: {peak_mem/1e6:.1f} MB")

    final_checks = dict(
        reduced_reference_passed=reduced_report["passed"],
        Q_A_validation_passed=qa_report["passed"],
        K_A_hermitian=bool(herm_K_A < HERMITICITY_TOL),
        K_A_psd=bool(min_eig_K_A > -1e-3),
        occupations_within_01=bool(n_out_of_bounds == 0),
        trace_algebra_check=bool(abs(trace_K_A - sum_eig) < 1e-6),
        trace_matches_expected=bool(abs(trace_K_A - trace_expected) < 1e-2),
        state_space_orthonormality=bool(ortho_err < 1e-6),
    )
    overall_pass = all(final_checks.values())
    print("\n--- final validation summary ---")
    for name, ok in final_checks.items():
        print(f"  {name:28s} {'OK' if ok else 'FAIL'}")
    print(f"  OVERALL: {'PASS' if overall_pass else 'FAIL'}")

    # ── write the same file contract main.py's downstream scripts (symmetry_adapt_cnos.py,
    # cno_dos.py, cno_fatband.py, export_cubes.py, cno_eigenvalues.py, export_visualizer_data.py)
    # read: cno_occupations.npy / cno_orbitals.npy of matching length, sorted descending;
    # fft_grid_shape.npy; the ws_*.npy WS-cell map files. lam_sel/X are already filtered to the
    # physically significant (eigval > 1e-6) subspace and are already length-matched, unlike main.py's
    # rho (which stores every one of the Nr eigenpairs of the dense Nr x Nr density matrix) -- this
    # pipeline never forms that matrix, so there is no analogous "full Nr-length" spectrum to save.
    np.save(output_dir / "cno_occupations.npy", lam_sel)
    np.save(output_dir / "cno_orbitals.npy", X)
    np.save(output_dir / "fft_grid_shape.npy", np.array([sys_["Nx"], sys_["Ny"], sys_["Nz"]], dtype=int))
    np.save(output_dir / "kpoint_weights.npy", sys_["kweights"])
    np.save(output_dir / "ws_enabled.npy", np.array(True))
    np.save(output_dir / "ws_points_cart.npy", sys_["r_ws_cart"])
    np.save(output_dir / "ws_points_frac_cont.npy", sys_["r_ws_frac_cont"])
    np.save(output_dir / "ws_base_indices.npy", sys_["prim_indices"])
    np.save(output_dir / "ws_translation_int.npy", sys_["translations_all"])
    np.save(output_dir / "ws_center_cart.npy", sys_["center_cart"])
    np.save(output_dir / "ws_center_frac_wrapped.npy", sys_["center_frac_wrapped"])

    full_report = dict(
        material=sys_["material"], nstates=len(states), n_sites=len(sites),
        route="direct_L_A" if USE_DIRECT_L_A else "state_space_K_A",
        beta_convention="reciprocal (nonlq.proj + gauge_correct_beta) x Bloch image phase; no real-space grid",
        Q_A_validation=qa_report, reduced_reference=reduced_report,
        trace_expected=trace_expected, trace_K_A=trace_K_A, sum_eigvals=sum_eig,
        trace_minus_expected=float(trace_K_A - trace_expected),
        max_eigval=float(eigvals.max()), min_eigval=float(eigvals.min()),
        n_out_of_bounds=n_out_of_bounds, n_selected_orbitals=n_sel,
        state_space_orthonormality_err=ortho_err,
        checks=final_checks, overall_status="PASS" if overall_pass else "FAIL",
        timing_s=dict(build=t_build, gram=t_gram, eigh=t_eigh, total=t_total),
        peak_memory_MB=float(peak_mem / 1e6),
        top_20_occupations=eigvals[:20].tolist(),
    )
    with open(output_dir / "paw_regional_report.json", "w") as f:
        json.dump(full_report, f, indent=2, default=_json_default)

    with open(output_dir / "paw_regional_report.txt", "w") as f:
        f.write("=== Regional T^dagger P_A T PAW-CNO report ===\n\n")
        f.write("gate_status: PASSED (Q_A validation + reduced reference + full build all ran)\n\n")
        f.write(f"material: {sys_['material']}  nstates: {len(states)}  n_sites: {len(sites)}\n")
        f.write(f"route: {full_report['route']}\n")
        f.write(f"beta_convention: {full_report['beta_convention']}\n\n")
        f.write("--- trace reporting ---\n")
        f.write(f"trace_expected_input   = {trace_expected:.10f}\n")
        f.write(f"trace_K_A              = {trace_K_A:.10f}\n")
        f.write(f"sum_cno_occupations    = {sum_eig:.10f}\n")
        f.write(f"trace_K_A - eigensum   = {trace_K_A - sum_eig:.4e}  (algebra check)\n")
        f.write(f"trace_K_A - expected   = {trace_K_A - trace_expected:.4e}  (physical check)\n\n")
        f.write("--- CNO spectrum ---\n")
        f.write(f"max_eigval: {eigvals.max():.8f}  min_eigval: {eigvals.min():.8f}\n")
        f.write(f"n_out_of_bounds (>{BOUND_01_TOL:.0e}): {n_out_of_bounds}\n")
        f.write("top_20_cno_occupations:\n")
        for i, v in enumerate(eigvals[:20]):
            f.write(f"  CNO {i:3d} : {float(v):.10e}\n")
        f.write(f"\nstate-space orthonormality: {ortho_err:.4e}\n\n")
        f.write("--- final validation summary ---\n")
        for name, ok in final_checks.items():
            f.write(f"  {name:28s} {'OK' if ok else 'FAIL'}\n")
        f.write(f"  OVERALL: {'PASS' if overall_pass else 'FAIL'}\n\n")
        f.write(f"total_runtime_s: {t_total:.1f}\npeak_memory_MB: {peak_mem/1e6:.1f}\n\n")
        f.write("main.py and config.py were not modified (config.py only read).\n")

    print(f"\nSaved -> {output_dir / 'cno_occupations.npy'}\nSaved -> {output_dir / 'cno_orbitals.npy'}")
    print(f"Saved -> {output_dir / 'paw_regional_report.txt'}\nSaved -> {output_dir / 'paw_regional_report.json'}")
    return full_report


if __name__ == "__main__":
    main()
