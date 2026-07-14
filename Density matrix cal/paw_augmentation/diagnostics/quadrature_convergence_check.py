"""
quadrature_convergence_check.py -- post-processing diagnostic (no VASP
rerun, no Nr x Nr matrix assembly on enlarged grids) isolating whether the
~4% trace excess measured in output/paw_density_matrix_report.txt (see
RESULTS.md's 2026-07-10 update) comes from real-space quadrature error in
build_real_space_S, from the WS-cell atom-image/coordinate handling
specifically, or from something else (normalization / formula).

Method
------
The existing (native) WAVECAR plane-wave coefficients Cg(G) already contain
ALL the frequency content of the pseudo wavefunction psi~_n (band-limited by
ENCUT) -- no VASP rerun can add anything new. Zero-padding Cg onto a larger
FFT grid and inverse-transforming is therefore EXACT band-limited
interpolation: it increases the density of real-space SAMPLE points of the
SAME continuous function, without changing the underlying physics. This lets
us test whether build_real_space_S's real-space quadrature (radial spline x
spherical harmonic, evaluated pointwise on the FFT grid) converges toward
the reciprocal-space reference (paw.nonlq.proj(), which uses no real-space
grid at all and is not subject to this error) as the sampling grid is
refined -- exactly a quadrature-convergence study, not a physics change.

For each grid factor f in {1, 2, 3} (native (11,11,73), 2x, 3x per axis):
  1. Zero-pad each representative occupied band's Cg onto (f*Nx0, f*Ny0,
     f*Nz0) and inverse-FFT with the CORRECT (grid-size-dependent) sqrt(Nr_f)
     normalization -- never reusing the native Nr (Nr_f is recomputed and
     threaded through every normalization at that grid factor).
  2. Compute beta_realspace = <p~_i|psi~_n> via a direct real-space
     projector sum (same radial splines / Qij / normalization convention as
     build_real_space_S's contrib = Bblock^T Qij Bblock / Nr, just applied
     to a band's wavefunction instead of the identity), at that grid's
     sample points, for BOTH:
       (a) primitive-cell box coordinates (ix/Nx_f, iy/Ny_f, iz/Nz_f) @ latvec
       (b) WS-mapped coordinates (ws_cell.build_ws_grid_map on the SAME
           f*Nx0 grid, config.py's WS_CENTER -- read-only)
  3. Compare against beta_reciprocal (paw.nonlq.proj(), grid-independent,
     computed once per k-point) -- max/RMS difference.
  4. Build the small (nb x nb) PAW-corrected band Gram matrix
     S_corrected = S_ps + beta^H Qij beta using beta_realspace (never an
     Nr x Nr matrix -- beta is (nb, n_proj_total), n_proj_total is a small
     handful of projector channels per atom), and report its diagonal
     mean/range and max off-diagonal, for both (a) and (b), plus the
     (grid-independent) reciprocal-space reference for comparison.

Real-space beta derivation (k-dependence -- important, see below)
-------------------------------------------------------------------------
beta_n,i = <p~_i|psi_n> is a property of a SPECIFIC Bloch state (n,k), unlike
S_hat = 1 + sum_i |p~_i> Qij <p~_j|, which is k-INDEPENDENT (build_real_space_
S's "no Bloch phase needed" is correct for THAT operator, and is not touched
by anything below). Expanding the projector integral over all space as a sum
over unit-cell images and using Bloch periodicity psi(r+R) = exp(2*pi*i*k.R)
* psi(r) (main.py's own sign convention: psi_nk(r) = exp(2*pi*i*k.r_frac) *
u_nk(r)) gives, for grid points r restricted to ONE cell/WS box:

    beta_n,i = integral_cell dr  psi_n(r) *
               sum_image  exp(-2*pi*i * k_frac . n_image) * p~_i(r - Rimg)^*

where Rimg = atom_position + n_image @ latvec runs over the same atom-image
translations build_real_space_S already loops over, and n_image is the
INTEGER lattice translation for that image (not its Cartesian version). Two
consequences, both required and both implemented below:
  (i)  the wavefunction samples fed into the real-space beta sum must be the
       FULL Bloch psi_n(r) = exp(2*pi*i*k.r_frac)*u_n(r), not the bare
       cell-periodic u_n(r);
  (ii) each atom-image's contribution must be weighted by the per-image
       scalar phase exp(-2*pi*i*k_frac.n_image) BEFORE summing over images.
Both vanish (=1) at k=0 (Gamma), which is why an EARLIER version of this
script (missing both) converged perfectly at ik=1 but showed a
grid-refinement-independent plateau at every non-Gamma k-point tested --
exactly the signature of a missing k-phase, not of quadrature. See
RESULTS.md's dated entry for the numbers from that first (buggy) run, kept
for the record rather than silently overwritten.

Does NOT modify main.py or config.py (config.py is only read, for MATERIAL /
WS_CENTER / WS_TRANSLATION_SEARCH_RANGE). Does NOT rerun VASP. Does NOT
build any Nr x Nr matrix at grid factor 2 or 3 (build_real_space_S is never
called here). Does NOT globally rescale S and does NOT clip occupations --
this script never touches the production S/D/eigenvalue pipeline at all.

Writes output/quadrature_convergence_report.{json,txt}.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "helper functions"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for paw_overlap (paw_augmentation/)
from ws_cell import read_poscar_structure, parse_ws_center, build_ws_grid_map  # noqa: E402

from vaspwfc import vaspwfc  # noqa: E402
from paw import nonlq  # noqa: E402
from sph_harm import sph_r  # noqa: E402
from paw_overlap import load_pawpp, build_qij_block, offdiag_maxabs  # noqa: E402
from ase.io import read as ase_read  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

GRID_FACTORS = [1, 2, 3]
KPOINT_FRACTIONS = [0.0, 0.5, 1.0]   # spread across the mesh, like the preflight gate
ATOM_IMAGE_NMAX_PRIM = 4
DIST_PRUNE = 16.0                    # Angstrom, same as build_real_space_S's default

# Convergence-classification thresholds -- see main()'s "interpretation" block.
CONVERGENCE_SHRINK_FACTOR = 3.0      # worst-case offdiag must shrink by >= this, native->3x
CLOSE_TO_RECIP_MULTIPLE = 100.0      # ...and land within this multiple of the (near-exact) reciprocal reference
CLOSE_TO_RECIP_ABS_FLOOR = 1e-3      # ...or at least below this absolute bar


# ── zero-padded (band-limited-interpolated) real-space sampling ────────────

def zero_pad_ifft(Ck, gvec, grid_factor, base_ngrid):
    """Zero-pad (nb, nG) plane-wave coefficients onto a grid_factor-times
    larger FFT grid and inverse-transform -- EXACT band-limited
    interpolation of the SAME pseudo wavefunction (no new physics, no VASP
    rerun). Uses the grid_factor-scaled sqrt(Nr) normalization -- the
    native Nr is never reused here.
    """
    Nx0, Ny0, Nz0 = base_ngrid
    Nx, Ny, Nz = Nx0 * grid_factor, Ny0 * grid_factor, Nz0 * grid_factor
    Nr = Nx * Ny * Nz
    nb = Ck.shape[0]
    gx, gy, gz = gvec[:, 0] % Nx, gvec[:, 1] % Ny, gvec[:, 2] % Nz
    cg = np.zeros((nb, Nx, Ny, Nz), dtype=np.complex128)
    cg[:, gx, gy, gz] = Ck
    u = np.fft.ifftn(cg, axes=(1, 2, 3)) * np.sqrt(Nr)
    return u.reshape(nb, Nr), (Nx, Ny, Nz), Nr


# ── real-space beta for a batch of bands (never Nr x Nr) ───────────────────

def real_space_beta_for_bands(pawpp, elements_idx, atom_cart, latvec, r_grid_cart,
                               psi_r_bands, Nr, nmax, k_frac, dist_prune=DIST_PRUNE):
    """
    beta_n,i = <p~_i|psi_n> for a batch of BLOCH states via a direct
    real-space sum -- same radial-spline/Qij/normalization convention as
    build_real_space_S (paw_density_matrix.py), but returning only the small
    (nb, n_proj_total) projector-overlap array. Never allocates an Nr x Nr
    matrix regardless of how large Nr is.

    See the module docstring's "Real-space beta derivation" section for why
    BOTH of the following are required (unlike build_real_space_S's S_hat,
    which is k-independent and needs neither):
      psi_r_bands : (nb, Nr) complex -- the FULL Bloch wavefunction values
                    psi_n(r) = exp(2*pi*i*k_frac.r_frac) * u_n(r) at
                    r_grid_cart. NOT the bare cell-periodic u_n(r).
      k_frac      : (3,) fractional k-point (same convention as r_frac
                    elsewhere in this codebase) -- used for the per-atom-
                    image translation phase exp(-2*pi*i*k_frac.n_image)
                    applied to each image's contribution before summing.

    Nr : the grid THIS call is using (must be recomputed per grid factor by
         the caller, never reused from a different grid).
    """
    natoms = atom_cart.shape[0]
    nb = psi_r_bands.shape[0]
    ns = np.arange(-nmax, nmax + 1)
    n1, n2, n3 = [a.ravel() for a in np.meshgrid(ns, ns, ns, indexing='ij')]
    all_n = np.column_stack([n1, n2, n3])              # integer lattice translations
    all_n_cart = all_n @ latvec
    image_phase = np.exp(-2j * np.pi * (all_n @ np.asarray(k_frac)))  # (ntrans,)
    centroid = r_grid_cart.mean(axis=0)

    beta_blocks = []
    for iatom in range(natoms):
        pp = pawpp[elements_idx[iatom]]
        rmax_eff = pp.proj_rmax * (pp.NPSRNL - 1) / pp.NPSRNL
        lmmax = pp.lmmax
        beta_atom = np.zeros((lmmax, nb), dtype=np.complex128)

        images_cart = atom_cart[iatom] + all_n_cart
        d_centroid = np.linalg.norm(images_cart - centroid[None, :], axis=1)
        candidate_idx = np.where(d_centroid < dist_prune)[0]

        for ii in candidate_idx:
            Rimg = images_cart[ii]
            disp = r_grid_cart - Rimg[None, :]
            dist = np.linalg.norm(disp, axis=1)
            mask = dist <= rmax_eff
            if not mask.any():
                continue
            disp_m = disp[mask]
            dist_m = dist[mask]
            c_m = psi_r_bands[:, mask]  # (nb, npts)

            Bblock = np.zeros((lmmax, mask.sum()), dtype=np.complex128)
            rproj_ylm = [sph_r(disp_m, l).T for l in range(pp.proj_l.max() + 1)]
            iL = 0
            for l, spl_r in zip(pp.proj_l, pp.spl_rproj):
                TLP1 = 2 * l + 1
                rad = spl_r(dist_m)
                Bblock[iL:iL + TLP1, :] = rad * rproj_ylm[l]
                iL += TLP1
            Bblock *= np.sqrt(np.linalg.det(latvec)) * image_phase[ii]

            beta_atom += Bblock @ c_m.T  # (lmmax, npts) @ (npts, nb) -> (lmmax, nb)

        beta_blocks.append(beta_atom / np.sqrt(Nr))

    return np.concatenate(beta_blocks, axis=0).T  # (nb, n_proj_total)


def paw_corrected_gram(Ck_raw, beta, qij_block):
    S_ps = Ck_raw.conj() @ Ck_raw.T
    S_aug = beta.conj() @ qij_block @ beta.T
    return S_ps, S_ps + S_aug


def gram_summary(S):
    diag = np.diag(S).real
    return dict(diag_mean=float(diag.mean()), diag_min=float(diag.min()),
                diag_max=float(diag.max()), max_offdiag=float(offdiag_maxabs(S)))


def beta_diff_stats(beta_a, beta_b):
    diff = beta_a - beta_b
    return dict(max_abs=float(np.max(np.abs(diff))),
                rms=float(np.sqrt(np.mean(np.abs(diff) ** 2))))


def representative_kpoints(nkpts):
    return sorted(set(max(1, min(nkpts, round(f * (nkpts - 1)) + 1)) for f in KPOINT_FRACTIONS))


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)!r}")


def main():
    import config  # read-only

    material = config.MATERIAL
    data_dir = Path(__file__).resolve().parent.parent.parent / "Data" / material
    potcar_path = data_dir / "POTCAR"
    if not potcar_path.exists():
        print(f"BLOCKED: no POTCAR for {material}; this diagnostic needs PAW radial data. Aborting.")
        return

    print(f"=== Real-space quadrature convergence check: {material} ===\n")
    t_start = time.time()

    latvec, species, counts, atom_symbols, atom_numbers, frac_coords, cart_coords = \
        read_poscar_structure(data_dir / "POSCAR")
    pawpp = load_pawpp(potcar_path)
    pawpp_elements = [pp.element.split('_')[0] for pp in pawpp]
    elements_idx = [pawpp_elements.index(s) for s in atom_symbols]

    wfc = vaspwfc(str(data_dir / "WAVECAR"), lsorbit=False)
    base_ngrid = tuple(int(x) for x in wfc._ngrid)
    atoms = ase_read(str(data_dir / "POSCAR"))

    center_cart, _, _ = parse_ws_center(config.WS_CENTER, config.WS_CENTER_COORD_TYPE, latvec)
    ws_nmax = config.WS_TRANSLATION_SEARCH_RANGE + 1

    kpoints = representative_kpoints(wfc._nkpts)
    print(f"native grid: {base_ngrid}   representative k-points: {kpoints}\n")

    # ── per-k, grid-independent data (computed once) ────────────────────────
    per_k = {}
    for ik in kpoints:
        kvec = wfc._kvecs[ik - 1]
        occ_all = wfc._occs[0, ik - 1, :]
        bands = np.where(occ_all > 1e-6)[0] + 1
        if len(bands) == 0:
            continue
        gvec = wfc.gvectors(ik)
        Ck_raw = np.stack([wfc.readBandCoeff(ispin=1, ikpt=ik, iband=int(ib), norm=False)
                            for ib in bands])

        proj = nonlq(atoms, wfc._encut, pawpp, k=kvec, lgam=wfc._lgam, gamma_half=wfc._gam_half)
        assert list(proj.element_idx) == list(elements_idx), \
            "atom ordering mismatch between nonlq and real-space elements_idx"
        qij_block = build_qij_block(pawpp, elements_idx)
        beta_recip = np.stack([proj.proj(Ck_raw[i]) for i in range(len(bands))])
        S_ps_ref, S_corr_recip = paw_corrected_gram(Ck_raw, beta_recip, qij_block)
        recip_summary = gram_summary(S_corr_recip)

        print(f"ik={ik:4d}  k_frac={np.round(kvec, 4).tolist()}  nbands={len(bands)}  "
              f"reciprocal reference: max_offdiag={recip_summary['max_offdiag']:.3e}  "
              f"diag=[{recip_summary['diag_min']:.5f},{recip_summary['diag_max']:.5f}]")

        per_k[ik] = dict(kvec=kvec, bands=bands, gvec=gvec, Ck_raw=Ck_raw,
                          qij_block=qij_block, beta_recip=beta_recip,
                          recip_summary=recip_summary)
    print()

    # ── grid-factor loop: build the (grid-dependent, k-independent) WS map ──
    # and primitive-cell coordinates ONCE per factor, reused across k-points.
    results = []
    for f in GRID_FACTORS:
        Nxf, Nyf, Nzf = base_ngrid[0] * f, base_ngrid[1] * f, base_ngrid[2] * f
        Nr_f = Nxf * Nyf * Nzf
        print(f"--- grid factor {f}x: ({Nxf},{Nyf},{Nzf})  Nr={Nr_f} ---")
        t0 = time.time()

        ix, iy, iz = [a.ravel() for a in np.mgrid[0:Nxf, 0:Nyf, 0:Nzf]]
        r_prim_frac = np.column_stack([ix / Nxf, iy / Nyf, iz / Nzf])
        r_prim_cart = r_prim_frac @ latvec

        r_ws_cart, r_ws_frac_cont, prim_indices, translations_all = build_ws_grid_map(
            latvec, (Nxf, Nyf, Nzf), center_cart, nmax=ws_nmax,
        )
        base_flat = (prim_indices[:, 0].astype(np.int64) * Nyf
                     + prim_indices[:, 1]) * Nzf + prim_indices[:, 2]
        print(f"  WS map + primitive grid built in {time.time()-t0:.1f}s")

        for ik, kd in per_k.items():
            t1 = time.time()
            k_frac = kd["kvec"]
            u_bands, ngrid_f, Nr_check = zero_pad_ifft(kd["Ck_raw"], kd["gvec"], f, base_ngrid)
            assert Nr_check == Nr_f and ngrid_f == (Nxf, Nyf, Nzf)

            # FULL Bloch wavefunction psi_n(r) = exp(2*pi*i*k.r_frac)*u_n(r)
            # -- required for beta (see module docstring), unlike S_hat.
            psi_prim_bands = u_bands * np.exp(2j * np.pi * (r_prim_frac @ k_frac))[None, :]
            beta_prim = real_space_beta_for_bands(
                pawpp, elements_idx, cart_coords, latvec, r_prim_cart,
                psi_prim_bands, Nr_f, nmax=ATOM_IMAGE_NMAX_PRIM, k_frac=k_frac,
            )
            _, S_corr_prim = paw_corrected_gram(kd["Ck_raw"], beta_prim, kd["qij_block"])

            psi_ws_bands = (u_bands[:, base_flat]
                             * np.exp(2j * np.pi * (r_ws_frac_cont @ k_frac))[None, :])
            beta_ws = real_space_beta_for_bands(
                pawpp, elements_idx, cart_coords, latvec, r_ws_cart,
                psi_ws_bands, Nr_f, nmax=ws_nmax, k_frac=k_frac,
            )
            _, S_corr_ws = paw_corrected_gram(kd["Ck_raw"], beta_ws, kd["qij_block"])

            entry = dict(
                ik=int(ik), grid_factor=f, ngrid=list(ngrid_f), Nr=int(Nr_f),
                nbands=int(len(kd["bands"])), elapsed_s=float(time.time() - t1),
                beta_diff_prim=beta_diff_stats(beta_prim, kd["beta_recip"]),
                beta_diff_ws=beta_diff_stats(beta_ws, kd["beta_recip"]),
                gram_prim=gram_summary(S_corr_prim),
                gram_ws=gram_summary(S_corr_ws),
                gram_reciprocal_reference=kd["recip_summary"],
            )
            results.append(entry)
            print(f"  ik={ik:4d}  +{entry['elapsed_s']:.1f}s  "
                  f"beta_diff: prim(max={entry['beta_diff_prim']['max_abs']:.3e}, "
                  f"rms={entry['beta_diff_prim']['rms']:.3e})  "
                  f"ws(max={entry['beta_diff_ws']['max_abs']:.3e}, "
                  f"rms={entry['beta_diff_ws']['rms']:.3e})")
            print(f"            gram: prim(offdiag={entry['gram_prim']['max_offdiag']:.3e}, "
                  f"diag=[{entry['gram_prim']['diag_min']:.4f},{entry['gram_prim']['diag_max']:.4f}])  "
                  f"ws(offdiag={entry['gram_ws']['max_offdiag']:.3e}, "
                  f"diag=[{entry['gram_ws']['diag_min']:.4f},{entry['gram_ws']['diag_max']:.4f}])")
        print()

    # ── aggregate across k-points (worst case per grid factor) + interpret ──
    by_factor = {f: [r for r in results if r["grid_factor"] == f] for f in GRID_FACTORS}

    def _worst(entries, *key_path):
        vals = []
        for e in entries:
            d = e
            for k in key_path:
                d = d[k]
            vals.append(d)
        return max(vals) if vals else float("nan")

    agg = {}
    for f in GRID_FACTORS:
        e = by_factor[f]
        agg[f] = dict(
            worst_beta_diff_prim_max=_worst(e, "beta_diff_prim", "max_abs"),
            worst_beta_diff_ws_max=_worst(e, "beta_diff_ws", "max_abs"),
            worst_gram_prim_offdiag=_worst(e, "gram_prim", "max_offdiag"),
            worst_gram_ws_offdiag=_worst(e, "gram_ws", "max_offdiag"),
            worst_gram_reciprocal_offdiag=_worst(e, "gram_reciprocal_reference", "max_offdiag"),
        )

    f_lo, f_hi = GRID_FACTORS[0], GRID_FACTORS[-1]
    recip_level = agg[f_lo]["worst_gram_reciprocal_offdiag"]  # grid-independent; any factor's value works
    close_bar = max(recip_level * CLOSE_TO_RECIP_MULTIPLE, CLOSE_TO_RECIP_ABS_FLOOR)

    def _shrink(key):
        v_lo, v_hi = agg[f_lo][key], agg[f_hi][key]
        return v_lo / v_hi if v_hi > 0 else float("inf")

    prim_shrink = _shrink("worst_gram_prim_offdiag")
    ws_shrink = _shrink("worst_gram_ws_offdiag")
    prim_converging = bool(prim_shrink >= CONVERGENCE_SHRINK_FACTOR
                            and agg[f_hi]["worst_gram_prim_offdiag"] < close_bar)
    ws_converging = bool(ws_shrink >= CONVERGENCE_SHRINK_FACTOR
                          and agg[f_hi]["worst_gram_ws_offdiag"] < close_bar)

    if prim_converging and ws_converging:
        diagnosis = "quadrature_confirmed"
        recommendation = (
            "1. Real-space quadrature is confirmed as the residual error source: both the "
            "primitive-cell and WS-mapped real-space beta/Gram results converge toward the "
            "reciprocal-space reference as the grid is refined. Recommend a reciprocal-space / "
            "atom-centered low-rank PAW construction for the production S matrix (the existing "
            "nonlq/Qij machinery already gives an accurate, grid-independent correction at the "
            "band-pair level) rather than relying on real-space grid refinement, which would "
            "require an impractically fine production FFT grid to reach comparable accuracy."
        )
    elif prim_converging and not ws_converging:
        diagnosis = "ws_handling_is_separate_source"
        recommendation = (
            "2. Primitive-cell real-space quadrature converges toward the reciprocal reference, "
            "but the WS-mapped coordinates do NOT converge at a comparable rate -- WS "
            "atom-image / coordinate handling is an additional, separate error source beyond "
            "plain quadrature. Recommend a targeted WS-cell correction (re-check the atom-image "
            "search range and translation bookkeeping used when evaluating the augmentation "
            "term at WS-mapped coordinates) rather than a general reciprocal-space rebuild."
        )
    elif (not prim_converging) and (not ws_converging):
        diagnosis = "inconclusive_check_normalization"
        recommendation = (
            "Neither the primitive-cell nor the WS-mapped real-space result converges toward "
            "the reciprocal-space reference with grid refinement. This points at a "
            "normalization or formula issue in build_real_space_S / the beta convention (not "
            "simple quadrature undersampling), which refining the grid cannot fix. Do NOT "
            "choose between the reciprocal-space rebuild or a WS-targeted fix yet -- "
            "re-examine the real-space S formula and normalization factors first."
        )
    else:  # ws_converging and not prim_converging -- unexpected/asymmetric
        diagnosis = "unexpected_asymmetric_convergence"
        recommendation = (
            "WS-mapped coordinates converged but the primitive-cell coordinates did not -- this "
            "is not one of the anticipated failure modes and should be re-examined manually "
            "before choosing a fix; the beta/Gram computation may have a bug specific to the "
            "primitive-cell grid path."
        )

    print("--- convergence summary (worst case over k-points) ---")
    for f in GRID_FACTORS:
        a = agg[f]
        print(f"  f={f}x  gram_prim_offdiag={a['worst_gram_prim_offdiag']:.3e}  "
              f"gram_ws_offdiag={a['worst_gram_ws_offdiag']:.3e}  "
              f"reciprocal_ref={a['worst_gram_reciprocal_offdiag']:.3e}")
    print(f"  prim shrink ({f_lo}x->{f_hi}x): {prim_shrink:.2f}x  "
          f"(converging={prim_converging})")
    print(f"  ws   shrink ({f_lo}x->{f_hi}x): {ws_shrink:.2f}x  "
          f"(converging={ws_converging})")
    print(f"\nDIAGNOSIS: {diagnosis}")
    print(f"RECOMMENDATION: {recommendation}")
    print(f"\nTotal runtime: {time.time()-t_start:.1f}s")

    # ── write reports ────────────────────────────────────────────────────
    report = dict(
        material=material, data_dir=str(data_dir), native_grid=list(base_ngrid),
        grid_factors=GRID_FACTORS, kpoints=kpoints,
        ws_center=config.WS_CENTER, ws_center_coord_type=config.WS_CENTER_COORD_TYPE,
        ws_translation_search_range_nmax=ws_nmax,
        thresholds=dict(
            convergence_shrink_factor=CONVERGENCE_SHRINK_FACTOR,
            close_to_recip_multiple=CLOSE_TO_RECIP_MULTIPLE,
            close_to_recip_abs_floor=CLOSE_TO_RECIP_ABS_FLOOR,
        ),
        per_kpoint_reciprocal_reference={
            int(ik): kd["recip_summary"] for ik, kd in per_k.items()
        },
        results=results,
        aggregate=agg,
        prim_shrink_factor=prim_shrink, ws_shrink_factor=ws_shrink,
        prim_converging=prim_converging, ws_converging=ws_converging,
        diagnosis=diagnosis, recommendation=recommendation,
        note="This script never builds an Nr x Nr matrix, never rescales S, "
             "and never clips occupations -- it does not touch the production "
             "S/D/eigenvalue pipeline at all.",
    )
    json_path = OUT / "quadrature_convergence_report.json"
    with open(json_path, "w") as f_json:
        json.dump(report, f_json, indent=2, default=_json_default)

    txt_path = OUT / "quadrature_convergence_report.txt"
    with open(txt_path, "w") as f_txt:
        f_txt.write("=== Real-space quadrature convergence check ===\n\n")
        f_txt.write(f"material: {material}\n")
        f_txt.write(f"native_grid: {base_ngrid}\n")
        f_txt.write(f"grid_factors: {GRID_FACTORS}\n")
        f_txt.write(f"representative_kpoints: {kpoints}\n")
        f_txt.write(f"ws_center: {config.WS_CENTER} ({config.WS_CENTER_COORD_TYPE})\n\n")

        f_txt.write("--- per-kpoint reciprocal-space reference (grid-independent) ---\n")
        for ik, kd in per_k.items():
            rs = kd["recip_summary"]
            f_txt.write(f"  ik={ik:4d}  max_offdiag={rs['max_offdiag']:.4e}  "
                        f"diag=[{rs['diag_min']:.5f},{rs['diag_max']:.5f}]\n")
        f_txt.write("\n")

        f_txt.write("--- per (k, grid_factor) results ---\n")
        for e in results:
            f_txt.write(f"  ik={e['ik']:4d}  f={e['grid_factor']}x  ngrid={e['ngrid']}  Nr={e['Nr']}\n")
            f_txt.write(f"    beta_diff_prim: max={e['beta_diff_prim']['max_abs']:.4e}  "
                        f"rms={e['beta_diff_prim']['rms']:.4e}\n")
            f_txt.write(f"    beta_diff_ws  : max={e['beta_diff_ws']['max_abs']:.4e}  "
                        f"rms={e['beta_diff_ws']['rms']:.4e}\n")
            f_txt.write(f"    gram_prim     : max_offdiag={e['gram_prim']['max_offdiag']:.4e}  "
                        f"diag_mean={e['gram_prim']['diag_mean']:.5f}  "
                        f"diag=[{e['gram_prim']['diag_min']:.5f},{e['gram_prim']['diag_max']:.5f}]\n")
            f_txt.write(f"    gram_ws       : max_offdiag={e['gram_ws']['max_offdiag']:.4e}  "
                        f"diag_mean={e['gram_ws']['diag_mean']:.5f}  "
                        f"diag=[{e['gram_ws']['diag_min']:.5f},{e['gram_ws']['diag_max']:.5f}]\n")
        f_txt.write("\n")

        f_txt.write("--- convergence summary (worst case over k-points) ---\n")
        for f in GRID_FACTORS:
            a = agg[f]
            f_txt.write(f"  f={f}x  gram_prim_offdiag={a['worst_gram_prim_offdiag']:.4e}  "
                        f"gram_ws_offdiag={a['worst_gram_ws_offdiag']:.4e}  "
                        f"reciprocal_ref={a['worst_gram_reciprocal_offdiag']:.4e}\n")
        f_txt.write(f"  prim shrink ({f_lo}x->{f_hi}x): {prim_shrink:.2f}x  (converging={prim_converging})\n")
        f_txt.write(f"  ws   shrink ({f_lo}x->{f_hi}x): {ws_shrink:.2f}x  (converging={ws_converging})\n\n")

        f_txt.write(f"DIAGNOSIS: {diagnosis}\n\n")
        f_txt.write(f"RECOMMENDATION: {recommendation}\n\n")
        f_txt.write("Note: this script never builds an Nr x Nr matrix, never rescales S, and "
                    "never clips occupations -- it does not touch the production S/D/eigenvalue "
                    "pipeline at all. main.py and config.py were not modified (config.py was "
                    "only read).\n")

    print(f"\nSaved -> {json_path}\nSaved -> {txt_path}")


if __name__ == '__main__':
    main()
