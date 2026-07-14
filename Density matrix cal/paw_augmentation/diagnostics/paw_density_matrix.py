"""
PAW-augmentation-corrected real-space CNO density matrix.

Extends paw_overlap.py's band-pair overlap correction to a full site-restricted
(WS-cell) density-matrix / natural-orbital-occupation calculation.

Correct occupation eigenproblem (this is NOT a generalized eigenvalue problem
D v = lambda S v -- an earlier version of this script solved that, which is
wrong; see below for why):

D is built (in build_density_matrix) as an OUTER-PRODUCT coefficient sum,

    D_rr' = sum_nk w_k f_nk c_r(nk) c_r'(nk)*

i.e. the density operator is rho_hat = sum_rr' D_rr' |chi_r><chi_r'|, where
{|chi_r>} is the (non-orthogonal) real-space grid basis with metric
<chi_r|chi_r'> = S_rr'. Solving rho_hat|phi> = lambda|phi> for
|phi> = sum_r y_r |chi_r> gives, after expanding <chi_r'|phi> = sum_s S_r's y_s:

    (D S) y = lambda y

NOT D v = lambda S v (that equation is what you'd solve if D were instead the
covariant matrix elements <chi_r|rho_hat|chi_r'>, which it is not -- it's the
outer-product coefficient matrix). D S is not Hermitian, but it is similar to
the manifestly Hermitian M = S^(1/2) D S^(1/2) via conjugation by S^(1/2)
(S^(1/2) (D S) S^(-1/2) = S^(1/2) D S^(1/2) = M), so DS has the same real
eigenvalues as M, and if M y' = lambda y' (y' plain-orthonormal, from
np.linalg.eigh), then x = S^(-1/2) y' satisfies (DS) x = lambda x and
x^H S x = delta_ij (S-metric-orthonormal) -- this is solved via
solve_paw_cno() below.

The metric S itself:

    S[r, r'] = delta(r, r') + sum_{atom images R} sum_ij  p~_i(r - R) Qij p~_j*(r' - R)

is the position-space representation of the PAW operator
S_hat = 1 + sum_i |p~_i> Qij <p~_j|, evaluated directly at the ACTUAL
(possibly WS-cell-unwrapped, multi-cell) Cartesian grid coordinates -- no
Bloch phase needed, because S_hat is a k-independent local operator (unlike
the band-pair overlap, which lives at fixed k). This reuses only pawpotcar's
radial projector splines + Qij (paw.py) and sph_harm.sph_r -- no pysbt, no
real-space AE partial-wave reconstruction.

Does NOT run or modify main.py/config.py; reads config.py and ws_cell.py
(read-only imports) to reproduce the exact WS-cell setup for whatever
MATERIAL/OUTPUT_SUBDIR config.py currently specifies, and writes all outputs
under paw_augmentation/output/, never under Data/*/output/. Reads WAVECAR/
POSCAR/POTCAR/EIGENVAL from Density matrix cal/Data/<MATERIAL>/ by default
(live data) -- pass an explicit data_dir to main() to point at a snapshot
instead (e.g. for a frozen/mismatched-data regression check).

--------------------------------------------------------------------------
2026-07-10 update: preflight gate + trace-reporting fix
--------------------------------------------------------------------------
This experimental pipeline is expensive (the Nr x Nr real-space S/D build
and the two Nr x Nr eigh's took ~11 minutes / ~1.25 GB each for WSe2_mono's
current grid) and, as of the "data integrity incident" documented in
RESULTS.md, is not safe to assume valid just because a POTCAR happens to be
present next to a WAVECAR. Before doing that expensive work, main() now runs
preflight_paw_overlap_check() -- the existing, cheap, few-k-point reciprocal-
space band-pair overlap check from paw_overlap.py -- as a go/no-go gate, and
aborts cleanly (writing a short BLOCKED report, no huge arrays touched) if
the augmentation correction does not substantially improve orthogonality
over the plain pseudo-wavefunction overlap. See preflight_paw_overlap_check's
docstring for the exact pass/fail criterion.

The trace bookkeeping in the report was also corrected/expanded: the matrix-
algebra identity Tr(M) = Tr(D@S) = sum(eigvals) is checked (this is only an
eigensolver/arithmetic self-consistency check), and SEPARATELY an
independent, physically-meaningful expected value
trace_expected_input = sum_k w_k * sum_n f_nk (computed inside
build_density_matrix using the exact same band-selection/occupation array
that builds D, so it cannot silently diverge from what D encodes) is compared
against it -- see the "trace reporting" section of main() and
PawOverlapCorrector's docstring below for what each field means.

SOC (LSORBIT) and Fermi-window occupation mode are both explicitly detected
and aborted on (not implemented in build_density_matrix), rather than
silently producing a wrong answer -- deferred to future work, see
RESULTS.md.
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh as sc_eigh

# Force line-buffered stdout: when this script's output is redirected to a
# file (e.g. run in the background), Python otherwise block-buffers instead
# of flushing per line, so prints only appear once the process exits -- no
# use for watching a multi-minute run in progress. This makes every print()
# show up immediately regardless of where stdout is going.
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # for config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "helper functions"))  # for ws_cell
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for paw_overlap (paw_augmentation/)
from ws_cell import read_poscar_structure, parse_ws_center, build_ws_grid_map  # noqa: E402

from vaspwfc import vaspwfc  # noqa: E402
from sph_harm import sph_r  # noqa: E402
from paw_overlap import (  # noqa: E402
    load_pawpp, PawOverlapCorrector, offdiag_maxabs, diag_stats,
)

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)


# ── preflight gate (cheap, few k-points, reuses paw_overlap.py unmodified) ──

class PawPreflightError(RuntimeError):
    """Raised when the existing reciprocal-space PAW overlap correction does
    not substantially improve occupied-band orthogonality -- signals a
    WAVECAR/POTCAR pair (mismatched, wrong potential, etc.) that should not
    be trusted for the expensive real-space PAW-corrected density matrix."""


SOFT_POTENTIAL_TOL = 1e-4      # uncorrected offdiag already this small -> no augmentation problem to fix
MIN_IMPROVEMENT_FACTOR = 10.0  # corrected must beat uncorrected by at least this factor ...
CORRECTED_ABS_TOL = 1e-3       # ... AND land below this absolute bar (unless soft-potential case)


def _representative_kpoints(nkpts):
    """First / quarter / half / three-quarter / last k-point, deduplicated
    and clipped to a valid range -- deliberately not hardcoded to specific
    indices like TASK_BRIEF.md's [1,2,50,150,300] so this works for any
    material's k-mesh size, not just the ones that happen to have >=300
    k-points."""
    raw = [1, max(1, nkpts // 4), max(1, nkpts // 2), max(1, (3 * nkpts) // 4), nkpts]
    return sorted(set(k for k in raw if 1 <= k <= nkpts))


def preflight_paw_overlap_check(wavecar, poscar, potcar, ispin=1, verbose=True):
    """
    Cheap (few-k-point, band-pair-only, nb x nb) sanity gate that must pass
    before the expensive Nr x Nr real-space S/D construction below is
    attempted.

    Uses the EXISTING, already-validated reciprocal-space PAW overlap
    machinery (paw_overlap.PawOverlapCorrector) -- not re-derived here -- to
    check that the augmentation correction pulls the occupied-band Gram
    matrix substantially closer to the identity than the plain pseudo-
    wavefunction overlap. This is exactly the goal-(a) diagnostic from
    TASK_BRIEF.md, reused here as a go/no-go gate rather than a one-off
    report.

    Pass criterion, per representative k-point:
      - if the UNCORRECTED max|offdiag| is already < SOFT_POTENTIAL_TOL,
        the correction isn't needed (soft potential) -- passes trivially.
      - otherwise, the CORRECTED max|offdiag| must be < CORRECTED_ABS_TOL
        AND at least MIN_IMPROVEMENT_FACTOR times smaller than the
        uncorrected value.
    The overall gate passes only if every checked k-point passes AND at
    least one k-point was actually evaluated (an all-skipped run, e.g. every
    k-point having fewer than 2 occupied bands, is NOT a pass).

    Returns a report dict (always -- callers decide whether to raise/abort)
    with a top-level 'passed' bool and per-k-point detail. Never touches
    Nr x Nr arrays.
    """
    corr = PawOverlapCorrector(str(wavecar), str(poscar), str(potcar))
    nkpts = corr.wfc._nkpts
    kpoints = _representative_kpoints(nkpts)

    rows = []
    all_ok = True
    n_evaluated = 0
    for ik in kpoints:
        bands, S_ps, S_corr = corr.overlaps(ik, ispin=ispin)
        if len(bands) < 2:
            rows.append(dict(ik=int(ik), nbands=int(len(bands)), skipped=True,
                              reason="fewer than 2 occupied bands"))
            if verbose:
                print(f"  [preflight] ik={ik:4d}  skipped (fewer than 2 occupied bands)")
            continue

        before = offdiag_maxabs(S_ps)
        after = offdiag_maxabs(S_corr)
        dmin, dmax, dmad = diag_stats(S_corr)
        n_evaluated += 1

        if before < SOFT_POTENTIAL_TOL:
            ok = True
            reason = "soft potential (uncorrected already below tolerance)"
            improvement = float("inf")
        else:
            improvement = before / max(after, 1e-300)
            ok = bool(after < CORRECTED_ABS_TOL and improvement >= MIN_IMPROVEMENT_FACTOR)
            reason = f"improvement={improvement:.2f}x, corrected={after:.3e}"
        all_ok = all_ok and ok

        rows.append(dict(
            ik=int(ik), nbands=int(len(bands)),
            offdiag_uncorrected=float(before), offdiag_corrected=float(after),
            improvement_factor=float(improvement),
            corrected_diag_range=[float(dmin), float(dmax)],
            corrected_diag_mean_abs_dev=float(dmad),
            passed=ok, reason=reason,
        ))
        if verbose:
            print(f"  [preflight] ik={ik:4d} nbands={len(bands):3d}  "
                  f"uncorrected={before:.3e}  corrected={after:.3e}  "
                  f"{'OK' if ok else 'FAIL'} ({reason})")

    passed = bool(all_ok and n_evaluated > 0)
    return dict(
        kpoints_checked=kpoints,
        n_evaluated=n_evaluated,
        soft_potential_tol=SOFT_POTENTIAL_TOL,
        min_improvement_factor=MIN_IMPROVEMENT_FACTOR,
        corrected_abs_tol=CORRECTED_ABS_TOL,
        rows=rows,
        passed=passed,
    )


# ── real-space PAW metric S (unchanged) ─────────────────────────────────────

def build_real_space_S(pawpp, elements_idx, atom_cart, latvec, r_grid_cart,
                        nmax=4, dist_prune=16.0, verbose=True):
    """
    Build the (Nr, Nr) real-space PAW metric matrix

        S = I + sum_images  Breal_image^H  @ Qij_image @ Breal_image

    pawpp        : list of pawpotcar, one per element type
    elements_idx : element-type index (into pawpp) for each atom in the cell
    atom_cart    : (natoms, 3) Cartesian atom positions (base cell)
    latvec       : (3,3) lattice vectors
    r_grid_cart  : (Nr, 3) Cartesian coordinates of the grid points (WS or box)
    nmax         : periodic-image search range for atoms (same convention as
                   ws_cell.build_ws_grid_map's nmax)
    dist_prune   : Angstrom; atom images farther than this from the grid's
                   centroid are skipped outright (cheap prune before the
                   expensive per-point distance calc)
    """
    natoms = atom_cart.shape[0]
    ns = np.arange(-nmax, nmax + 1)
    n1, n2, n3 = [a.ravel() for a in np.meshgrid(ns, ns, ns, indexing='ij')]
    all_n = np.column_stack([n1, n2, n3])
    all_n_cart = all_n @ latvec

    centroid = r_grid_cart.mean(axis=0)
    Nr = r_grid_cart.shape[0]

    S = np.eye(Nr, dtype=np.complex128)

    t0 = time.time()
    n_images_used = 0
    for iatom in range(natoms):
        pp = pawpp[elements_idx[iatom]]
        rmax_eff = pp.proj_rmax * (pp.NPSRNL - 1) / pp.NPSRNL
        Qij = pp.get_Qij()

        images_cart = atom_cart[iatom] + all_n_cart  # (nimg, 3)
        d_centroid = np.linalg.norm(images_cart - centroid[None, :], axis=1)
        candidate_idx = np.where(d_centroid < dist_prune)[0]

        if verbose:
            print(f"  atom {iatom + 1}/{natoms}  ({len(candidate_idx)} candidate "
                  f"images within {dist_prune} Ang)  elapsed={time.time()-t0:.1f}s")

        for ii in candidate_idx:
            Rimg = images_cart[ii]
            disp = r_grid_cart - Rimg[None, :]          # (Nr, 3)
            dist = np.linalg.norm(disp, axis=1)
            mask = dist <= rmax_eff
            npts = int(mask.sum())
            if npts == 0:
                continue

            n_images_used += 1
            disp_m = disp[mask]
            dist_m = dist[mask]

            # radial part per projector channel (spline in real-space, same
            # convention as pawpotcar.spl_rproj / paw.nonlr.calc_rproj)
            Bblock = np.zeros((pp.lmmax, npts), dtype=np.float64)
            rproj_ylm = [sph_r(disp_m, l).T for l in range(pp.proj_l.max() + 1)]
            iL = 0
            for l, spl_r in zip(pp.proj_l, pp.spl_rproj):
                TLP1 = 2 * l + 1
                rad = spl_r(dist_m)
                Bblock[iL:iL + TLP1, :] = rad * rproj_ylm[l]
                iL += TLP1
            Bblock *= np.sqrt(np.linalg.det(latvec))  # sqrt(cell volume), nonlr convention

            idx_pts = np.where(mask)[0]
            # S[r, r'] += sum_ij Bblock[i, r] * Qij[i,j] * Bblock[j, r']
            #
            # Normalization: beta_n,i = <p~_i|psi~_n> = (1/sqrt(Nr)) * sum_r
            # c_n(r) * P_i(r), because c_n(r) = sqrt(Nr) * ifftn(Cg_n)(r) is
            # the sqrt(Nr)-normalized grid convention shared with D's
            # construction (see build_density_matrix / main.py), while P_i(r)
            # here is the plain (nonlr-convention) real-space projector value
            # with no extra grid normalization. Verified against the already
            # validated reciprocal-space nonlq.proj() result for a real band
            # in _test_beta_consistency.py (agreement to a few % -- residual
            # difference is real-space quadrature error on this comparatively
            # coarse 11x11x73 grid, not a normalization mismatch). Since S
            # here is sandwiched between two such beta's (m and n), the
            # combined factor is 1/Nr, not 1/sqrt(Nr).
            contrib = (Bblock.T @ Qij @ Bblock) / Nr  # (npts, npts), real
            S[np.ix_(idx_pts, idx_pts)] += contrib

    return S, n_images_used


# ── density matrix D, now also returning the independent expected trace ────

def build_density_matrix(wfc, kfrac_all, kweights, ispin, Nr, ngrid,
                          r_for_phase, prim_indices, occ_tol=1e-6, verbose=True):
    """Same accumulation as main.py's rho loop, but using RAW (norm=False)
    band coefficients -- correct convention for pairing with the PAW-corrected
    metric S (see module docstring).

    Also accumulates and returns trace_expected_input = sum_k w_k * sum_n
    f_nk, using the EXACT SAME per-k `occ` array (band selection by
    occ_tol, spin-degeneracy halving when max(occ)>1.5) that builds D --
    by construction, not a separately re-derived formula, so it cannot
    silently diverge from what D actually encodes.

    Returns (D, info) where info is a dict with trace_expected_input,
    halving_triggered, occ_tol, ispin, n_kpoints.
    """
    Nx, Ny, Nz = ngrid
    D = np.zeros((Nr, Nr), dtype=np.complex128)
    trace_expected_input = 0.0
    halving_triggered = False
    t0 = time.time()
    t_prev = t0

    for ik in range(1, wfc._nkpts + 1):
        wk = kweights[ik - 1]
        k_frac = kfrac_all[ik - 1]

        occ_all = wfc._occs[ispin - 1, ik - 1, :]
        bands = np.where(occ_all > occ_tol)[0] + 1
        occ = occ_all[bands - 1]
        if len(bands) and np.max(occ) > 1.5:
            occ = occ / 2.0
            halving_triggered = True
        if len(bands) == 0:
            continue

        trace_expected_input += wk * float(occ.sum())

        gvec = wfc.gvectors(ik)
        nG = gvec.shape[0]
        gx, gy, gz = gvec[:, 0] % Nx, gvec[:, 1] % Ny, gvec[:, 2] % Nz

        Ck = np.stack([wfc.readBandCoeff(ispin=ispin, ikpt=ik, iband=int(ib), norm=False)
                       for ib in bands])
        nb = len(bands)
        cg = np.zeros((nb, Nx, Ny, Nz), dtype=np.complex128)
        cg[:, gx, gy, gz] = Ck
        u = np.fft.ifftn(cg, axes=(1, 2, 3)) * np.sqrt(Nr)

        if prim_indices is not None:
            psi = u[:, prim_indices[:, 0], prim_indices[:, 1], prim_indices[:, 2]]
        else:
            psi = u.reshape(nb, Nr)
        psi = psi * np.exp(2j * np.pi * (r_for_phase @ k_frac))[None, :]

        D += wk * (psi.T @ (occ[:, None] * psi).conj())

        if verbose and (ik == 1 or ik % 10 == 0 or ik == wfc._nkpts):
            now = time.time()
            print(f"  k {ik:4d}/{wfc._nkpts}  wk={wk:.6f}  bands={nb}  "
                  f"+{now - t_prev:.1f}s since last print  elapsed={now - t0:.1f}s")
            t_prev = now

    info = dict(
        trace_expected_input=float(trace_expected_input),
        halving_triggered=bool(halving_triggered),
        occ_tol=occ_tol, ispin=ispin, n_kpoints=int(wfc._nkpts),
    )
    return D, info


def solve_paw_cno(D, S, min_s_eval_tol=1e-10):
    """
    Correct PAW-CNO occupation eigenproblem: eigenvalues of D @ S (equivalently
    of the Hermitian M = S^(1/2) D S^(1/2)), NOT of the generalized problem
    D v = lambda S v -- see module docstring for the derivation of why.

    Returns
    -------
    eigvals : (Nr,) real, sorted descending
    eigvecs : (Nr, Nr) columns are S-metric-orthonormal (x_i^H S x_j = delta_ij),
              in the same grid-coefficient representation as D/S.
    diag    : dict of diagnostics (see keys below)
    """
    Nr = D.shape[0]
    t0 = time.time()

    herm_err_D_in = float(np.max(np.abs(D - D.conj().T)))
    herm_err_S_in = float(np.max(np.abs(S - S.conj().T)))
    D = 0.5 * (D + D.conj().T)
    S = 0.5 * (S + S.conj().T)
    print(f"  [1/6] Hermitized D, S ({Nr}x{Nr})  "
          f"(pre-symmetrization |D-D^H|={herm_err_D_in:.2e}, |S-S^H|={herm_err_S_in:.2e})  "
          f"+{time.time()-t0:.1f}s")

    t1 = time.time()
    print(f"  [2/6] diagonalizing S (eigh, {Nr}x{Nr}) ...")
    s_eval, s_vec = np.linalg.eigh(S)
    print(f"        done  +{time.time()-t1:.1f}s")
    s_positive_definite = bool(s_eval.min() > min_s_eval_tol)
    if not s_positive_definite:
        print(f"  S is not positive definite: min(s_eval)={s_eval.min():.4e} "
              f"<= tol={min_s_eval_tol:.1e}")
        print(f"  smallest 10 S eigenvalues: {np.sort(s_eval)[:10]}")
        raise ValueError(
            f"S metric is not positive definite (min eigenvalue {s_eval.min():.4e} "
            f"<= {min_s_eval_tol:.1e}) -- cannot form S^(1/2)/S^(-1/2)."
        )
    print(f"        min(s_eval)={s_eval.min():.4e}  max(s_eval)={s_eval.max():.4e}")

    t1 = time.time()
    print("  [3/6] building S^(1/2), S^(-1/2) ...")
    S_half = (s_vec * np.sqrt(s_eval)) @ s_vec.conj().T
    S_inv_half = (s_vec * (1.0 / np.sqrt(s_eval))) @ s_vec.conj().T
    print(f"        done  +{time.time()-t1:.1f}s")

    t1 = time.time()
    print("  [4/6] building M = S^(1/2) D S^(1/2) ...")
    M = S_half @ D @ S_half
    herm_err_M = float(np.max(np.abs(M - M.conj().T)))
    trace_M_pre_sym = float(np.trace(M).real)
    M = 0.5 * (M + M.conj().T)
    print(f"        |M-M^H|={herm_err_M:.2e}  Tr(M)={trace_M_pre_sym:.8f}  "
          f"done  +{time.time()-t1:.1f}s")

    t1 = time.time()
    print(f"  [5/6] diagonalizing M (eigh, {Nr}x{Nr}) ...")
    eigvals, yvecs = np.linalg.eigh(M)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    yvecs = yvecs[:, order]
    print(f"        done  +{time.time()-t1:.1f}s")

    t1 = time.time()
    print("  [6/6] converting eigenvectors back (S^(-1/2) yvecs) and checking "
          "S-orthonormality ...")
    eigvecs = S_inv_half @ yvecs

    # S-metric orthonormality error of the returned eigenvectors: should be ~0
    check = eigvecs.conj().T @ S @ eigvecs
    s_orthonorm_err = float(np.max(np.abs(check - np.eye(check.shape[0]))))
    print(f"        done  +{time.time()-t1:.1f}s  total solve_paw_cno time "
          f"{time.time()-t0:.1f}s")

    trace_DS = float(np.trace(D @ S).real)
    trace_M = float(np.trace(M).real)  # post-symmetrization, matches eigvals.sum() by construction
    sum_eigvals = float(eigvals.sum())
    diag = dict(
        herm_err_D_in=herm_err_D_in,
        herm_err_S_in=herm_err_S_in,
        herm_err_M=herm_err_M,
        s_positive_definite=s_positive_definite,
        min_s_eval=float(s_eval.min()),
        max_eigval=float(eigvals.max()),
        min_eigval=float(eigvals.min()),
        n_out_of_bounds=int(np.sum((eigvals < -1e-3) | (eigvals > 1 + 1e-3))),
        bound_01_tol=1e-3,
        bound_01_passed=bool(np.sum((eigvals < -1e-3) | (eigvals > 1 + 1e-3)) == 0),
        trace_M=trace_M,
        trace_DS=trace_DS,
        sum_eigvals=sum_eigvals,
        trace_M_minus_eigensum=float(trace_M - sum_eigvals),
        trace_DS_minus_eigensum=float(trace_DS - sum_eigvals),
        s_orthonorm_err=s_orthonorm_err,
        s_orthonorm_passed=bool(s_orthonorm_err < 1e-6),
    )
    return eigvals, eigvecs, diag


def main(data_dir=None):
    import config  # read-only

    material = config.MATERIAL
    ispin = config.ISPIN

    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent.parent / "Data" / material
    else:
        data_dir = Path(data_dir)

    print(f"=== PAW-corrected CNO density matrix: {material} / "
          f"{config.OUTPUT_SUBDIR} ===\n")

    # ── scope guards: SOC and Fermi-window occupation mode are not ─────────
    # implemented in build_density_matrix and are explicitly deferred (see
    # module docstring "2026-07-10 update") rather than silently producing a
    # wrong answer.
    if getattr(config, "LSORBIT", False):
        _write_blocked_report(
            material, config, data_dir,
            reason="LSORBIT=True (SOC): not implemented in this experimental "
                   "script's build_density_matrix (deferred; see RESULTS.md).",
        )
        print("BLOCKED: SOC (LSORBIT=True) is not supported by this experimental "
              "script yet. Aborting before any expensive computation.")
        return
    if getattr(config, "RESTRICT_TO_FERMI_WINDOW", False):
        _write_blocked_report(
            material, config, data_dir,
            reason="RESTRICT_TO_FERMI_WINDOW=True: build_density_matrix only "
                   "implements the occ_tol-threshold band-selection convention "
                   "(matching main.py's non-Fermi-window path); the Fermi-window "
                   "selected-subspace convention is not implemented here "
                   "(deferred; see RESULTS.md).",
        )
        print("BLOCKED: Fermi-window occupation mode is not supported by this "
              "experimental script yet. Aborting before any expensive computation.")
        return

    potcar_path = data_dir / "POTCAR"
    if not potcar_path.exists():
        _write_blocked_report(
            material, config, data_dir,
            reason=f"No POTCAR at {potcar_path}; cannot build any PAW correction "
                   "for this material.",
        )
        print(f"BLOCKED: no POTCAR for {material}. Aborting before any expensive computation.")
        return

    wavecar_path = data_dir / "WAVECAR"
    poscar_path = data_dir / "POSCAR"

    # ── preflight gate: cheap, few-k-point, must pass before the expensive ──
    # Nr x Nr real-space S/D construction is attempted.
    print("--- Preflight: reciprocal-space PAW overlap check (existing "
          "paw_overlap.py machinery, unmodified) ---")
    preflight = preflight_paw_overlap_check(wavecar_path, poscar_path, potcar_path, ispin=ispin)
    print(f"Preflight: {'PASSED' if preflight['passed'] else 'FAILED'} "
          f"({preflight['n_evaluated']} k-point(s) evaluated)\n")

    if not preflight["passed"]:
        _write_blocked_report(
            material, config, data_dir,
            reason="Preflight reciprocal-space PAW overlap check failed: the "
                   "augmentation correction did not substantially improve "
                   "occupied-band orthogonality at the checked k-points. This is "
                   "the expected, correct outcome for a mismatched WAVECAR/POTCAR "
                   "pair (see RESULTS.md 'data integrity incident') -- the gate "
                   "is working as intended, not itself a bug.",
            preflight=preflight,
        )
        print("BLOCKED: preflight PAW overlap check failed. Aborting before the "
              "expensive real-space S/D construction. See "
              f"{OUT / 'paw_density_matrix_report.txt'} for the full preflight "
              "table.")
        return

    # ── everything below only runs once the preflight gate has passed ──────

    wfc = vaspwfc(str(wavecar_path), lsorbit=False)
    Nx, Ny, Nz = wfc._ngrid
    Nr = Nx * Ny * Nz
    print(f"WAVECAR: nkpts={wfc._nkpts} nbands={wfc._nbands} ngrid=({Nx},{Ny},{Nz}) Nr={Nr}")

    latvec, species, counts, atom_symbols, atom_numbers, frac_coords, cart_coords = \
        read_poscar_structure(poscar_path)
    volume = abs(np.dot(latvec[0], np.cross(latvec[1], latvec[2])))
    print(f"POSCAR: volume={volume:.4f} Ang^3  atoms={atom_symbols}")

    pawpp = load_pawpp(potcar_path)
    pawpp_elements = [pp.element.split('_')[0] for pp in pawpp]
    elements_idx = [pawpp_elements.index(s) for s in atom_symbols]
    print(f"POTCAR: elements={pawpp_elements}  per-atom idx={elements_idx}")

    # k-points: EIGENVAL as in main.py
    def _read_eigenval(path, nkpts_expected, nbands_expected):
        with open(path) as fh:
            lines = fh.readlines()
        nkpts = int(lines[5].split()[1])
        nbands = int(lines[5].split()[2])
        if nkpts != nkpts_expected or nbands != nbands_expected:
            raise ValueError("EIGENVAL/WAVECAR dimension mismatch")
        kfrac = np.zeros((nkpts, 3))
        kweights = np.zeros(nkpts)
        idx = 6
        for ik in range(nkpts):
            while not lines[idx].split():
                idx += 1
            kline = lines[idx].split()
            kfrac[ik] = [float(x) for x in kline[:3]]
            kweights[ik] = float(kline[3])
            idx += 1
            idx += nbands_expected
        kweights /= kweights.sum()
        return kfrac, kweights

    kfrac_all, kweights = _read_eigenval(data_dir / "EIGENVAL", wfc._nkpts, wfc._nbands)

    # WS cell (replicate main.py; config.py values, read-only)
    center_cart, center_frac_cont, center_frac_wrapped = parse_ws_center(
        config.WS_CENTER, config.WS_CENTER_COORD_TYPE, latvec
    )
    print(f"WS center: {config.WS_CENTER} -> {np.round(center_cart, 4)} Ang")
    r_ws_cart, r_ws_frac_cont, prim_indices, translations_all = build_ws_grid_map(
        latvec, (Nx, Ny, Nz), center_cart, nmax=config.WS_TRANSLATION_SEARCH_RANGE
    )
    assert len(r_ws_cart) == Nr

    # ── Build corrected metric S (k-independent, built once) ──────────────
    print("\nBuilding real-space PAW metric S on WS grid ...")
    t0 = time.time()
    S_ws, n_img = build_real_space_S(
        pawpp, elements_idx, cart_coords, latvec, r_ws_cart,
        nmax=config.WS_TRANSLATION_SEARCH_RANGE + 1,
    )
    print(f"  done in {time.time()-t0:.1f}s, {n_img} atom-images contributed")
    herm_err_S = float(np.max(np.abs(S_ws - S_ws.conj().T)))
    print(f"  |S-S^dagger|_max = {herm_err_S:.2e}")

    # ── Build density matrix D from RAW coefficients ───────────────────────
    print("\nBuilding density matrix D (raw/un-renormalized coefficients) ...")
    D_ws, d_info = build_density_matrix(
        wfc, kfrac_all, kweights, ispin, Nr, (Nx, Ny, Nz),
        r_ws_frac_cont, prim_indices,
    )
    herm_err_D = float(np.max(np.abs(D_ws - D_ws.conj().T)))
    trace_raw_D = float(np.trace(D_ws).real)
    print(f"  |D-D^dagger|_max = {herm_err_D:.2e}  Tr(D)={trace_raw_D:.6f}")

    np.save(OUT / "S_ws.npy", S_ws)
    np.save(OUT / "D_ws_raw.npy", D_ws)

    # ── Same-dataset uncorrected baseline: plain eigh(D), no S ─────────────
    print("\nUncorrected (same-dataset) baseline: eigh(D_ws), no metric correction ...")
    eigvals_uncorr = np.linalg.eigvalsh(0.5 * (D_ws + D_ws.conj().T))
    eigvals_uncorr = np.sort(eigvals_uncorr)[::-1]
    print(f"Top 20 uncorrected: {[round(float(v), 6) for v in eigvals_uncorr[:20]]}")
    print(f"  max={eigvals_uncorr.max():.6f}  sum={eigvals_uncorr.sum():.6f}")
    np.save(OUT / "cno_occupations_uncorrected_samedata.npy", eigvals_uncorr)

    # ── Solve the CORRECT PAW-CNO occupation eigenproblem ───────────────────
    print("\nSolving PAW-CNO occupation eigenproblem (eigenvalues of D @ S, "
          "via Hermitian S^(1/2) D S^(1/2)) ...")
    t0 = time.time()
    eigvals, eigvecs, diag = solve_paw_cno(D_ws, S_ws)
    print(f"  done in {time.time()-t0:.1f}s")

    top20 = eigvals[:20]
    n_occ = int(np.sum(eigvals > 1e-6))

    print(f"Top 20 corrected CNO occupations: {[round(float(v), 6) for v in top20]}")
    print(f"Sum={diag['sum_eigvals']:.6f}  N(>1e-6)={n_occ}  "
          f"N(outside [0,1] by >1e-3)={diag['n_out_of_bounds']}")
    print(f"min={diag['min_eigval']:.6f}  max={diag['max_eigval']:.6f}")

    np.save(OUT / "cno_occupations_corrected.npy", eigvals)
    np.save(OUT / "cno_orbitals_corrected.npy", eigvecs)

    # ── Debug-only comparison: the INCORRECT generalized-eigenproblem route ─
    print("\n[debug only -- incorrect/inverse-metric test] "
          "generalized eigh(D, S) (D v = lambda S v) ...")
    t0 = time.time()
    D_ws_h = 0.5 * (D_ws + D_ws.conj().T)
    S_ws_h = 0.5 * (S_ws + S_ws.conj().T)
    try:
        eigvals_debug, _ = sc_eigh(D_ws_h, S_ws_h)
        eigvals_debug = np.sort(eigvals_debug)[::-1]
        print(f"  done in {time.time()-t0:.1f}s")
        print(f"  [debug/incorrect] top 20: {[round(float(v), 6) for v in eigvals_debug[:20]]}")
        print(f"  [debug/incorrect] max={eigvals_debug.max():.6f}  "
              f"sum={eigvals_debug.sum():.6f}")
    except np.linalg.LinAlgError as e:
        print(f"  [debug/incorrect] eigh(D,S) failed (not fatal, debug-only): {e}")
        eigvals_debug = None

    # ── trace reporting block ───────────────────────────────────────────────
    trace_paw_M = diag["trace_M"]
    trace_DS = diag["trace_DS"]
    sum_paw_cno_occupations = diag["sum_eigvals"]
    trace_expected_input = d_info["trace_expected_input"]
    trace_M_minus_eigensum = diag["trace_M_minus_eigensum"]
    trace_DS_minus_eigensum = diag["trace_DS_minus_eigensum"]
    trace_paw_minus_expected = trace_paw_M - trace_expected_input
    trace_relative_error = (trace_paw_minus_expected / trace_expected_input
                             if trace_expected_input != 0 else float("nan"))

    occupation_mode = "occ_threshold"  # RESTRICT_TO_FERMI_WINDOW guarded-out above
    if d_info["halving_triggered"]:
        trace_convention = (
            "occ_threshold selection with occupations >1.5 halved to a per-spin "
            "convention before accumulation (matches main.py); "
            "trace_expected_input is therefore NOT the spin-summed electron "
            "count -- it is the expected occupied-SPATIAL-orbital count per WS "
            "cell under the halved per-spin convention."
        )
    else:
        trace_convention = (
            "occ_threshold selection; occupations in this dataset never "
            "exceeded 1.5, so the >1.5 halving branch was a no-op -- "
            "occupations were already stored per-spin-orbital. "
            "trace_expected_input is the expected occupied-spatial-orbital "
            "count per WS cell under that as-stored per-spin convention, "
            "still not a spin-summed electron count."
        )

    print("\n--- trace reporting ---")
    print(f"trace_raw_D (Re Tr(D), raw pseudo-coefficient trace only)   = {trace_raw_D:.8f}")
    print(f"trace_paw_M (Re Tr(M), corrected matrix trace)              = {trace_paw_M:.8f}")
    print(f"trace_DS (Re Tr(D@S))                                        = {trace_DS:.8f}")
    print(f"sum_paw_cno_occupations (eigvals.sum())                      = {sum_paw_cno_occupations:.8f}")
    print(f"trace_expected_input (sum_k w_k sum_n f_nk, same convention  = {trace_expected_input:.8f}")
    print(f"                      as build_density_matrix)")
    print(f"trace_M_minus_eigensum (algebra/eigensolver check)           = {trace_M_minus_eigensum:.4e}")
    print(f"trace_DS_minus_eigensum (algebra/eigensolver check)          = {trace_DS_minus_eigensum:.4e}")
    print(f"trace_paw_minus_expected (PHYSICAL normalization check)      = {trace_paw_minus_expected:.4e}")
    print(f"trace_relative_error                                          = {trace_relative_error:.4e}")
    print(f"occupation_mode                                               = {occupation_mode}")

    # ── validation summary ──────────────────────────────────────────────────
    checks = dict(
        preflight_passed=preflight["passed"],
        S_hermitian=bool(herm_err_S < 1e-8),
        S_positive_definite=diag["s_positive_definite"],
        D_hermitian=bool(herm_err_D < 1e-8),
        M_hermitian=bool(diag["herm_err_M"] < 1e-6),
        eigenvectors_S_orthonormal=diag["s_orthonorm_passed"],
        occupations_within_01_bound=diag["bound_01_passed"],
        trace_M_eq_DS_eq_eigensum=bool(
            abs(trace_M_minus_eigensum) < 1e-6 and abs(trace_DS_minus_eigensum) < 1e-6
        ),
    )
    all_passed = all(checks.values())
    print("\n--- validation summary ---")
    for name, ok in checks.items():
        print(f"  {name:32s} {'OK' if ok else 'FAIL'}")
    print(f"  OVERALL: {'PASS' if all_passed else 'FAIL'}")

    # ── write report ─────────────────────────────────────────────────────────
    with open(OUT / "paw_density_matrix_report.txt", "w") as f:
        f.write("=== PAW-corrected CNO density matrix report ===\n\n")
        f.write(f"gate_status: PASSED (preflight) -- full computation ran\n\n")
        f.write("Correct PAW-CNO occupations are eigenvalues of S^(1/2) D S^(1/2)\n"
                "(equivalently of D @ S), NOT of the generalized problem\n"
                "D v = lambda S v -- see module docstring / solve_paw_cno() for\n"
                "the derivation.\n\n")
        f.write(f"material: {material}  output_subdir: {config.OUTPUT_SUBDIR}\n")
        f.write(f"data_dir: {data_dir}\n")
        f.write(f"ws_center: {config.WS_CENTER} ({config.WS_CENTER_COORD_TYPE}) "
                f"-> {center_cart.tolist()} Ang\n")
        f.write(f"grid: ({Nx},{Ny},{Nz})  Nr={Nr}\n")
        f.write(f"n_atom_images_in_S: {n_img}\n\n")

        f.write("--- preflight reciprocal-space PAW overlap check ---\n")
        f.write(f"kpoints_checked: {preflight['kpoints_checked']}\n")
        f.write(f"soft_potential_tol: {preflight['soft_potential_tol']:.1e}  "
                f"min_improvement_factor: {preflight['min_improvement_factor']:.1f}  "
                f"corrected_abs_tol: {preflight['corrected_abs_tol']:.1e}\n")
        for row in preflight["rows"]:
            if row.get("skipped"):
                f.write(f"  ik={row['ik']:4d}  SKIPPED ({row['reason']})\n")
            else:
                f.write(f"  ik={row['ik']:4d}  nbands={row['nbands']:3d}  "
                        f"uncorrected={row['offdiag_uncorrected']:.4e}  "
                        f"corrected={row['offdiag_corrected']:.4e}  "
                        f"{'OK' if row['passed'] else 'FAIL'} ({row['reason']})\n")
        f.write(f"preflight passed: {preflight['passed']}\n\n")

        f.write("--- Hermiticity / metric validity ---\n")
        f.write(f"herm_err_S (pre-symmetrization): {herm_err_S:.4e}\n")
        f.write(f"herm_err_D (pre-symmetrization): {herm_err_D:.4e}\n")
        f.write(f"herm_err_M (pre-symmetrization): {diag['herm_err_M']:.4e}\n")
        f.write(f"min_s_eval: {diag['min_s_eval']:.6e}   "
                f"S_positive_definite: {diag['s_positive_definite']}\n")
        f.write(f"S-metric orthonormality error of returned eigenvectors "
                f"(max|X^H S X - I|): {diag['s_orthonorm_err']:.4e}   "
                f"passed: {diag['s_orthonorm_passed']}\n\n")

        f.write("--- occupation bound ---\n")
        f.write(f"min_eigval: {diag['min_eigval']:.8f}\n")
        f.write(f"max_eigval: {diag['max_eigval']:.8f}\n")
        f.write(f"n_eigval_outside_[0,1]_by_gt_{diag['bound_01_tol']:.0e}: "
                f"{diag['n_out_of_bounds']}   passed: {diag['bound_01_passed']}\n\n")

        f.write("--- trace reporting ---\n")
        f.write("Definitions:\n")
        f.write("  trace_raw_D              = Re Tr(D) -- RAW PSEUDO-COEFFICIENT trace only;\n"
                "                              this is NOT the physical particle number.\n")
        f.write("  trace_paw_M              = Re Tr(M) -- the PAW-corrected matrix trace,\n"
                "                              M = S^(1/2) D S^(1/2).\n")
        f.write("  trace_DS                 = Re Tr(D @ S); must equal trace_paw_M.\n")
        f.write("  sum_paw_cno_occupations  = eigvals.sum(); must equal both traces above.\n")
        f.write("  trace_expected_input     = sum_k w_k * sum_n f_nk, computed inside\n"
                "                              build_density_matrix using the exact same\n"
                "                              band-selection/per-spin occupation convention\n"
                "                              used to build D.\n")
        f.write("  trace_M_minus_eigensum, trace_DS_minus_eigensum:\n"
                "                              ALGEBRA/EIGENSOLVER self-consistency checks only\n"
                "                              (matrix trace vs. its own eigenvalue sum) -- these\n"
                "                              being ~0 does not by itself validate the physics.\n")
        f.write("  trace_paw_minus_expected, trace_relative_error:\n"
                "                              the INDEPENDENT PHYSICAL NORMALIZATION check --\n"
                "                              corrected matrix trace vs. the expected occupied\n"
                "                              count computed directly from the input occupation\n"
                "                              numbers, entirely independent of the eigensolver.\n")
        f.write(f"  trace_convention          = {trace_convention}\n")
        f.write(f"  occupation_mode           = {occupation_mode}\n\n")

        f.write(f"trace_raw_D               = {trace_raw_D:.10f}\n")
        f.write(f"trace_paw_M               = {trace_paw_M:.10f}\n")
        f.write(f"trace_DS                  = {trace_DS:.10f}\n")
        f.write(f"sum_paw_cno_occupations   = {sum_paw_cno_occupations:.10f}\n")
        f.write(f"trace_expected_input      = {trace_expected_input:.10f}\n")
        f.write(f"trace_M_minus_eigensum    = {trace_M_minus_eigensum:.4e}\n")
        f.write(f"trace_DS_minus_eigensum   = {trace_DS_minus_eigensum:.4e}\n")
        f.write(f"trace_paw_minus_expected  = {trace_paw_minus_expected:.4e}\n")
        f.write(f"trace_relative_error      = {trace_relative_error:.4e}\n\n")

        f.write("--- validation summary ---\n")
        for name, ok in checks.items():
            f.write(f"  {name:32s} {'OK' if ok else 'FAIL'}\n")
        f.write(f"  OVERALL: {'PASS' if all_passed else 'FAIL'}\n\n")

        f.write(f"uncorrected (same-data, plain eigh(D), no S) max_eigval: "
                f"{eigvals_uncorr.max():.8f}\n")
        f.write(f"uncorrected top_20: {[round(float(v), 6) for v in eigvals_uncorr[:20]]}\n\n")

        f.write("top_20_corrected_cno_occupations:\n")
        for i, v in enumerate(top20):
            f.write(f"  CNO {i:3d} : {float(v):.10e}\n")
        f.write("\ncno_orbitals_corrected.npy: eigenvectors of D @ S (S-metric "
                 "orthonormal, X^H S X = I) -- NOT the plain X^H X = I convention "
                 "main.py's cno_orbitals.npy uses.\n")

        f.write("\n--- future work (not resolved here) ---\n")
        f.write("- Real-space quadrature convergence of build_real_space_S has not been\n"
                "  systematically checked against grid density (see RESULTS.md's\n"
                "  _test_beta_consistency.py note on 'a few %' residual disagreement,\n"
                "  attributed to but not rigorously isolated as quadrature error).\n")
        f.write("- The exact regional projector T^dagger P_A T (restricting the FULL PAW\n"
                "  operator, not just its plane-wave part, to the WS cell) is not\n"
                "  implemented -- build_real_space_S evaluates the augmentation term at\n"
                "  actual (possibly WS-unwrapped) Cartesian coordinates but does not\n"
                "  separately verify this equals the theoretically exact regional\n"
                "  restriction operator.\n")
        f.write("- SOC (LSORBIT=True) and Fermi-window occupation mode are both detected\n"
                "  and cause a clean BLOCKED abort (see module docstring) -- neither is\n"
                "  implemented in build_density_matrix.\n")

        if eigvals_debug is not None:
            f.write("\n[DEBUG ONLY -- INCORRECT / inverse-metric test, NOT the "
                    "physical result] generalized eigh(D, S) (D v = lambda S v):\n")
            f.write(f"  top_20: {[round(float(v), 6) for v in eigvals_debug[:20]]}\n")
            f.write(f"  max={eigvals_debug.max():.8f}  sum={eigvals_debug.sum():.8f}\n")

    print(f"\nSaved report -> {OUT / 'paw_density_matrix_report.txt'}")


def _write_blocked_report(material, config, data_dir, reason, preflight=None):
    """Write a short 'blocked' report for a guard/gate abort that happened
    BEFORE any expensive Nr x Nr computation -- no large arrays are touched
    or overwritten in this path."""
    with open(OUT / "paw_density_matrix_report.txt", "w") as f:
        f.write("=== PAW-corrected CNO density matrix report ===\n\n")
        f.write("gate_status: BLOCKED -- aborted before the expensive real-space "
                "S/D construction\n\n")
        f.write(f"material: {material}  output_subdir: {getattr(config, 'OUTPUT_SUBDIR', '?')}\n")
        f.write(f"data_dir: {data_dir}\n")
        f.write(f"LSORBIT: {getattr(config, 'LSORBIT', None)}\n")
        f.write(f"RESTRICT_TO_FERMI_WINDOW: {getattr(config, 'RESTRICT_TO_FERMI_WINDOW', None)}\n\n")
        f.write(f"reason: {reason}\n\n")
        if preflight is not None:
            f.write("--- preflight reciprocal-space PAW overlap check ---\n")
            f.write(f"kpoints_checked: {preflight['kpoints_checked']}\n")
            f.write(f"soft_potential_tol: {preflight['soft_potential_tol']:.1e}  "
                    f"min_improvement_factor: {preflight['min_improvement_factor']:.1f}  "
                    f"corrected_abs_tol: {preflight['corrected_abs_tol']:.1e}\n")
            for row in preflight["rows"]:
                if row.get("skipped"):
                    f.write(f"  ik={row['ik']:4d}  SKIPPED ({row['reason']})\n")
                else:
                    f.write(f"  ik={row['ik']:4d}  nbands={row['nbands']:3d}  "
                            f"uncorrected={row['offdiag_uncorrected']:.4e}  "
                            f"corrected={row['offdiag_corrected']:.4e}  "
                            f"{'OK' if row['passed'] else 'FAIL'} ({row['reason']})\n")
            f.write(f"preflight passed: {preflight['passed']}\n")
        f.write("\nNote: any D_ws_raw.npy / S_ws.npy / cno_*_corrected.npy files already\n"
                "present in this output/ directory are from a PREVIOUS run and are NOT\n"
                "representative of this (blocked) run -- they were not touched or\n"
                "regenerated here.\n")


if __name__ == '__main__':
    main()
