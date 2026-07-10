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
Bloch phase needed, because S_hat is a k-independent local real-space
operator (unlike the band-pair overlap, which lives at fixed k). This reuses
only pawpotcar's radial projector splines + Qij (paw.py) and sph_harm.sph_r
-- no pysbt, no real-space AE partial-wave reconstruction.

Does NOT run or modify main.py/config.py; reads config.py and ws_cell.py
(read-only imports) to reproduce the exact WS-cell setup for whatever
MATERIAL/OUTPUT_SUBDIR config.py currently specifies, and writes all outputs
under paw_augmentation/output/, never under Data/*/output/. Reads WAVECAR/
POSCAR/POTCAR/EIGENVAL from Density matrix cal/Data/<MATERIAL>/ by default
(live data) -- pass an explicit data_dir to main() to point at a snapshot
instead (e.g. for a frozen/mismatched-data regression check).
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "helper functions"))  # for ws_cell
from ws_cell import read_poscar_structure, parse_ws_center, build_ws_grid_map  # noqa: E402

from vaspwfc import vaspwfc  # noqa: E402
from sph_harm import sph_r  # noqa: E402
from paw_overlap import load_pawpp  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)


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


def build_density_matrix(wfc, kfrac_all, kweights, ispin, Nr, ngrid,
                          r_for_phase, prim_indices, occ_tol=1e-6, verbose=True):
    """Same accumulation as main.py's rho loop, but using RAW (norm=False)
    band coefficients -- correct convention for pairing with the PAW-corrected
    metric S (see module docstring)."""
    Nx, Ny, Nz = ngrid
    D = np.zeros((Nr, Nr), dtype=np.complex128)
    t0 = time.time()
    t_prev = t0

    for ik in range(1, wfc._nkpts + 1):
        wk = kweights[ik - 1]
        k_frac = kfrac_all[ik - 1]

        occ_all = wfc._occs[ispin - 1, ik - 1, :]
        bands = np.where(occ_all > occ_tol)[0] + 1
        occ = occ_all[bands - 1]
        if np.max(occ) > 1.5:
            occ = occ / 2.0
        if len(bands) == 0:
            continue

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

    return D


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

    D = 0.5 * (D + D.conj().T)
    S = 0.5 * (S + S.conj().T)
    print(f"  [1/6] Hermitized D, S ({Nr}x{Nr})  +{time.time()-t0:.1f}s")

    t1 = time.time()
    print(f"  [2/6] diagonalizing S (eigh, {Nr}x{Nr}) ...")
    s_eval, s_vec = np.linalg.eigh(S)
    print(f"        done  +{time.time()-t1:.1f}s")
    if s_eval.min() <= min_s_eval_tol:
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
    M = 0.5 * (M + M.conj().T)
    print(f"        done  +{time.time()-t1:.1f}s")

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

    tr_DS = float(np.trace(D @ S).real)
    diag = dict(
        min_s_eval=float(s_eval.min()),
        max_eigval=float(eigvals.max()),
        min_eigval=float(eigvals.min()),
        n_out_of_bounds=int(np.sum((eigvals < -1e-3) | (eigvals > 1 + 1e-3))),
        sum_eigvals=float(eigvals.sum()),
        tr_DS=tr_DS,
        sum_vs_tr_DS_diff=float(abs(eigvals.sum() - tr_DS)),
        s_orthonorm_err=s_orthonorm_err,
    )
    return eigvals, eigvecs, diag


def main(data_dir=None):
    import config  # read-only

    material = config.MATERIAL
    ispin = config.ISPIN
    # Live data by default -- Data/WSe2_mono now has a matching plain-W
    # WAVECAR+POTCAR pair (the earlier mismatch from a concurrent VASP run
    # has been resolved), so the data_snapshot/ workaround from that incident
    # is no longer used unless a caller explicitly passes data_dir (e.g. to
    # rerun against a frozen snapshot for comparison).
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "Data" / material
    else:
        data_dir = Path(data_dir)

    print(f"=== PAW-corrected CNO density matrix: {material} / "
          f"{config.OUTPUT_SUBDIR} ===\n")

    wfc = vaspwfc(str(data_dir / "WAVECAR"), lsorbit=config.LSORBIT)
    Nx, Ny, Nz = wfc._ngrid
    Nr = Nx * Ny * Nz
    print(f"WAVECAR: nkpts={wfc._nkpts} nbands={wfc._nbands} ngrid=({Nx},{Ny},{Nz}) Nr={Nr}")

    latvec, species, counts, atom_symbols, atom_numbers, frac_coords, cart_coords = \
        read_poscar_structure(data_dir / "POSCAR")
    volume = abs(np.dot(latvec[0], np.cross(latvec[1], latvec[2])))
    print(f"POSCAR: volume={volume:.4f} Ang^3  atoms={atom_symbols}")

    pawpp = load_pawpp(data_dir / "POTCAR")
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
    D_ws = build_density_matrix(
        wfc, kfrac_all, kweights, ispin, Nr, (Nx, Ny, Nz),
        r_ws_frac_cont, prim_indices,
    )
    herm_err_D = float(np.max(np.abs(D_ws - D_ws.conj().T)))
    tr_D = float(np.trace(D_ws).real)
    print(f"  |D-D^dagger|_max = {herm_err_D:.2e}  Tr(D)={tr_D:.6f}")

    N_check = np.trace(D_ws @ S_ws).real
    print(f"  Tr(D S) (should be close to total occupied-electron count) = {N_check:.6f}")

    np.save(OUT / "S_ws.npy", S_ws)
    np.save(OUT / "D_ws_raw.npy", D_ws)

    # ── Same-dataset uncorrected baseline: plain eigh(D), no S ─────────────
    # Isolates exactly the effect of the S correction (identical D matrix,
    # built from the identical WAVECAR snapshot) rather than comparing
    # against the pre-existing Data/WSe2_mono/output/W_center CNO result,
    # which was built by main.py from a different (older) WAVECAR state
    # (norm=True convention, 17 bands/13 occupied vs this snapshot's 15
    # bands/9 occupied) -- see RESULTS.md.
    print("\nUncorrected (same-dataset) baseline: eigh(D_ws), no metric correction ...")
    eigvals_uncorr = np.linalg.eigvalsh(0.5 * (D_ws + D_ws.conj().T))
    eigvals_uncorr = np.sort(eigvals_uncorr)[::-1]
    print(f"Top 20 uncorrected: {[round(float(v), 6) for v in eigvals_uncorr[:20]]}")
    print(f"  max={eigvals_uncorr.max():.6f}  sum={eigvals_uncorr.sum():.6f}")
    np.save(OUT / "cno_occupations_uncorrected_samedata.npy", eigvals_uncorr)

    # ── Solve the CORRECT PAW-CNO occupation eigenproblem: eigenvalues of ──
    # D @ S (via Hermitian M = S^(1/2) D S^(1/2)) -- see module docstring and
    # solve_paw_cno() for why this replaces the generalized problem
    # D v = lambda S v used by an earlier version of this script.
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
    print(f"min(S eigenvalue)={diag['min_s_eval']:.4e}  "
          f"Tr(D S)={diag['tr_DS']:.6f}  "
          f"|sum(eigvals) - Tr(D S)|={diag['sum_vs_tr_DS_diff']:.4e}  "
          f"S-orthonormality error of eigenvectors={diag['s_orthonorm_err']:.4e}")

    np.save(OUT / "cno_occupations_corrected.npy", eigvals)
    # NOTE: these are eigenvectors of DS / M=S^(1/2)DS^(1/2) (see solve_paw_cno),
    # normalized as v^H S v = 1 (S-metric), NOT the plain v^H v = 1 convention
    # main.py's cno_orbitals.npy uses. Do not feed directly into export_cubes.py
    # or other main.py-pipeline tools without accounting for this -- the
    # real-space amplitude/normalization differs from the uncorrected convention.
    np.save(OUT / "cno_orbitals_corrected.npy", eigvecs)

    # ── Debug-only comparison: the INCORRECT generalized-eigenproblem route ─
    # D v = lambda S v (equivalent to eigenvalues of S^(-1/2) D S^(-1/2), the
    # INVERSE-metric direction) is NOT the physical occupation spectrum for
    # this D/S construction -- see module docstring. Kept only so the wrong
    # and right answers can be compared side by side; never use eigvals_debug
    # as the reported result.
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

    with open(OUT / "paw_density_matrix_report.txt", "w") as f:
        f.write("=== PAW-corrected CNO density matrix report ===\n\n")
        f.write("Correct PAW-CNO occupations are eigenvalues of S^(1/2) D S^(1/2)\n"
                "(equivalently of D @ S), NOT of the generalized problem\n"
                "D v = lambda S v -- see module docstring / solve_paw_cno() for\n"
                "the derivation.\n\n")
        f.write(f"material: {material}  output_subdir: {config.OUTPUT_SUBDIR}\n")
        f.write(f"data_dir: {data_dir}\n")
        f.write(f"ws_center: {config.WS_CENTER} ({config.WS_CENTER_COORD_TYPE}) "
                f"-> {center_cart.tolist()} Ang\n")
        f.write(f"grid: ({Nx},{Ny},{Nz})  Nr={Nr}\n")
        f.write(f"n_atom_images_in_S: {n_img}\n")
        f.write(f"herm_err_S: {herm_err_S:.4e}\n")
        f.write(f"herm_err_D: {herm_err_D:.4e}\n")
        f.write(f"min_s_eval: {diag['min_s_eval']:.6e}\n")
        f.write(f"Tr(D): {tr_D:.8f}\n")
        f.write(f"Tr(D S): {diag['tr_DS']:.8f}\n")
        f.write(f"\nuncorrected (same-data, plain eigh(D), no S) max_eigval: "
                f"{eigvals_uncorr.max():.8f}\n")
        f.write(f"uncorrected top_20: {[round(float(v), 6) for v in eigvals_uncorr[:20]]}\n")
        f.write(f"\nsum_corrected_eigvals: {diag['sum_eigvals']:.8f}\n")
        f.write(f"n_eigval_gt_1e-6: {n_occ}\n")
        f.write(f"n_eigval_outside_[0,1]_by_gt_1e-3: {diag['n_out_of_bounds']}\n")
        f.write(f"min_eigval: {diag['min_eigval']:.8f}\n")
        f.write(f"max_eigval: {diag['max_eigval']:.8f}\n")
        f.write(f"|sum(eigvals) - Tr(D S)|: {diag['sum_vs_tr_DS_diff']:.4e}\n")
        f.write(f"S-metric orthonormality error of returned eigenvectors "
                f"(max|eigvecs^H S eigvecs - I|): {diag['s_orthonorm_err']:.4e}\n")
        f.write("top_20_corrected_cno_occupations:\n")
        for i, v in enumerate(top20):
            f.write(f"  CNO {i:3d} : {float(v):.10e}\n")
        f.write("\ncno_orbitals_corrected.npy: eigenvectors of D @ S (S-metric "
                 "orthonormal, v^H S v = 1) -- NOT the plain v^H v = 1 convention "
                 "main.py's cno_orbitals.npy uses.\n")
        if eigvals_debug is not None:
            f.write("\n[DEBUG ONLY -- INCORRECT / inverse-metric test, NOT the "
                    "physical result] generalized eigh(D, S) (D v = lambda S v):\n")
            f.write(f"  top_20: {[round(float(v), 6) for v in eigvals_debug[:20]]}\n")
            f.write(f"  max={eigvals_debug.max():.8f}  sum={eigvals_debug.sum():.8f}\n")

    print(f"\nSaved report -> {OUT / 'paw_density_matrix_report.txt'}")


if __name__ == '__main__':
    main()
