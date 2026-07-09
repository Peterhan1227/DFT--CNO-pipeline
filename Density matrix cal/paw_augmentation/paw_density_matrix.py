"""
PAW-augmentation-corrected real-space CNO density matrix.

Extends paw_overlap.py's band-pair overlap correction to a full site-restricted
(WS-cell) density-matrix / natural-orbital-occupation calculation, following
TASK_BRIEF.md section 2's "generalized eigenvalue problem" recommendation:

    D v = lambda * S v

instead of main.py's plain  rho v = lambda v.

Key idea (derivation in RESULTS.md): treat the Nr real-space FFT/WS grid
points as a basis {|r>}. Because IFFT is a unitary map between the
(zero-padded) plane-wave coefficient vector and the real-space grid vector
(Parseval), this basis is *exactly* orthonormal under the plane-wave-only
inner product -- so D (built exactly as main.py builds its density matrix,
but from RAW un-renormalized coefficients) needs no change. Only the metric
changes:

    S[r, r'] = delta(r, r') + sum_{atom images R} sum_ij  p~_i(r - R) Qij p~_j*(r' - R)

This is the position-space representation of the PAW operator
S_hat = 1 + sum_i |p~_i> Qij <p~_j|, evaluated directly at the ACTUAL
(possibly WS-cell-unwrapped, multi-cell) Cartesian grid coordinates -- no
Bloch phase needed, because S_hat is a k-independent local real-space
operator (unlike the band-pair overlap, which lives at fixed k). This reuses
only pawpotcar's radial projector splines + Qij (paw.py) and sph_harm.sph_r
-- no pysbt, no real-space AE partial-wave reconstruction.

Does NOT run or modify main.py/config.py; reads config.py and ws_cell.py
(read-only imports) to reproduce the exact WSe2 W_center setup, and writes
all outputs under paw_augmentation/output/, never under Data/*/output/.
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh as sc_eigh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for config, ws_cell
from ws_cell import read_poscar_structure, parse_ws_center, build_ws_grid_map  # noqa: E402

from vaspwfc import vaspwfc  # noqa: E402
from sph_harm import sph_r  # noqa: E402
from paw_overlap import load_pawpp  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)


def build_real_space_S(pawpp, elements_idx, atom_cart, latvec, r_grid_cart,
                        nmax=4, dist_prune=16.0):
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

    n_images_used = 0
    for iatom in range(natoms):
        pp = pawpp[elements_idx[iatom]]
        rmax_eff = pp.proj_rmax * (pp.NPSRNL - 1) / pp.NPSRNL
        Qij = pp.get_Qij()

        images_cart = atom_cart[iatom] + all_n_cart  # (nimg, 3)
        d_centroid = np.linalg.norm(images_cart - centroid[None, :], axis=1)
        candidate_idx = np.where(d_centroid < dist_prune)[0]

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

        if verbose and (ik == 1 or ik % 40 == 0 or ik == wfc._nkpts):
            print(f"  k {ik:4d}/{wfc._nkpts}  wk={wk:.6f}  bands={nb}  "
                  f"elapsed={time.time()-t0:.1f}s")

    return D


def main():
    import config  # read-only

    material = config.MATERIAL
    ispin = config.ISPIN
    # NOTE: reading from the frozen snapshot, not Data/<material> directly --
    # see diagnostics.py header / RESULTS.md: Data/WSe2_mono/WAVECAR was
    # overwritten mid-task by what looks like the user's own concurrent VASP
    # work, so all results in this folder are pinned to a snapshot taken at
    # 2026-07-09T14:20:28Z for internal self-consistency.
    data_dir = Path(__file__).resolve().parent / "data_snapshot" / material

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

    # ── Solve generalized eigenproblem D v = lambda S v ────────────────────
    print("\nSolving generalized eigenproblem D v = lambda S v ...")
    t0 = time.time()
    D_ws_h = 0.5 * (D_ws + D_ws.conj().T)
    S_ws_h = 0.5 * (S_ws + S_ws.conj().T)
    try:
        eigvals, eigvecs = sc_eigh(D_ws_h, S_ws_h)
    except np.linalg.LinAlgError as e:
        print(f"  eigh(D,S) failed: {e}")
        raise
    print(f"  done in {time.time()-t0:.1f}s")

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    top20 = eigvals[:20]
    n_occ = int(np.sum(eigvals > 1e-6))
    n_out_of_bounds = int(np.sum((eigvals < -1e-3) | (eigvals > 1 + 1e-3)))

    print(f"Top 20 corrected CNO occupations: {[round(float(v), 6) for v in top20]}")
    print(f"Sum={eigvals.sum():.6f}  N(>1e-6)={n_occ}  "
          f"N(outside [0,1] by >1e-3)={n_out_of_bounds}")
    print(f"min={eigvals.min():.6f}  max={eigvals.max():.6f}")

    np.save(OUT / "cno_occupations_corrected.npy", eigvals)
    # NOTE: these are eigenvectors of the GENERALIZED problem D v = lambda S v,
    # normalized as v^H S v = 1 (S-metric), NOT the plain v^H v = 1 convention
    # main.py's cno_orbitals.npy uses. Do not feed directly into export_cubes.py
    # or other main.py-pipeline tools without accounting for this -- the
    # real-space amplitude/normalization differs from the uncorrected convention.
    np.save(OUT / "cno_orbitals_corrected.npy", eigvecs)

    with open(OUT / "paw_density_matrix_report.txt", "w") as f:
        f.write("=== PAW-corrected CNO density matrix report ===\n\n")
        f.write(f"material: {material}  output_subdir: {config.OUTPUT_SUBDIR}\n")
        f.write(f"ws_center: {config.WS_CENTER} ({config.WS_CENTER_COORD_TYPE}) "
                f"-> {center_cart.tolist()} Ang\n")
        f.write(f"grid: ({Nx},{Ny},{Nz})  Nr={Nr}\n")
        f.write(f"n_atom_images_in_S: {n_img}\n")
        f.write(f"herm_err_S: {herm_err_S:.4e}\n")
        f.write(f"herm_err_D: {herm_err_D:.4e}\n")
        f.write(f"Tr(D): {tr_D:.8f}\n")
        f.write(f"Tr(D S): {N_check:.8f}\n")
        f.write(f"\nuncorrected (same-data, plain eigh(D), no S) max_eigval: "
                f"{eigvals_uncorr.max():.8f}\n")
        f.write(f"uncorrected top_20: {[round(float(v), 6) for v in eigvals_uncorr[:20]]}\n")
        f.write(f"\nsum_corrected_eigvals: {eigvals.sum():.8f}\n")
        f.write(f"n_eigval_gt_1e-6: {n_occ}\n")
        f.write(f"n_eigval_outside_[0,1]_by_gt_1e-3: {n_out_of_bounds}\n")
        f.write(f"min_eigval: {eigvals.min():.8f}\n")
        f.write(f"max_eigval: {eigvals.max():.8f}\n")
        f.write("top_20_corrected_cno_occupations:\n")
        for i, v in enumerate(top20):
            f.write(f"  CNO {i:3d} : {float(v):.10e}\n")
        f.write("\ncno_orbitals_corrected.npy: eigenvectors of D v = lambda S v, "
                 "normalized v^H S v = 1 (S-metric) -- NOT the plain v^H v = 1 "
                 "convention main.py's cno_orbitals.npy uses.\n")

    print(f"\nSaved report -> {OUT / 'paw_density_matrix_report.txt'}")


if __name__ == '__main__':
    main()
