"""
check_projector_idempotency.py — structural check on main.py's density matrix.

main.py builds, per k-point, psi_nk(r) on the WS-cell grid and accumulates

    D = sum_k w_k P_a P_k P_a         where P_k = sum_n(occ at k) |psi_nk><psi_nk|

P_a is the "restrict to these Nr grid coordinates" operator (a coordinate
projection, not built as an explicit matrix here -- it's implicit in only ever
working with Nr grid points at all). P_k is the per-k occupied-Bloch-state
projector, built directly from IFFT'd, Bloch-phased WAVECAR coefficients using
the EXACT SAME convention main.py uses (readBandCoeff(norm=True), same
occupied-band selection logic, same WS-cell reindexing) -- this script does
not modify or re-run main.py, it independently reproduces its per-k building
block from the WAVECAR + main.py's own saved output, so the check is a real
cross-validation, not a tautology.

Two idempotency properties are checked:

  1. P_a^2 = P_a: trivially true as long as the Nr WS-cell grid points are
     pairwise distinct (a genuine coordinate relabelling, not a coordinate
     collapsing several physical points onto one label). Checked via
     uniqueness of the grid coordinates, not by literally building an Nr x Nr
     coordinate-projection matrix (that would just be the identity restricted
     to distinct rows -- the content of the check is in "are the rows
     distinct", which is what's verified).

  2. P_k^2 = P_k for a handful of k sampled across the mesh: this is NOT
     trivial -- it depends on the occupied psi_nk at that k actually being
     mutually orthonormal (guaranteed by unitary IFFT/scatter ONLY if the raw
     WAVECAR coefficients themselves are orthonormal at that k, which is
     exactly what breaks down for PAW potentials with large augmentation, e.g.
     W_sv or even plain W's 5d channel -- see paw_augmentation/RESULTS.md).
     This is thus an independent, first-principles way of re-detecting that
     same issue, at the level of an explicit Nr x Nr projector rather than a
     small band-pair overlap matrix.

As a third, end-to-end check, the full D = sum_k w_k P_k is independently
rebuilt from the WAVECAR (identical accumulation to main.py's own loop) and
compared against the ACTUAL saved Data/<MATERIAL>/output/<OUTPUT_SUBDIR>/
density_matrix.npy -- this validates that main.py's saved output really is
what its own formula says it should be, not just that the formula is
self-consistent in the abstract.

Read-only: does not modify or re-run main.py/config.py. Reads WAVECAR/
EIGENVAL from Data/<MATERIAL>/ and main.py's saved outputs from
Data/<MATERIAL>/output/<OUTPUT_SUBDIR>/, exactly as configured in config.py.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

PIPELINE_DIR = Path(__file__).resolve().parent.parent   # Density matrix cal/
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(PIPELINE_DIR / "helper functions"))

from vaspwfc import vaspwfc  # noqa: E402
from config import (  # noqa: E402
    MATERIAL, LSORBIT, OUTPUT_SUBDIR, ISPIN,
    RESTRICT_TO_FERMI_WINDOW, EFERMI, FERMI_WINDOW_EV,
    USE_WS_CELL,
)

N_SAMPLE_K = 5   # how many k-points (spread across the mesh) to test P_k^2=P_k on


def read_eigenval(path, nkpts_expected, nbands_expected):
    """Identical to main.py's _read_eigenval."""
    with open(path) as fh:
        lines = fh.readlines()
    nkpts = int(lines[5].split()[1])
    nbands = int(lines[5].split()[2])
    if nkpts != nkpts_expected or nbands != nbands_expected:
        raise ValueError(
            f"EIGENVAL has ({nkpts} k-pts, {nbands} bands) "
            f"but WAVECAR has ({nkpts_expected}, {nbands_expected})"
        )
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


def occupied_bands_and_occ(wfc, ik, ispin, band_energies, restrict_fermi, efermi, window, occ_tol=1e-6):
    """Identical selection logic to main.py's density-matrix loop."""
    if restrict_fermi:
        mask = np.abs(band_energies[ik - 1] - efermi) <= window
        bands = np.where(mask)[0] + 1
        occ = np.ones(len(bands), dtype=float)
    else:
        occ_all = wfc._occs[ispin - 1, ik - 1, :]
        bands = np.where(occ_all > occ_tol)[0] + 1
        occ = occ_all[bands - 1]
        if len(occ) and np.max(occ) > 1.5:
            occ = occ / 2.0
    return bands, occ


def build_psi_k(wfc, ik, ispin, bands, Nx, Ny, Nz, Nr, r_for_phase, prim_indices, k_frac):
    """Identical construction to main.py's _to_psi, for the non-SOC case."""
    gvec = wfc.gvectors(ik)
    nG = gvec.shape[0]
    gx, gy, gz = gvec[:, 0] % Nx, gvec[:, 1] % Ny, gvec[:, 2] % Nz

    Ck = np.stack([wfc.readBandCoeff(ispin=ispin, ikpt=ik, iband=int(ib), norm=True)
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
    return psi


def main():
    data_dir = PIPELINE_DIR / "Data" / MATERIAL
    output_dir = data_dir / "output" / OUTPUT_SUBDIR

    print(f"=== Projector idempotency check: {MATERIAL} / {OUTPUT_SUBDIR} ===\n")

    if LSORBIT:
        raise NotImplementedError("This check only implements the non-SOC (LSORBIT=False) "
                                  "branch of main.py's _to_psi; extend build_psi_k for SOC.")

    wfc = vaspwfc(str(data_dir / "WAVECAR"), lsorbit=LSORBIT)
    Nx, Ny, Nz = np.load(output_dir / "fft_grid_shape.npy").astype(int)
    Nr = Nx * Ny * Nz
    print(f"WAVECAR: nkpts={wfc._nkpts} nbands={wfc._nbands} ngrid=({Nx},{Ny},{Nz}) Nr={Nr}")

    band_energies = None
    if RESTRICT_TO_FERMI_WINDOW:
        # main.py requires EIGENVAL for this branch; replicate that requirement.
        kfrac_all, kweights, band_energies_full = None, None, None
        with open(data_dir / "EIGENVAL") as fh:
            lines = fh.readlines()
        nkpts = int(lines[5].split()[1])
        nbands = int(lines[5].split()[2])
        kfrac_all = np.zeros((nkpts, 3))
        kweights = np.zeros(nkpts)
        band_energies = np.zeros((nkpts, nbands))
        idx = 6
        for ik in range(nkpts):
            while not lines[idx].split():
                idx += 1
            kline = lines[idx].split()
            kfrac_all[ik] = [float(x) for x in kline[:3]]
            kweights[ik] = float(kline[3])
            idx += 1
            for ib in range(nbands):
                band_energies[ik, ib] = float(lines[idx].split()[1])
                idx += 1
        kweights /= kweights.sum()
    else:
        kfrac_all, kweights = read_eigenval(data_dir / "EIGENVAL", wfc._nkpts, wfc._nbands)

    kweights_saved = np.load(output_dir / "kpoint_weights.npy")
    if not np.allclose(kweights, kweights_saved, atol=1e-8):
        print("  WARNING: recomputed k-weights differ from the saved kpoint_weights.npy "
              "-- EIGENVAL may have changed since main.py's run.")

    # Trust THIS run's own recorded metadata for whether WS-cell mode was used,
    # not live config.py (which may have changed since) and not file existence
    # (a later USE_WS_CELL=False run reusing the same OUTPUT_SUBDIR leaves any
    # earlier run's ws_*.npy files sitting there untouched -- main.py only
    # writes them inside `if USE_WS_CELL:`, so their mere presence does NOT
    # mean this specific run used them).
    meta_text = (output_dir / "cno_metadata.txt").read_text()
    ws_line = next(line for line in meta_text.splitlines() if line.startswith("USE_WS_CELL"))
    ws_enabled = ws_line.split(":", 1)[1].strip() == "True"
    print(f"  (from {output_dir/'cno_metadata.txt'}: USE_WS_CELL={ws_enabled} for this run)")
    if ws_enabled:
        r_for_phase = np.load(output_dir / "ws_points_frac_cont.npy")
        prim_indices = np.load(output_dir / "ws_base_indices.npy").astype(int)
    else:
        ix, iy, iz = [a.ravel() for a in np.mgrid[0:Nx, 0:Ny, 0:Nz]]
        r_for_phase = np.column_stack([ix / Nx, iy / Ny, iz / Nz])
        prim_indices = None

    # ── Check 1: P_a^2 = P_a  (grid-point distinctness) ────────────────────────
    print(f"\n[1] P_a^2 = P_a check: are the {Nr} grid coordinates pairwise distinct?")
    r_round = np.round(r_for_phase, 6)
    n_unique = len(np.unique(r_round, axis=0))
    print(f"    unique coordinates: {n_unique} / {Nr}  "
          f"{'OK -- P_a is a genuine (idempotent) coordinate projection' if n_unique == Nr else 'FAILED -- duplicate grid points, P_a not idempotent'}")

    # ── Check 2: P_k^2 = P_k for a sample of k-points ──────────────────────────
    sample_iks = sorted(set(
        [1] + [max(1, min(wfc._nkpts, round(f * wfc._nkpts))) for f in (0.25, 0.5, 0.75, 1.0)]
    ))[:N_SAMPLE_K]
    print(f"\n[2] P_k^2 = P_k check at {len(sample_iks)} sampled k-points: {sample_iks}")
    t0 = time.time()
    for ik in sample_iks:
        bands, occ = occupied_bands_and_occ(
            wfc, ik, ISPIN, band_energies, RESTRICT_TO_FERMI_WINDOW, EFERMI, FERMI_WINDOW_EV
        )
        if len(bands) == 0:
            print(f"    ik={ik:4d}  no occupied bands -- skipped")
            continue
        psi = build_psi_k(wfc, ik, ISPIN, bands, Nx, Ny, Nz, Nr,
                           r_for_phase, prim_indices, kfrac_all[ik - 1])
        # P_k = sum_n(occ) |psi_n><psi_n| -- binary occupied-subspace projector,
        # i.e. occ folded in as 0/1 inclusion (not as a fractional weight), since
        # idempotency is a property of the OCCUPIED-SUBSPACE projector, not of
        # main.py's k-weighted, occupation-weighted accumulator.
        P_k = psi.T @ psi.conj()
        resid = P_k @ P_k - P_k
        err = float(np.max(np.abs(resid)))
        norm_Pk = float(np.max(np.abs(P_k)))
        print(f"    ik={ik:4d}  n_occ={len(bands):3d}  "
              f"max|P_k^2 - P_k|={err:.4e}  (max|P_k| entry ={norm_Pk:.4e})  "
              f"+{time.time()-t0:.1f}s")

    # ── Check 3: rebuild full D = sum_k w_k P_k, compare to main.py's saved D ──
    print(f"\n[3] Rebuilding full D = sum_k w_k P_k from the WAVECAR "
          f"({wfc._nkpts} k-points) and comparing to the saved density_matrix.npy ...")
    D_saved = np.load(output_dir / "density_matrix.npy")
    D_check = np.zeros((Nr, Nr), dtype=np.complex128)
    t0 = time.time()
    for ik in range(1, wfc._nkpts + 1):
        bands, occ = occupied_bands_and_occ(
            wfc, ik, ISPIN, band_energies, RESTRICT_TO_FERMI_WINDOW, EFERMI, FERMI_WINDOW_EV
        )
        if len(bands) == 0:
            continue
        psi = build_psi_k(wfc, ik, ISPIN, bands, Nx, Ny, Nz, Nr,
                           r_for_phase, prim_indices, kfrac_all[ik - 1])
        D_check += kweights[ik - 1] * (psi.T @ (occ[:, None] * psi).conj())
        if ik == 1 or ik % 20 == 0 or ik == wfc._nkpts:
            print(f"    k {ik:4d}/{wfc._nkpts}  elapsed={time.time()-t0:.1f}s")

    diff = D_check - D_saved
    max_abs_diff = float(np.max(np.abs(diff)))
    max_rel_diff = max_abs_diff / float(np.max(np.abs(D_saved)))
    tr_check = float(np.trace(D_check).real)
    tr_saved = float(np.trace(D_saved).real)
    print(f"\n    max|D_check - D_saved|       = {max_abs_diff:.4e}")
    print(f"    max|diff| / max|D_saved|      = {max_rel_diff:.4e}")
    print(f"    Tr(D_check) = {tr_check:.8f}   Tr(D_saved) = {tr_saved:.8f}   "
          f"|diff| = {abs(tr_check - tr_saved):.4e}")
    print("    " + ("OK -- saved D matches independent rebuild"
                     if max_abs_diff < 1e-8 else
                     "MISMATCH -- saved D does not match an independent rebuild "
                     "of main.py's own formula (check EIGENVAL/config haven't "
                     "changed since that run)"))


if __name__ == '__main__':
    main()
