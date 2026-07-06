"""
main_v2.py — CNO density matrix via direct Fourier sum on the FFT grid.

Purpose: verify that the explicit plane-wave sum reproduces the ifftn result
to numerical precision.  Every aspect of the calculation is identical to
main.py EXCEPT that np.fft.ifftn is never called in the production path.

Convention confirmed from main.py:
  1. ifftn(cg, axes=(1,2,3)) * sqrt(Nr)  gives u_nk with sum_j |u_j|^2 = 1.
  2. Bloch phase exp(2πi k . r_frac_cont) is applied separately.
  3. Combined: psi_nk(r) = (1/sqrt(Nr)) * sum_G C_G * exp(2πi (G+k) . r_frac_cont).
  4. In WS mode, r_frac_cont is the continuous (unwrapped) fractional coord of
     each WS grid point; G.n is an integer so exp(2πi G.r_ws) = exp(2πi G.r_prim).
  5. Therefore evaluating psi directly at r_ws_frac_cont with mode='psi' and
     norm_factor=1/sqrt(Nr) is exactly equivalent to the FFT path.

Output directory: Data/<MATERIAL>/output/v2_direct_same_grid/
"""
import sys
import time
import numpy as np
from pathlib import Path

from vaspwfc import vaspwfc
from config import (
    MATERIAL, LSORBIT, ISPIN,
    RESTRICT_TO_FERMI_WINDOW, EFERMI, FERMI_WINDOW_EV,
    USE_WS_CELL, WS_CENTER, WS_CENTER_COORD_TYPE, WS_TRANSLATION_SEARCH_RANGE,
)
from ws_cell import read_poscar_structure, parse_ws_center, build_ws_grid_map
from direct_fourier import fourier_eval_bands, benchmark_one_kpoint

# ── config ────────────────────────────────────────────────────────────────────
CHUNK_SIZE      = 4096    # real-space points per Fourier-sum chunk
RUN_BENCHMARK   = True    # print estimated runtime before full loop
DIAG_N_BANDS    = 3       # bands to compare in FFT vs direct diagnostic
DIAG_N_KPTS     = 2       # k-points to test in diagnostic
SAVE_DIAG_NPZ   = True    # save point-by-point comparison to .npz

OUTPUT_SUBDIR_V2 = "v2_direct_same_grid"

# ─────────────────────────────────────────────────────────── helpers ──────────

def _read_eigenval(path, nkpts_expected, nbands_expected):
    with open(path) as fh:
        lines = fh.readlines()
    nkpts  = int(lines[5].split()[1])
    nbands = int(lines[5].split()[2])
    if nkpts != nkpts_expected or nbands != nbands_expected:
        raise ValueError(
            f"EIGENVAL has ({nkpts} k-pts, {nbands} bands) "
            f"but WAVECAR has ({nkpts_expected}, {nbands_expected})"
        )
    kfrac    = np.zeros((nkpts, 3))
    kweights = np.zeros(nkpts)
    energies = np.zeros((nkpts, nbands))
    idx = 6
    for ik in range(nkpts):
        while not lines[idx].split():
            idx += 1
        kline        = lines[idx].split()
        kfrac[ik]    = [float(x) for x in kline[:3]]
        kweights[ik] = float(kline[3])
        idx += 1
        for ib in range(nbands):
            energies[ik, ib] = float(lines[idx].split()[1])
            idx += 1
    kweights /= kweights.sum()
    return kfrac, kweights, energies


# ─────────────────────────────────────────────────────────────── setup ────────

print("=== main_v2.py: CNO via direct Fourier sum (same FFT grid) ===\n")
t_start = time.perf_counter()

ispin                    = ISPIN
restrict_to_fermi_window = RESTRICT_TO_FERMI_WINDOW
efermi                   = EFERMI if restrict_to_fermi_window else None
fermi_window_ev          = FERMI_WINDOW_EV if restrict_to_fermi_window else None

if LSORBIT:
    raise NotImplementedError(
        "main_v2.py: LSORBIT=True path is not verified for the direct Fourier sum. "
        "Set LSORBIT=False or use main.py for SOC calculations."
    )

data_dir   = Path(__file__).resolve().parent / "Data" / MATERIAL
output_dir = data_dir / "output" / OUTPUT_SUBDIR_V2
output_dir.mkdir(parents=True, exist_ok=True)

# ── WAVECAR ───────────────────────────────────────────────────────────────────
wfc = vaspwfc(str(data_dir / "WAVECAR"), lsorbit=False, lgamma=False)
Nx, Ny, Nz = wfc._ngrid
Nr = Nx * Ny * Nz
norm_v2 = 1.0 / np.sqrt(Nr)
print(f"WAVECAR  : nkpts={wfc._nkpts}  nbands={wfc._nbands}  "
      f"ngrid=({Nx},{Ny},{Nz})  Nr={Nr}")
np.save(output_dir / "fft_grid_shape.npy", np.array([Nx, Ny, Nz], dtype=int))

# ── POSCAR ────────────────────────────────────────────────────────────────────
latvec, species, counts, atom_symbols, atom_numbers, frac_coords, cart_coords = \
    read_poscar_structure(data_dir / "POSCAR")
volume = abs(np.dot(latvec[0], np.cross(latvec[1], latvec[2])))
print(f"POSCAR   : volume={volume:.4f} Å³  "
      f"atoms: {' '.join(f'{s}({c})' for s, c in zip(species, counts))}")

# ── k-points ──────────────────────────────────────────────────────────────────
eigenval_path          = data_dir / "EIGENVAL"
kfrac_all              = None
kweights               = None
band_energies          = None
kcoord_source          = None
kweight_source         = None
eigenval_mismatch_note = None
uniform_warning        = None

if eigenval_path.exists():
    try:
        kfrac_all, kweights, band_energies = _read_eigenval(
            eigenval_path, wfc._nkpts, wfc._nbands)
        kcoord_source  = "EIGENVAL"
        kweight_source = "EIGENVAL"
    except ValueError as e:
        eigenval_mismatch_note = str(e)

if kfrac_all is None:
    for attr in ("_kvecs", "_kpts", "kvecs", "kpts"):
        if hasattr(wfc, attr):
            arr = np.asarray(getattr(wfc, attr), dtype=float)
            if arr.shape == (wfc._nkpts, 3):
                kfrac_all     = arr
                kcoord_source = f"vaspwfc.{attr}"
                break
    if kfrac_all is None:
        raise RuntimeError("No fractional k-coordinates found.")

if kweights is None:
    for attr in ("_kweights", "_kwhts", "_weights", "kweights", "kwhts", "weights"):
        if hasattr(wfc, attr):
            val = getattr(wfc, attr)
            if val is not None and hasattr(val, "__len__") and len(val) == wfc._nkpts:
                w              = np.asarray(val, dtype=float)
                kweights       = w / w.sum()
                kweight_source = f"vaspwfc.{attr}"
                break

if kweights is None:
    kweights       = np.ones(wfc._nkpts, dtype=float) / wfc._nkpts
    kweight_source = "uniform fallback"
    uniform_warning = ("WARNING: uniform k-weights used — only valid for fully "
                       "unreduced BZ mesh (ISYM=0).")

np.save(output_dir / "kpoint_weights.npy", kweights)
print(f"k-points : coord={kcoord_source}  weights={kweight_source}")
if uniform_warning:
    print(f"  {uniform_warning}")

if restrict_to_fermi_window and band_energies is None:
    raise RuntimeError("Fermi-window filter requires a parseable EIGENVAL.")

# ── real-space grid (same as main.py) ─────────────────────────────────────────
# r_for_phase : (Nr, 3) fractional coords where psi is evaluated.
#   WS mode  → r_ws_frac_cont (continuous, unwrapped).
#   Prim mode → (ix/Nx, iy/Ny, iz/Nz) in C order matching mgrid[0:Nx,0:Ny,0:Nz].

r_for_phase      = None
prim_indices     = None
translations_all = None
center_cart = center_frac_cont = center_frac_wrapped = None

if USE_WS_CELL:
    center_cart, center_frac_cont, center_frac_wrapped = parse_ws_center(
        WS_CENTER, WS_CENTER_COORD_TYPE, latvec)
    print(f"WS cell  : center={WS_CENTER} ({WS_CENTER_COORD_TYPE})"
          f" → {np.round(center_cart, 4)} Å")
    print(f"           building map grid=({Nx},{Ny},{Nz})"
          f" nmax={WS_TRANSLATION_SEARCH_RANGE} ...", end="", flush=True)

    r_ws_cart, r_ws_frac_cont, prim_indices, translations_all = build_ws_grid_map(
        latvec, (Nx, Ny, Nz), center_cart, nmax=WS_TRANSLATION_SEARCH_RANGE)
    assert len(r_ws_cart) == Nr
    print(f"  done  Nr={Nr}")
    r_for_phase = r_ws_frac_cont

    np.save(output_dir / "ws_enabled.npy",            np.array(True))
    np.save(output_dir / "ws_points_cart.npy",         r_ws_cart)
    np.save(output_dir / "ws_points_frac_cont.npy",    r_ws_frac_cont)
    np.save(output_dir / "ws_base_indices.npy",         prim_indices)
    np.save(output_dir / "ws_translation_int.npy",     translations_all)
    np.save(output_dir / "ws_center_cart.npy",          center_cart)
    np.save(output_dir / "ws_center_frac_wrapped.npy",  center_frac_wrapped)
else:
    ix, iy, iz  = [a.ravel() for a in np.mgrid[0:Nx, 0:Ny, 0:Nz]]
    r_for_phase = np.column_stack([ix/Nx, iy/Ny, iz/Nz])
    np.save(output_dir / "ws_enabled.npy", np.array(False))

print()

# ─────────────────────────────── Step 0: optional benchmark ───────────────────
if RUN_BENCHMARK:
    print("--- Benchmark: timing one k-point ---")
    ik_bm  = 1
    gvec_bm = wfc.gvectors(ik_bm)
    nG_bm   = gvec_bm.shape[0]
    bands_bm = list(range(1, min(5, wfc._nbands) + 1))
    Ck_bm    = np.stack([wfc.readBandCoeff(ispin=ispin, ikpt=ik_bm, iband=ib, norm=True)
                         for ib in bands_bm])
    bm_sizes = [max(1, Nr // 4), Nr]
    benchmark_one_kpoint(Ck_bm, gvec_bm, kfrac_all[ik_bm - 1],
                         bm_sizes, chunk_size=CHUNK_SIZE)
    # Project full runtime
    n_bands_avg = max(1, wfc._nbands // 2)
    t_per_pt_approx = 2e-8  # rough estimate: 20 ns per (point, G-pair) for matmul
    t_full_est = wfc._nkpts * n_bands_avg * nG_bm * Nr * t_per_pt_approx
    print(f"    Full calc estimate: {t_full_est:.1f} s "
          f"(nkpts={wfc._nkpts}, bands≈{n_bands_avg}, nG≈{nG_bm}, Nr={Nr})")
    print()

# ────────────────────────── Step 1: diagnostic FFT vs direct Fourier ──────────
print("--- Step 1: Diagnostic comparison FFT vs direct Fourier ---")
diag_results = {}
diag_kpts = list(range(1, min(DIAG_N_KPTS + 1, wfc._nkpts + 1)))
diag_Gamma_found = False

for ik in diag_kpts:
    k_frac = kfrac_all[ik - 1]
    is_gamma = np.allclose(k_frac, 0.0, atol=1e-8)
    if is_gamma:
        diag_Gamma_found = True
    gvec = wfc.gvectors(ik)
    nG   = gvec.shape[0]
    gx   = gvec[:, 0] % Nx
    gy   = gvec[:, 1] % Ny
    gz   = gvec[:, 2] % Nz

    bands_diag = list(range(1, min(DIAG_N_BANDS + 1, wfc._nbands + 1)))
    Ck_diag    = np.stack([wfc.readBandCoeff(ispin=ispin, ikpt=ik, iband=ib, norm=True)
                           for ib in bands_diag])
    nb_d = len(bands_diag)

    # --- FFT path (same as main.py) ---
    cg = np.zeros((nb_d, Nx, Ny, Nz), dtype=np.complex128)
    cg[:, gx, gy, gz] = Ck_diag
    u_fft  = np.fft.ifftn(cg, axes=(1, 2, 3)) * np.sqrt(Nr)  # (nb, Nx, Ny, Nz)
    if prim_indices is not None:
        psi_fft = u_fft[:, prim_indices[:, 0], prim_indices[:, 1], prim_indices[:, 2]]
    else:
        psi_fft = u_fft.reshape(nb_d, Nr)
    psi_fft = psi_fft * np.exp(2j * np.pi * (r_for_phase @ k_frac))[None, :]

    # --- Direct Fourier path ---
    psi_dir = fourier_eval_bands(Ck_diag, gvec, k_frac, r_for_phase,
                                  mode='psi', norm_factor=norm_v2,
                                  chunk_size=CHUNK_SIZE, verbose=False)

    # --- Errors ---
    diff      = psi_dir - psi_fft
    max_abs   = float(np.max(np.abs(diff)))
    rms       = float(np.sqrt(np.mean(np.abs(diff)**2)))
    rel_fro   = float(np.linalg.norm(diff) / (np.linalg.norm(psi_fft) + 1e-300))
    # Phase-aligned: minimize |e^{ia} psi_dir - psi_fft|^2 globally
    dot       = np.vdot(psi_fft, psi_dir)   # sum over all bands and points
    best_phase = np.exp(-1j * np.angle(dot)) if abs(dot) > 0 else 1.0
    pa_max    = float(np.max(np.abs(best_phase * psi_dir - psi_fft)))

    norm_fft  = [float(np.linalg.norm(psi_fft[b])) for b in range(nb_d)]
    norm_dir  = [float(np.linalg.norm(psi_dir[b])) for b in range(nb_d)]

    diag_results[ik] = dict(k_frac=k_frac.tolist(), is_gamma=is_gamma,
                             max_abs=max_abs, rms=rms, rel_fro=rel_fro,
                             pa_max=pa_max, norm_fft=norm_fft, norm_dir=norm_dir)

    label = "Γ" if is_gamma else f"k={np.round(k_frac,3).tolist()}"
    print(f"  ik={ik} ({label}):  max|Δ|={max_abs:.2e}  rms={rms:.2e}"
          f"  rel_fro={rel_fro:.2e}  phase-aligned_max={pa_max:.2e}")
    for b in range(nb_d):
        print(f"    band {bands_diag[b]}: |psi_FFT|={norm_fft[b]:.8f}"
              f"  |psi_direct|={norm_dir[b]:.8f}")

    if SAVE_DIAG_NPZ:
        # Save complex values and errors for first k-point first band
        if ik == diag_kpts[0]:
            np.savez(output_dir / "diag_fft_vs_direct.npz",
                     s_frac=r_for_phase,
                     r_cart=r_for_phase @ latvec,
                     psi_fft_re=psi_fft[0].real, psi_fft_im=psi_fft[0].imag,
                     psi_dir_re=psi_dir[0].real, psi_dir_im=psi_dir[0].imag,
                     abs_error=np.abs(diff[0]))

if not diag_Gamma_found and wfc._nkpts > DIAG_N_KPTS:
    # Explicitly test Gamma if not in the first few k-points
    for ik in range(1, wfc._nkpts + 1):
        if np.allclose(kfrac_all[ik - 1], 0.0, atol=1e-8):
            k_frac = kfrac_all[ik - 1]
            gvec   = wfc.gvectors(ik)
            gx, gy, gz = gvec[:, 0] % Nx, gvec[:, 1] % Ny, gvec[:, 2] % Nz
            Ck_g   = np.stack([wfc.readBandCoeff(ispin=ispin, ikpt=ik, iband=ib, norm=True)
                                for ib in range(1, min(DIAG_N_BANDS+1, wfc._nbands+1))])
            nb_g   = Ck_g.shape[0]
            cg     = np.zeros((nb_g, Nx, Ny, Nz), dtype=np.complex128)
            cg[:, gx, gy, gz] = Ck_g
            u_g    = np.fft.ifftn(cg, axes=(1, 2, 3)) * np.sqrt(Nr)
            psi_fft_g = (u_g.reshape(nb_g, Nr) if prim_indices is None else
                         u_g[:, prim_indices[:,0], prim_indices[:,1], prim_indices[:,2]])
            psi_fft_g *= np.exp(2j * np.pi * (r_for_phase @ k_frac))[None, :]
            psi_dir_g  = fourier_eval_bands(Ck_g, gvec, k_frac, r_for_phase,
                                             mode='psi', norm_factor=norm_v2,
                                             chunk_size=CHUNK_SIZE, verbose=False)
            diff_g   = psi_dir_g - psi_fft_g
            print(f"  Γ-point (ik={ik}): max|Δ|={np.max(np.abs(diff_g)):.2e}"
                  f"  rms={np.sqrt(np.mean(np.abs(diff_g)**2)):.2e}")
            break
print()

# ──────────────────────────── Step 2: production density matrix ───────────────
print("--- Step 2: Production density matrix (direct Fourier, no ifftn) ---")
occ_tol      = 1e-6
rho          = np.zeros((Nr, Nr), dtype=np.complex128)
psi_norms_k1 = []
t_loop_start = time.perf_counter()

for ik in range(1, wfc._nkpts + 1):
    wk     = kweights[ik - 1]
    k_frac = kfrac_all[ik - 1]

    if restrict_to_fermi_window:
        mask  = np.abs(band_energies[ik - 1] - efermi) <= fermi_window_ev
        bands = np.where(mask)[0] + 1
        occ   = np.ones(len(bands), dtype=float)
        if len(bands) == 0:
            continue
    else:
        occ_all = wfc._occs[ispin - 1, ik - 1, :]
        bands   = np.where(occ_all > occ_tol)[0] + 1
        occ     = occ_all[bands - 1]
        if np.max(occ) > 1.5:
            occ = occ / 2.0

    gvec = wfc.gvectors(ik)
    nb   = len(bands)
    Ck   = np.stack([wfc.readBandCoeff(ispin=ispin, ikpt=ik, iband=ib, norm=True)
                     for ib in bands])

    # Direct Fourier: evaluates full psi at r_for_phase with norm 1/sqrt(Nr)
    psi = fourier_eval_bands(Ck, gvec, k_frac, r_for_phase,
                              mode='psi', norm_factor=norm_v2,
                              chunk_size=CHUNK_SIZE, verbose=False)

    rho += wk * (psi.T @ (occ[:, None] * psi).conj())

    if ik == 1:
        psi_norms_k1 = [float(np.linalg.norm(psi[i])) for i in range(min(3, nb))]

    if ik == 1 or ik % 20 == 0 or ik == wfc._nkpts:
        print(f"  k {ik:4d}/{wfc._nkpts}  wk={wk:.6f}  bands={nb}")

t_loop = time.perf_counter() - t_loop_start
print(f"  Loop time: {t_loop:.1f} s\n")

# ── diagonalize and save CNOs ─────────────────────────────────────────────────
np.save(output_dir / "density_matrix.npy", rho)
herm_err = float(np.max(np.abs(rho - rho.conj().T)))
tr_rho   = float(np.trace(rho).real)
print(f"rho: |rho-rho†|_max={herm_err:.2e}  Tr={tr_rho:.6f}")

eigvals, eigvecs = np.linalg.eigh(rho)
order   = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]
n_occupied = int(np.sum(eigvals > 1e-6))
top20      = eigvals[:20]

def _fmt(v):
    return 0.0 if abs(v) < 1e-12 else float(v)

print(f"Top 20: {[round(_fmt(v), 6) for v in top20]}")
print(f"Sum={eigvals.sum():.6f}  N(>1e-6)={n_occupied}")

np.save(output_dir / "cno_occupations.npy", eigvals)
np.save(output_dir / "cno_orbitals.npy",    eigvecs)

# ─────────────────────── Step 3: compare with main.py outputs ─────────────────
print("\n--- Step 3: Comparison with main.py outputs ---")
orig_dir = data_dir / "output"
try:
    from config import OUTPUT_SUBDIR as ORIG_SUBDIR
    orig_dir_full = orig_dir / ORIG_SUBDIR

    orig_rho     = np.load(orig_dir_full / "density_matrix.npy")
    orig_eigvals = np.load(orig_dir_full / "cno_occupations.npy")
    orig_eigvecs = np.load(orig_dir_full / "cno_orbitals.npy")

    rho_diff   = rho - orig_rho
    rho_fro    = float(np.linalg.norm(rho_diff, 'fro'))
    rho_rel    = rho_fro / (float(np.linalg.norm(orig_rho, 'fro')) + 1e-300)
    tr_diff    = abs(tr_rho - float(np.trace(orig_rho).real))
    herm_orig  = float(np.max(np.abs(orig_rho - orig_rho.conj().T)))

    n_cmp = min(20, len(eigvals), len(orig_eigvals))
    eig_diff = eigvals[:n_cmp] - orig_eigvals[:n_cmp]
    max_eig_diff = float(np.max(np.abs(eig_diff)))

    print(f"  Tr(rho_v2)={tr_rho:.8f}  Tr(rho_orig)={float(np.trace(orig_rho).real):.8f}"
          f"  ΔTr={tr_diff:.2e}")
    print(f"  |rho_v2 - rho_orig|_F = {rho_fro:.4e}  "
          f"rel = {rho_rel:.4e}")
    print(f"  Hermiticity: v2={herm_err:.2e}  orig={herm_orig:.2e}")
    print(f"  Max CNO eigenvalue difference (top {n_cmp}): {max_eig_diff:.4e}")
    print(f"  CNO eigenvalues comparison:")
    print(f"    {'CNO':>5}  {'v2':>14}  {'original':>14}  {'diff':>12}")
    for i in range(n_cmp):
        print(f"    {i:5d}  {_fmt(eigvals[i]):14.8f}  "
              f"{_fmt(orig_eigvals[i]):14.8f}  {eig_diff[i]:12.2e}")

    # CNO subspace overlaps for isolated and degenerate groups
    print(f"\n  CNO subspace overlaps (top 8):")
    n_ov = min(8, n_cmp)
    M = orig_eigvecs[:, :n_ov].conj().T @ eigvecs[:, :n_ov]  # (n_ov, n_ov)
    sv = np.linalg.svd(M, compute_uv=False)
    print(f"    Singular values of <phi_orig|phi_v2>: {np.round(sv, 6).tolist()}")

    # Phase-aligned overlap for (likely) isolated CNOs
    for i in range(min(4, n_ov)):
        ov = abs(float(orig_eigvecs[:, i].conj() @ eigvecs[:, i]))
        print(f"    |<phi_orig_{i}|phi_v2_{i}>| = {ov:.8f}")

    np.savez(output_dir / "comparison_with_original.npz",
             rho_frobenius=np.array(rho_fro),
             rho_relative=np.array(rho_rel),
             eig_diff=eig_diff,
             subspace_sv=sv)

except FileNotFoundError as e:
    print(f"  Comparison skipped — original output not found: {e}")

# ── metadata ─────────────────────────────────────────────────────────────────
t_total = time.perf_counter() - t_start
meta = output_dir / "cno_metadata_v2.txt"
with open(meta, "w") as f:
    f.write("=== main_v2.py metadata ===\n\n")
    f.write(f"method                  : direct Fourier sum on FFT grid\n")
    f.write(f"material                : {MATERIAL}\n")
    f.write(f"LSORBIT                 : {LSORBIT}\n")
    f.write(f"ispin                   : {ispin}\n")
    f.write(f"restrict_to_fermi_window: {restrict_to_fermi_window}\n")
    if restrict_to_fermi_window:
        f.write(f"efermi                  : {efermi} eV\n")
        f.write(f"fermi_window_ev         : {fermi_window_ev} eV\n")
    f.write(f"fft_grid                : ({Nx}, {Ny}, {Nz})\n")
    f.write(f"nr_total                : {Nr}\n")
    f.write(f"volume_Ang3             : {volume:.6f}\n")
    f.write(f"nkpts                   : {wfc._nkpts}\n")
    f.write(f"nbands                  : {wfc._nbands}\n")
    f.write(f"chunk_size              : {CHUNK_SIZE}\n")
    f.write(f"\n--- normalization ---\n")
    f.write(f"mode                    : psi (full Bloch wavefunction)\n")
    f.write(f"norm_factor             : 1/sqrt(Nr) = {norm_v2:.6e}\n")
    f.write(f"convention              : psi_nk(r) = (1/sqrt(Nr)) * sum_G C_G * exp(2pi i (G+k).r_frac)\n")
    f.write(f"sum_j |psi_j|^2         : = 1 (exactly, by Parseval on FFT grid)\n")
    f.write(f"\n--- wavefunction reconstruction ---\n")
    f.write(f"method                  : explicit Fourier sum (no ifftn in production)\n")
    f.write(f"fft_used_for            : diagnostic comparison only\n")
    f.write(f"original_uses           : u_nk (cell-periodic) then Bloch phase\n")
    f.write(f"v2_uses                 : full psi_nk directly at r_for_phase\n")
    f.write(f"equivalence             : exact (G.n = integer -> exp(2pi i G.n) = 1)\n")
    f.write(f"\n--- WS cell ---\n")
    f.write(f"USE_WS_CELL             : {USE_WS_CELL}\n")
    if USE_WS_CELL:
        f.write(f"ws_center_input         : {WS_CENTER}  ({WS_CENTER_COORD_TYPE})\n")
        f.write(f"ws_center_frac_wrapped  : {center_frac_wrapped.tolist()}\n")
        f.write(f"ws_translation_nmax     : {WS_TRANSLATION_SEARCH_RANGE}\n")
    f.write(f"\n--- diagnostic FFT vs direct ---\n")
    for ik, res in diag_results.items():
        f.write(f"ik={ik}  k={res['k_frac']}  Gamma={res['is_gamma']}\n")
        f.write(f"  max|Δ|={res['max_abs']:.3e}  rms={res['rms']:.3e}"
                f"  rel_fro={res['rel_fro']:.3e}  phase_aligned_max={res['pa_max']:.3e}\n")
        f.write(f"  norms_FFT={[f'{v:.6f}' for v in res['norm_fft']]}\n")
        f.write(f"  norms_direct={[f'{v:.6f}' for v in res['norm_dir']]}\n")
    f.write(f"\n--- density matrix ---\n")
    f.write(f"rho_hermiticity_error   : {herm_err:.4e}\n")
    f.write(f"Tr(rho)                 : {tr_rho:.8f}\n")
    if psi_norms_k1:
        f.write(f"sample_psi_norms_ik1    : {[f'{v:.6f}' for v in psi_norms_k1]}\n")
    f.write(f"\n--- CNO spectrum ---\n")
    f.write(f"sum_cno_occ             : {eigvals.sum():.10f}\n")
    f.write(f"n_eigenvalues_gt_1e-6   : {n_occupied}\n")
    f.write("top_20_cno_occupations  :\n")
    for i, v in enumerate(top20):
        f.write(f"  CNO {i:3d} : {_fmt(v):.10e}\n")
    f.write(f"\n--- performance ---\n")
    f.write(f"density_matrix_loop_s   : {t_loop:.2f}\n")
    f.write(f"total_runtime_s         : {t_total:.2f}\n")

print(f"\nSaved metadata → {meta}")
print(f"Output directory: {output_dir}")
print(f"Total runtime: {t_total:.1f} s")
