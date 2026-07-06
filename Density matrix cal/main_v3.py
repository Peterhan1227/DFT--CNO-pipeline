"""
main_v3.py — CNO density matrix via direct Fourier sum on a new WS-cell grid.

Purpose: avoid (1) evaluating psi only in the primitive FFT cell, (2) folding
grid points into the WS cell, and (3) adding manual translation phases.
Instead, build a uniform Cartesian grid that covers the chosen Wigner-Seitz
cell and evaluate the full Bloch wavefunction directly at those positions.

Physical convention:
  psi_nk(s) = (1/sqrt(Omega)) * sum_G C_G * exp(2pi i (G+k).s)
  Each retained WS point carries weight w = dx*dy*dz (Cartesian voxel volume).
  Wavefunctions are weighted by sqrt(w) before forming the density matrix so
  that sum_j |psi_j|^2 * w ≈ 1 and Tr(rho) ≈ n_occ regardless of resolution.

Fractional coordinates s = A^{-1} r are NOT reduced modulo 1; evaluating the
full (k+G).s exponential at the actual WS coordinates automatically includes
the correct Bloch translation phase.

Grid settings (edit here or pass via CLI flags — see below):
  WS_GRID_SHAPE          = None   -> compute from WS_GRID_SPACING_ANGSTROM
  WS_GRID_SPACING_ANGSTROM = None -> derive from FFT grid spacing

Output directory: Data/<MATERIAL>/output/v3_direct_ws_grid/
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
from ws_cell import read_poscar_structure, parse_ws_center
from direct_fourier import fourier_eval_bands, ws_membership

# ── grid configuration ────────────────────────────────────────────────────────
# Set WS_GRID_SHAPE = (nx, ny, nz) for an explicit bounding-box grid size.
# Set WS_GRID_SPACING_ANGSTROM for a target Cartesian spacing.
# If both are None, the code derives a spacing matching the FFT grid.
WS_GRID_SHAPE           = None    # e.g. (40, 40, 40)
WS_GRID_SPACING_ANGSTROM = 0.25   # e.g. 0.25

CHUNK_SIZE          = 4096        # Fourier-sum chunk over real-space points
WS_NMAX             = 2           # lattice-translation search range for WS test
WS_MARGIN           = 0.05        # fractional margin around bounding box
RHO_MEMORY_WARN_GB  = 4.0         # warn if density matrix exceeds this
OUTPUT_SUBDIR_V3    = "v3_direct_ws_grid"

# Convergence mode: set to a list of spacings (Å) to run at multiple resolutions
CONVERGENCE_SPACINGS = None  # e.g. [0.5, 0.35, 0.25]

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


def build_ws_bounding_box(center_cart, latvec, margin=0.05):
    """
    Compute a Cartesian bounding box that fully encloses the WS cell.

    The WS cell is bounded by bisector planes to lattice vectors R with |n|≤1.
    A safe bounding sphere has radius R_max = max(|R|/2) + margin*R_max.

    Returns
    -------
    box_lo : (3,) Cartesian lower bounds
    box_hi : (3,) Cartesian upper bounds
    R_max  : float — bounding sphere radius
    """
    ns = np.array([-1, 0, 1])
    n1, n2, n3 = np.meshgrid(ns, ns, ns, indexing='ij')
    all_n = np.column_stack([n1.ravel(), n2.ravel(), n3.ravel()])
    all_n = all_n[np.any(all_n != 0, axis=1)]
    R_cart = all_n @ latvec                           # (26, 3)
    R_norms = np.linalg.norm(R_cart, axis=1)
    R_max   = 0.5 * R_norms.max() * (1.0 + margin)

    box_lo = center_cart - R_max
    box_hi = center_cart + R_max
    return box_lo, box_hi, R_max


def build_ws_grid(center_cart, latvec, spacing, nmax=2, margin=0.05):
    """
    Build a uniform Cartesian grid covering the WS cell.

    Algorithm:
      1. Compute WS bounding box.
      2. Generate uniform Cartesian grid in box.
      3. Test WS membership for each point.
      4. Retain only inside-WS points.

    Parameters
    ----------
    center_cart : (3,) Cartesian WS center
    latvec      : (3, 3) lattice vectors
    spacing     : float — target Cartesian grid spacing in Angstrom
    nmax        : int   — translation search range for WS test
    margin      : float — fractional margin on bounding box

    Returns
    -------
    r_ws_cart   : (n_ws, 3) Cartesian coordinates of WS grid points
    s_ws_frac   : (n_ws, 3) fractional coordinates (NOT reduced mod 1)
    box_shape   : (3,) int — full bounding-box grid shape (nx, ny, nz)
    dV          : float — Cartesian voxel volume (Ang^3)
    """
    box_lo, box_hi, R_max = build_ws_bounding_box(center_cart, latvec, margin)

    nx = max(3, int(np.ceil((box_hi[0] - box_lo[0]) / spacing)))
    ny = max(3, int(np.ceil((box_hi[1] - box_lo[1]) / spacing)))
    nz = max(3, int(np.ceil((box_hi[2] - box_lo[2]) / spacing)))
    box_shape = (nx, ny, nz)

    dx = (box_hi[0] - box_lo[0]) / nx
    dy = (box_hi[1] - box_lo[1]) / ny
    dz = (box_hi[2] - box_lo[2]) / nz
    dV = dx * dy * dz

    # Build Cartesian grid: half-open [lo, hi)
    x = box_lo[0] + dx * (np.arange(nx) + 0.5)
    y = box_lo[1] + dy * (np.arange(ny) + 0.5)
    z = box_lo[2] + dz * (np.arange(nz) + 0.5)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    r_box = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # WS membership test
    mask = ws_membership(r_box, center_cart, latvec, nmax=nmax)

    r_ws_cart = r_box[mask]
    # Convert to fractional coordinates (NOT reduced mod 1)
    A_inv = np.linalg.inv(latvec)
    s_ws_frac = r_ws_cart @ A_inv

    return r_ws_cart, s_ws_frac, box_shape, dV


# ─────────────────────────────────────────────────────────────── setup ────────

print("=== main_v3.py: CNO via direct Fourier sum on new WS-cell grid ===\n")
t_start = time.perf_counter()

ispin                    = ISPIN
restrict_to_fermi_window = RESTRICT_TO_FERMI_WINDOW
efermi                   = EFERMI if restrict_to_fermi_window else None
fermi_window_ev          = FERMI_WINDOW_EV if restrict_to_fermi_window else None

if LSORBIT:
    raise NotImplementedError(
        "main_v3.py: LSORBIT=True is not verified for the direct Fourier sum. "
        "Use main.py for SOC calculations."
    )

data_dir   = Path(__file__).resolve().parent / "Data" / MATERIAL
output_dir = data_dir / "output" / OUTPUT_SUBDIR_V3
output_dir.mkdir(parents=True, exist_ok=True)

# ── WAVECAR ───────────────────────────────────────────────────────────────────
wfc = vaspwfc(str(data_dir / "WAVECAR"), lsorbit=False, lgamma=False)
Nx_fft, Ny_fft, Nz_fft = wfc._ngrid
Nr_fft = Nx_fft * Ny_fft * Nz_fft
print(f"WAVECAR  : nkpts={wfc._nkpts}  nbands={wfc._nbands}  "
      f"ngrid=({Nx_fft},{Ny_fft},{Nz_fft})  encut={wfc._encut:.1f}")

# ── POSCAR ────────────────────────────────────────────────────────────────────
latvec, species, counts, atom_symbols, atom_numbers, frac_coords, cart_coords = \
    read_poscar_structure(data_dir / "POSCAR")
volume = abs(np.dot(latvec[0], np.cross(latvec[1], latvec[2])))
norm_v3_physical = 1.0 / np.sqrt(volume)   # physical normalization: |psi|^2 integrates to 1
print(f"POSCAR   : volume={volume:.4f} Å³  norm_factor=1/sqrt(Ω)={norm_v3_physical:.4e}")

# ── k-points ──────────────────────────────────────────────────────────────────
eigenval_path  = data_dir / "EIGENVAL"
kfrac_all      = None
kweights       = None
band_energies  = None
kcoord_source  = None
kweight_source = None
uniform_warning = None

if eigenval_path.exists():
    try:
        kfrac_all, kweights, band_energies = _read_eigenval(
            eigenval_path, wfc._nkpts, wfc._nbands)
        kcoord_source  = "EIGENVAL"
        kweight_source = "EIGENVAL"
    except ValueError as e:
        print(f"  EIGENVAL mismatch: {e}")

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
    uniform_warning = "WARNING: uniform k-weights used (ISYM=0 assumed)."

np.save(output_dir / "kpoint_weights.npy", kweights)
print(f"k-points : coord={kcoord_source}  weights={kweight_source}")
if uniform_warning:
    print(f"  {uniform_warning}")

if restrict_to_fermi_window and band_energies is None:
    raise RuntimeError("Fermi-window filter requires a parseable EIGENVAL.")

# ── WS center ─────────────────────────────────────────────────────────────────
if not USE_WS_CELL:
    raise RuntimeError(
        "main_v3.py requires USE_WS_CELL=True. "
        "The WS cell defines the region of interest for the new grid."
    )

center_cart, center_frac_cont, center_frac_wrapped = parse_ws_center(
    WS_CENTER, WS_CENTER_COORD_TYPE, latvec)
print(f"WS cell  : center={WS_CENTER} ({WS_CENTER_COORD_TYPE})"
      f" → {np.round(center_cart, 4)} Å")

# ── derive grid spacing from FFT grid if not set ──────────────────────────────
if WS_GRID_SPACING_ANGSTROM is None and WS_GRID_SHAPE is None:
    a1_norm = np.linalg.norm(latvec[0])
    a2_norm = np.linalg.norm(latvec[1])
    a3_norm = np.linalg.norm(latvec[2])
    fft_spacing_avg = (a1_norm / Nx_fft + a2_norm / Ny_fft + a3_norm / Nz_fft) / 3.0
    target_spacing  = fft_spacing_avg
    print(f"  FFT grid spacings: {a1_norm/Nx_fft:.4f}, {a2_norm/Ny_fft:.4f}, "
          f"{a3_norm/Nz_fft:.4f} Å  →  target={target_spacing:.4f} Å")
else:
    target_spacing = WS_GRID_SPACING_ANGSTROM


def _run_at_spacing(spacing, label=""):
    """Build WS grid, evaluate psi, form density matrix, diagonalize."""

    print(f"\n{'='*60}")
    print(f"  Running with grid spacing={spacing:.4f} Å  {label}")

    # ── build WS grid ─────────────────────────────────────────────────────────
    print("  Building WS grid ...", end="", flush=True)
    t0 = time.perf_counter()

    if WS_GRID_SHAPE is not None:
        nx, ny, nz = WS_GRID_SHAPE
        box_lo, box_hi, R_max = build_ws_bounding_box(center_cart, latvec, WS_MARGIN)
        x = np.linspace(box_lo[0], box_hi[0], nx, endpoint=False) + (box_hi[0]-box_lo[0])/(2*nx)
        y = np.linspace(box_lo[1], box_hi[1], ny, endpoint=False) + (box_hi[1]-box_lo[1])/(2*ny)
        z = np.linspace(box_lo[2], box_hi[2], nz, endpoint=False) + (box_hi[2]-box_lo[2])/(2*nz)
        X, Y, Z  = np.meshgrid(x, y, z, indexing='ij')
        r_box    = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        mask     = ws_membership(r_box, center_cart, latvec, nmax=WS_NMAX)
        r_ws_cart = r_box[mask]
        A_inv     = np.linalg.inv(latvec)
        s_ws_frac = r_ws_cart @ A_inv
        dx = (box_hi[0] - box_lo[0]) / nx
        dy = (box_hi[1] - box_lo[1]) / ny
        dz = (box_hi[2] - box_lo[2]) / nz
        dV = dx * dy * dz
        box_shape = (nx, ny, nz)
    else:
        r_ws_cart, s_ws_frac, box_shape, dV = build_ws_grid(
            center_cart, latvec, spacing, nmax=WS_NMAX, margin=WS_MARGIN)

    n_ws = len(r_ws_cart)
    t_grid = time.perf_counter() - t0
    print(f"  done ({t_grid:.1f}s)")

    # Report grid stats
    box_lo, box_hi, R_max = build_ws_bounding_box(center_cart, latvec, WS_MARGIN)
    print(f"  Bounding box : [{box_lo.round(3)}] to [{box_hi.round(3)}] Å")
    print(f"  Box grid shape: {box_shape}  "
          f"({box_shape[0]*box_shape[1]*box_shape[2]:,} total bounding points)")
    print(f"  WS interior  : {n_ws:,} points  ({100*n_ws/(box_shape[0]*box_shape[1]*box_shape[2]):.1f}%)")
    print(f"  dV           : {dV:.6f} Å³")
    vol_approx = n_ws * dV
    print(f"  n_ws * dV    : {vol_approx:.4f} Å³  (vs Omega={volume:.4f} Å³, "
          f"ratio={vol_approx/volume:.4f})")
    print(f"  s_frac range : min={s_ws_frac.min(axis=0).round(3)}  "
          f"max={s_ws_frac.max(axis=0).round(3)}")

    # Memory check for density matrix
    rho_gb = (n_ws**2) * 16 / 1e9
    print(f"  Density matrix: ({n_ws},{n_ws}) complex128 = {rho_gb:.2f} GB")
    if rho_gb > RHO_MEMORY_WARN_GB:
        print(f"  WARNING: density matrix exceeds {RHO_MEMORY_WARN_GB} GB. "
              f"Consider a coarser grid (larger spacing).")

    # Weight for volume-quadrature: sqrt(dV) folded into psi
    sqrt_w = np.sqrt(dV)

    # Normalization: physical (1/sqrt(Omega)), then sqrt(dV) weight:
    # -> combined norm_factor = sqrt(dV) / sqrt(Omega) = sqrt(dV/Omega)
    norm_combined = np.sqrt(dV / volume)

    # ── density matrix loop ───────────────────────────────────────────────────
    print(f"  Building density matrix (norm_factor=sqrt(dV/Omega)={norm_combined:.4e}) ...")
    rho      = np.zeros((n_ws, n_ws), dtype=np.complex128)
    occ_tol  = 1e-6
    psi_norms_k1 = []
    t_loop_start = time.perf_counter()

    for ik in range(1, wfc._nkpts + 1):
        wk     = kweights[ik - 1]
        k_frac = kfrac_all[ik - 1]

        if restrict_to_fermi_window:
            mask_e = np.abs(band_energies[ik - 1] - efermi) <= fermi_window_ev
            bands  = np.where(mask_e)[0] + 1
            occ    = np.ones(len(bands), dtype=float)
            if len(bands) == 0:
                continue
        else:
            occ_all = wfc._occs[ispin - 1, ik - 1, :]
            bands   = np.where(occ_all > occ_tol)[0] + 1
            occ     = occ_all[bands - 1]
            if len(occ) > 0 and np.max(occ) > 1.5:
                occ = occ / 2.0

        if len(bands) == 0:
            continue

        gvec = wfc.gvectors(ik)
        nb   = len(bands)
        Ck   = np.stack([wfc.readBandCoeff(ispin=ispin, ikpt=ik, iband=ib, norm=True)
                         for ib in bands])

        # Evaluate full Bloch psi at WS fractional coords (NOT reduced mod 1)
        # norm_factor = sqrt(dV/Omega) so sum_j |psi_j|^2 ≈ 1 (converges with resolution)
        psi = fourier_eval_bands(Ck, gvec, k_frac, s_ws_frac,
                                  mode='psi', norm_factor=norm_combined,
                                  chunk_size=CHUNK_SIZE, verbose=False)
        # psi shape: (nb, n_ws) with sum_j |psi_j|^2 ≈ 1 per band

        rho += wk * (psi.T @ (occ[:, None] * psi).conj())

        if ik == 1:
            psi_norms_k1 = [float(np.linalg.norm(psi[i])) for i in range(min(3, nb))]

        if ik == 1 or ik % 20 == 0 or ik == wfc._nkpts:
            print(f"    k {ik:4d}/{wfc._nkpts}  wk={wk:.6f}  bands={nb}")

    t_loop = time.perf_counter() - t_loop_start

    # ── diagonalize ───────────────────────────────────────────────────────────
    np.save(output_dir / f"density_matrix_v3_{spacing:.4f}A.npy", rho)
    herm_err = float(np.max(np.abs(rho - rho.conj().T)))
    tr_rho   = float(np.trace(rho).real)
    print(f"\n  rho: |rho-rho†|_max={herm_err:.2e}  Tr={tr_rho:.6f}")

    eigvals, eigvecs = np.linalg.eigh(rho)
    order   = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    n_occ   = int(np.sum(eigvals > 1e-6))
    top20   = eigvals[:20]

    def _fmt(v):
        return 0.0 if abs(v) < 1e-12 else float(v)

    print(f"  Top 20: {[round(_fmt(v), 6) for v in top20]}")
    print(f"  Sum={eigvals.sum():.6f}  N(>1e-6)={n_occ}")
    print(f"  Loop time: {t_loop:.1f} s")

    np.save(output_dir / f"cno_occupations_v3_{spacing:.4f}A.npy", eigvals)
    np.save(output_dir / f"cno_orbitals_v3_{spacing:.4f}A.npy",    eigvecs)

    # Save WS-grid geometry + CNO values (irregular grid → npz, no cube)
    # For the full bounding grid: place WS values and set outside to NaN
    np.savez(output_dir / f"ws_grid_data_v3_{spacing:.4f}A.npz",
             r_ws_cart=r_ws_cart,
             s_ws_frac=s_ws_frac,
             dV=np.array(dV),
             norm_combined=np.array(norm_combined),
             volume=np.array(volume),
             cno_occupations=eigvals,
             cno_orbitals_top10=eigvecs[:, :min(10, n_occ)],
             center_cart=center_cart,
             box_shape=np.array(box_shape))

    # ── compare with original (main.py) ───────────────────────────────────────
    from config import OUTPUT_SUBDIR as ORIG_SUBDIR
    orig_dir_full = data_dir / "output" / ORIG_SUBDIR
    if (orig_dir_full / "cno_occupations.npy").exists():
        orig_eigvals = np.load(orig_dir_full / "cno_occupations.npy")
        n_cmp = min(20, len(eigvals), len(orig_eigvals))
        eig_diff = eigvals[:n_cmp] - orig_eigvals[:n_cmp]
        print(f"\n  CNO eigenvalue comparison with main.py (top {n_cmp}):")
        print(f"    {'CNO':>5}  {'v3':>14}  {'original':>14}  {'diff':>12}")
        for i in range(n_cmp):
            print(f"    {i:5d}  {_fmt(eigvals[i]):14.8f}  "
                  f"{_fmt(orig_eigvals[i]):14.8f}  {eig_diff[i]:12.4e}")
        print(f"  Max CNO eigenvalue diff: {np.max(np.abs(eig_diff)):.4e}")
        print(f"  Sum(v3)={eigvals.sum():.6f}  Sum(orig)={orig_eigvals.sum():.6f}")
    else:
        print(f"  (Comparison with main.py skipped — original outputs not found)")

    return dict(spacing=spacing, n_ws=n_ws, dV=dV, vol_approx=vol_approx,
                tr_rho=tr_rho, herm_err=herm_err, eigvals=eigvals,
                t_loop=t_loop, psi_norms_k1=psi_norms_k1,
                box_shape=box_shape, norm_combined=norm_combined)


# ─────────────────────────────────────────────────── main calculation ─────────

if CONVERGENCE_SPACINGS is not None:
    # Grid convergence mode
    all_results = []
    for sp in CONVERGENCE_SPACINGS:
        res = _run_at_spacing(sp, label=f"(convergence scan)")
        all_results.append(res)

    print("\n\n=== Grid convergence summary ===")
    print(f"{'spacing(Å)':>12}  {'n_ws':>10}  {'vol_approx':>12}  "
          f"{'Tr(rho)':>10}  {'CNO[0]':>12}  {'CNO[1]':>12}")
    for r in all_results:
        cno0 = r['eigvals'][0] if len(r['eigvals']) > 0 else float('nan')
        cno1 = r['eigvals'][1] if len(r['eigvals']) > 1 else float('nan')
        print(f"  {r['spacing']:10.4f}  {r['n_ws']:10,}  {r['vol_approx']:12.4f}  "
              f"{r['tr_rho']:10.6f}  {cno0:12.8f}  {cno1:12.8f}")
else:
    # Single run at default spacing
    result = _run_at_spacing(target_spacing)

# ── metadata ──────────────────────────────────────────────────────────────────
t_total = time.perf_counter() - t_start
meta = output_dir / "cno_metadata_v3.txt"
with open(meta, "w") as f:
    f.write("=== main_v3.py metadata ===\n\n")
    f.write(f"method                  : direct Fourier sum on new WS-cell grid\n")
    f.write(f"material                : {MATERIAL}\n")
    f.write(f"ispin                   : {ispin}\n")
    f.write(f"restrict_to_fermi_window: {restrict_to_fermi_window}\n")
    if restrict_to_fermi_window:
        f.write(f"efermi                  : {efermi} eV\n")
        f.write(f"fermi_window_ev         : {fermi_window_ev} eV\n")
    f.write(f"fft_grid_ref            : ({Nx_fft},{Ny_fft},{Nz_fft})  Nr={Nr_fft}\n")
    f.write(f"volume_Ang3             : {volume:.6f}\n")
    f.write(f"ws_center               : {WS_CENTER}  ({WS_CENTER_COORD_TYPE})\n")
    f.write(f"ws_center_cart          : {center_cart.round(6).tolist()}\n")
    f.write(f"\n--- normalization convention ---\n")
    f.write(f"physical_norm_factor    : 1/sqrt(Omega) = {norm_v3_physical:.6e}\n")
    f.write(f"volume_weight           : w_i = dV (Cartesian voxel volume)\n")
    f.write(f"psi_weighted            : psi_tilde = psi * sqrt(w_i)\n")
    f.write(f"combined_norm_factor    : sqrt(dV/Omega)\n")
    f.write(f"target_norm_sum         : sum_j |psi_tilde_j|^2 -> 1  (converges as dV->0)\n")
    f.write(f"density_matrix          : rho[r,r'] = sum_nk wk * occ * psi_tilde(r) * psi_tilde*(r')\n")
    f.write(f"expected_trace          : Tr(rho) ≈ n_occ (per spin)\n")
    f.write(f"\n--- comparison with main.py convention ---\n")
    f.write(f"main.py norm_factor     : 1/sqrt(Nr)  (discrete, exact Parseval on FFT grid)\n")
    f.write(f"main.py sum_j |psi|^2   : = 1 exactly\n")
    f.write(f"v3 sum_j |psi_tilde|^2  : ≈ 1 (converges, not exact for finite grid)\n")
    f.write(f"both: Tr(rho)           : ≈ n_occ\n")
    f.write(f"\n--- wavefunction evaluation ---\n")
    f.write(f"mode                    : psi (full Bloch: exp(2pi i (G+k).s_frac))\n")
    f.write(f"s_frac                  : NOT reduced mod 1\n")
    f.write(f"no_folding              : WS positions evaluated directly\n")
    f.write(f"no_translation_phases   : (k+G).s already includes correct Bloch phase\n")
    f.write(f"\n--- total runtime ---\n")
    f.write(f"total_runtime_s         : {t_total:.2f}\n")

print(f"\nSaved metadata → {meta}")
print(f"Output directory: {output_dir}")
print(f"Total runtime: {t_total:.1f} s")
