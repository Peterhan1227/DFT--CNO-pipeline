import numpy as np
from pathlib import Path
from vaspwfc import vaspwfc
from config import MATERIAL, LSORBIT


# ── helpers ───────────────────────────────────────────────────────────────────

def _ask(prompt, default):
    """Prompt with a default; return default if the user just presses Enter."""
    ans = input(f"  {prompt} [{default}]: ").strip()
    return ans if ans else str(default)


def get_kpoint_weights(wfc):
    """Read k-point weights from vaspwfc; fall back to uniform if unavailable.
    Returns (normalized_weights, source_name).
    """
    nkpts = wfc._nkpts
    for attr in ("_kweights", "_kwhts", "_weights", "kweights", "kwhts", "weights"):
        if not hasattr(wfc, attr):
            continue
        val = getattr(wfc, attr)
        if val is not None and hasattr(val, "__len__") and len(val) == nkpts:
            w = np.asarray(val, dtype=float)
            return w / w.sum(), attr
    return np.ones(nkpts, dtype=float) / nkpts, "uniform fallback"


def _read_eigenval_energies(path, nkpts_expected, nbands_expected):
    """Return band energies array (nkpts, nbands) from VASP EIGENVAL.

    Reads nkpts/nbands from the file header and validates them against the
    WAVECAR values so a line-mode EIGENVAL is caught before it causes a
    confusing index error.
    """
    with open(path) as fh:
        lines = fh.readlines()
    nkpts  = int(lines[5].split()[1])
    nbands = int(lines[5].split()[2])
    if nkpts != nkpts_expected or nbands != nbands_expected:
        raise ValueError(
            f"EIGENVAL mismatch: file has nkpts={nkpts}, nbands={nbands} "
            f"but WAVECAR has nkpts={nkpts_expected}, nbands={nbands_expected}.\n"
            f"The Fermi-window filter requires the BZ-mesh EIGENVAL, "
            f"not the line-mode one."
        )
    energies = np.zeros((nkpts, nbands), dtype=float)
    idx = 6
    for ik in range(nkpts):
        while idx < len(lines) and not lines[idx].split():
            idx += 1
        idx += 1   # skip k-point header line
        for ib in range(nbands):
            energies[ik, ib] = float(lines[idx].split()[1])
            idx += 1
    return energies


# ── configuration ─────────────────────────────────────────────────────────────

print("=== Wavecar_to_Coeff: density matrix construction ===\n")

ispin = int(_ask("Spin channel  (1 = up / non-polarised,  2 = down)", 1))

ans = _ask("Restrict to Fermi window?  (y/n)", "n").lower()
restrict_to_fermi_window = ans.startswith("y")

if restrict_to_fermi_window:
    efermi          = float(_ask("Fermi energy (eV)", 5.9837))
    fermi_window_ev = float(_ask("Window half-width (eV)", 5.0))
else:
    efermi = fermi_window_ev = None

print()


# ── paths ─────────────────────────────────────────────────────────────────────

data_dir      = Path(__file__).resolve().parent / "Data" / MATERIAL
wavecar_path  = data_dir / "WAVECAR"
eigenval_path = data_dir / "EIGENVAL"
poscar_path   = data_dir / "POSCAR"
output_dir    = data_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)
rho_file        = output_dir / "density_matrix.npy"
grid_shape_file = output_dir / "fft_grid_shape.npy"
metadata_path   = output_dir / "cno_metadata.txt"


# ── load files ────────────────────────────────────────────────────────────────

# WAVECAR
wfc = vaspwfc(str(wavecar_path), lsorbit=LSORBIT)
print(f"WAVECAR  : nkpts={wfc._nkpts}  nbands={wfc._nbands}  "
      f"ngrid={tuple(wfc._ngrid)}  encut={wfc._encut}")

# Line-mode k-point paths give meaningless BZ averages.
kweights, kw_source = get_kpoint_weights(wfc)
is_uniform = bool(np.allclose(kweights, kweights[0]))
print(f"k-weights: source={kw_source}  uniform={is_uniform}  sum={kweights.sum():.6f}")
np.save(output_dir / "kpoint_weights.npy", kweights)

# POSCAR – lattice vectors and volume
with open(poscar_path) as fh:
    pl = fh.readlines()
scale  = float(pl[1])
latvec = scale * np.array([[float(x) for x in pl[i].split()] for i in range(2, 5)])
volume = abs(np.dot(latvec[0], np.cross(latvec[1], latvec[2])))
print(f"POSCAR   : volume={volume:.4f} Å³")

# FFT grid
Nx, Ny, Nz = wfc._ngrid
Nr = Nx * Ny * Nz
np.save(grid_shape_file, np.array([Nx, Ny, Nz], dtype=int))
print(f"FFT grid : ({Nx}, {Ny}, {Nz})  Nr={Nr}")

# EIGENVAL – only when Fermi-window filtering is on
band_energies = None
if restrict_to_fermi_window:
    if not eigenval_path.exists():
        raise FileNotFoundError(f"Fermi-window filtering requires EIGENVAL at: {eigenval_path}")
    band_energies = _read_eigenval_energies(eigenval_path, wfc._nkpts, wfc._nbands)
    n_win = int(np.sum(np.abs(band_energies - efermi) <= fermi_window_ev))
    print(f"EIGENVAL : Fermi window [{efermi - fermi_window_ev:.3f}, {efermi + fermi_window_ev:.3f}] eV  "
          f"({n_win}/{band_energies.size} states in window)")

print()


# ── build density matrix:  rho = Σ_k w_k  Σ_n f_nk |u_nk><u_nk| ─────────────

occ_tol = 1e-6
rho = np.zeros((Nr, Nr), dtype=np.complex128)

for ik in range(1, wfc._nkpts + 1):
    wk      = kweights[ik - 1]
    occ_all = wfc._occs[ispin - 1, ik - 1, :]
    bands   = np.where(occ_all > occ_tol)[0] + 1   # 1-indexed
    occ     = occ_all[bands - 1]

    if restrict_to_fermi_window and band_energies is not None:
        ek        = band_energies[ik - 1, bands - 1]
        in_window = np.abs(ek - efermi) <= fermi_window_ev
        bands, occ = bands[in_window], occ[in_window]
        if len(bands) == 0:
            continue

    if np.max(occ) > 1.5:
        occ = occ / 2.0

    gvec = wfc.gvectors(ik)
    nG   = gvec.shape[0]
    gx, gy, gz = gvec[:, 0] % Nx, gvec[:, 1] % Ny, gvec[:, 2] % Nz

    Ck = np.stack([wfc.readBandCoeff(ispin=ispin, ikpt=ik, iband=ib, norm=True)
                   for ib in bands])

    coeff_grid = np.zeros((len(bands), Nx, Ny, Nz), dtype=np.complex128)
    if LSORBIT:
        # SOC: Ck is (nbands, 2*nG) — first nG cols are spin-up, next nG are spin-down.
        # Density matrix is the spin-trace: rho += w*(Psi_up†*f*Psi_up + Psi_dn†*f*Psi_dn)
        coeff_grid_dn = np.zeros_like(coeff_grid)
        coeff_grid[:, gx, gy, gz]    = Ck[:, :nG]
        coeff_grid_dn[:, gx, gy, gz] = Ck[:, nG:]
        Psi_up = np.fft.ifftn(coeff_grid,    axes=(1, 2, 3)).reshape(len(bands), Nr) * np.sqrt(Nr)
        Psi_dn = np.fft.ifftn(coeff_grid_dn, axes=(1, 2, 3)).reshape(len(bands), Nr) * np.sqrt(Nr)
        rho += wk * (Psi_up.T @ (occ[:, None] * Psi_up).conj()
                   + Psi_dn.T @ (occ[:, None] * Psi_dn).conj())
    else:
        coeff_grid[:, gx, gy, gz] = Ck
        Psi          = np.fft.ifftn(coeff_grid, axes=(1, 2, 3)) * np.sqrt(Nr)
        Psi          = Psi.reshape(len(bands), Nr)
        weighted_Psi = occ[:, None] * Psi
        rho         += wk * (Psi.T @ weighted_Psi.conj())

    if ik == 1 or ik % 20 == 0 or ik == wfc._nkpts:
        print(f"  k {ik:4d}/{wfc._nkpts}  wk={wk:.6f}  bands={len(bands)}")

print(f"\nrho: finite={np.isfinite(rho).all()}  "
      f"|rho-rho†|_max={np.max(np.abs(rho - rho.conj().T)):.2e}  "
      f"Tr={np.trace(rho).real:.6f}")
np.save(rho_file, rho)
print(f"Saved density matrix → {rho_file}  ({rho_file.stat().st_size / 1024**2:.2f} MB)")


# ── CNO diagonalization ───────────────────────────────────────────────────────

print("\n--- CNO diagonalization ---")
eigvals, eigvecs = np.linalg.eigh(rho)
order   = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

n_occupied = int(np.sum(eigvals > 1e-6))
top20      = eigvals[:20]


def _fmt(v):
    return 0.0 if abs(v) < 1e-12 else float(v)


print(f"Top 20 occupations : {[_fmt(v) for v in top20]}")
print(f"Sum={eigvals.sum():.6f}  N(>1e-6)={n_occupied}  "
      f"max={_fmt(eigvals[0]):.4e}  min={_fmt(eigvals[-1]):.4e}")

occ_file = output_dir / "cno_occupations.npy"
orb_file = output_dir / "cno_orbitals.npy"
np.save(occ_file, eigvals)
np.save(orb_file, eigvecs)
print(f"Saved CNO occupations → {occ_file}")
print(f"Saved CNO orbitals    → {orb_file}  ({orb_file.stat().st_size / 1024**2:.2f} MB)")


# ── metadata ──────────────────────────────────────────────────────────────────

with open(metadata_path, "w") as f:
    f.write(f"ispin                   : {ispin}\n")
    f.write(f"restrict_to_fermi_window: {restrict_to_fermi_window}\n")
    if restrict_to_fermi_window:
        f.write(f"efermi                  : {efermi} eV\n")
        f.write(f"fermi_window_ev         : {fermi_window_ev} eV\n")
    f.write(f"k_weight_source         : {kw_source}\n")
    f.write(f"k_weights_uniform       : {is_uniform}\n")
    f.write(f"fft_grid                : ({Nx}, {Ny}, {Nz})\n")
    f.write(f"volume_Ang3             : {volume:.6f}\n")
    f.write(f"Tr(rho)                 : {np.trace(rho).real:.8f}\n")
    f.write(f"sum_cno_occ             : {eigvals.sum():.10f}\n")
    f.write(f"n_eigenvalues_gt_1e-6   : {n_occupied}\n")
    f.write("top_20_cno_occupations  :\n")
    for i, v in enumerate(top20):
        f.write(f"  CNO {i:3d} : {v:.10e}\n")
print(f"Saved metadata → {metadata_path}")
