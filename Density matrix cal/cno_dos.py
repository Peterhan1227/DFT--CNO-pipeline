"""
cno_dos.py — CNO-projected density of states over the full Brillouin zone.

Reads:
  WAVECAR_dos (or WAVECAR)      full-BZ DOS/NSCF wavefunction
  EIGENVAL_dos (or EIGENVAL)    k-weights and eigenvalues for that run
  output/<SUBDIR>/cno_orbitals.npy / cno_occupations.npy  (from main.py)
  DOSCAR (optional)             E_F, energy range, and VASP total DOS

Writes:
  output/<SUBDIR>/cno_dos/cno_projected_dos.npz
  output/<SUBDIR>/cno_dos/<output_file>   (PNG plot)
"""
from __future__ import annotations

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from config import MATERIAL, LSORBIT, OUTPUT_SUBDIR, ISPIN
from vaspwfc import vaspwfc


# ── user settings ─────────────────────────────────────────────────────────────
cno_start         = 0              # first CNO index to include
cno_count         = 8             # number of CNOs starting from cno_start
sigma             = 0.03           # Gaussian broadening width (eV)
emin              = None           # lower energy bound (eV rel. to E_F); None = auto
emax              = None           # upper energy bound (eV rel. to E_F); None = auto
nedos             = 2000           # number of points in the energy grid
output_file       = "cno_projected_dos.png"   # saved inside cno_dos/ subdirectory
compare_total_dos = False          # overlay reconstructed total DOS and VASP DOSCAR


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_eigenval_dos(path, nkpts_expected, nbands_expected, ispin):
    """Parse a BZ-mesh EIGENVAL. Returns (kfrac, kweights, energies).

    energies shape: (nkpts, nbands) for the requested spin channel.
    Raises ValueError if dimensions do not match WAVECAR.
    """
    with open(path) as fh:
        lines = fh.readlines()

    try:
        ispin_file = int(lines[0].split()[3])
    except Exception:
        ispin_file = 1

    nkpts  = int(lines[5].split()[1])
    nbands = int(lines[5].split()[2])
    if nkpts != nkpts_expected or nbands != nbands_expected:
        raise ValueError(
            f"EIGENVAL has ({nkpts} k-pts, {nbands} bands) "
            f"but WAVECAR has ({nkpts_expected}, {nbands_expected})"
        )

    # column in the per-band lines that holds the energy for our spin channel
    e_col = ispin if ispin_file == 2 else 1

    kfrac    = np.zeros((nkpts, 3))
    kweights = np.zeros(nkpts)
    energies = np.zeros((nkpts, nbands))
    idx = 6
    for ik in range(nkpts):
        while idx < len(lines) and not lines[idx].split():
            idx += 1
        kline         = lines[idx].split()
        kfrac[ik]     = [float(x) for x in kline[:3]]
        kweights[ik]  = float(kline[3])
        idx += 1
        for ib in range(nbands):
            energies[ik, ib] = float(lines[idx].split()[e_col])
            idx += 1

    kweights /= kweights.sum()
    return kfrac, kweights, energies


def _parse_doscar(path):
    """Parse the VASP DOSCAR header and total DOS block.

    Returns (efermi, emin, emax, nedos, energy_arr, total_dos_arr).
    total_dos_arr sums both spin channels for ISPIN=2.
    Returns all None on failure.
    """
    try:
        with open(path) as fh:
            lines = fh.readlines()

        # line 5 (0-indexed): EMAX  EMIN  NEDOS  EFERMI  weight
        parts5   = lines[5].split()
        emax_d   = float(parts5[0])
        emin_d   = float(parts5[1])
        nedos_d  = int(parts5[2])
        efermi_d = float(parts5[3])

        energy_d = np.zeros(nedos_d)
        dos_d    = np.zeros(nedos_d)
        ncols    = len(lines[6].split())   # 3 = nspin=1, 5 = nspin=2
        for i in range(nedos_d):
            parts       = lines[6 + i].split()
            energy_d[i] = float(parts[0])
            if ncols >= 5:
                dos_d[i] = float(parts[1]) + float(parts[2])   # up + down
            else:
                dos_d[i] = float(parts[1])

        return efermi_d, emin_d, emax_d, nedos_d, energy_d, dos_d

    except Exception as exc:
        print(f"WARNING: failed to parse DOSCAR ({exc}).", file=sys.stderr)
        return None, None, None, None, None, None


# ── paths ─────────────────────────────────────────────────────────────────────

base_dir   = Path(__file__).resolve().parent
data_dir   = base_dir / "Data" / MATERIAL
output_dir = data_dir / "output" / OUTPUT_SUBDIR
dos_dir    = output_dir / "cno_dos"
dos_dir.mkdir(parents=True, exist_ok=True)

# DOS WAVECAR/EIGENVAL — prefer _dos suffix, fall back to plain names
wavecar_dos_path  = (data_dir / "WAVECAR_dos"  if (data_dir / "WAVECAR_dos").exists()
                     else data_dir / "WAVECAR")
eigenval_dos_path = (data_dir / "EIGENVAL_dos" if (data_dir / "EIGENVAL_dos").exists()
                     else data_dir / "EIGENVAL")
poscar_path       = data_dir / "POSCAR"
doscar_path       = data_dir / "DOSCAR"

cno_orb_file    = output_dir / "cno_orbitals.npy"
cno_occ_file    = output_dir / "cno_occupations.npy"
grid_shape_file = output_dir / "fft_grid_shape.npy"

for p, label in [
    (cno_orb_file,       "cno_orbitals.npy"),
    (cno_occ_file,       "cno_occupations.npy"),
    (grid_shape_file,    "fft_grid_shape.npy"),
    (wavecar_dos_path,   "WAVECAR_dos / WAVECAR"),
    (eigenval_dos_path,  "EIGENVAL_dos / EIGENVAL"),
]:
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}  [{label}]")

# ── load CNO data ─────────────────────────────────────────────────────────────

cno_orbs = np.load(cno_orb_file)
cno_occ  = np.load(cno_occ_file)
Nr, n_cno_avail = cno_orbs.shape
Nx_cno, Ny_cno, Nz_cno = np.load(grid_shape_file).astype(int)

cno_count = min(cno_count, n_cno_avail - cno_start)
if not (0 <= cno_start < n_cno_avail):
    raise ValueError(f"cno_start={cno_start} out of range [0, {n_cno_avail - 1}]")
if cno_count <= 0:
    raise ValueError(
        f"No CNOs to project: cno_start={cno_start}, "
        f"cno_count={cno_count}, n_cno_avail={n_cno_avail}"
    )
cno_indices = list(range(cno_start, cno_start + cno_count))
n_cnos      = len(cno_indices)

# ── optional DOSCAR ───────────────────────────────────────────────────────────

efermi        = None
doscar_energy = None
doscar_dos    = None

if doscar_path.exists():
    efermi, _emin_d, _emax_d, _nedos_d, doscar_energy, doscar_dos = \
        _parse_doscar(doscar_path)
    if efermi is not None:
        print(f"DOSCAR   : E_F={efermi:.6f} eV  NEDOS={_nedos_d}  "
              f"[{_emin_d:.3f}, {_emax_d:.3f}] eV")

if efermi is None:
    efermi = 0.0
    if not doscar_path.exists():
        print("WARNING: DOSCAR not found; E_F set to 0.0 eV.")

# ── load DOS WAVECAR ──────────────────────────────────────────────────────────

print(f"\nLoading DOS WAVECAR: {wavecar_dos_path.name} ...")
wfc    = vaspwfc(str(wavecar_dos_path), lsorbit=LSORBIT)
Nx, Ny, Nz = wfc._ngrid
Nr_wfc = Nx * Ny * Nz
nkpts  = wfc._nkpts
nbands = wfc._nbands
print(f"WAVECAR  : nkpts={nkpts}  nbands={nbands}  ngrid=({Nx},{Ny},{Nz})")

if (Nx, Ny, Nz) != (Nx_cno, Ny_cno, Nz_cno):
    raise ValueError(
        f"DOS WAVECAR grid ({Nx},{Ny},{Nz}) does not match "
        f"CNO grid ({Nx_cno},{Ny_cno},{Nz_cno}) from fft_grid_shape.npy.\n"
        "Ensure the DOS run used the same ENCUT and cell as the density matrix run."
    )
if Nr_wfc != Nr:
    raise ValueError(f"Nr mismatch: WAVECAR has {Nr_wfc}, CNO has {Nr}.")

# ── k-weights and eigenvalues ─────────────────────────────────────────────────

print(f"Loading eigenvalues : {eigenval_dos_path.name} ...")
kfrac, kweights, band_energies = _read_eigenval_dos(
    eigenval_dos_path, nkpts, nbands, ISPIN
)
wsum = kweights.sum()
print(f"k-weights: sum={wsum:.6f}  min={kweights.min():.6f}  max={kweights.max():.6f}")
if abs(wsum - 1.0) > 0.01:
    print(f"WARNING: k-weights sum deviates from 1 ({wsum:.6f}); check EIGENVAL.")

# ── sanity check ──────────────────────────────────────────────────────────────

print(f"\nSanity check:")
print(f"  nkpts        = {nkpts}")
print(f"  nbands       = {nbands}")
print(f"  n_cno_avail  = {n_cno_avail}")
print(f"  cno_indices  = {cno_indices}")
for idx in cno_indices:
    print(f"    CNO {idx:3d}  occupation = {cno_occ[idx]:.8e}")

# ── WS mode setup ─────────────────────────────────────────────────────────────
# Mirror the _to_psi logic from main.py / cno_fatband.py exactly.

_ws_enabled_file = output_dir / "ws_enabled.npy"
ws_mode          = _ws_enabled_file.exists() and bool(np.load(_ws_enabled_file))

if ws_mode:
    for fname, p in [("ws_base_indices.npy",    output_dir / "ws_base_indices.npy"),
                     ("ws_points_frac_cont.npy", output_dir / "ws_points_frac_cont.npy")]:
        if not p.exists():
            raise FileNotFoundError(
                f"WS mode active but {fname} not found: {p}\n"
                "Rerun main.py to regenerate WS map files."
            )
    _prim_indices = np.load(output_dir / "ws_base_indices.npy")
    _r_for_phase  = np.load(output_dir / "ws_points_frac_cont.npy")
    print(f"WS mode  : True  prim_indices={_prim_indices.shape}  "
          f"r_ws_frac_cont={_r_for_phase.shape}")

    def _to_psi(u_3d, k_frac):
        u_ws = u_3d[_prim_indices[:, 0], _prim_indices[:, 1], _prim_indices[:, 2]]
        return u_ws * np.exp(2j * np.pi * (_r_for_phase @ k_frac))

else:
    _ix, _iy, _iz = [a.ravel() for a in np.mgrid[0:Nx, 0:Ny, 0:Nz]]
    _r_for_phase   = np.column_stack([_ix / Nx, _iy / Ny, _iz / Nz])
    print("WS mode  : False")

    def _to_psi(u_3d, k_frac):
        return u_3d.reshape(Nr) * np.exp(2j * np.pi * (_r_for_phase @ k_frac))

# ── prepare CNO matrix ────────────────────────────────────────────────────────

_cnos_raw   = np.stack([cno_orbs[:, idx] for idx in cno_indices])    # (n_cnos, Nr)
_norms      = np.sqrt(np.sum(np.abs(_cnos_raw) ** 2, axis=1, keepdims=True))
cnos_conj   = (_cnos_raw / _norms).conj()                             # (n_cnos, Nr)

# ── projection loop ───────────────────────────────────────────────────────────
# P_i[n,k] = |<CNO_i | psi_nk>|^2
# cnos_conj @ psi evaluates all CNOs at once per (k, band) with one matvec.

print(f"\nComputing |<CNO_i | psi_nk>|² for {n_cnos} CNO(s), "
      f"{nbands} bands × {nkpts} k-points ...")

weights_all = np.zeros((n_cnos, nbands, nkpts))

for ik in range(1, nkpts + 1):
    k_frac = kfrac[ik - 1]
    gvec   = wfc.gvectors(ik)
    gx, gy, gz = gvec[:, 0] % Nx, gvec[:, 1] % Ny, gvec[:, 2] % Nz
    nG     = len(gx)

    for ib in range(1, nbands + 1):
        coeff      = wfc.readBandCoeff(ispin=ISPIN, ikpt=ik, iband=ib, norm=True)
        coeff_grid = np.zeros((Nx, Ny, Nz), dtype=np.complex128)

        if LSORBIT:
            coeff_grid_dn = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
            coeff_grid[gx, gy, gz]    = coeff[:nG]
            coeff_grid_dn[gx, gy, gz] = coeff[nG:]
            psi_up = _to_psi(np.fft.ifftn(coeff_grid)    * np.sqrt(Nr), k_frac)
            psi_dn = _to_psi(np.fft.ifftn(coeff_grid_dn) * np.sqrt(Nr), k_frac)
            weights_all[:, ib - 1, ik - 1] = (np.abs(cnos_conj @ psi_up) ** 2
                                             + np.abs(cnos_conj @ psi_dn) ** 2)
        else:
            coeff_grid[gx, gy, gz] = coeff
            psi = _to_psi(np.fft.ifftn(coeff_grid) * np.sqrt(Nr), k_frac)
            weights_all[:, ib - 1, ik - 1] = np.abs(cnos_conj @ psi) ** 2

    if ik == 1 or ik % 20 == 0 or ik == nkpts:
        print(f"  k {ik:4d}/{nkpts}  max P: {weights_all[:, :, ik - 1].max():.4f}")

# ── energy grid (E - E_F) ─────────────────────────────────────────────────────

e_nk = band_energies - efermi    # (nkpts, nbands), relative to E_F

_pad    = 5 * sigma
emin_ev = emin if emin is not None else float(e_nk.min()) - _pad
emax_ev = emax if emax is not None else float(e_nk.max()) + _pad

energy_grid = np.linspace(emin_ev, emax_ev, nedos)    # (nedos,)
gauss_norm  = 1.0 / (sigma * np.sqrt(2.0 * np.pi))

# ── Gaussian broadening ───────────────────────────────────────────────────────
# DOS_i(E) = sum_{n,k} w_k * P_i[n,k] * G(E - e_nk, sigma)
# Vectorised over bands and energy grid per k-point; avoids large 3-D arrays.

print(f"\nApplying Gaussian broadening  sigma={sigma} eV ...")

dos_arr = np.zeros((n_cnos, nedos))

for ik in range(nkpts):
    enk_k = e_nk[ik]                                              # (nbands,)
    diff  = energy_grid[None, :] - enk_k[:, None]                 # (nbands, nedos)
    gauss = gauss_norm * np.exp(-0.5 * (diff / sigma) ** 2)       # (nbands, nedos)
    # (n_cnos, nbands) @ (nbands, nedos) → (n_cnos, nedos)
    dos_arr += kweights[ik] * (weights_all[:, :, ik] @ gauss)

# ── optional total DOS reconstruction ────────────────────────────────────────
# Reconstructed from eigenvalues using the same sigma, for visual comparison.

total_dos_recon = None
if compare_total_dos:
    print("Reconstructing total DOS from eigenvalues ...")
    spin_degen      = 1 if (LSORBIT or ISPIN == 2) else 2
    total_dos_recon = np.zeros(nedos)
    for ik in range(nkpts):
        enk_k = e_nk[ik]
        diff  = energy_grid[None, :] - enk_k[:, None]
        gauss = gauss_norm * np.exp(-0.5 * (diff / sigma) ** 2)
        total_dos_recon += spin_degen * kweights[ik] * gauss.sum(axis=0)

# ── save ──────────────────────────────────────────────────────────────────────

npz_path = dos_dir / "cno_projected_dos.npz"
np.savez(
    npz_path,
    energy_grid = energy_grid,
    cno_indices = np.array(cno_indices),
    cno_dos     = dos_arr,
    sigma       = np.float64(sigma),
    efermi      = np.float64(efermi),
)
print(f"\nSaved projected DOS → {npz_path}")

if total_dos_recon is not None:
    np.savez(
        dos_dir / "total_dos_reconstructed.npz",
        energy_grid = energy_grid,
        total_dos   = total_dos_recon,
        sigma       = np.float64(sigma),
        efermi      = np.float64(efermi),
    )

# ── plot ──────────────────────────────────────────────────────────────────────

print("Plotting ...")

fig, axes = plt.subplots(
    n_cnos, 1,
    figsize=(max(5, 0.8 * (emax_ev - emin_ev)), max(2.0 * n_cnos, 5)),
    sharex=True,
    squeeze=False,
)
axes = axes.ravel()

# global y-limit: largest peak across all CNOs, with 15 % headroom
ymax = float(dos_arr.max()) * 1.15
if ymax == 0.0:
    ymax = 1.0

for ci, (idx, ax) in enumerate(zip(cno_indices, axes)):
    d = dos_arr[ci]    # (nedos,)

    if compare_total_dos:
        if total_dos_recon is not None:
            _td = total_dos_recon / total_dos_recon.max() * ymax
            ax.fill_between(energy_grid, 0, _td, alpha=0.10, color="grey")
            ax.plot(energy_grid, _td, color="grey", lw=0.6,
                    label="Total DOS (eigen., norm.)")
        if doscar_dos is not None:
            _de = doscar_energy - efermi
            _dd = np.interp(energy_grid, _de, doscar_dos, left=0.0, right=0.0)
            if _dd.max() > 0:
                _dd = _dd / _dd.max() * ymax
                ax.fill_between(energy_grid, 0, _dd, alpha=0.10, color="steelblue")
                ax.plot(energy_grid, _dd, color="steelblue", lw=0.6,
                        label="VASP DOSCAR (norm.)")

    ax.fill_between(energy_grid, 0, d, alpha=0.60, color="crimson")
    ax.plot(energy_grid, d, color="crimson", lw=0.9)
    ax.axvline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xlim(emin_ev, emax_ev)
    ax.set_ylim(0, ymax)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_ylabel(f"CNO {idx}\nocc={cno_occ[idx]:.4f}", fontsize=7, rotation=0,
                  labelpad=48, va="center")

    if ci == 0 and compare_total_dos:
        ax.legend(fontsize=5, loc="upper right")
    if ci < n_cnos - 1:
        ax.tick_params(labelbottom=False)

axes[-1].set_xlabel(r"$E - E_F$  (eV)", fontsize=8)
fig.supylabel("Projected DOS (arb. units)", fontsize=8)
fig.suptitle(
    f"{MATERIAL}  —  CNO-projected DOS   "
    rf"$\sigma$={sigma} eV   E$_F$={efermi:.4f} eV",
    fontsize=9,
)
plt.tight_layout()

out_path = dos_dir / output_file
fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved plot          → {out_path}")
print("\nAll done.")
