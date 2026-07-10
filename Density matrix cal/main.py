import sys
import numpy as np
from pathlib import Path
from vaspwfc import vaspwfc
from config import (
    MATERIAL, LSORBIT, OUTPUT_SUBDIR,
    ISPIN, RESTRICT_TO_FERMI_WINDOW, EFERMI, FERMI_WINDOW_EV,
    USE_WS_CELL,
    WS_CENTER, WS_CENTER_COORD_TYPE, WS_TRANSLATION_SEARCH_RANGE,
)

sys.path.insert(0, str(Path(__file__).resolve().parent / "helper functions"))
from ws_cell import read_poscar_structure, parse_ws_center, build_ws_grid_map


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_eigenval(path, nkpts_expected, nbands_expected):
    """Parse a BZ-mesh EIGENVAL; return (kfrac, kweights, energies).

    Raises ValueError if the file dimensions do not match expectations,
    so callers can fall back gracefully.
    """
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


print("=== Wavecar_to_Coeff: density matrix construction ===\n")

ispin                    = ISPIN
restrict_to_fermi_window = RESTRICT_TO_FERMI_WINDOW
efermi                   = EFERMI if RESTRICT_TO_FERMI_WINDOW else None
fermi_window_ev          = FERMI_WINDOW_EV if RESTRICT_TO_FERMI_WINDOW else None

print(f"ispin={ispin}  restrict_to_fermi_window={restrict_to_fermi_window}"
      + (f"  efermi={efermi} eV  window=±{fermi_window_ev} eV"
         if restrict_to_fermi_window else ""))
print()


# ── paths ─────────────────────────────────────────────────────────────────────

data_dir   = Path(__file__).resolve().parent / "Data" / MATERIAL
output_dir = data_dir / "output" / OUTPUT_SUBDIR
output_dir.mkdir(parents=True, exist_ok=True)


# ── load WAVECAR ──────────────────────────────────────────────────────────────

wfc = vaspwfc(str(data_dir / "WAVECAR"), lsorbit=LSORBIT)
Nx, Ny, Nz = wfc._ngrid
Nr = Nx * Ny * Nz
print(f"WAVECAR  : nkpts={wfc._nkpts}  nbands={wfc._nbands}  "
      f"ngrid=({Nx},{Ny},{Nz})  encut={wfc._encut:.1f}")
np.save(output_dir / "fft_grid_shape.npy", np.array([Nx, Ny, Nz], dtype=int))


# ── load POSCAR ───────────────────────────────────────────────────────────────

latvec, species, counts, atom_symbols, atom_numbers, frac_coords, cart_coords = \
    read_poscar_structure(data_dir / "POSCAR")
volume = abs(np.dot(latvec[0], np.cross(latvec[1], latvec[2])))
print(f"POSCAR   : volume={volume:.4f} Å³  "
      f"atoms: {' '.join(f'{s}({c})' for s, c in zip(species, counts))}")


# ── k-points: coordinates and weights ────────────────────────────────────────
# Priority:
#   1. EIGENVAL — preferred source for both kfrac and kweights.
#   2. vaspwfc attributes — used if EIGENVAL is absent or dimension-mismatched.
#   3. Uniform fallback — last resort; valid only for a fully unreduced k mesh.

eigenval_path          = data_dir / "EIGENVAL"
kfrac_all              = None
kweights               = None
band_energies          = None   # populated only when EIGENVAL is parsed
kcoord_source          = None
kweight_source         = None
uniform_warning        = None
eigenval_mismatch_note = None

if eigenval_path.exists():
    try:
        kfrac_all, kweights, band_energies = _read_eigenval(
            eigenval_path, wfc._nkpts, wfc._nbands
        )
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
        raise RuntimeError(
            "No fractional k-coordinates found. "
            "Provide a BZ-mesh EIGENVAL or ensure vaspwfc exposes _kvecs/_kpts."
        )

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
    uniform_warning = (
        "WARNING: uniform k-weights used. "
        "This is only physically valid for a fully unreduced BZ mesh (ISYM=0)."
    )

np.save(output_dir / "kpoint_weights.npy", kweights)
kweights_uniform = bool(np.allclose(kweights, kweights[0]))
print(f"k-points : coord={kcoord_source}  weights={kweight_source}")
if uniform_warning:
    print(f"  {uniform_warning}")

kfrac_max_comp = float(np.max(np.abs(kfrac_all)))
kfrac_warning  = (
    f"WARNING: max |k_frac| component = {kfrac_max_comp:.4f} > 1.5; "
    "check that these are reduced fractional reciprocal coordinates."
    if kfrac_max_comp > 1.5 else None
)

if restrict_to_fermi_window:
    if band_energies is None:
        raise RuntimeError(
            "Fermi window filtering requires band energies from EIGENVAL, "
            "but EIGENVAL was not successfully parsed. "
            "Check that the BZ-mesh EIGENVAL matches the WAVECAR."
        )
    n_win = int(np.sum(np.abs(band_energies - efermi) <= fermi_window_ev))
    print(f"  Fermi window [{efermi-fermi_window_ev:.3f}, {efermi+fermi_window_ev:.3f}] eV "
          f"({n_win}/{band_energies.size} states in window)")


# ── WS cell / Bloch-phase setup ───────────────────────────────────────────────
# r_for_phase : (Nr, 3) fractional coords passed to exp(2πi k·r).
#   WS mode  → r_ws_frac_cont, continuous (unwrapped), possibly outside [0,1).
#   Prim mode → [ix/Nx, iy/Ny, iz/Nz] in the same C-order as ifftn reshape.
# prim_indices: (Nr, 3) int — for WS mode, maps each WS point to its FFT index.

r_for_phase       = None
prim_indices      = None
translations_all  = None
center_cart = center_frac_cont = center_frac_wrapped = None

if USE_WS_CELL:
    center_cart, center_frac_cont, center_frac_wrapped = parse_ws_center(
        WS_CENTER, WS_CENTER_COORD_TYPE, latvec
    )
    print(f"WS cell  : center={WS_CENTER} ({WS_CENTER_COORD_TYPE})"
          f" → {np.round(center_cart, 4)} Å")
    print(f"           building map grid=({Nx},{Ny},{Nz})"
          f" nmax={WS_TRANSLATION_SEARCH_RANGE} ...", end="", flush=True)

    r_ws_cart, r_ws_frac_cont, prim_indices, translations_all = build_ws_grid_map(
        latvec, (Nx, Ny, Nz), center_cart, nmax=WS_TRANSLATION_SEARCH_RANGE
    )
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

print()


# ── density matrix loop ───────────────────────────────────────────────────────
# rho[r,r'] = Σ_nk  w_k * f_nk * psi_nk(r) * psi_nk*(r')
# psi_nk(r) = exp(2πi k·r_frac) * u_nk(r)      [Bloch phase always applied]
# u_nk from IFFT of plane-wave coefficients.

def _to_psi(u_batch, k_frac):
    """Map (nb, Nx, Ny, Nz) → (nb, Nr) full Bloch wavefunctions."""
    if prim_indices is not None:   # WS mode: reindex to WS images
        psi = u_batch[:, prim_indices[:, 0], prim_indices[:, 1], prim_indices[:, 2]]
    else:
        psi = u_batch.reshape(u_batch.shape[0], Nr)
    return psi * np.exp(2j * np.pi * (r_for_phase @ k_frac))[None, :]


occ_tol      = 1e-6
rho          = np.zeros((Nr, Nr), dtype=np.complex128)
psi_norms_k1 = []

for ik in range(1, wfc._nkpts + 1):
    wk     = kweights[ik - 1]
    k_frac = kfrac_all[ik - 1]

    if restrict_to_fermi_window:
        # All bands within the energy window get occupation 1 (including empty conduction bands)
        mask  = np.abs(band_energies[ik - 1] - efermi) <= fermi_window_ev
        bands = np.where(mask)[0] + 1   # 1-indexed
        occ   = np.ones(len(bands), dtype=float)
        if len(bands) == 0:
            continue
    else:
        occ_all = wfc._occs[ispin - 1, ik - 1, :]
        bands   = np.where(occ_all > occ_tol)[0] + 1   # 1-indexed
        occ     = occ_all[bands - 1]
        if np.max(occ) > 1.5:   # spin-degenerate: VASP stores f=2; halve to get per-spin
            occ = occ / 2.0

    gvec = wfc.gvectors(ik)
    nG   = gvec.shape[0]
    gx, gy, gz = gvec[:, 0] % Nx, gvec[:, 1] % Ny, gvec[:, 2] % Nz

    Ck = np.stack([wfc.readBandCoeff(ispin=ispin, ikpt=ik, iband=ib, norm=True)
                   for ib in bands])
    nb = len(bands)
    cg = np.zeros((nb, Nx, Ny, Nz), dtype=np.complex128)

    if LSORBIT:
        cg_dn = np.zeros_like(cg)
        cg[:, gx, gy, gz]    = Ck[:, :nG]
        cg_dn[:, gx, gy, gz] = Ck[:, nG:]
        psi_up = _to_psi(np.fft.ifftn(cg,    axes=(1, 2, 3)) * np.sqrt(Nr), k_frac)
        psi_dn = _to_psi(np.fft.ifftn(cg_dn, axes=(1, 2, 3)) * np.sqrt(Nr), k_frac)
        rho += wk * (psi_up.T @ (psi_up).conj()
                   + psi_dn.T @ (psi_dn).conj())
        if ik == 1:
            psi_norms_k1 = [float(np.linalg.norm(psi_up[i])) for i in range(min(3, nb))]
    else:
        cg[:, gx, gy, gz] = Ck
        psi = _to_psi(np.fft.ifftn(cg, axes=(1, 2, 3)) * np.sqrt(Nr), k_frac)
        rho += wk * (psi.T @ (occ[:, None] * psi).conj())
        if ik == 1:
            psi_norms_k1 = [float(np.linalg.norm(psi[i])) for i in range(min(3, nb))]

    if ik == 1 or ik % 20 == 0 or ik == wfc._nkpts:
        print(f"  k {ik:4d}/{wfc._nkpts}  wk={wk:.6f}  bands={nb}")


# ── save density matrix and diagonalize ───────────────────────────────────────

np.save(output_dir / "density_matrix.npy", rho)
herm_err = float(np.max(np.abs(rho - rho.conj().T)))
tr_rho   = float(np.trace(rho).real)
print(f"\nrho: |rho-rho†|_max={herm_err:.2e}  Tr={tr_rho:.6f}")

eigvals, eigvecs = np.linalg.eigh(rho)
order   = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]
n_occupied = int(np.sum(eigvals > 1e-6))
top20      = eigvals[:20]

def _fmt(v):
    return 0.0 if abs(v) < 1e-12 else float(v)

print(f"Top 20 : {[round(_fmt(v), 6) for v in top20]}")
print(f"Sum={eigvals.sum():.6f}  N(>1e-6)={n_occupied}")

np.save(output_dir / "cno_occupations.npy", eigvals)
np.save(output_dir / "cno_orbitals.npy",    eigvecs)


# ── metadata (all diagnostics go here, not to terminal) ───────────────────────

meta = output_dir / "cno_metadata.txt"
with open(meta, "w") as f:

    f.write("=== CNO density matrix metadata ===\n\n")

    f.write(f"material                : {MATERIAL}\n")
    f.write(f"LSORBIT                 : {LSORBIT}\n")
    f.write(f"ispin                   : {ispin}\n")
    f.write(f"restrict_to_fermi_window: {restrict_to_fermi_window}\n")
    if restrict_to_fermi_window:
        f.write(f"efermi                  : {efermi} eV\n")
        f.write(f"fermi_window_ev         : {fermi_window_ev} eV\n")
    f.write(f"fft_grid                : ({Nx}, {Ny}, {Nz})\n")
    f.write(f"volume_Ang3             : {volume:.6f}\n")
    f.write(f"nkpts                   : {wfc._nkpts}\n")
    f.write(f"nbands                  : {wfc._nbands}\n")

    f.write("\n--- k-point coordinates ---\n")
    f.write(f"kcoord_source           : {kcoord_source}\n")
    if eigenval_mismatch_note:
        f.write(f"eigenval_mismatch_note  : {eigenval_mismatch_note}\n")
    n_show = min(3, len(kfrac_all))
    for i in range(n_show):
        f.write(f"kfrac[{i:>4d}]             : {kfrac_all[i].tolist()}\n")
    if len(kfrac_all) > n_show:
        f.write(f"  ...\n")
        for i in range(max(n_show, len(kfrac_all) - n_show), len(kfrac_all)):
            f.write(f"kfrac[{i:>4d}]             : {kfrac_all[i].tolist()}\n")
    if kfrac_warning:
        f.write(f"{kfrac_warning}\n")

    f.write("\n--- k-point weights ---\n")
    f.write(f"kweight_source          : {kweight_source}\n")
    f.write(f"kweight_sum             : {kweights.sum():.8f}\n")
    f.write(f"kweight_min             : {kweights.min():.8f}\n")
    f.write(f"kweight_max             : {kweights.max():.8f}\n")
    f.write(f"kweights_uniform        : {kweights_uniform}\n")
    if uniform_warning:
        f.write(f"{uniform_warning}\n")

    f.write("\n--- Wigner-Seitz cell ---\n")
    f.write(f"USE_WS_CELL             : {USE_WS_CELL}\n")
    if USE_WS_CELL:
        f.write(f"ws_center_input         : {WS_CENTER}  ({WS_CENTER_COORD_TYPE})\n")
        f.write(f"ws_center_frac_wrapped  : {center_frac_wrapped.tolist()}\n")
        f.write(f"ws_center_cart_Ang      : {[round(x, 6) for x in center_cart.tolist()]}\n")
        f.write(f"ws_translation_nmax     : {WS_TRANSLATION_SEARCH_RANGE}\n")
        trans_min = translations_all.min(axis=0).tolist()
        trans_max = translations_all.max(axis=0).tolist()
        n_unique  = len(np.unique(translations_all, axis=0))
        f.write(f"ws_translation_min      : {trans_min}\n")
        f.write(f"ws_translation_max      : {trans_max}\n")
        f.write(f"ws_n_unique_translations: {n_unique}\n")

    f.write("\n--- physical ---\n")
    f.write("bloch_phase             : psi_nk(r) = exp(2*pi*i * k_frac . r_frac_cont) * u_nk(r)\n")
    f.write(f"rho_hermiticity_error   : {herm_err:.4e}\n")
    f.write(f"Tr(rho)                 : {tr_rho:.8f}\n")
    f.write("spin_trace_note         : occupations > 1.5 were halved before accumulation;\n"
            "                          Tr counts spatial orbitals, not spin-orbitals.\n"
            "                          Expected Tr ~ 4 for primitive Si (4 occupied spatial bands).\n")
    if psi_norms_k1:
        f.write(f"sample_psi_norms_ik1    : {[f'{v:.6f}' for v in psi_norms_k1]}\n")

    f.write("\n--- CNO spectrum ---\n")
    f.write(f"sum_cno_occ             : {eigvals.sum():.10f}\n")
    f.write(f"n_eigenvalues_gt_1e-6   : {n_occupied}\n")
    f.write("top_20_cno_occupations  :\n")
    for i, v in enumerate(top20):
        f.write(f"  CNO {i:3d} : {_fmt(v):.10e}\n")
    if len(top20) > 1:
        f.write("gaps_between_top_20     :\n")
        for i in range(len(top20) - 1):
            f.write(f"  gap {i:2d}→{i+1:<2d} : {_fmt(top20[i] - top20[i+1]):.4e}\n")

print(f"Saved metadata → {meta}")
