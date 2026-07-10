"""
cno_fatband.py

Project one or more saved CNO orbitals onto every Bloch state from a line-mode
band-structure WAVECAR and plot a CNO-colored band structure.

Reads KPOINTS directly (no KLABELS needed), following plot_vasp_bs.py.
k-axis distances are computed from the actual EIGENVAL k-points via the
reciprocal lattice from POSCAR, giving physical Å⁻¹ units.

Does NOT recompute rho. Does NOT diagonalize the density matrix.
Only projects one saved CNO onto line-mode Bloch states.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from config import MATERIAL, LSORBIT, OUTPUT_SUBDIR
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from vaspwfc import vaspwfc
from config import EFERMI

# ── user settings ─────────────────────────────────────────────────────────────
ispin       = 1        # 1 = spin-up (or non-spin-polarised), 2 = spin-down
cno_indices = list(range(0, 5))   # e.g. list(range(0, 9))  or  [1, 3, 4, 5]
efermi      = EFERMI
ylim      = [-15, 15]
cmap      = LinearSegmentedColormap.from_list("wt_red", ["white", "crimson"])
linewidth = 2.5
interpolate_for_plot      = False
interp_points_per_segment = 3

# ──3D CNO snapshot (optional; requires the standalone CNO-Visualizer package) ──
# When on, each plotted CNO also gets a rotating 3D density-isosurface GIF saved
# next to its fatband PNG — a VESTA-like view (no phase) for reports.  The GIF is
# rendered from the SAME cnos_sym_adapted vector being plotted, so it always
# matches the fatband.  Set render_cno_3d = False to skip entirely.
render_cno_3d    = True
cno_3d_iso       = 0.5          # isosurface level as a fraction of max |psi|^2
cno_3d_replicate = (2, 2, 2)    # primitive cells drawn around the orbital
cno_3d_seconds     = 10.0        # clip length in seconds (camera turns the whole time)
cno_3d_fps         = 15         # GIF frame rate (lower = smaller file)
cno_3d_deg_per_sec = 12.0       # camera turn speed in deg/s (lower = slower)
cno_3d_spin_axis   = "z"        # camera turns about this axis ("x"/"y"/"z")

# ── paths ─────────────────────────────────────────────────────────────────────
base_dir        = base_dir = __import__("pathlib").Path(__file__).resolve().parent
data_dir        = base_dir / "Data" / MATERIAL
output_dir      = data_dir / "output" / OUTPUT_SUBDIR
wavecar_path    = data_dir / "WAVECAR_lm"
kpoints_path    = data_dir / "KPOINTS"
eigenval_path   = data_dir / "EIGENVAL_lm"
poscar_path     = data_dir / "POSCAR"
cno_orb_file    = output_dir / "cnos_sym_adapted.npy"
cno_occ_file    = output_dir / "cno_occupations.npy"
grid_shape_file = output_dir / "fft_grid_shape.npy"
fatband_dir     = output_dir / "cno_fatband"
fatband_dir.mkdir(parents=True, exist_ok=True)

# ── WS mode setup ─────────────────────────────────────────────────────────────
# In WS mode the CNO vector is ordered by WS grid points.  The line-mode Bloch
# state must be evaluated on the SAME WS grid with the SAME Bloch phase that
# was used when building the density matrix:
#   psi_{n,k}(r_ws_p) = u_{n,k}(r_prim_p) * exp(2πi k · r_ws_frac_cont_p)
# This is the identical _to_psi logic used in Wavecar_to_Coeff.py.
_ws_enabled_file = output_dir / "ws_enabled.npy"
ws_mode = _ws_enabled_file.exists() and bool(np.load(_ws_enabled_file))


# ── data containers (from plot_vasp_bs.py) ────────────────────────────────────

@dataclass
class KPath:
    n_per_segment: int
    coord_type: str
    nodes_frac: np.ndarray   # (n_nodes, 3)
    labels: list              # length n_nodes
    raw_points: np.ndarray   # (2*n_segments, 3)
    raw_labels: list

    @property
    def n_segments(self) -> int:
        return len(self.labels) - 1


@dataclass
class EigenvalData:
    ispin: int
    nelect: float
    nkpts: int
    nbands: int
    kpts_frac: np.ndarray    # (nkpts, 3)
    energies: np.ndarray     # (nspin, nkpts, nbands)


# ── k-path helpers (ported from plot_vasp_bs.py) ──────────────────────────────

def pretty_label(label: str) -> str:
    clean = label.strip()
    if clean.upper() in {"G", "GAMMA", "\\GAMMA", "Γ"}:
        return r"$\Gamma$"
    return clean


def parse_kpoint_line(line: str):
    before, *after = line.split("!", 1)
    parts = before.split()
    if len(parts) < 3:
        raise ValueError(f"Cannot parse k-point line: {line!r}")
    coords = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
    label = ""
    if after:
        label = after[0].strip().split()[0] if after[0].strip() else ""
    elif len(parts) >= 4:
        label = parts[3].strip()
    return coords, label


def read_kpoints_line_mode(path) -> KPath:
    """Read a VASP line-mode KPOINTS file (no KLABELS needed)."""
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh]

    if len(lines) < 5:
        raise ValueError("KPOINTS seems too short to be a line-mode file.")

    n_per_segment = int(float(lines[1].split()[0]))

    mode = lines[2].strip().lower()
    if not (mode.startswith("l") or "line" in mode):
        raise ValueError(
            f"Expected line-mode KPOINTS; line 3 is: {lines[2]!r}"
        )

    coord_str = lines[3].strip().lower()
    if coord_str.startswith("r"):
        coord_type = "reciprocal"
    elif coord_str.startswith("c") or coord_str.startswith("k"):
        coord_type = "cartesian"
    else:
        raise ValueError(f"Unrecognised KPOINTS coordinate type: {lines[3]!r}")

    point_lines = [ln for ln in lines[4:] if ln.strip()]
    if len(point_lines) % 2 != 0:
        raise ValueError("Line-mode KPOINTS must have an even number of endpoint lines.")

    raw_points, raw_labels = [], []
    for ln in point_lines:
        coords, label = parse_kpoint_line(ln)
        raw_points.append(coords)
        raw_labels.append(label)

    raw_points = np.array(raw_points, dtype=float)
    n_segments = len(raw_points) // 2

    nodes  = [raw_points[0]]
    labels = [raw_labels[0] or "K0"]
    for seg in range(n_segments):
        end_idx = 2 * seg + 1
        nodes.append(raw_points[end_idx])
        labels.append(raw_labels[end_idx] or f"K{seg + 1}")

    return KPath(
        n_per_segment=n_per_segment,
        coord_type=coord_type,
        nodes_frac=np.array(nodes, dtype=float),
        labels=labels,
        raw_points=raw_points,
        raw_labels=raw_labels,
    )


def read_poscar_lattice(path) -> Optional[np.ndarray]:
    """Return direct lattice rows (Å). Returns None if file missing."""
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        lines = fh.readlines()
    scale   = float(lines[1].split()[0])
    lattice = np.array(
        [[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)],
        dtype=float,
    )
    if scale > 0:
        lattice *= scale
    else:
        vol = abs(np.linalg.det(lattice))
        lattice *= (abs(scale) / vol) ** (1.0 / 3.0)
    return lattice


def reciprocal_lattice(poscar_path) -> Optional[np.ndarray]:
    """Return reciprocal lattice rows b_i in Å⁻¹ (b_i · a_j = 2π δ_ij)."""
    A = read_poscar_lattice(poscar_path)
    if A is None:
        return None
    return 2.0 * np.pi * np.linalg.inv(A).T


def frac_to_cart_k(kpts: np.ndarray, recip: Optional[np.ndarray]) -> np.ndarray:
    if recip is None:
        return np.array(kpts, dtype=float)
    return np.array(kpts, dtype=float) @ recip


def read_eigenval(path) -> EigenvalData:
    """Robust EIGENVAL parser from plot_vasp_bs.py. Returns (nspin, nkpts, nbands)."""
    with open(path) as fh:
        lines = fh.readlines()

    try:
        ispin_ev = int(lines[0].split()[3])
    except Exception:
        ispin_ev = 1

    nelect = float(lines[5].split()[0])
    nkpts  = int(lines[5].split()[1])
    nbands = int(lines[5].split()[2])
    nspin  = 2 if ispin_ev == 2 else 1

    kpts     = np.zeros((nkpts, 3), dtype=float)
    energies = np.zeros((nspin, nkpts, nbands), dtype=float)
    idx = 6

    for ik in range(nkpts):
        while idx < len(lines) and not lines[idx].split():
            idx += 1
        kline = lines[idx].split()
        kpts[ik] = [float(kline[0]), float(kline[1]), float(kline[2])]
        idx += 1
        for ib in range(nbands):
            parts = lines[idx].split()
            energies[0, ik, ib] = float(parts[1])
            if nspin == 2:
                energies[1, ik, ib] = float(parts[2])
            idx += 1

    return EigenvalData(
        ispin=ispin_ev, nelect=nelect, nkpts=nkpts, nbands=nbands,
        kpts_frac=kpts, energies=energies,
    )


def cumulative_kdistance(kpts_frac: np.ndarray, recip: Optional[np.ndarray]) -> np.ndarray:
    """x-axis from actual EIGENVAL k-points (physical Å⁻¹ if POSCAR available)."""
    k_cart = frac_to_cart_k(kpts_frac, recip)
    dist   = np.zeros(len(k_cart), dtype=float)
    if len(k_cart) > 1:
        dist[1:] = np.cumsum(np.linalg.norm(np.diff(k_cart, axis=0), axis=1))
    return dist


def tick_positions(kpath: KPath, recip: Optional[np.ndarray]):
    """Tick x-positions and pretty labels from KPOINTS path nodes."""
    if kpath.coord_type == "cartesian":
        nodes_c = np.array(kpath.nodes_frac, dtype=float)
    else:
        nodes_c = frac_to_cart_k(kpath.nodes_frac, recip)
    ticks = np.zeros(len(nodes_c), dtype=float)
    if len(nodes_c) > 1:
        ticks[1:] = np.cumsum(np.linalg.norm(np.diff(nodes_c, axis=0), axis=1))
    labels = [pretty_label(lb) for lb in kpath.labels]
    return ticks, labels


def align_ticks(ticks: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Rescale tick positions to exactly match the EIGENVAL x-axis endpoint."""
    if ticks[-1] == 0 or x[-1] == 0:
        return ticks
    return ticks * (x[-1] / ticks[-1])


# ── segment builder for LineCollection ───────────────────────────────────────

def make_segments(x, y, w, interpolate, n_interp):
    """Build LineCollection segments and per-segment average colors.

    If interpolate is True, linearly interpolate the scalar quantities x, y,
    and w between neighboring k-points. This is purely visual smoothing of
    plotted scalars — no wavefunction interpolation is performed.
    """
    if interpolate and n_interp > 1:
        xi, yi, wi = [], [], []
        for i in range(len(x) - 1):
            xi.append(np.linspace(x[i], x[i + 1], n_interp, endpoint=False))
            yi.append(np.linspace(y[i], y[i + 1], n_interp, endpoint=False))
            wi.append(np.linspace(w[i], w[i + 1], n_interp, endpoint=False))
        xi.append([x[-1]]); yi.append([y[-1]]); wi.append([w[-1]])
        x = np.concatenate(xi)
        y = np.concatenate(yi)
        w = np.concatenate(wi)
    pts    = np.array([x, y]).T
    segs   = np.stack([pts[:-1], pts[1:]], axis=1)
    colors = (w[:-1] + w[1:]) * 0.5
    return segs, colors


# ── load saved CNO data ───────────────────────────────────────────────────────
cno_orbs = np.load(cno_orb_file)
cno_occ  = np.load(cno_occ_file)
Nx, Ny, Nz = np.load(grid_shape_file).astype(int)
Nr = Nx * Ny * Nz

print(f"Loaded CNO orbitals   : {cno_orb_file}  shape={cno_orbs.shape}")
print(f"Loaded CNO occupations: {cno_occ_file}  n={len(cno_occ)}")
print(f"FFT grid shape        : ({Nx}, {Ny}, {Nz})  Nr={Nr}")
print(f"WS mode               : {ws_mode}")

if ws_mode:
    _ws_files = {
        "ws_base_indices.npy":    output_dir / "ws_base_indices.npy",
        "ws_points_frac_cont.npy": output_dir / "ws_points_frac_cont.npy",
    }
    for fname, p in _ws_files.items():
        if not p.exists():
            raise FileNotFoundError(
                f"WS mode active but {fname} not found: {p}\n"
                "Rerun Wavecar_to_Coeff.py to regenerate WS map files."
            )
    _prim_indices   = np.load(output_dir / "ws_base_indices.npy")
    _r_for_phase    = np.load(output_dir / "ws_points_frac_cont.npy")
    print(f"WS map loaded         : prim_indices={_prim_indices.shape}  "
          f"r_ws_frac_cont={_r_for_phase.shape}")

    def _to_psi(u_3d: np.ndarray, k_frac: np.ndarray) -> np.ndarray:
        """Reindex u grid to WS ordering and apply Bloch phase."""
        u_ws = u_3d[_prim_indices[:, 0], _prim_indices[:, 1], _prim_indices[:, 2]]
        return u_ws * np.exp(2j * np.pi * (_r_for_phase @ k_frac))

else:
    _ix, _iy, _iz = [a.ravel() for a in np.mgrid[0:Nx, 0:Ny, 0:Nz]]
    _r_for_phase   = np.column_stack([_ix / Nx, _iy / Ny, _iz / Nz])

    def _to_psi(u_3d: np.ndarray, k_frac: np.ndarray) -> np.ndarray:
        """Flatten u grid and apply Bloch phase at primitive positions."""
        return u_3d.reshape(Nr) * np.exp(2j * np.pi * (_r_for_phase @ k_frac))

if cno_orbs.shape[0] != Nr:
    raise ValueError(
        f"cno_orbitals.npy has {cno_orbs.shape[0]} grid points "
        f"but fft_grid_shape gives Nr={Nr}."
    )
n_cno_avail = cno_orbs.shape[1]
for idx in cno_indices:
    if idx < 0 or idx >= n_cno_avail:
        raise ValueError(
            f"cno_indices contains {idx}, which is out of range; "
            f"cno_orbitals.npy has {n_cno_avail} columns (0–{n_cno_avail - 1})."
        )

# ── prepare selected CNOs ─────────────────────────────────────────────────────
# Build (n_cnos, Nr) matrix of unit-normalised CNO row vectors.
_cnos_raw   = np.stack([cno_orbs[:, idx] for idx in cno_indices])       # (n_cnos, Nr)
_norms      = np.sqrt(np.sum(np.abs(_cnos_raw) ** 2, axis=1, keepdims=True))
cnos_matrix = _cnos_raw / _norms
cnos_conj   = cnos_matrix.conj()   # precomputed for fast inner products

print(f"\nSelected CNO indices  : {cno_indices}")
for idx in cno_indices:
    print(f"  CNO {idx:3d} occupation : {cno_occ[idx]:.8e}")

# ── load line-mode WAVECAR ────────────────────────────────────────────────────
wfc = vaspwfc(str(wavecar_path), lsorbit=LSORBIT)

if tuple(wfc._ngrid) != (Nx, Ny, Nz):
    raise ValueError(
        f"WAVECAR FFT grid {tuple(wfc._ngrid)} does not match "
        f"saved grid ({Nx}, {Ny}, {Nz}) from fft_grid_shape.npy."
    )

# ── build k-axis from KPOINTS + POSCAR (no KLABELS) ──────────────────────────
kpath   = read_kpoints_line_mode(kpoints_path)
recip   = reciprocal_lattice(poscar_path)

if recip is None:
    print("WARNING: POSCAR not found; x-axis will be in fractional reciprocal units, not Å⁻¹.",
          file=sys.stderr)

eig = read_eigenval(eigenval_path)

if eig.nkpts != wfc._nkpts:
    raise ValueError(
        f"EIGENVAL has {eig.nkpts} k-points but WAVECAR has {wfc._nkpts}."
    )

kdistances      = cumulative_kdistance(eig.kpts_frac, recip)
kticks, klabels = tick_positions(kpath, recip)
kticks          = align_ticks(kticks, kdistances)

# energies: (nbands, nkpts) for convenient per-band indexing
spin_idx         = ispin - 1
nbands           = eig.nbands
nkpts            = eig.nkpts
energies_shifted = (eig.energies[spin_idx] - efermi).T   # (nbands, nkpts)

print(f"\nEIGENVAL : nkpts={nkpts}, nbands={nbands}, nelect={eig.nelect}")
print(f"KPOINTS  : {' → '.join(klabels)}")

# ── compute |<CNO_i | psi_nk>|² for every CNO, band, and k-point ─────────────
# weights_all[ci, b, k] = |<CNO_i | psi_nk>|²
# cnos_conj @ psi evaluates all inner products in one matrix-vector multiply.
n_cnos = len(cno_indices)
print(f"\nComputing |<CNO | psi_nk>|² for {n_cnos} CNO(s), "
      f"{nbands} bands × {nkpts} k-points...")

weights_all = np.zeros((n_cnos, nbands, nkpts))

for ik in range(1, nkpts + 1):
    k_frac = eig.kpts_frac[ik - 1]
    gvec   = wfc.gvectors(ik)
    gx     = gvec[:, 0] % Nx
    gy     = gvec[:, 1] % Ny
    gz     = gvec[:, 2] % Nz
    nG     = len(gx)

    for ib in range(1, nbands + 1):
        coeff      = wfc.readBandCoeff(ispin=ispin, ikpt=ik, iband=ib, norm=True)
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

    if ik == 1 or ik % 10 == 0 or ik == nkpts:
        w_sum_k = weights_all[:, :, ik - 1].sum(axis=0)
        print(f"  k-point {ik:4d}/{nkpts}  max summed weight: {w_sum_k.max():.4f}")

# ── optional: set up the 3D isosurface snapshot renderer ─────────────────────
_render_3d = render_cno_3d
if _render_3d:
    try:
        from cno_visualizer.snapshot import render_density_gif
        sys.path.insert(0, str(base_dir / "helper functions"))
        from ws_cell import read_poscar_structure
        _lat3d, _, _, _asym3d, _, _, _acart3d = read_poscar_structure(poscar_path)
        print("3D snapshots         : ON  (CNO-Visualizer found)")
    except Exception as exc:
        print(f"3D snapshots         : OFF — CNO-Visualizer unavailable ({exc})")
        _render_3d = False

# ── per-CNO output: weights, CSV, metadata, plots ────────────────────────────
print()
for ci, idx in enumerate(cno_indices):
    w = weights_all[ci]   # (nbands, nkpts) for this CNO only

    # save weights
    np.save(fatband_dir / f"cno_{idx:03d}_weights.npy", w)

    # projection CSV
    csv_path = fatband_dir / f"cno_{idx:03d}_projection_table.csv"
    with open(csv_path, "w") as fh:
        fh.write("k_index,band_index,k_distance,energy_eV,weight\n")
        for ib in range(nbands):
            for ik in range(nkpts):
                fh.write(f"{ik},{ib},{kdistances[ik]:.8f},"
                         f"{energies_shifted[ib, ik]:.6f},{w[ib, ik]:.8e}\n")

    # metadata
    meta_path = fatband_dir / f"cno_{idx:03d}_fatband_metadata.txt"
    with open(meta_path, "w") as fh:
        fh.write(f"cno_index               : {idx}\n")
        fh.write(f"cno_occupation          : {cno_occ[idx]:.10e}\n")
        fh.write(f"total_weight            : {w.sum():.10f}\n")
        fh.write(f"max_weight              : {w.max():.10f}\n")
        fh.write(f"efermi                  : {efermi}\n")
        fh.write(f"ispin                   : {ispin}\n")
        fh.write(f"grid shape (Nx, Ny, Nz) : ({Nx}, {Ny}, {Nz})\n")
        fh.write(f"nkpts                   : {nkpts}\n")
        fh.write(f"nbands                  : {nbands}\n")
        fh.write(f"k-axis source           : KPOINTS + POSCAR (Ang^-1)\n")
        fh.write(f"ws_mode                 : {ws_mode}\n")
        fh.write("NOTE: weight = |<CNO | psi_nk>|^2\n")
        fh.write("NOTE: psi_nk(r) = u_nk(r) * exp(2*pi*i * k_frac . r_frac_cont)\n")
        fh.write("NOTE: in WS mode r_frac_cont is the continuous (unwrapped) WS coord\n")
        fh.write("NOTE: interpolation (if used) is only for plotting scalar "
                 "weights/energies, not wavefunctions.\n")

    vmin = 0.0
    vmax = w.max()

    # LineCollection plot
    all_segs, all_colors = [], []
    for b in range(nbands):
        segs, seg_colors = make_segments(
            kdistances, energies_shifted[b], w[b],
            interpolate_for_plot, interp_points_per_segment,
        )
        all_segs.append(segs)
        all_colors.append(seg_colors)
    all_segs   = np.concatenate(all_segs,   axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    for b in range(nbands):
        ax.plot(kdistances, energies_shifted[b], 'k:', linewidth=0.5, zorder=1)
    lc = LineCollection(all_segs, cmap=cmap, linewidth=linewidth, zorder=2)
    lc.set_array(all_colors)
    lc.set_clim(vmin, vmax)
    ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, pad=0.02)
    cb.set_label(r"$|\langle\mathrm{CNO}\,|\,\psi_{nk}\rangle|^2$", fontsize=10)
    for tick in kticks:
        ax.axvline(tick, color="0.75", lw=0.8, zorder=0)
    ax.axhline(0.0, color="k", lw=0.8, ls="--", zorder=0)
    ax.set_xticks(kticks)
    ax.set_xticklabels(klabels)
    ax.set_ylabel(r"$E - E_F$ (eV)")
    ax.set_xlabel("k-path")
    ax.set_title(f"CNO {idx}  (occ={cno_occ[idx]:.4f})  fatband")
    ax.set_xlim(kdistances[0], kdistances[-1])
    ax.set_ylim(ylim)
    plt.tight_layout()
    lc_path = fatband_dir / f"cno_{idx:03d}_colored_fatband.png"
    fig.savefig(lc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # scatter plot
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for b in range(nbands):
        ax2.plot(kdistances, energies_shifted[b], 'k:', linewidth=0.5, zorder=1)
    sc = None
    for b in range(nbands):
        sc = ax2.scatter(
            kdistances, energies_shifted[b],
            c=w[b], cmap=cmap, s=6,
            vmin=vmin, vmax=vmax, linewidths=0,
        )
    if sc is not None:
        fig2.colorbar(sc, ax=ax2, pad=0.02).set_label(
            r"$|\langle\mathrm{CNO}\,|\,\psi_{nk}\rangle|^2$", fontsize=10)
    for tick in kticks:
        ax2.axvline(tick, color="0.75", lw=0.8, zorder=0)
    ax2.axhline(0.0, color="k", lw=0.8, ls="--", zorder=0)
    ax2.set_xticks(kticks)
    ax2.set_xticklabels(klabels)
    ax2.set_ylabel(r"$E - E_F$ (eV)")
    ax2.set_xlabel("k-path")
    ax2.set_title(f"CNO {idx}  (occ={cno_occ[idx]:.4f})  fatband (scatter)")
    ax2.set_ylim(ylim)
    ax2.set_xlim(kdistances[0], kdistances[-1])
    plt.tight_layout()
    sc_path = fatband_dir / f"cno_{idx:03d}_scatter_fatband.png"
    fig2.savefig(sc_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"  CNO {idx:3d}  max_w={vmax:.4f}  → {lc_path.name},  {sc_path.name}")

    # 3D density-isosurface GIF (VESTA-like, no phase) of this same CNO.
    if _render_3d:
        try:
            u3d = cno_orbs[:, idx]
            if ws_mode:
                g3 = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
                g3[_prim_indices[:, 0], _prim_indices[:, 1], _prim_indices[:, 2]] = u3d
            else:
                g3 = u3d.reshape(Nx, Ny, Nz)
            gif_path = fatband_dir / f"cno_{idx:03d}_structure.gif"
            render_density_gif(
                g3, _lat3d, _acart3d, _asym3d, str(gif_path),
                iso_fraction=cno_3d_iso, replication=cno_3d_replicate,
                seconds=cno_3d_seconds, fps=cno_3d_fps,
                deg_per_sec=cno_3d_deg_per_sec, spin_axis=cno_3d_spin_axis,
            )
            print(f"  CNO {idx:3d}  3D isosurface → {gif_path.name}")
        except Exception as exc:
            print(f"  CNO {idx:3d}  3D render skipped: {exc}")

print("\nAll done.")
