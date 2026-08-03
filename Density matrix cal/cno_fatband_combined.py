"""
cno_fatband_combined.py

Combined multi-CNO fatband plot, built entirely from cno_fatband.py's
already-saved output (cno_{idx:03d}_weights.npy) -- does NOT reload the
WAVECAR or recompute any projections. Each selected CNO gets its own solid
color; per-(band,k) opacity encodes |<CNO|psi_nk>|^2, so overlapping CNO
character shows up as layered/blended color instead of competing on a
single shared white-to-red scale.

Reads:
  output/<SUBDIR>/cno_fatband/cno_{idx:03d}_weights.npy   (from cno_fatband.py)
  output/<SUBDIR>/cno_occupations.npy
  KPOINTS, POSCAR, EIGENVAL_lm (or EIGENVAL)               (for the k-axis/energy axis only)

Writes:
  output/<SUBDIR>/cno_fatband/<output_file>   (combined PNG)

Run cno_fatband.py first (with cno_indices covering everything you might
want to combine) so the weight files this script reads actually exist.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from config import MATERIAL, OUTPUT_SUBDIR
from config import EFERMI

# ── user settings ────────────────────────────────────────────────────────────
ispin              = 1
cno_indices        = [0, 1, 2]      # which saved CNOs to combine
cno_colors         = None                  # None = auto (tab10-ish); or a list of
                                            # matplotlib colors, same length as cno_indices
efermi             = EFERMI
ylim               = [-10, 10]
linewidth          = 3.0
max_alpha          = 0.95                  # opacity at weight == vmax (per CNO, see below)
per_cno_vmax       = None                  # None = each CNO normalized to its OWN max weight;
                                            # or a float to use one shared scale for all CNOs
output_file        = "cno_combined_fatband.png"
interpolate_for_plot      = False
interp_points_per_segment = 3

# ── paths ─────────────────────────────────────────────────────────────────────
base_dir      = Path(__file__).resolve().parent
data_dir      = base_dir / "Data" / MATERIAL
output_dir    = data_dir / "output" / OUTPUT_SUBDIR
fatband_dir   = output_dir / "cno_fatband"
kpoints_path  = data_dir / "KPOINTS"
poscar_path   = data_dir / "POSCAR"
eigenval_path = (data_dir / "EIGENVAL_lm" if (data_dir / "EIGENVAL_lm").exists()
                  else data_dir / "EIGENVAL")
cno_occ_file  = output_dir / "cno_occupations.npy"


# ── k-path / EIGENVAL helpers (ported from cno_fatband.py, unchanged) ────────

@dataclass
class KPath:
    n_per_segment: int
    coord_type: str
    nodes_frac: np.ndarray
    labels: list
    raw_points: np.ndarray
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
    kpts_frac: np.ndarray
    energies: np.ndarray


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
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    if len(lines) < 5:
        raise ValueError("KPOINTS seems too short to be a line-mode file.")
    n_per_segment = int(float(lines[1].split()[0]))
    mode = lines[2].strip().lower()
    if not (mode.startswith("l") or "line" in mode):
        raise ValueError(f"Expected line-mode KPOINTS; line 3 is: {lines[2]!r}")
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
    nodes = [raw_points[0]]
    labels = [raw_labels[0] or "K0"]
    for seg in range(n_segments):
        end_idx = 2 * seg + 1
        nodes.append(raw_points[end_idx])
        labels.append(raw_labels[end_idx] or f"K{seg + 1}")
    return KPath(n_per_segment=n_per_segment, coord_type=coord_type,
                 nodes_frac=np.array(nodes, dtype=float), labels=labels,
                 raw_points=raw_points, raw_labels=raw_labels)


def read_poscar_lattice(path) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        lines = fh.readlines()
    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)],
                        dtype=float)
    if scale > 0:
        lattice *= scale
    else:
        vol = abs(np.linalg.det(lattice))
        lattice *= (abs(scale) / vol) ** (1.0 / 3.0)
    return lattice


def reciprocal_lattice(poscar_path) -> Optional[np.ndarray]:
    A = read_poscar_lattice(poscar_path)
    if A is None:
        return None
    return 2.0 * np.pi * np.linalg.inv(A).T


def frac_to_cart_k(kpts: np.ndarray, recip: Optional[np.ndarray]) -> np.ndarray:
    if recip is None:
        return np.array(kpts, dtype=float)
    return np.array(kpts, dtype=float) @ recip


def read_eigenval(path) -> EigenvalData:
    with open(path) as fh:
        lines = fh.readlines()
    try:
        ispin_ev = int(lines[0].split()[3])
    except Exception:
        ispin_ev = 1
    nelect = float(lines[5].split()[0])
    nkpts = int(lines[5].split()[1])
    nbands = int(lines[5].split()[2])
    nspin = 2 if ispin_ev == 2 else 1
    kpts = np.zeros((nkpts, 3), dtype=float)
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
    return EigenvalData(ispin=ispin_ev, nelect=nelect, nkpts=nkpts, nbands=nbands,
                         kpts_frac=kpts, energies=energies)


def cumulative_kdistance(kpts_frac: np.ndarray, recip: Optional[np.ndarray]) -> np.ndarray:
    k_cart = frac_to_cart_k(kpts_frac, recip)
    dist = np.zeros(len(k_cart), dtype=float)
    if len(k_cart) > 1:
        dist[1:] = np.cumsum(np.linalg.norm(np.diff(k_cart, axis=0), axis=1))
    return dist


def tick_positions(kpath: KPath, recip: Optional[np.ndarray]):
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
    if ticks[-1] == 0 or x[-1] == 0:
        return ticks
    return ticks * (x[-1] / ticks[-1])


def make_segments(x, y, w, interpolate, n_interp):
    """Build LineCollection segments and per-segment average weight (0-1
    range expected upstream)."""
    if interpolate and n_interp > 1:
        xi, yi, wi = [], [], []
        for i in range(len(x) - 1):
            xi.append(np.linspace(x[i], x[i + 1], n_interp, endpoint=False))
            yi.append(np.linspace(y[i], y[i + 1], n_interp, endpoint=False))
            wi.append(np.linspace(w[i], w[i + 1], n_interp, endpoint=False))
        xi.append([x[-1]]); yi.append([y[-1]]); wi.append([w[-1]])
        x = np.concatenate(xi); y = np.concatenate(yi); w = np.concatenate(wi)
    pts = np.array([x, y]).T
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    w_avg = (w[:-1] + w[1:]) * 0.5
    return segs, w_avg


# ── load axis data (no WAVECAR needed -- KPOINTS/POSCAR/EIGENVAL only) ───────

kpath = read_kpoints_line_mode(kpoints_path)
recip = reciprocal_lattice(poscar_path)
if recip is None:
    print("WARNING: POSCAR not found; x-axis will be in fractional reciprocal units, not Å⁻¹.",
          file=sys.stderr)
eig = read_eigenval(eigenval_path)

kdistances = cumulative_kdistance(eig.kpts_frac, recip)
kticks, klabels = tick_positions(kpath, recip)
kticks = align_ticks(kticks, kdistances)

spin_idx = ispin - 1
nbands = eig.nbands
nkpts = eig.nkpts
energies_shifted = (eig.energies[spin_idx] - efermi).T   # (nbands, nkpts)

print(f"EIGENVAL : nkpts={nkpts}, nbands={nbands}")
print(f"KPOINTS  : {' → '.join(klabels)}")

# ── load cached per-CNO weights (from cno_fatband.py) ────────────────────────

cno_occ = np.load(cno_occ_file) if cno_occ_file.exists() else None

weights_by_cno = {}
for idx in cno_indices:
    wpath = fatband_dir / f"cno_{idx:03d}_weights.npy"
    if not wpath.exists():
        raise FileNotFoundError(
            f"{wpath} not found -- run cno_fatband.py with cno_indices covering "
            f"{idx} first (its cno_indices setting must include every CNO you "
            "want to combine here)."
        )
    w = np.load(wpath)
    if w.shape != (nbands, nkpts):
        raise ValueError(f"{wpath} has shape {w.shape}, expected ({nbands},{nkpts}) "
                          "from the current EIGENVAL -- stale weights file?")
    weights_by_cno[idx] = w
    occ_str = f"{cno_occ[idx]:.4f}" if cno_occ is not None else "?"
    print(f"  loaded CNO {idx:3d} (occ={occ_str})  max_w={w.max():.4f}")

if cno_colors is None:
    palette = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("tab20").colors)
    cno_colors = [palette[i % len(palette)] for i in range(len(cno_indices))]
cno_colors = [to_rgb(c) for c in cno_colors]

# ── combined plot ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5.5))
for b in range(nbands):
    ax.plot(kdistances, energies_shifted[b], 'k:', linewidth=0.5, zorder=1)

legend_handles = []
for (idx, w), color in zip(weights_by_cno.items(), cno_colors):
    vmax = per_cno_vmax if per_cno_vmax is not None else w.max()
    vmax = vmax if vmax > 0 else 1.0

    all_segs, all_alpha = [], []
    for b in range(nbands):
        segs, w_avg = make_segments(kdistances, energies_shifted[b], w[b],
                                     interpolate_for_plot, interp_points_per_segment)
        all_segs.append(segs)
        all_alpha.append(w_avg)
    all_segs = np.concatenate(all_segs, axis=0)
    all_alpha = np.clip(np.concatenate(all_alpha) / vmax, 0.0, 1.0) * max_alpha

    rgba = np.tile(np.array([*color, 0.0]), (len(all_segs), 1))
    rgba[:, 3] = all_alpha

    lc = LineCollection(all_segs, colors=rgba, linewidth=linewidth, zorder=2)
    ax.add_collection(lc)

    occ_str = f"{cno_occ[idx]:.4f}" if cno_occ is not None else "?"
    legend_handles.append(plt.Line2D([0], [0], color=color, lw=3,
                                      label=f"CNO {idx} (occ={occ_str})"))

for tick in kticks:
    ax.axvline(tick, color="0.75", lw=0.8, zorder=0)
ax.axhline(0.0, color="k", lw=0.8, ls="--", zorder=0)
ax.set_xticks(kticks)
ax.set_xticklabels(klabels)
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_xlabel("k-path")
ax.set_title(f"Combined CNO fatband: {cno_indices}")
ax.set_xlim(kdistances[0], kdistances[-1])
ax.set_ylim(ylim)
ax.legend(handles=legend_handles, fontsize=8, loc="best")
plt.tight_layout()

out_path = fatband_dir / output_file
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved -> {out_path}")
