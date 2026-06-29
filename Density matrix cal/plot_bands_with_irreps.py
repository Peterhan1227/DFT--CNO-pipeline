#!/usr/bin/env python3
"""
plot_bands_with_irreps.py

Plots a VASP line-mode band structure with optional irrep labels at
high-symmetry k-points from an IrRep JSON output file.

Reads the same files as cno_fatband.py:
  Data/<MATERIAL>/KPOINTS       (line-mode k-path)
  Data/<MATERIAL>/EIGENVAL_lm   (line-mode band energies)
  Data/<MATERIAL>/POSCAR        (lattice vectors for Ang^-1 x-axis)

MATERIAL and EFERMI default to values in config.py (same as all other scripts).

Irrep label convention (Bilbao / Mulliken):
  GM5+  ->  Gamma_5^+      W1 -> W_1      L3- -> L_3^-

Usage (run from the "Density matrix cal" folder):
  python plot_bands_with_irreps.py
  python plot_bands_with_irreps.py --irrep-json ../Irrep/output/test_si_irrep.json
  python plot_bands_with_irreps.py --material CoSn --efermi 5.98 --ylim -4 4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from config import MATERIAL, EFERMI

# ── project paths (mirrors cno_fatband.py) ───────────────────────────────────
base_dir    = Path(__file__).resolve().parent
data_dir    = base_dir / "Data" / MATERIAL
output_dir  = data_dir / "output"
irrep_dir   = base_dir.parent / "Irrep" / "output"

output_dir.mkdir(parents=True, exist_ok=True)

# ── user settings (edit here or use CLI args) ─────────────────────────────────
ylim = [-10, 15]


# ─────────────────────────────────────────────────── data containers ──────────
# (identical to the structures in cno_fatband.py)

@dataclass
class KPath:
    n_per_segment: int
    coord_type:    str
    nodes_frac:    np.ndarray   # (n_nodes, 3)
    labels:        list

    @property
    def n_segments(self) -> int:
        return len(self.labels) - 1


@dataclass
class EigenvalData:
    ispin:     int
    nelect:    float
    nkpts:     int
    nbands:    int
    kpts_frac: np.ndarray   # (nkpts, 3)
    energies:  np.ndarray   # (nspin, nkpts, nbands)


# ──────────────────────────────────────────────────── k-path helpers ──────────
# Ported verbatim from cno_fatband.py / plot_vasp_bs.py.

def pretty_label(label: str) -> str:
    if label.strip().upper() in {"G", "GAMMA", "GM", "GA"}:
        return r"$\Gamma$"
    return label.strip()


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


def read_kpoints_line_mode(path: Path) -> KPath:
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh]

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

    nodes  = [raw_points[0]]
    labels = [raw_labels[0] or "K0"]
    for seg in range(n_segments):
        nodes.append(raw_points[2 * seg + 1])
        labels.append(raw_labels[2 * seg + 1] or f"K{seg + 1}")

    return KPath(
        n_per_segment=n_per_segment,
        coord_type=coord_type,
        nodes_frac=np.array(nodes, dtype=float),
        labels=labels,
    )


def read_poscar_lattice(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
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
        lattice *= (abs(scale) / abs(np.linalg.det(lattice))) ** (1.0 / 3.0)
    return lattice


def reciprocal_lattice(poscar: Path) -> Optional[np.ndarray]:
    A = read_poscar_lattice(poscar)
    return None if A is None else 2.0 * np.pi * np.linalg.inv(A).T


def frac_to_cart_k(kpts: np.ndarray, recip: Optional[np.ndarray]) -> np.ndarray:
    return np.array(kpts, dtype=float) @ recip if recip is not None else np.array(kpts, dtype=float)


def read_eigenval(path: Path) -> EigenvalData:
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

    return EigenvalData(ispin_ev, nelect, nkpts, nbands, kpts, energies)


def cumulative_kdistance(kpts_frac: np.ndarray, recip: Optional[np.ndarray]) -> np.ndarray:
    k_cart = frac_to_cart_k(kpts_frac, recip)
    dist   = np.zeros(len(k_cart), dtype=float)
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
    return ticks, [pretty_label(lb) for lb in kpath.labels]


def align_ticks(ticks: np.ndarray, x: np.ndarray) -> np.ndarray:
    if ticks[-1] == 0 or x[-1] == 0:
        return ticks
    return ticks * (x[-1] / ticks[-1])


# ───────────────────────────────────────────────── IrRep JSON parsing ─────────

@dataclass
class IrrepBlock:
    kpname:      str
    energy_mean: float      # absolute eV, as stored in JSON
    dimension:   int
    irrep_label: str        # dominant irrep, e.g. "GM5+", "W1"


def _unwrap(obj):
    if isinstance(obj, dict):
        if obj.get("@class") == "array":
            return np.array(obj["data"])
        return {k: _unwrap(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unwrap(x) for x in obj]
    return obj


def _kpname_from_irreps(irreps_list: list[dict]) -> str:
    prefixes: list[str] = []
    for blk in irreps_list:
        for irname in blk:
            m = re.match(r'^([A-Za-z]+)', irname)
            if m:
                prefixes.append(m.group(1).upper())
    return max(set(prefixes), key=prefixes.count) if prefixes else ""


def load_irrep_json(path: Path) -> list[IrrepBlock]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    blocks: list[IrrepBlock] = []
    for entry in raw.get("characters and irreps", []):
        for kp in entry["subspace"]["k points"]:
            e_means   = np.array(_unwrap(kp["energies_mean"]))
            dims      = np.array(_unwrap(kp["dimensions"]), dtype=int)
            irreps_raw = kp["irreps"]
            irreps_u = [
                {irn: (_unwrap(v) if isinstance(v, dict) else v)
                 for irn, v in blk.items()}
                for blk in irreps_raw
            ]
            kpname = _kpname_from_irreps(irreps_u)
            for ib, (e, d) in enumerate(zip(e_means, dims)):
                blk_d = irreps_u[ib] if ib < len(irreps_u) else {}
                label = (max(blk_d, key=lambda k: float(blk_d[k][0]))
                         if blk_d else "?")
                blocks.append(IrrepBlock(kpname, float(e), int(d), label))
    return blocks


# ─────────────────────────────────── irrep label formatting (LaTeX) ───────────

_KNAME_LATEX = {
    "GM": r"\Gamma", "G": r"\Gamma", "GA": r"\Gamma",
    "A": "A",  "H": "H",  "K": "K",  "L": "L",  "M": "M",
    "N": "N",  "P": "P",  "R": "R",  "T": "T",  "U": "U",
    "V": "V",  "W": "W",  "X": "X",  "Y": "Y",  "Z": "Z",
    "D": r"\Delta", "DT": r"\Delta", "LD": r"\Lambda",
    "S": r"\Sigma", "SM": r"\Sigma", "F": "F",
}
_IRREP_RE = re.compile(r'^([A-Za-z]+)(\d+)([+-]?)(.*)$')

_NODE_TO_KPNAME = {
    "G": "GM", "GAMMA": "GM", "GM": "GM", "GA": "GM",
    "A": "A",  "H": "H",  "K": "K",  "L": "L",  "M": "M",
    "N": "N",  "P": "P",  "R": "R",  "T": "T",  "U": "U",
    "V": "V",  "W": "W",  "X": "X",  "Y": "Y",  "Z": "Z",
}


def fmt_irrep(label: str) -> str:
    m = _IRREP_RE.match(label.strip())
    if m is None:
        return label
    prefix, idx, sign, _ = m.groups()
    kname = _KNAME_LATEX.get(prefix.upper(), prefix)
    sup = f"^{{{sign}}}" if sign else ""
    return f"${kname}_{{{idx}}}{sup}$"


def match_nodes_to_blocks(
    kpath: KPath, blocks: list[IrrepBlock]
) -> dict[int, list[IrrepBlock]]:
    by_kpname: dict[str, list[IrrepBlock]] = {}
    for blk in blocks:
        by_kpname.setdefault(blk.kpname, []).append(blk)
    result: dict[int, list[IrrepBlock]] = {}
    for i, lbl in enumerate(kpath.labels):
        cname = _NODE_TO_KPNAME.get(lbl.upper().strip(), lbl.upper().strip())
        if cname in by_kpname:
            result[i] = by_kpname[cname]
    return result


# ─────────────────────────────────────────────────────────── main plot ─────────

def plot_bands(
    material:          str,
    efermi:            float,
    irrep_json:        Optional[Path],
    ylim_arg:          Optional[tuple[float, float]],
    output:            Optional[Path],
    figsize:           tuple[float, float],
    dpi:               int,
    show:              bool,
    label_fontsize:    float,
    label_offset_frac: float,
) -> None:

    kpoints_path  = base_dir / "Data" / material / "KPOINTS"
    eigenval_path = base_dir / "Data" / material / "EIGENVAL_lm"
    poscar_path   = base_dir / "Data" / material / "POSCAR"

    for p in (kpoints_path, eigenval_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Not found: {p}\n"
                f"  Expected line-mode files in Data/{material}/"
            )

    kpath = read_kpoints_line_mode(kpoints_path)
    eig   = read_eigenval(eigenval_path)
    recip = reciprocal_lattice(poscar_path)

    if recip is None:
        print("WARNING: POSCAR not found; x-axis in fractional units.",
              file=sys.stderr)

    # x-axis — exactly as in cno_fatband.py
    kdistances      = cumulative_kdistance(eig.kpts_frac, recip)
    kticks, klabels = tick_positions(kpath, recip)
    kticks          = align_ticks(kticks, kdistances)

    # energies shaped (nbands, nkpts) — same as cno_fatband.py's energies_shifted
    energies_shifted = (eig.energies[0] - efermi).T   # (nbands, nkpts)

    print(f"EIGENVAL_lm : nkpts={eig.nkpts}, nbands={eig.nbands}")
    print(f"KPOINTS     : {' -> '.join(kpath.labels)}")
    print(f"E_Fermi     : {efermi} eV")

    # ── load irreps ────────────────────────────────────────────────────────────
    blocks: list[IrrepBlock] = []
    node_blocks: dict[int, list[IrrepBlock]] = {}

    if irrep_json is not None and irrep_json.exists():
        blocks = load_irrep_json(irrep_json)
        node_blocks = match_nodes_to_blocks(kpath, blocks)
        kpnames_found = sorted({b.kpname for b in blocks})
        print(f"Irrep JSON  : {irrep_json.name}  "
              f"({len(blocks)} blocks: {', '.join(kpnames_found)})")
        for i, blist in node_blocks.items():
            print(f"  node [{i}] '{kpath.labels[i]}' -> "
                  f"{blist[0].kpname} ({len(blist)} blocks)")
    elif irrep_json is not None:
        print(f"WARNING: IrRep JSON not found: {irrep_json}", file=sys.stderr)

    # ── figure — exactly mirroring cno_fatband.py structure ───────────────────
    fig, ax = plt.subplots(figsize=figsize)

    # bands: one ax.plot call per band, k-axis = kdistances, y = energies
    for b in range(eig.nbands):
        ax.plot(kdistances, energies_shifted[b], color="k", linewidth=0.8, zorder=2)

    # high-symmetry tick lines
    for tick in kticks:
        ax.axvline(tick, color="0.75", lw=0.8, zorder=0)
    ax.axhline(0.0, color="k", lw=0.8, ls="--", zorder=0)

    # ── irrep labels ───────────────────────────────────────────────────────────
    if node_blocks:
        _ylim = ylim_arg if ylim_arg else ylim
        e_lo, e_hi = _ylim[0], _ylim[1]
        xspan  = kdistances[-1] - kdistances[0]
        dx     = label_offset_frac * xspan
        min_dy = (e_hi - e_lo) * 0.02

        for node_idx, blist in node_blocks.items():
            x_tick  = kticks[node_idx]
            is_last = (node_idx == len(kpath.labels) - 1)
            x_lbl   = x_tick - dx if is_last else x_tick + dx
            ha      = "right" if is_last else "left"

            placed_ys: list[float] = []
            for blk in sorted(blist, key=lambda b: b.energy_mean):
                e_rel = blk.energy_mean - efermi
                if e_rel < e_lo or e_rel > e_hi:
                    continue
                y_place = e_rel
                for prev_y in placed_ys:
                    if abs(y_place - prev_y) < min_dy:
                        y_place = prev_y + min_dy
                placed_ys.append(y_place)
                ax.text(x_lbl, y_place, fmt_irrep(blk.irrep_label),
                        fontsize=label_fontsize, ha=ha, va="center",
                        zorder=5, clip_on=True)

    # ── axes ───────────────────────────────────────────────────────────────────
    ax.set_xticks(kticks)
    ax.set_xticklabels(klabels)
    ax.set_xlim(kdistances[0], kdistances[-1])
    ax.set_ylabel(r"$E - E_F$ (eV)")
    _ylim = ylim_arg if ylim_arg else ylim
    ax.set_ylim(_ylim[0], _ylim[1])

    fig.tight_layout()

    # ── save ───────────────────────────────────────────────────────────────────
    if output is None:
        suffix = "_irreps" if node_blocks else ""
        output = base_dir / "Data" / material / "output" / f"band_structure{suffix}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────── CLI ───────

def main() -> None:
    # Default irrep JSON for whichever material config.py selects
    default_irrep = None
    if MATERIAL == "Si":
        candidate = irrep_dir / "test_si_irrep.json"
        if candidate.exists():
            default_irrep = candidate

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--material",    default=MATERIAL,
                   help=f"Material subfolder under Data/  (default from config.py: {MATERIAL})")
    p.add_argument("--efermi",      type=float, default=EFERMI,
                   help=f"Fermi energy eV  (default from config.py: {EFERMI})")
    p.add_argument("--irrep-json",  default=str(default_irrep) if default_irrep else None,
                   metavar="JSON",
                   help="IrRep JSON file for labels  "
                        f"(default for Si: {default_irrep})")
    p.add_argument("--no-irreps",   action="store_true",
                   help="Skip irrep labels even if JSON is found")
    p.add_argument("--ylim",        type=float, nargs=2, default=None,
                   metavar=("EMIN", "EMAX"),
                   help=f"Energy window relative to E_F  (default: {ylim})")
    p.add_argument("--out",         default=None, metavar="FILE",
                   help="Override output path  (default: Data/<material>/output/band_structure[_irreps].png)")
    p.add_argument("--no-show",     action="store_true")
    p.add_argument("--dpi",         type=int,   default=150)
    p.add_argument("--figsize",     type=float, nargs=2, default=(9.0, 5.0))
    p.add_argument("--label-fontsize",  type=float, default=7.0)
    p.add_argument("--label-offset",    type=float, default=0.015)
    args = p.parse_args()

    irrep_json = None
    if not args.no_irreps and args.irrep_json:
        irrep_json = Path(args.irrep_json)

    plot_bands(
        material=args.material,
        efermi=args.efermi,
        irrep_json=irrep_json,
        ylim_arg=tuple(args.ylim) if args.ylim else None,
        output=Path(args.out) if args.out else None,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        show=not args.no_show,
        label_fontsize=args.label_fontsize,
        label_offset_frac=args.label_offset,
    )


if __name__ == "__main__":
    main()
