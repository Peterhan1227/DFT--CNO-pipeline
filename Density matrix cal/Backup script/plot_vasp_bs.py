#!/usr/bin/env python3
"""
plot_vasp_bs.py

Self-contained VASP band-structure plotter.

Reads:
  - KPOINTS  : VASP line-mode high-symmetry path + labels
  - EIGENVAL : band eigenvalues from a direct/non-SCF band-structure calculation
  - POSCAR   : lattice vectors, used to convert reciprocal fractional k-points to Å^-1 distance
  - OUTCAR   : optional, used to auto-detect E-fermi if --efermi is not supplied

No KLABELS file is needed.

Typical use:
  python plot_vasp_bs.py --efermi 5.4321 --ylim -2 2
  python plot_vasp_bs.py --out band.png --ylim -4 4
  python plot_vasp_bs.py --spin both --ylim -2 2

Notes:
  - This assumes your KPOINTS is VASP line-mode.
  - If POSCAR is missing, the script still plots, but the x-axis distance is in fractional
    reciprocal-coordinate units rather than physical Å^-1.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------- data containers ----------------------------- #

@dataclass
class KPath:
    n_per_segment: int
    coord_type: str
    nodes_frac: np.ndarray          # shape (n_nodes, 3), reciprocal fractional coords if coord_type is reciprocal
    labels: list[str]               # length n_nodes
    raw_points: np.ndarray          # shape (2*n_segments, 3), as read from KPOINTS
    raw_labels: list[str]           # length 2*n_segments

    @property
    def n_segments(self) -> int:
        return len(self.labels) - 1


@dataclass
class EigenvalData:
    ispin: int
    nelect: float
    nkpts: int
    nbands: int
    kpts_frac: np.ndarray           # shape (nkpts, 3), normally reciprocal fractional coords
    energies: np.ndarray            # shape (nspin, nkpts, nbands)


# ----------------------------- basic utilities ----------------------------- #

def pretty_label(label: str) -> str:
    """Make common high-symmetry labels look nice on the plot."""
    clean = label.strip()
    upper = clean.upper()
    if upper in {"G", "GAMMA", "\\GAMMA", "Γ"}:
        return r"$\Gamma$"
    # Convert labels like SIGMA if you ever use them; otherwise leave as-is.
    return clean


def parse_kpoint_line(line: str) -> tuple[np.ndarray, str]:
    """
    Parse one KPOINTS coordinate line.

    Accepts formats like:
      0.000000 0.000000 0.000000 ! GAMMA
      0.500000 0.000000 0.500000 X
      0.500000 0.000000 0.500000

    Returns:
      coords, label
    """
    before_comment, *after_comment = line.split("!", 1)
    parts = before_comment.split()

    if len(parts) < 3:
        raise ValueError(f"Cannot parse k-point line: {line!r}")

    coords = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)

    label = ""
    if after_comment:
        label = after_comment[0].strip().split()[0] if after_comment[0].strip() else ""
    elif len(parts) >= 4:
        label = parts[3].strip()

    return coords, label


def read_kpoints_line_mode(filename: str = "KPOINTS") -> KPath:
    """
    Read VASP line-mode KPOINTS file and extract the high-symmetry path.

    For VASP line-mode, the path is given as pairs:
      A
      B

      B
      C

      C
      D

    The path nodes become A-B-C-D.
    """
    with open(filename, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    if len(lines) < 5:
        raise ValueError("KPOINTS seems too short to be a line-mode file.")

    try:
        n_per_segment = int(float(lines[1].split()[0]))
    except Exception as exc:
        raise ValueError("Could not read number of k-points per segment from KPOINTS line 2.") from exc

    mode = lines[2].strip().lower()
    if not (mode.startswith("l") or "line" in mode):
        raise ValueError(
            "This script expects a VASP line-mode KPOINTS file. "
            f"KPOINTS line 3 is: {lines[2]!r}"
        )

    coord_type = lines[3].strip().lower()
    if coord_type.startswith("r"):
        coord_type = "reciprocal"
    elif coord_type.startswith("c") or coord_type.startswith("k"):
        coord_type = "cartesian"
    else:
        raise ValueError(
            "Could not identify KPOINTS coordinate type from line 4. "
            "Expected Reciprocal or Cartesian."
        )

    point_lines = [line for line in lines[4:] if line.strip()]
    if len(point_lines) % 2 != 0:
        raise ValueError(
            "Line-mode KPOINTS should contain an even number of endpoint lines "
            "(two endpoints per segment)."
        )

    raw_points = []
    raw_labels = []
    for line in point_lines:
        coords, label = parse_kpoint_line(line)
        raw_points.append(coords)
        raw_labels.append(label)

    raw_points = np.array(raw_points, dtype=float)
    n_segments = len(raw_points) // 2

    # Construct path nodes: first segment start, then every segment end.
    nodes = [raw_points[0]]
    labels = [raw_labels[0] or "K0"]

    for seg in range(n_segments):
        end_idx = 2 * seg + 1
        label = raw_labels[end_idx] or f"K{seg+1}"
        nodes.append(raw_points[end_idx])
        labels.append(label)

    return KPath(
        n_per_segment=n_per_segment,
        coord_type=coord_type,
        nodes_frac=np.array(nodes, dtype=float),
        labels=labels,
        raw_points=raw_points,
        raw_labels=raw_labels,
    )


def read_poscar_lattice(filename: str = "POSCAR") -> Optional[np.ndarray]:
    """
    Read direct lattice vectors from POSCAR.

    Returns:
      direct lattice matrix A with rows a1, a2, a3 in Å.
    """
    if not os.path.exists(filename):
        return None

    with open(filename, "r") as f:
        lines = f.readlines()

    if len(lines) < 5:
        raise ValueError("POSCAR seems too short.")

    scale = float(lines[1].split()[0])
    lattice = np.array(
        [[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)],
        dtype=float,
    )

    # Standard positive scale factor: multiply lattice vectors.
    # Negative scale factor means target cell volume in VASP; handle that too.
    if scale > 0:
        lattice *= scale
    else:
        target_volume = abs(scale)
        current_volume = abs(np.linalg.det(lattice))
        lattice *= (target_volume / current_volume) ** (1.0 / 3.0)

    return lattice


def reciprocal_lattice_from_poscar(filename: str = "POSCAR") -> Optional[np.ndarray]:
    """
    Return reciprocal lattice matrix B with rows b1, b2, b3 in Å^-1,
    where b_i · a_j = 2π δ_ij.
    """
    lattice = read_poscar_lattice(filename)
    if lattice is None:
        return None

    # If A has rows a_i, then B = 2π * inv(A).T has rows b_i.
    return 2.0 * np.pi * np.linalg.inv(lattice).T


def frac_to_cart_k(kpts: np.ndarray, recip_lattice: Optional[np.ndarray]) -> np.ndarray:
    """
    Convert reciprocal fractional k-points to Cartesian reciprocal coordinates.

    If recip_lattice is None, return kpts unchanged as a fallback.
    """
    if recip_lattice is None:
        return np.array(kpts, dtype=float)
    return np.array(kpts, dtype=float) @ recip_lattice


# ----------------------------- EIGENVAL parser ----------------------------- #

def read_eigenval(filename: str = "EIGENVAL") -> EigenvalData:
    """
    Read VASP EIGENVAL.

    Returns energies with shape:
      (nspin, nkpts, nbands)

    For ISPIN=1, nspin=1.
    For ISPIN=2, nspin=2, using spin-up and spin-down eigenvalue columns.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    if len(lines) < 8:
        raise ValueError("EIGENVAL seems too short.")

    header0 = lines[0].split()
    try:
        ispin = int(header0[3])
    except Exception:
        # Most VASP EIGENVAL files have ISPIN as the 4th number on line 1.
        # If unavailable, assume non-spin-polarized.
        ispin = 1

    try:
        nelect = float(lines[5].split()[0])
        nkpts = int(lines[5].split()[1])
        nbands = int(lines[5].split()[2])
    except Exception as exc:
        raise ValueError("Could not read nelect, nkpts, nbands from EIGENVAL line 6.") from exc

    nspin = 2 if ispin == 2 else 1
    kpts = np.zeros((nkpts, 3), dtype=float)
    energies = np.zeros((nspin, nkpts, nbands), dtype=float)

    idx = 6

    for ik in range(nkpts):
        # Skip blank lines before k-point header.
        while idx < len(lines) and not lines[idx].split():
            idx += 1

        if idx >= len(lines):
            raise ValueError(f"Unexpected end of EIGENVAL before k-point {ik+1}.")

        kline = lines[idx].split()
        if len(kline) < 3:
            raise ValueError(f"Could not parse k-point header at line {idx+1}: {lines[idx]!r}")

        kpts[ik] = [float(kline[0]), float(kline[1]), float(kline[2])]
        idx += 1

        for ib in range(nbands):
            if idx >= len(lines):
                raise ValueError(f"Unexpected end of EIGENVAL at k-point {ik+1}, band {ib+1}.")

            parts = lines[idx].split()
            if len(parts) < 2:
                raise ValueError(f"Could not parse band line at line {idx+1}: {lines[idx]!r}")

            if nspin == 1:
                energies[0, ik, ib] = float(parts[1])
            else:
                if len(parts) < 3:
                    raise ValueError(
                        f"ISPIN=2 expected two energy columns at line {idx+1}: {lines[idx]!r}"
                    )
                energies[0, ik, ib] = float(parts[1])  # spin up
                energies[1, ik, ib] = float(parts[2])  # spin down

            idx += 1

    return EigenvalData(
        ispin=ispin,
        nelect=nelect,
        nkpts=nkpts,
        nbands=nbands,
        kpts_frac=kpts,
        energies=energies,
    )


def read_efermi_from_outcar(filename: str = "OUTCAR") -> Optional[float]:
    """Read the last E-fermi value from OUTCAR, if available."""
    if not os.path.exists(filename):
        return None

    efermi = None
    pattern = re.compile(r"E-fermi\s*:\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)")

    with open(filename, "r", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                efermi = float(match.group(1))

    return efermi


# ----------------------------- k-axis construction ----------------------------- #

def cumulative_kdistance_from_eigenval(
    kpts_frac: np.ndarray,
    recip_lattice: Optional[np.ndarray],
) -> np.ndarray:
    """
    Build the x-axis directly from the actual k-points in EIGENVAL.

    This avoids KLABELS and avoids assumptions about integer division.
    """
    k_cart = frac_to_cart_k(kpts_frac, recip_lattice)
    distances = np.zeros(len(k_cart), dtype=float)

    if len(k_cart) <= 1:
        return distances

    steps = np.linalg.norm(np.diff(k_cart, axis=0), axis=1)
    distances[1:] = np.cumsum(steps)

    return distances


def tick_positions_from_kpoints_path(
    kpath: KPath,
    recip_lattice: Optional[np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """
    Compute tick positions from the path nodes listed in KPOINTS.

    This does not require KLABELS. It uses the high-symmetry endpoints directly.
    """
    if kpath.coord_type == "cartesian":
        nodes_cart = np.array(kpath.nodes_frac, dtype=float)
    else:
        nodes_cart = frac_to_cart_k(kpath.nodes_frac, recip_lattice)

    ticks = np.zeros(len(nodes_cart), dtype=float)
    if len(nodes_cart) > 1:
        ticks[1:] = np.cumsum(np.linalg.norm(np.diff(nodes_cart, axis=0), axis=1))

    labels = [pretty_label(label) for label in kpath.labels]
    return ticks, labels


def align_ticks_to_actual_xaxis(
    ticks_from_path: np.ndarray,
    x_from_eigenval: np.ndarray,
) -> np.ndarray:
    """
    Small robustness helper.

    The KPOINTS path distance and EIGENVAL cumulative distance should agree in scale.
    Tiny differences can appear from repeated endpoints and floating precision. Rescale the
    tick positions to exactly match the final x value from EIGENVAL.
    """
    if len(ticks_from_path) == 0:
        return ticks_from_path

    if ticks_from_path[-1] == 0 or x_from_eigenval[-1] == 0:
        return ticks_from_path

    scale = x_from_eigenval[-1] / ticks_from_path[-1]
    return ticks_from_path * scale


# ----------------------------- plotting ----------------------------- #

def plot_bands(
    kpoints_file: str = "KPOINTS",
    eigenval_file: str = "EIGENVAL",
    poscar_file: str = "POSCAR",
    outcar_file: str = "OUTCAR",
    efermi: Optional[float] = None,
    ylim: Optional[list[float]] = None,
    spin: str = "both",
    output: Optional[str] = None,
    figsize: tuple[float, float] = (8.0, 5.0),
    dpi: int = 300,
    show: bool = True,
) -> None:
    kpath = read_kpoints_line_mode(kpoints_file)
    eig = read_eigenval(eigenval_file)
    recip_lattice = reciprocal_lattice_from_poscar(poscar_file)

    if recip_lattice is None:
        print(
            "WARNING: POSCAR not found. Plotting k-distance in fractional reciprocal-coordinate units, "
            "not physical Å^-1.",
            file=sys.stderr,
        )

    if efermi is None:
        efermi = read_efermi_from_outcar(outcar_file)

    if efermi is None:
        print(
            "WARNING: Could not determine E-fermi from OUTCAR and --efermi was not supplied. "
            "Plotting absolute energies.",
            file=sys.stderr,
        )
        efermi = 0.0
        ylabel = "Energy (eV)"
    else:
        ylabel = r"$E - E_F$ (eV)"

    x = cumulative_kdistance_from_eigenval(eig.kpts_frac, recip_lattice)
    ticks, labels = tick_positions_from_kpoints_path(kpath, recip_lattice)
    ticks = align_ticks_to_actual_xaxis(ticks, x)

    if len(x) != eig.nkpts:
        raise RuntimeError(f"Internal error: len(x)={len(x)} but nkpts={eig.nkpts}")

    expected = kpath.n_segments * kpath.n_per_segment
    if expected != eig.nkpts:
        print(
            "NOTE: KPOINTS suggests "
            f"{kpath.n_segments} segments × {kpath.n_per_segment} points/segment = {expected} points, "
            f"but EIGENVAL contains {eig.nkpts} k-points. "
            "I will trust the actual EIGENVAL k-points for plotting.",
            file=sys.stderr,
        )

    energies = eig.energies - efermi

    plt.figure(figsize=figsize)

    if eig.ispin == 2 and spin in {"both", "up"}:
        for ib in range(eig.nbands):
            plt.plot(x, energies[0, :, ib], lw=0.9, color="tab:blue", label="spin up" if ib == 0 and spin == "both" else None)

    if eig.ispin == 2 and spin in {"both", "down"}:
        for ib in range(eig.nbands):
            plt.plot(x, energies[1, :, ib], lw=0.9, color="tab:orange", ls="--", label="spin down" if ib == 0 and spin == "both" else None)

    if eig.ispin != 2:
        for ib in range(eig.nbands):
            plt.plot(x, energies[0, :, ib], lw=0.9, color="tab:blue")

    for tick in ticks:
        plt.axvline(tick, color="0.75", lw=0.8, zorder=0)

    plt.axhline(0.0, color="k", lw=0.8, ls="--")
    plt.xticks(ticks=ticks, labels=labels)
    plt.xlim(x[0], x[-1])
    plt.ylabel(ylabel)
    plt.xlabel("k-path")

    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    if eig.ispin == 2 and spin == "both":
        plt.legend(frameon=False)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=dpi, bbox_inches="tight")
        print(f"Saved band structure plot to {output}")

    if show:
        plt.show()
    else:
        plt.close()


# ----------------------------- command-line interface ----------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a VASP line-mode band structure using only KPOINTS, EIGENVAL, POSCAR, and optionally OUTCAR."
    )
    parser.add_argument("--kpoints", default="KPOINTS", help="VASP line-mode KPOINTS file. Default: KPOINTS")
    parser.add_argument("--eigenval", default="EIGENVAL", help="VASP EIGENVAL file. Default: EIGENVAL")
    parser.add_argument("--poscar", default="POSCAR", help="VASP POSCAR file. Default: POSCAR")
    parser.add_argument("--outcar", default="OUTCAR", help="VASP OUTCAR file for auto E-fermi. Default: OUTCAR")
    parser.add_argument("--efermi", type=float, default=None, help="Fermi energy in eV. Overrides OUTCAR auto-detection.")
    parser.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("EMIN", "EMAX"), help="Energy window, e.g. --ylim -2 2")
    parser.add_argument("--spin", choices=["up", "down", "both"], default="both", help="For ISPIN=2 EIGENVAL. Default: both")
    parser.add_argument("--out", default=None, help="Save figure to file, e.g. band.png")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive plot window.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI when saving. Default: 300")
    parser.add_argument("--figsize", type=float, nargs=2, default=(8.0, 5.0), metavar=("W", "H"), help="Figure size in inches.")

    args = parser.parse_args()

    plot_bands(
        kpoints_file=args.kpoints,
        eigenval_file=args.eigenval,
        poscar_file=args.poscar,
        outcar_file=args.outcar,
        efermi=args.efermi,
        ylim=args.ylim,
        spin=args.spin,
        output=args.out,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
