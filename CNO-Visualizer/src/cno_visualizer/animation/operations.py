"""Rotation symmetry operations about a basis center, as 4x4 transforms.

Rotations are specified by an axis (a Cartesian vector, a crystallographic
direction ``[uvw]`` resolved through the lattice, or a named axis ``x/y/z``) and
an angle (directly, or as an ``n``-fold order → ``360/n``).  The 4x4 matrix
applies the rotation *about the center* so a site-symmetry operation keeps the
center fixed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

AxisLike = Union[str, Sequence[float], np.ndarray]


def _parse_miller(s: str) -> list:
    """Parse a crystallographic direction into 3 indices.

    Accepts spaced/comma forms ('1 1 1', '1,-1,0') and the compact concatenated
    form ('111', '001', '1-10' where a leading '-' negates the next digit).
    """
    s = s.strip().strip("[]()")
    if (" " in s) or ("," in s):
        toks = [t for t in re.split(r"[ ,]+", s) if t]
    else:
        toks, i, sign = [], 0, 1
        while i < len(s):
            ch = s[i]
            if ch in "+-":
                sign = -1 if ch == "-" else 1
                i += 1
                continue
            if ch.isdigit():
                toks.append(str(sign * int(ch)))
                sign = 1
                i += 1
            else:
                raise ValueError(f"cannot parse axis direction {s!r}")
    if len(toks) != 3:
        raise ValueError(f"axis direction must have 3 indices, got {s!r}")
    return [float(t) for t in toks]


@dataclass(frozen=True)
class RotationOp:
    label: str
    axis: np.ndarray       # unit Cartesian vector through the center
    angle_deg: float
    center: np.ndarray     # Cartesian


def axis_to_cartesian(axis: AxisLike, lattice: Optional[np.ndarray]) -> np.ndarray:
    """Resolve an axis spec to a unit Cartesian vector.

    Strings: ``x``/``y``/``z`` (Cartesian axes) or a crystallographic direction
    like ``111``, ``[1 1 1]``, ``1,-1,0`` resolved as ``[uvw] @ lattice`` (so the
    axis follows the real lattice, which matters for non-orthogonal cells).
    """
    if isinstance(axis, str):
        s = axis.strip().lower()
        named = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
        if s in named:
            v = np.asarray(named[s], dtype=np.float64)
        else:
            uvw = np.asarray(_parse_miller(s), dtype=np.float64)
            v = uvw if lattice is None else uvw @ np.asarray(lattice, dtype=np.float64)
    else:
        v = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        raise ValueError("axis vector has zero length")
    return v / norm


def make_rotation(
    center: Sequence[float],
    axis: AxisLike = "z",
    *,
    n_fold: Optional[int] = None,
    angle_deg: Optional[float] = None,
    lattice: Optional[np.ndarray] = None,
    label: Optional[str] = None,
) -> RotationOp:
    """Build a :class:`RotationOp`.  Give either ``n_fold`` or ``angle_deg``."""
    if n_fold is None and angle_deg is None:
        raise ValueError("specify either n_fold or angle_deg")
    if n_fold is not None:
        if int(n_fold) < 1:
            raise ValueError("n_fold must be >= 1")
        angle = 360.0 / float(n_fold)
    else:
        angle = float(angle_deg)
    unit = axis_to_cartesian(axis, lattice)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    if label is None:
        axis_str = axis if isinstance(axis, str) else "axis"
        label = (f"C{int(n_fold)} rotation about [{axis_str}]" if n_fold is not None
                 else f"{angle:.0f} deg about [{axis_str}]")
    return RotationOp(label=label, axis=unit, angle_deg=angle, center=center)


def rotation_matrix_4x4(axis_unit: np.ndarray, angle_rad: float, center: np.ndarray) -> np.ndarray:
    """4x4 homogeneous matrix: rotate by ``angle_rad`` about ``axis_unit`` through ``center``."""
    ax = np.asarray(axis_unit, dtype=np.float64).reshape(3)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    x, y, z = ax
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    R = np.eye(3) + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = c - R @ c
    return M


__all__ = ["RotationOp", "axis_to_cartesian", "make_rotation", "rotation_matrix_4x4"]
