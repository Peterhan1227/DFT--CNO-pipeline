"""Rotation symmetry operations about a basis center, as 4x4 transforms.

Rotations are specified by an axis (a Cartesian vector, a crystallographic
direction ``[uvw]`` resolved through the lattice, or a named axis ``x/y/z``) and
an angle (directly, or as an ``n``-fold order → ``360/n``).  The 4x4 matrix
applies the rotation *about the center* so a site-symmetry operation keeps the
center fixed.
"""

from __future__ import annotations

import math
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


@dataclass(frozen=True)
class ScrewOp:
    """A screw rotation: a proper rotation about ``axis`` (through ``center``) plus an
    intrinsic translation *along that same axis*.  Because the translation is parallel
    to the rotation axis, rotation and translation commute, so applying the operation
    ``t`` times (``t`` need not be an integer, for smooth animation) is simply "rotate
    by ``t * angle_deg`` and translate by ``t * translation``" about the fixed ``center``
    — no compounding/step-accumulation error.

    ``n_fold`` applications return the lattice to a pure translation along the axis
    (a symmetry of the infinite crystal); e.g. a 4_1 screw axis needs 4 applications.
    """

    label: str
    axis: np.ndarray          # unit Cartesian vector, the screw axis direction
    angle_deg: float          # rotation per application
    translation: np.ndarray  # Cartesian translation per application (parallel to axis)
    center: np.ndarray       # any point on the screw axis (Cartesian)


@dataclass(frozen=True)
class GlideOp:
    """A glide reflection: a mirror across the plane through ``center`` with normal
    ``normal``, plus an intrinsic in-plane translation.

    A reflection has determinant -1 and is not continuously connected to the identity
    through rigid rotations, so it is animated as a "flip through the plane": the
    component of every point along ``normal`` is scaled by ``cos(pi * t)`` (``+1`` at
    ``t=0`` -> ``0`` at ``t=0.5`` (flattened into the plane) -> ``-1`` at ``t=1`` (the
    exact mirror image)), while the in-plane glide translation grows linearly with
    ``t``.  Two applications (``t: 0 -> 2``) return the lattice to a pure in-plane
    translation (a symmetry of the infinite crystal).
    """

    label: str
    normal: np.ndarray        # unit Cartesian vector, mirror-plane normal
    translation: np.ndarray  # Cartesian in-plane glide vector per application
    center: np.ndarray        # any point on the mirror plane (Cartesian)


def screw_matrix_4x4(
    axis_unit: np.ndarray, angle_rad: float, translation: np.ndarray, center: np.ndarray
) -> np.ndarray:
    """4x4 matrix: rotate by ``angle_rad`` about ``axis_unit`` through ``center``, then
    add ``translation`` (parallel to the axis, so order does not matter)."""
    M = rotation_matrix_4x4(axis_unit, angle_rad, center)
    M[:3, 3] += np.asarray(translation, dtype=np.float64).reshape(3)
    return M


def glide_matrix_4x4(
    normal_unit: np.ndarray, t: float, translation: np.ndarray, center: np.ndarray
) -> np.ndarray:
    """4x4 matrix for the glide "flip" at progress ``t`` (see :class:`GlideOp`)."""
    n = np.asarray(normal_unit, dtype=np.float64).reshape(3)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    scale = math.cos(math.pi * float(t))
    R = np.eye(3, dtype=np.float64) - (1.0 - scale) * np.outer(n, n)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = c - R @ c + np.asarray(translation, dtype=np.float64).reshape(3) * float(t)
    return M


__all__ = [
    "RotationOp",
    "ScrewOp",
    "GlideOp",
    "axis_to_cartesian",
    "make_rotation",
    "rotation_matrix_4x4",
    "screw_matrix_4x4",
    "glide_matrix_4x4",
]
