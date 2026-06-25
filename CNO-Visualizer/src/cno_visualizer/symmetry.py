"""Symmetry scaffolding for later animation work.

Nothing in this module is invoked from the interactive viewer today; it exists so
that follow-on work can plug in symmetry-operation animations and overlap
calculations without changing the viewer's public surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class SymmetryOperation:
    """A crystallographic-style symmetry operation in fractional coordinates.

    The application rule is

        r_new = (r_frac - center_frac) @ rotation_frac.T + center_frac + translation_frac
    """

    label: str
    rotation_frac: np.ndarray   # (3, 3)
    translation_frac: np.ndarray  # (3,)
    center_frac: np.ndarray     # (3,)

    def apply_frac(self, r_frac: np.ndarray) -> np.ndarray:
        r = np.asarray(r_frac, dtype=np.float64)
        return (r - self.center_frac) @ self.rotation_frac.T + self.center_frac + self.translation_frac

    def apply_cart(self, r_cart: np.ndarray, lattice: np.ndarray) -> np.ndarray:
        latvec = np.asarray(lattice, dtype=np.float64)
        invL = np.linalg.inv(latvec)
        r_frac = np.asarray(r_cart, dtype=np.float64) @ invL
        return self.apply_frac(r_frac) @ latvec

    def apply_mesh(self, mesh, lattice: np.ndarray):
        """Return a transformed copy of ``mesh`` (PyVista ``PolyData`` etc.)."""
        new = mesh.copy(deep=True)
        new.points = self.apply_cart(np.asarray(new.points), lattice)
        return new


def _identity() -> SymmetryOperation:
    return SymmetryOperation(
        label="identity",
        rotation_frac=np.eye(3),
        translation_frac=np.zeros(3),
        center_frac=np.zeros(3),
    )


def _inversion(center_frac=(0.0, 0.0, 0.0)) -> SymmetryOperation:
    return SymmetryOperation(
        label="inversion",
        rotation_frac=-np.eye(3),
        translation_frac=np.zeros(3),
        center_frac=np.asarray(center_frac, dtype=np.float64),
    )


def _swap_xy(center_frac=(0.0, 0.0, 0.0)) -> SymmetryOperation:
    rot = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return SymmetryOperation(
        label="swap_xy",
        rotation_frac=rot,
        translation_frac=np.zeros(3),
        center_frac=np.asarray(center_frac, dtype=np.float64),
    )


def _c3_111(center_frac=(0.0, 0.0, 0.0)) -> SymmetryOperation:
    """Cyclic 3-fold permutation around the [111] axis: (x, y, z) -> (z, x, y)."""
    rot = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return SymmetryOperation(
        label="c3_111",
        rotation_frac=rot,
        translation_frac=np.zeros(3),
        center_frac=np.asarray(center_frac, dtype=np.float64),
    )


def built_in_operations() -> Dict[str, SymmetryOperation]:
    """Return demo symmetry operations.

    These are illustrative.  They are *not* guaranteed to be physical symmetries
    of any particular loaded material — animation work should use space-group
    information from elsewhere when correctness is required.
    """
    return {
        "identity": _identity(),
        "inversion": _inversion(),
        "swap_xy": _swap_xy(),
        "c3_111": _c3_111(),
    }


def compute_symmetry_overlap(*args, **kwargs):
    """Placeholder for the overlap integral ``<psi | T psi>``.

    Computing a *physical* overlap requires the full complex field, not just the
    visible isosurface; the original and transformed fields must be evaluated on
    a common grid; interpolation and gauge conventions must be handled
    explicitly.  Approximating the overlap from mesh vertices alone is wrong.

    Implementing this correctly is deferred — call sites must surface the
    :class:`NotImplementedError` rather than silently using a fake result.
    """
    raise NotImplementedError(
        "compute_symmetry_overlap requires evaluating <psi | T psi> on a common "
        "field grid with explicit interpolation and gauge handling; mesh-vertex "
        "approximations are not physically meaningful."
    )


__all__ = [
    "SymmetryOperation",
    "built_in_operations",
    "compute_symmetry_overlap",
]
