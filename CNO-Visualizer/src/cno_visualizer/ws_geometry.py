"""Wigner–Seitz cell geometry built independently from the volumetric data.

The displayed WS polyhedron does not depend on the irregular ``points_cart`` —
it is constructed analytically from the lattice and the user-chosen ``center``
via half-space inequalities.  This means the boundary is exact and the same
half-spaces can be reused to clip the Delaunay tetrahedralization in
:mod:`cno_visualizer.field`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection


@dataclass(frozen=True)
class WSGeometry:
    A: np.ndarray             # (m, 3)  — half-space normals
    b: np.ndarray             # (m,)    — offsets, inequality is A·r + b <= 0
    interior: np.ndarray      # (3,)    — strictly-inside feasible point
    vertices: np.ndarray      # (V, 3)  — polyhedron vertices in Cartesian coords
    faces: np.ndarray         # flat VTK face array
    polyhedron: object        # pyvista.PolyData; lazily imported to keep tests light

    @property
    def volume(self) -> float:
        return float(ConvexHull(self.vertices).volume)


def ws_halfspaces(
    lattice: np.ndarray,
    center: np.ndarray,
    n_max: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(A, b)`` such that the WS cell at ``center`` is ``A @ r + b <= 0``.

    For every nonzero lattice translation ``R = n @ lattice`` (``n`` integer, ``|n_i| <= n_max``)
    the bisecting half-plane between ``center`` and ``center + R`` is

        R · (r - center) <= |R|² / 2.

    Re-arranged into the canonical ``A·r + b <= 0`` form, this gives

        A = R / |R|,    b = -R·center/|R| - |R|/2.
    """
    lattice = np.asarray(lattice, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    if lattice.shape != (3, 3):
        raise ValueError(f"lattice must be (3, 3), got {lattice.shape}")
    if n_max < 1:
        raise ValueError(f"n_max must be >= 1, got {n_max}")

    rng = np.arange(-n_max, n_max + 1)
    n1, n2, n3 = np.meshgrid(rng, rng, rng, indexing="ij")
    n = np.stack([n1.ravel(), n2.ravel(), n3.ravel()], axis=1)
    nonzero = np.any(n != 0, axis=1)
    n = n[nonzero]
    R = n.astype(np.float64) @ lattice            # (m, 3)
    R_norm = np.linalg.norm(R, axis=1)            # (m,)
    A = R / R_norm[:, None]
    b = -(A @ center) - 0.5 * R_norm
    return A, b


def _build_polyhedron(A: np.ndarray, b: np.ndarray, interior: np.ndarray):
    """Return ``(vertices, faces, pyvista.PolyData)`` for the WS polyhedron."""
    import pyvista as pv

    halfspaces = np.hstack([A, b[:, None]])
    hsi = HalfspaceIntersection(halfspaces, interior)
    vertices = np.unique(np.round(hsi.intersections, decimals=9), axis=0)
    hull = ConvexHull(vertices)
    faces_flat = []
    for simplex in hull.simplices:
        faces_flat.extend([3, int(simplex[0]), int(simplex[1]), int(simplex[2])])
    faces = np.asarray(faces_flat, dtype=np.int64)
    poly = pv.PolyData(vertices, faces).clean()
    return vertices, faces, poly


def ws_polyhedron(
    lattice: np.ndarray,
    center: np.ndarray,
    n_max: int = 3,
) -> WSGeometry:
    """Build the WS polyhedron and return the full :class:`WSGeometry` bundle."""
    A, b = ws_halfspaces(lattice, center, n_max=n_max)
    interior = np.asarray(center, dtype=np.float64).reshape(3)
    # Sanity: the chosen center must be strictly inside every half-space.
    margins = A @ interior + b
    if not np.all(margins < -1e-9):
        worst = float(np.max(margins))
        raise ValueError(
            f"WS center is not strictly inside its own bisectors (worst margin {worst:.3e})."
        )
    vertices, faces, poly = _build_polyhedron(A, b, interior)
    return WSGeometry(
        A=A,
        b=b,
        interior=interior,
        vertices=vertices,
        faces=faces,
        polyhedron=poly,
    )


def points_inside_ws(
    geometry: WSGeometry,
    points: np.ndarray,
    tol: float = 1e-9,
) -> np.ndarray:
    """Return a boolean mask of points satisfying all WS half-space inequalities."""
    points = np.asarray(points, dtype=np.float64)
    return np.all((geometry.A @ points.T + geometry.b[:, None]) <= tol, axis=0)


__all__ = ["WSGeometry", "ws_halfspaces", "ws_polyhedron", "points_inside_ws"]
