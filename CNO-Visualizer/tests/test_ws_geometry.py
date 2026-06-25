"""Wigner-Seitz cell tests (spec §18 items 12-14)."""

from __future__ import annotations

import numpy as np
import pytest

from cno_visualizer.ws_geometry import ws_halfspaces, ws_polyhedron


def test_center_strictly_inside_inequalities():
    lattice = np.diag([3.0, 4.0, 5.0])
    center = np.array([0.6, -0.4, 0.1])
    A, b = ws_halfspaces(lattice, center)
    margins = A @ center + b
    assert np.all(margins < -1e-9), "WS center is on or outside a bisector"


def test_vertices_satisfy_all_inequalities():
    lattice = np.diag([3.0, 4.0, 5.0])
    center = np.zeros(3)
    geom = ws_polyhedron(lattice, center)
    margins = (geom.A @ geom.vertices.T + geom.b[:, None])
    assert margins.max() < 1e-6, f"WS vertex violates a half-space (max {margins.max():.3e})"


def test_volume_matches_primitive_cell_volume():
    lattice = np.array(
        [
            [3.5, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 5.5],
        ]
    )
    center = np.zeros(3)
    geom = ws_polyhedron(lattice, center)
    expected = abs(np.linalg.det(lattice))
    np.testing.assert_allclose(geom.volume, expected, rtol=1e-3)


def test_translation_invariance():
    """Translating the center by a lattice vector translates the polyhedron rigidly."""
    lattice = np.diag([3.0, 3.0, 3.0])
    geom_origin = ws_polyhedron(lattice, np.zeros(3))
    shifted_center = lattice[0]  # exact lattice vector
    geom_shifted = ws_polyhedron(lattice, shifted_center)

    # Vertex sets are translated by the same lattice vector — sort to match.
    v0 = np.sort(geom_origin.vertices, axis=0)
    v1 = np.sort(geom_shifted.vertices - shifted_center, axis=0)
    np.testing.assert_allclose(v1, v0, atol=1e-9)
    np.testing.assert_allclose(geom_shifted.volume, geom_origin.volume, rtol=1e-6)


def test_n_max_too_small_raises_or_returns_inflated_polyhedron():
    """With ``n_max=1`` the diagonal neighbors are still included; with n_max < 1 we reject."""
    lattice = np.eye(3)
    with pytest.raises(ValueError):
        ws_halfspaces(lattice, np.zeros(3), n_max=0)
