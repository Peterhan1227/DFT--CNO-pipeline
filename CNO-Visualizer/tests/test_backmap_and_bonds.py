"""Regular-grid back-mapping, periodic bonds, and WS clipping (post-fix behavior)."""

from __future__ import annotations

import numpy as np
import pytest

from cno_visualizer.crystal import build_bonded_crystal, find_bonds
from cno_visualizer.data import CNOData
from cno_visualizer.field import (
    build_crystal_volume,
    build_ws_block_volume,
    clip_surface_to_ws,
    contour_density,
    grid_to_flat_index,
    set_active_cno,
)
from cno_visualizer.ws_geometry import points_inside_ws, ws_polyhedron


def test_grid_to_flat_index_inverts_base_indices(tiny_ws_npz):
    """WS back-map must reproduce |psi|^2 placed at base_indices (the cube relation)."""
    data = CNOData.from_npz(tiny_ws_npz)
    Nx, Ny, Nz = (int(x) for x in data.grid_shape)
    psi = data.cno(0)

    # Reference: exactly what the exporter does to write its .cube.
    ref = np.zeros((Nx, Ny, Nz))
    bi = data.base_indices
    ref[bi[:, 0], bi[:, 1], bi[:, 2]] = np.abs(psi) ** 2

    G = grid_to_flat_index(data)
    via_lookup = np.abs(psi[G]) ** 2
    np.testing.assert_allclose(via_lookup, ref, atol=1e-12)


def test_crystal_volume_periodic_wrap(tiny_ws_npz):
    """The +1 wrap layer must carry the same values as the opposite face."""
    pytest.importorskip("pyvista")
    data = CNOData.from_npz(tiny_ws_npz)
    Nx, Ny, Nz = (int(x) for x in data.grid_shape)
    volume = build_crystal_volume(data, (1, 1, 1))
    # point_to_psi_index = qref.ravel(order="F") with qref shape (Nx+1, Ny+1, Nz+1).
    p2q = volume.point_to_psi_index.reshape((Nx + 1, Ny + 1, Nz + 1), order="F")
    # The wrap layer (i = Nx) must map to the same values as the first face (i = 0).
    np.testing.assert_array_equal(p2q[0, :, :], p2q[Nx, :, :])
    np.testing.assert_array_equal(p2q[:, 0, :], p2q[:, Ny, :])
    np.testing.assert_array_equal(p2q[:, :, 0], p2q[:, :, Nz])


def test_crystal_volume_density_matches_backmap(tiny_ws_npz):
    pytest.importorskip("pyvista")
    data = CNOData.from_npz(tiny_ws_npz)
    volume = build_crystal_volume(data, (1, 1, 1))
    rho_max = set_active_cno(volume, data, 0)
    assert np.isclose(rho_max, float(np.max(np.abs(data.cno(0)) ** 2)))


def test_periodic_bonds_connect_across_cells():
    """A single atom whose only neighbours are periodic images must still bond."""
    lattice = np.diag([1.4, 1.4, 1.4])
    symbols = ["C"]
    atoms_cart = np.zeros((1, 3))

    # Naive single-cell search finds nothing (one isolated atom).
    assert find_bonds(atoms_cart, symbols, bond_scale=1.15) == []

    draw_pos, draw_sym, bonds = build_bonded_crystal(
        symbols, atoms_cart, lattice, (1, 1, 1), bond_scale=1.15
    )
    # Simple-cubic: the central atom bonds to its 6 axial images.
    assert len(bonds) == 6
    assert draw_pos.shape[0] == 7  # central + 6 neighbours drawn (no dangling sticks)
    # Every drawn bond references a valid drawn-atom index.
    for i, j in bonds:
        assert 0 <= i < draw_pos.shape[0]
        assert 0 <= j < draw_pos.shape[0]


def test_clip_surface_to_ws_keeps_only_interior(tiny_ws_npz):
    pytest.importorskip("pyvista")
    data = CNOData.from_npz(tiny_ws_npz)
    geom = ws_polyhedron(data.lattice, data.ws_center_cart)
    volume = build_ws_block_volume(data, n_cells=1)
    rho_max = set_active_cno(volume, data, 0)
    # This fixture's density only spans ~[0.25, 1.0]*max, so contour at 0.5*max.
    surface = contour_density(volume, 0.5 * rho_max)
    if surface is None:
        pytest.skip("contour empty for this fixture")
    clipped = clip_surface_to_ws(surface, geom)
    if clipped.n_points == 0:
        pytest.skip("clip removed everything for this fixture")
    inside = points_inside_ws(geom, np.asarray(clipped.points), tol=1e-6)
    assert np.all(inside), "clipped surface escapes the WS cell"
