"""Finite-volume WS data must remain on its saved unwrapped FFT mesh."""

from __future__ import annotations

import numpy as np
import pytest

from cno_visualizer.data import CNOData
from cno_visualizer.field import build_expanded_ws_volume, grid_to_flat_index, set_active_cno


def _expanded_ws_data() -> CNOData:
    # A 3x3x3 unwrapped mesh represented with a 2x2x2 periodic base grid.
    # It deliberately has 27 saved samples, greater than prod(grid_shape)=8.
    actual = np.stack(
        np.meshgrid(np.arange(3), np.arange(3), np.arange(3), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    grid_shape = np.array([2, 2, 2])
    base = actual % grid_shape[None, :]
    translations = actual // grid_shape[None, :]
    frac = actual / grid_shape[None, :]
    values = np.exp(-np.sum((frac - 0.5) ** 2, axis=1))[None, :].astype(np.complex128)
    return CNOData.from_ws_arrays(
        values, grid_shape, np.diag([2.0, 2.0, 2.0]),
        base_indices=base, translations=translations,
        ws_center_cart=np.array([1.0, 1.0, 1.0]),
        points_frac_cont=frac,
        atom_symbols=["H"], atoms_cart=np.zeros((1, 3)),
    )


def test_expanded_ws_keeps_all_unwrapped_samples():
    data = _expanded_ws_data()
    assert data.expanded_ws
    actual = data.base_indices + data.translations * data.grid_shape[None, :]
    assert len(np.unique(actual, axis=0)) == data.n_points == 27
    with pytest.raises(ValueError, match="cannot be folded"):
        grid_to_flat_index(data)


def test_expanded_ws_hexahedra_use_only_saved_vertices():
    pytest.importorskip("pyvista")
    data = _expanded_ws_data()
    volume = build_expanded_ws_volume(data)
    # A 3x3x3 vertex block has exactly 2x2x2 complete hexahedra.
    assert volume.mesh.n_points == 27
    assert volume.mesh.n_cells == 8
    rho_max = set_active_cno(volume, data, 0)
    assert np.isclose(rho_max, np.max(np.abs(data.cno(0)) ** 2))
