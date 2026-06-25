"""Periodicity and integration tests (spec §18 items 15, 17)."""

from __future__ import annotations

import numpy as np
import pytest

from cno_visualizer.data import CNOData
from cno_visualizer.field import (
    build_primitive_volume,
    contour_density,
    color_surface_by_phase,
    set_active_cno,
)


def test_ws_integer_relation(tiny_ws_npz):
    data = CNOData.from_npz(tiny_ws_npz)
    actual_indices = data.base_indices + data.translations * data.grid_shape

    # Re-derive Cartesian coords from those integer indices and check they
    # match points_cart up to floating noise (which proves the relation).
    frac = actual_indices.astype(np.float64) / data.grid_shape
    expected_cart = frac @ data.lattice
    np.testing.assert_allclose(expected_cart, data.points_cart, atol=1e-9)


def test_contour_empty_when_iso_too_high(grid_order_npz):
    """An impossibly high isovalue must yield ``None`` (not crash) so the viewer
    can preserve the last good surface."""
    data = CNOData.from_npz(grid_order_npz)
    volume = build_primitive_volume(data)
    rho_max = set_active_cno(volume, data, 0)
    huge_iso = rho_max * 1e6
    result = contour_density(volume, huge_iso)
    assert result is None


def test_contour_carries_psi_real_and_imag(tiny_primitive_npz):
    """`contour_density` must pass psi_real / psi_imag through to the surface,
    otherwise phase coloring cannot be derived from interpolated complex parts.
    """
    pytest.importorskip("pyvista")
    data = CNOData.from_npz(tiny_primitive_npz)
    volume = build_primitive_volume(data)
    rho_max = set_active_cno(volume, data, 0)
    surface = contour_density(volume, 0.05 * rho_max)
    if surface is None:
        pytest.skip("contour was empty for this fixture; not informative")
    assert "psi_real" in surface.point_data
    assert "psi_imag" in surface.point_data
    phase = color_surface_by_phase(surface, offset=0.0)
    assert phase.shape == (surface.n_points,)
    assert np.all(phase >= 0.0)
    assert np.all(phase < 2 * np.pi + 1e-9)
