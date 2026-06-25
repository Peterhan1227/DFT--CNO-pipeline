"""Loader / validation tests (spec §18 items 1-7, 16)."""

from __future__ import annotations

import numpy as np
import pytest

from cno_visualizer.data import CNOData, CNODataError


def test_loads_valid_npz(tiny_primitive_npz):
    data = CNOData.from_npz(tiny_primitive_npz)
    assert data.format_version == "cno-visualizer-v1"
    assert data.n_cnos == 2
    assert data.cno_values.dtype.kind == "c"
    assert data.cno_values.shape[1] == int(np.prod(data.grid_shape))
    assert not data.ws_enabled
    assert data.source_path == tiny_primitive_npz


def test_unsupported_version(bad_version_npz):
    with pytest.raises(CNODataError, match="format_version"):
        CNOData.from_npz(bad_version_npz)


def test_missing_required_field(missing_field_npz):
    with pytest.raises(CNODataError, match="Missing required fields"):
        CNOData.from_npz(missing_field_npz)


def test_cno_shape_validation(shape_mismatch_npz):
    with pytest.raises(CNODataError, match="does not match"):
        CNOData.from_npz(shape_mismatch_npz)


def test_singular_lattice(singular_lattice_npz):
    with pytest.raises(CNODataError, match="singular"):
        CNOData.from_npz(singular_lattice_npz)


def test_nan_cno_rejected(nan_cno_npz):
    with pytest.raises(CNODataError, match="NaN"):
        CNOData.from_npz(nan_cno_npz)


def test_lattice_row_convention(tiny_primitive_npz, cubic_lattice):
    data = CNOData.from_npz(tiny_primitive_npz)
    np.testing.assert_allclose(data.lattice, cubic_lattice)
    # r_cart = r_frac @ lattice, so e_x in frac maps to a1 = lattice[0].
    np.testing.assert_allclose(
        np.array([1.0, 0.0, 0.0]) @ data.lattice, cubic_lattice[0]
    )


def test_invalid_cno_index_raises(tiny_primitive_npz):
    data = CNOData.from_npz(tiny_primitive_npz)
    with pytest.raises(CNODataError):
        data.cno(data.n_cnos)
    with pytest.raises(CNODataError):
        data.cno(-1)


def test_density_and_phase_unchanged_by_offset(tiny_primitive_npz):
    data = CNOData.from_npz(tiny_primitive_npz)
    rho0 = data.density(0)
    rho1 = data.density(0)
    np.testing.assert_array_equal(rho0, rho1)
    p0 = data.phase(0, offset=0.0)
    p_shift = data.phase(0, offset=1.234)
    # Same density-derived geometry, but the phase has shifted.
    np.testing.assert_allclose(rho0, np.abs(data.cno(0)) ** 2)
    assert np.allclose(np.mod(p_shift - p0, 2 * np.pi), 1.234) or np.allclose(
        np.mod(p_shift - p0, 2 * np.pi), np.mod(1.234, 2 * np.pi)
    )
