"""Synthetic .npz fixtures — no exporter, no real DFT data needed."""

from __future__ import annotations

import numpy as np
import pytest


def _save(path, **arrays):
    np.savez_compressed(path, **arrays)
    return path


@pytest.fixture
def primitive_grid_shape():
    return (6, 5, 4)


@pytest.fixture
def cubic_lattice():
    # Slightly asymmetric so a transposition would be detectable downstream.
    return np.diag([4.0, 5.0, 6.0])


@pytest.fixture
def tiny_primitive_npz(tmp_path, primitive_grid_shape, cubic_lattice):
    """Two CNOs on a tiny primitive grid; psi has a closed-form analytic form."""
    Nx, Ny, Nz = primitive_grid_shape
    grid_shape = np.asarray(primitive_grid_shape, dtype=np.int32)
    lattice = cubic_lattice.astype(np.float64)

    ii, jj, kk = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij")
    frac = np.stack([ii / Nx, jj / Ny, kk / Nz], axis=-1)
    cart = frac @ lattice
    x, y, z = cart[..., 0], cart[..., 1], cart[..., 2]

    psi0 = np.exp(2j * np.pi * frac[..., 0]) * np.exp(-(x**2 + y**2 + z**2) * 0.05)
    psi1 = np.exp(1j * (x + y + z)) * np.exp(-(x**2 + y**2 + z**2) * 0.1)
    cno_values = np.stack(
        [psi0.ravel(order="C").astype(np.complex128), psi1.ravel(order="C").astype(np.complex128)],
        axis=0,
    )

    path = tmp_path / "Si_cno_visualizer_data.npz"
    return _save(
        path,
        format_version=np.array("cno-visualizer-v1"),
        cno_values=cno_values,
        cno_occupations=np.array([1.95, 1.50], dtype=np.float64),
        cno_indices=np.array([0, 1], dtype=np.int64),
        grid_shape=grid_shape,
        lattice=lattice,
        atom_symbols=np.asarray(["Si", "Si"]),
        atom_numbers=np.asarray([14, 14], dtype=np.int64),
        atoms_frac=np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], dtype=np.float64),
        atoms_cart=np.array([[0.0, 0.0, 0.0], [1.0, 1.25, 1.5]], dtype=np.float64),
        ws_enabled=np.array(False),
    )


@pytest.fixture
def grid_order_npz(tmp_path, primitive_grid_shape, cubic_lattice):
    """A single CNO whose value uniquely identifies the (i, j, k) triple.

    The integer-encoded amplitude makes axis transpositions easy to detect.
    """
    Nx, Ny, Nz = primitive_grid_shape
    grid_shape = np.asarray(primitive_grid_shape, dtype=np.int32)
    ii, jj, kk = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij")
    psi3 = 100.0 * ii + 10.0 * jj + 1.0 * kk
    # Promote to complex; phase is fixed so density == |psi|^2 == (encoded value)^2.
    psi = psi3.astype(np.complex128)
    psi_flat = psi.ravel(order="C")[None, :]  # shape (1, Nr)
    path = tmp_path / "order_data.npz"
    return _save(
        path,
        format_version=np.array("cno-visualizer-v1"),
        cno_values=psi_flat,
        cno_occupations=np.array([1.0], dtype=np.float64),
        cno_indices=np.array([0], dtype=np.int64),
        grid_shape=grid_shape,
        lattice=cubic_lattice.astype(np.float64),
        atom_symbols=np.asarray(["H"]),
        atom_numbers=np.asarray([1], dtype=np.int64),
        atoms_frac=np.zeros((1, 3), dtype=np.float64),
        atoms_cart=np.zeros((1, 3), dtype=np.float64),
        ws_enabled=np.array(False),
    )


@pytest.fixture
def tiny_ws_npz(tmp_path):
    """A WS-mode dataset: cubic lattice, ws points = primitive grid (shifted by an int translation).

    The fixture chooses translations so the integer relation
    ``actual_indices = base_indices + translations * grid_shape`` is non-trivial.
    """
    grid_shape = np.asarray((4, 4, 4), dtype=np.int32)
    lattice = np.diag([4.0, 4.0, 4.0]).astype(np.float64)
    Nx, Ny, Nz = (int(s) for s in grid_shape)
    Nr = Nx * Ny * Nz

    base = np.array(
        [(ix, iy, iz) for ix in range(Nx) for iy in range(Ny) for iz in range(Nz)],
        dtype=np.int64,
    )  # (Nr, 3)
    # Translate the second half of points by (-1, 0, 0) so translations are interesting.
    translations = np.zeros_like(base)
    translations[Nr // 2 :, 0] = -1

    frac = base / grid_shape + translations
    cart = frac @ lattice
    psi = np.exp(2j * np.pi * frac[:, 0]) * np.exp(-np.sum(frac**2, axis=1) * 0.5)
    cno_values = psi.astype(np.complex128)[None, :]

    path = tmp_path / "ws_data.npz"
    return _save(
        path,
        format_version=np.array("cno-visualizer-v1"),
        cno_values=cno_values,
        cno_occupations=np.array([1.0], dtype=np.float64),
        cno_indices=np.array([0], dtype=np.int64),
        grid_shape=grid_shape,
        lattice=lattice,
        atom_symbols=np.asarray(["H"]),
        atom_numbers=np.asarray([1], dtype=np.int64),
        atoms_frac=np.zeros((1, 3), dtype=np.float64),
        atoms_cart=np.zeros((1, 3), dtype=np.float64),
        ws_enabled=np.array(True),
        points_cart=cart.astype(np.float64),
        points_frac_cont=frac.astype(np.float64),
        base_indices=base,
        translations=translations,
        ws_center_cart=np.array([2.0, 2.0, 2.0], dtype=np.float64),
        ws_center_frac=np.array([0.5, 0.5, 0.5], dtype=np.float64),
    )


@pytest.fixture
def bad_version_npz(tmp_path, primitive_grid_shape, cubic_lattice):
    path = tmp_path / "bad_version.npz"
    Nr = int(np.prod(primitive_grid_shape))
    return _save(
        path,
        format_version=np.array("cno-visualizer-v999"),
        cno_values=np.zeros((1, Nr), dtype=np.complex128),
        cno_occupations=np.array([1.0]),
        cno_indices=np.array([0], dtype=np.int64),
        grid_shape=np.asarray(primitive_grid_shape, dtype=np.int32),
        lattice=cubic_lattice.astype(np.float64),
        atom_symbols=np.asarray(["H"]),
        atom_numbers=np.asarray([1], dtype=np.int64),
        atoms_frac=np.zeros((1, 3)),
        atoms_cart=np.zeros((1, 3)),
        ws_enabled=np.array(False),
    )


@pytest.fixture
def missing_field_npz(tmp_path):
    """Save a deliberately incomplete file — missing cno_values."""
    path = tmp_path / "missing.npz"
    return _save(
        path,
        format_version=np.array("cno-visualizer-v1"),
        grid_shape=np.asarray((2, 2, 2), dtype=np.int32),
    )


@pytest.fixture
def singular_lattice_npz(tmp_path, primitive_grid_shape):
    path = tmp_path / "singular.npz"
    Nr = int(np.prod(primitive_grid_shape))
    return _save(
        path,
        format_version=np.array("cno-visualizer-v1"),
        cno_values=np.zeros((1, Nr), dtype=np.complex128),
        cno_occupations=np.array([1.0]),
        cno_indices=np.array([0], dtype=np.int64),
        grid_shape=np.asarray(primitive_grid_shape, dtype=np.int32),
        # rank-deficient lattice (rows linearly dependent)
        lattice=np.array(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        ),
        atom_symbols=np.asarray(["H"]),
        atom_numbers=np.asarray([1], dtype=np.int64),
        atoms_frac=np.zeros((1, 3)),
        atoms_cart=np.zeros((1, 3)),
        ws_enabled=np.array(False),
    )


@pytest.fixture
def nan_cno_npz(tmp_path, primitive_grid_shape, cubic_lattice):
    path = tmp_path / "nan_cno.npz"
    Nr = int(np.prod(primitive_grid_shape))
    cno = np.zeros((1, Nr), dtype=np.complex128)
    cno[0, 0] = np.nan + 0j
    return _save(
        path,
        format_version=np.array("cno-visualizer-v1"),
        cno_values=cno,
        cno_occupations=np.array([1.0]),
        cno_indices=np.array([0], dtype=np.int64),
        grid_shape=np.asarray(primitive_grid_shape, dtype=np.int32),
        lattice=cubic_lattice.astype(np.float64),
        atom_symbols=np.asarray(["H"]),
        atom_numbers=np.asarray([1], dtype=np.int64),
        atoms_frac=np.zeros((1, 3)),
        atoms_cart=np.zeros((1, 3)),
        ws_enabled=np.array(False),
    )


@pytest.fixture
def shape_mismatch_npz(tmp_path, primitive_grid_shape, cubic_lattice):
    """cno_values width does not match prod(grid_shape)."""
    path = tmp_path / "wrong_shape.npz"
    Nr_wrong = int(np.prod(primitive_grid_shape)) + 7
    return _save(
        path,
        format_version=np.array("cno-visualizer-v1"),
        cno_values=np.zeros((1, Nr_wrong), dtype=np.complex128),
        cno_occupations=np.array([1.0]),
        cno_indices=np.array([0], dtype=np.int64),
        grid_shape=np.asarray(primitive_grid_shape, dtype=np.int32),
        lattice=cubic_lattice.astype(np.float64),
        atom_symbols=np.asarray(["H"]),
        atom_numbers=np.asarray([1], dtype=np.int64),
        atoms_frac=np.zeros((1, 3)),
        atoms_cart=np.zeros((1, 3)),
        ws_enabled=np.array(False),
    )
