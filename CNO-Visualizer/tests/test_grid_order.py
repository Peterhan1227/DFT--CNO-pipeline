"""Primitive-grid axis ordering test (spec §18 item 8).

The synthetic field ``psi[i, j, k] = 100*i + 10*j + k`` makes axis transposition
detectable: probing at three asymmetric (i, j, k) points must yield the encoded
amplitude.  Failure here means C/F ravel order was confused.
"""

from __future__ import annotations

import numpy as np

from cno_visualizer.data import CNOData
from cno_visualizer.field import build_primitive_volume


def test_primitive_volume_axis_order(grid_order_npz, primitive_grid_shape):
    data = CNOData.from_npz(grid_order_npz)
    volume = build_primitive_volume(data)
    grid = volume.mesh

    # Attach scalars manually (mirrors what set_active_cno does).
    Nx, Ny, Nz = primitive_grid_shape
    psi = data.cno(0)
    psi_mapped = psi[volume.point_to_psi_index]
    grid.point_data["rho"] = np.real(psi_mapped) ** 2

    # Probe three asymmetric (i, j, k) corners and verify the encoded amplitude.
    points = np.asarray(grid.points)
    fractions = points @ np.linalg.inv(data.lattice)
    ii = np.round(fractions[:, 0] * Nx).astype(int) % Nx
    jj = np.round(fractions[:, 1] * Ny).astype(int) % Ny
    kk = np.round(fractions[:, 2] * Nz).astype(int) % Nz
    expected = (100 * ii + 10 * jj + kk).astype(np.float64)

    np.testing.assert_allclose(
        np.real(psi_mapped),
        expected,
        atol=1e-9,
        err_msg="StructuredGrid scalars are not in Fortran-order; check the ravel.",
    )

    # Sanity probes at asymmetric NumPy indices.  The structured grid carries a
    # one-layer periodic wrap (dims = N + 1), so the VTK Fortran-order stride uses
    # the actual grid dimensions, not (Nx, Ny, Nz).
    Dx, Dy, Dz = grid.dimensions  # (Nx + 1, Ny + 1, Nz + 1)
    for target in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (3, 2, 1)]:
        ti, tj, tk = target
        p = ti + tj * Dx + tk * Dx * Dy
        encoded = 100 * ti + 10 * tj + tk
        assert np.isclose(np.real(psi_mapped[p]), encoded), (
            f"Mismatch at (i,j,k)={target}: got {psi_mapped[p].real}, expected {encoded}"
        )
