"""Volumetric mesh construction and density contouring.

The orbital is always rendered on a **regular structured grid** that is built by
back-mapping the stored values onto the FFT grid ``(Nx, Ny, Nz)`` and then
contouring ``|psi|^2``.  This is exactly what an external cube viewer (VESTA)
does, so the isosurface matches the ``.cube`` ground truth bit-for-bit.

Why a regular grid even for WS-mode data
----------------------------------------
``|CNO(r)|^2`` is lattice-periodic, so the value stored for the WS representative
point ``q`` is also the value at the regular grid point ``base_indices[q]``.  We
therefore invert ``base_indices`` to a lookup table ``G[ix, iy, iz] = q`` and
build a :class:`pyvista.StructuredGrid` whose geometry follows the (possibly
non-orthogonal) lattice.  An irregular Delaunay tetrahedralization of the wrapped
point cloud is **not** used — it produced degenerate tetrahedra and a lumpy
surface.  The Wigner–Seitz cell is still available analytically (see
:mod:`cno_visualizer.ws_geometry`); the "WS view" simply contours the same
regular grid over a small block of cells and clips the surface to that polyhedron.

NumPy vs VTK ordering
---------------------
A :class:`pyvista.StructuredGrid` built from three ``(Ni, Nj, Nk)`` coordinate
arrays stores its points in Fortran order (first index fastest).  Every point
array attached to it must therefore be flattened with ``ravel(order="F")``; the
``point_to_psi_index`` map below is built the same way so a single fancy-index
``psi[point_to_psi_index]`` lands each value on the correct vertex.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from cno_visualizer.data import CNOData
from cno_visualizer.ws_geometry import WSGeometry


@dataclass
class Volume:
    """A structured volumetric mesh plus its sampling layout.

    ``mesh`` is a :class:`pyvista.StructuredGrid`.  ``point_to_psi_index`` maps
    each mesh vertex to a flat index into a ``cno_values`` row, so the geometry is
    built once and only the scalar arrays change when the CNO changes.
    """

    mesh: object
    point_to_psi_index: np.ndarray
    is_ws: bool


# --------------------------------------------------------------- index back-map
def grid_to_flat_index(data: CNOData) -> np.ndarray:
    """Return ``G`` with ``G[ix, iy, iz]`` = the flat ``cno_values`` column index.

    * WS mode — invert ``base_indices`` (``base_indices[q] = (ix, iy, iz)``).
    * Primitive mode — ``cno_values`` is the C-order flattening of ``(Nx, Ny, Nz)``.
    """
    Nx, Ny, Nz = (int(x) for x in data.grid_shape)
    if data.ws_enabled and data.base_indices is not None:
        bi = np.asarray(data.base_indices, dtype=np.int64)
        G = np.full((Nx, Ny, Nz), -1, dtype=np.int64)
        G[np.mod(bi[:, 0], Nx), np.mod(bi[:, 1], Ny), np.mod(bi[:, 2], Nz)] = np.arange(
            bi.shape[0], dtype=np.int64
        )
        if np.any(G < 0):
            missing = int(np.count_nonzero(G < 0))
            raise ValueError(
                f"base_indices do not cover all {Nx * Ny * Nz} FFT grid points "
                f"({missing} missing); cannot back-map to a regular grid."
            )
        return G
    return np.arange(Nx * Ny * Nz, dtype=np.int64).reshape(Nx, Ny, Nz)


# ------------------------------------------------------------------- builders
def _structured_from_index_range(
    data: CNOData,
    i_start: Sequence[int],
    i_stop: Sequence[int],
) -> Volume:
    """Build a structured grid spanning fractional indices ``[i_start, i_stop]``.

    The range is inclusive of ``i_stop`` (one extra layer) so that contours close
    seamlessly across the periodic boundary; the extra layer wraps via modular
    indexing back onto the stored values.
    """
    import pyvista as pv

    Nx, Ny, Nz = (int(x) for x in data.grid_shape)
    G = grid_to_flat_index(data)

    ax_i = np.arange(int(i_start[0]), int(i_stop[0]) + 1)
    ax_j = np.arange(int(i_start[1]), int(i_stop[1]) + 1)
    ax_k = np.arange(int(i_start[2]), int(i_stop[2]) + 1)
    ii, jj, kk = np.meshgrid(ax_i, ax_j, ax_k, indexing="ij")

    frac = np.stack([ii / Nx, jj / Ny, kk / Nz], axis=-1)
    cart = frac @ data.lattice
    grid = pv.StructuredGrid(
        np.ascontiguousarray(cart[..., 0], dtype=np.float64),
        np.ascontiguousarray(cart[..., 1], dtype=np.float64),
        np.ascontiguousarray(cart[..., 2], dtype=np.float64),
    )

    qref = G[np.mod(ii, Nx), np.mod(jj, Ny), np.mod(kk, Nz)]
    p_to_q = qref.ravel(order="F").astype(np.int64)
    return Volume(mesh=grid, point_to_psi_index=p_to_q, is_ws=bool(data.ws_enabled))


def build_crystal_volume(
    data: CNOData,
    replication: Tuple[int, int, int] = (1, 1, 1),
) -> Volume:
    """Regular grid over ``replication`` primitive cells starting at the origin.

    This is the default view: it reproduces the periodic ``|psi|^2`` exactly as a
    cube viewer would and tiles cleanly to a supercell.
    """
    Nx, Ny, Nz = (int(x) for x in data.grid_shape)
    rx, ry, rz = (max(1, int(r)) for r in replication)
    return _structured_from_index_range(data, (0, 0, 0), (rx * Nx, ry * Ny, rz * Nz))


def build_ws_block_volume(data: CNOData, n_cells: int = 1) -> Volume:
    """Regular grid over a ``(2*n_cells+1)^3`` block centred on the WS cell.

    The caller is expected to clip the contoured surface to the WS polyhedron via
    :func:`clip_surface_to_ws`; the block must be large enough to contain it.
    """
    Nx, Ny, Nz = (int(x) for x in data.grid_shape)
    if data.ws_center_frac is not None:
        c = np.floor(np.asarray(data.ws_center_frac, dtype=np.float64)).astype(int)
    else:
        c = np.zeros(3, dtype=int)
    n = max(1, int(n_cells))
    i_start = ((c - n) * np.array([Nx, Ny, Nz])).astype(int)
    i_stop = ((c + n + 1) * np.array([Nx, Ny, Nz])).astype(int)
    return _structured_from_index_range(data, i_start, i_stop)


# Backwards-compatible alias (old name used a single primitive cell).
def build_primitive_volume(data: CNOData) -> Volume:
    return build_crystal_volume(data, (1, 1, 1))


# --------------------------------------------------------------------- updates
def set_active_cno(volume: Volume, data: CNOData, local_index: int) -> float:
    """Attach ``rho``, ``psi_real``, ``psi_imag`` for ``local_index``.

    Returns ``np.max(rho)`` so the caller can derive ``rho_iso = fraction * max_rho``.
    """
    psi = data.cno(local_index)
    psi_mapped = psi[volume.point_to_psi_index]
    rho = np.abs(psi_mapped) ** 2
    volume.mesh.point_data["rho"] = rho.astype(np.float64)
    volume.mesh.point_data["psi_real"] = psi_mapped.real.astype(np.float64)
    volume.mesh.point_data["psi_imag"] = psi_mapped.imag.astype(np.float64)
    volume.mesh.set_active_scalars("rho")
    return float(np.max(rho))


def contour_density(volume: Volume, rho_iso: float):
    """Contour the density at ``rho_iso``; ``psi_real``/``psi_imag`` ride along.

    Returns ``None`` if the contour is empty (e.g. ``rho_iso`` above the field
    maximum); the caller is expected to preserve the previous surface.
    """
    if not np.isfinite(rho_iso) or rho_iso <= 0.0:
        return None
    try:
        surface = volume.mesh.contour(
            isosurfaces=[float(rho_iso)],
            scalars="rho",
            compute_normals=False,
        )
    except Exception as exc:
        warnings.warn(f"Contour failed at rho_iso={rho_iso:.3e}: {exc}")
        return None
    if surface is None or surface.n_points == 0:
        return None
    return surface


def clip_surface_to_ws(surface, ws_geometry: WSGeometry, tol: float = 1e-6):
    """Clip ``surface`` to the Wigner–Seitz polyhedron via its active half-spaces.

    Only the half-spaces that actually touch a polyhedron vertex (the real faces)
    are used, so a handful of plane clips suffice instead of all neighbour
    bisectors.  Returns the clipped surface (possibly empty).
    """
    if surface is None or surface.n_points == 0:
        return surface
    A = np.asarray(ws_geometry.A, dtype=np.float64)
    b = np.asarray(ws_geometry.b, dtype=np.float64)
    V = np.asarray(ws_geometry.vertices, dtype=np.float64)
    # A face plane is "active" if some vertex lies on it.
    margins = A @ V.T + b[:, None]                       # (m, V)
    active = np.flatnonzero(np.min(np.abs(margins), axis=1) < tol)
    clipped = surface
    for i in active:
        n = A[i]
        r0 = -b[i] * n                                   # point on the plane (|n| = 1)
        try:
            # Keep the half where A·r + b <= 0 (the interior, containing the centre).
            clipped = clipped.clip(normal=n, origin=r0, invert=True)
        except Exception:
            continue
        if clipped.n_points == 0:
            break
    return clipped


def color_surface_by_phase(surface, offset: float = 0.0) -> np.ndarray:
    """Compute the wrapped phase on ``surface`` and assign it as active scalars.

    Vertices where ``|psi|`` is numerically zero are masked to the median phase
    value to avoid visually-noisy speckle along nodal lines.
    """
    real = np.asarray(surface.point_data["psi_real"])
    imag = np.asarray(surface.point_data["psi_imag"])
    mag = np.hypot(real, imag)
    phase = np.mod(np.arctan2(imag, real) + float(offset), 2.0 * np.pi)
    if mag.size:
        mask = mag < 1e-10
        if np.any(mask) and not np.all(mask):
            phase[mask] = float(np.median(phase[~mask]))
    surface.point_data["phase"] = phase
    surface.set_active_scalars("phase")
    return phase


__all__ = [
    "Volume",
    "grid_to_flat_index",
    "build_crystal_volume",
    "build_ws_block_volume",
    "build_primitive_volume",
    "set_active_cno",
    "contour_density",
    "clip_surface_to_ws",
    "color_surface_by_phase",
]
