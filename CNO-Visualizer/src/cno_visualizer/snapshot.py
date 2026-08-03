"""VESTA-like density-isosurface snapshots and gentle rock GIFs (no phase).

Same rendering technique as the symmetry animation (PyVista/VTK + imageio): dark
gradient background, a full xyz coordinate frame, atoms + bonds, and a single
solid translucent isosurface (phase ignored, imitating VESTA).  No operation is
applied and the lattice is **not** rotated — the structure and its coordinate axes
stay rigid together while the camera *rocks* slowly back and forth, giving a 3-D
parallax view that loops seamlessly.

Reuses the regular-grid volume builder, the crystal helpers, and the animation
scene helpers; the viewer (scene/controls/state) is never imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from cno_visualizer.animation.builder import apply_render_quality, build_coordinate_frame
from cno_visualizer.crystal import (
    build_atom_glyphs,
    build_bond_glyphs,
    build_bonded_crystal,
    build_local_bonded_crystal,
)
from cno_visualizer.data import CNOData
from cno_visualizer.field import (
    build_crystal_volume,
    build_ws_block_volume,
    clip_surface_to_ws,
    contour_density,
    set_active_cno,
)
from cno_visualizer.ws_geometry import ws_polyhedron


@dataclass(frozen=True)
class RegionalCNOMap:
    """Visualization-only description of an explicit finite-volume WS map.

    This deliberately contains no projector, PAW, or DFT logic.  It is a small
    transport object so upstream callers can ask the visualizer to render the
    saved CNO rows without knowing how the mesh is constructed.
    """

    grid_shape: Sequence[int]
    base_indices: np.ndarray
    translations: np.ndarray
    ws_center_cart: np.ndarray
    points_frac_cont: np.ndarray | None = None
    points_cart: np.ndarray | None = None

    @classmethod
    def from_quadrature(cls, quadrature, ws_center_cart) -> "RegionalCNOMap":
        """Adapt any saved-map object exposing the documented map attributes."""
        return cls(
            grid_shape=tuple(int(v) for v in quadrature.sample_grid_shape),
            base_indices=np.asarray(quadrature.base_indices, dtype=np.int64),
            translations=np.asarray(quadrature.translations, dtype=np.int64),
            ws_center_cart=np.asarray(ws_center_cart, dtype=np.float64),
            points_frac_cont=np.asarray(quadrature.points_frac_cont, dtype=np.float64),
            points_cart=(None if quadrature.points_cart is None
                         else np.asarray(quadrature.points_cart, dtype=np.float64)),
        )


def render_cno_gif(
    data: CNOData,
    cno_index: int = 0,
    output: str = "cno_structure.gif",
    *,
    iso_fraction: float = 0.5,
    replication: Tuple[int, int, int] = (2, 2, 2),
    surface_color: Tuple[float, float, float] = (0.25, 0.55, 0.95),
    surface_opacity: float = 1.0,
    background: str = "dark",
    show_atoms: bool = True,
    show_bonds: bool = True,
    show_axes: bool = True,
    show_ws: bool = True,
    context_radius: float = 3.1,
    seconds: float = 8.0,
    fps: int = 15,
    deg_per_sec: float = 12.0,
    camera_pull: float = 4.5,
    spin_axis="x",
    window_size: Sequence[int] = (640, 640),
) -> str:
    """Render a CNO density isosurface as a slow camera-turn GIF/MP4 (or still PNG).

    The structure (isosurface + atoms + bonds) and its xyz coordinate frame are
    static; the camera turns about ``spin_axis`` (``"x"``/``"y"``/``"z"`` or a
    3-vector, kept vertical on screen) at a **constant** ``deg_per_sec`` in one
    direction for ``seconds`` (so the clip simply stops partway — no need to complete
    a full revolution).  ``iso_fraction`` is the level as a fraction of
    ``max(|psi|^2)``; ``camera_pull`` sets the viewing distance (larger = further).
    """
    import pyvista as pv

    volume = (build_ws_block_volume(data, n_cells=1) if data.expanded_ws
              else build_crystal_volume(data, replication))
    rho_max = set_active_cno(volume, data, int(cno_index))
    surface = contour_density(volume, float(iso_fraction) * rho_max)
    geometry = None
    if data.expanded_ws:
        assert data.ws_center_cart is not None
        geometry = ws_polyhedron(data.lattice, data.ws_center_cart)
        surface = clip_surface_to_ws(surface, geometry)

    pl = pv.Plotter(off_screen=True, window_size=tuple(window_size))
    apply_render_quality(pl, background)

    if data.expanded_ws:
        draw_pos, draw_sym, bonds = build_local_bonded_crystal(
            data.atom_symbols, data.atoms_cart, data.lattice, data.ws_center_cart,
            radius=context_radius,
        )
        atom_scale = 0.22
    else:
        draw_pos, draw_sym, bonds = build_bonded_crystal(
            data.atom_symbols, data.atoms_cart, data.lattice, replication
        )
        atom_scale = 0.40
    # Center + scale the view on the orbital region (the isosurface).
    if surface is not None and surface.n_points:
        b = np.asarray(surface.bounds, dtype=np.float64)
        center = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
        extent = 0.5 * float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]])) + 1.0
    elif len(draw_pos):
        center = draw_pos.mean(axis=0)
        extent = float(np.max(np.linalg.norm(draw_pos - center, axis=1))) + 1.5
    else:
        center, extent = np.zeros(3), 5.0

    if surface is not None and surface.n_points:
        pl.add_mesh(
            surface, color=surface_color, opacity=float(surface_opacity),
            smooth_shading=True, specular=0.3, specular_power=15,
            ambient=0.3, diffuse=0.8, show_scalar_bar=False, name="iso",
        )
    if geometry is not None and show_ws:
        pl.add_mesh(
            geometry.polyhedron, color=(0.25, 0.90, 0.95), style="wireframe",
            line_width=1.5, opacity=0.42, name="wigner_seitz_boundary",
        )
    if show_atoms and len(draw_pos):
        atom_mesh, atom_rgb = build_atom_glyphs(draw_pos, draw_sym, radius_scale=atom_scale)
        if atom_mesh.n_points:
            pl.add_mesh(atom_mesh, scalars=atom_rgb, rgb=True, preference="cell",
                        pbr=True, metallic=0.15, roughness=0.45, name="atoms")
    if show_bonds and bonds:
        bond_mesh = build_bond_glyphs(
            draw_pos, bonds, radius=0.05 if data.expanded_ws else 0.08
        )
        if bond_mesh.n_points:
            pl.add_mesh(bond_mesh, color=(0.55, 0.55, 0.58),
                        pbr=True, metallic=0.2, roughness=0.5,
                        opacity=0.62 if data.expanded_ws else 1.0, name="bonds")
    if show_axes:
        build_coordinate_frame(pl, center, extent)

    # Static camera frame; the rock rotates only the viewpoint about world-Z.
    # Camera turns about `spin_axis` (default x), kept vertical on screen.
    _named = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
    n = np.asarray(_named.get(spin_axis, spin_axis), dtype=np.float64)
    n /= np.linalg.norm(n)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp = np.cross(n, ref)
    perp /= np.linalg.norm(perp)
    base_view = n * 0.6 + perp           # oblique view that orbits about n
    base_view /= np.linalg.norm(base_view)
    up = n
    dist = extent * float(camera_pull)

    def _rot(ang: float) -> np.ndarray:
        x, y, z = n
        c, s, C = np.cos(ang), np.sin(ang), 1.0 - np.cos(ang)
        return np.array([
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ])

    def _place(az_rad: float) -> None:
        pl.camera.position = tuple(center + (_rot(az_rad) @ base_view) * dist)
        pl.camera.focal_point = tuple(center)
        pl.camera.up = tuple(up)
        try:
            pl.renderer.reset_camera_clipping_range()
        except Exception:
            pass

    out = str(output)
    if out.lower().endswith(".png"):
        _place(0.0)
        pl.screenshot(out)
        pl.close()
        return out

    n_frames = max(1, int(round(float(seconds) * int(fps))))
    if out.lower().endswith(".gif"):
        pl.open_gif(out, fps=int(fps))
    else:
        pl.open_movie(out, framerate=int(fps))
    rate = np.radians(float(deg_per_sec))
    for f in range(n_frames):
        # Constant-speed turn in one direction; stops wherever the clip ends.
        _place(rate * (f / int(fps)))
        pl.write_frame()
    pl.close()
    return out


def render_density_gif(
    cno_grid: np.ndarray,
    lattice: np.ndarray,
    atoms_cart: np.ndarray,
    atom_symbols: Sequence,
    output: str = "cno_structure.gif",
    *,
    iso_fraction: float = 0.5,
    replication: Tuple[int, int, int] = (2, 2, 2),
    regional_map: RegionalCNOMap | None = None,
    **kwargs,
) -> str:
    """Convenience entry: render straight from a regular ``(Nx, Ny, Nz)`` density grid.

    ``cno_grid`` may be complex or real (only ``|.|^2`` matters).  Back-map your
    orbital to the regular grid, then call this.  Extra keyword args pass through to
    :func:`render_cno_gif` (e.g. ``seconds``, ``fps``, ``rock_deg``, ``camera_pull``,
    ``surface_color``, ``surface_opacity``).
    """
    grid = np.asarray(cno_grid)
    if regional_map is None:
        if grid.ndim != 3:
            raise ValueError(f"cno_grid must be 3-D (Nx, Ny, Nz), got shape {grid.shape}")
        cv = grid.reshape(-1, order="C").astype(np.complex128)[None, :]
        data = CNOData.from_arrays(
            cv, grid.shape, lattice, atom_symbols=atom_symbols, atoms_cart=atoms_cart
        )
    else:
        if grid.ndim != 1:
            raise ValueError(
                "A RegionalCNOMap requires one CNO row with shape (n_saved_samples,)"
            )
        data = CNOData.from_ws_arrays(
            grid, regional_map.grid_shape, lattice,
            base_indices=regional_map.base_indices,
            translations=regional_map.translations,
            ws_center_cart=regional_map.ws_center_cart,
            points_frac_cont=regional_map.points_frac_cont,
            points_cart=regional_map.points_cart,
            atom_symbols=atom_symbols,
            atoms_cart=atoms_cart,
        )
    return render_cno_gif(
        data, 0, output, iso_fraction=iso_fraction, replication=replication, **kwargs
    )


__all__ = ["RegionalCNOMap", "render_cno_gif", "render_density_gif"]
