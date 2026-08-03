"""Scene orchestrator — holds the PyVista plotter and named actor handles.

The orbital is contoured on a regular structured grid (see
:mod:`cno_visualizer.field`).  Two views are available:

* ``crystal`` (default) — the periodic ``|psi|^2`` contoured over the requested
  ``replication`` of primitive cells; reproduces a cube viewer exactly.
* ``ws`` — the same field contoured over a small block and clipped to the
  analytic Wigner–Seitz polyhedron, showing one localized orbital in its cell.

Incremental updates target single actors so slider events stay cheap:

* phase offset → in-place ``surface.point_data["phase"][:] = ...`` + ``render()``.
* opacity      → ``actor.prop.opacity = value`` + ``render()``.
* visibility   → ``actor.SetVisibility(bool)`` + ``render()``.
* isovalue / CNO → re-contour the cached volume; replace one actor.
* view mode    → rebuild the volume, re-contour, redraw the WS overlay.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from cno_visualizer.colormaps import is_real_orbital, resolve_cmap
from cno_visualizer.crystal import (
    build_atom_glyphs,
    build_bond_glyphs,
    build_bonded_crystal,
    build_local_bonded_crystal,
    replicated_cell_edges,
)
from cno_visualizer.data import CNOData
from cno_visualizer.field import (
    Volume,
    build_crystal_volume,
    build_ws_block_volume,
    clip_surface_to_ws,
    color_surface_by_phase,
    contour_density,
    set_active_cno,
)
from cno_visualizer.state import ViewerState
from cno_visualizer.ws_geometry import WSGeometry, ws_polyhedron


@dataclass
class _Caches:
    volume: Optional[Volume] = None
    ws_geometry: Optional[WSGeometry] = None
    rho_max: float = 0.0                  # max of |psi|^2 over current CNO
    is_real: bool = False                 # current CNO is (numerically) real
    current_surface: Optional[object] = None
    last_good_surface: Optional[object] = None


class Viewer:
    """Coordinates :class:`CNOData`, :class:`ViewerState`, and a PyVista plotter."""

    def __init__(self, data: CNOData, state: ViewerState):
        import pyvista as pv

        self.data = data
        self.state = state
        state.clamp_to(data.n_cnos)
        if state.view_mode == "ws" and not data.ws_enabled:
            warnings.warn(
                "view_mode='ws' requested but data is not WS-enabled; using crystal view."
            )
            state.view_mode = "crystal"
        if data.expanded_ws:
            # This field is one physical finite-volume WS sample set.  It is
            # not a periodic cube, so a replicated crystal contour would be a
            # visualization artefact.  Keep the regional view and its exact
            # WS boundary as the default.
            state.view_mode = "ws"
            state.show_axes = False
            state.show_cells = False

        self.plotter = pv.Plotter(off_screen=state.off_screen)
        self.actors: Dict[str, object] = {}
        self._caches = _Caches()
        self._setup_render_quality()

        # The WS polyhedron is built once if WS data is present; it is both the
        # overlay wireframe and the clip boundary for the WS view.
        if data.ws_enabled and data.ws_center_cart is not None:
            try:
                self._caches.ws_geometry = ws_polyhedron(
                    data.lattice, data.ws_center_cart
                )
            except Exception as exc:  # pragma: no cover - defensive
                warnings.warn(f"Could not build WS polyhedron: {exc}")

        self._build_volume()
        self._refresh_after_cno_change()
        self._build_crystal_context()
        if data.expanded_ws:
            self._focus_expanded_ws()

        if state.show_axes:
            self.plotter.show_axes()

    # ------------------------------------------------------------------ quality
    def _setup_render_quality(self) -> None:
        """Gradient background, anti-aliasing, ambient occlusion, soft lighting."""
        pl = self.plotter
        if self.state.background == "light":
            pl.set_background("#e9edf2", top="#f8fafc")
        else:
            pl.set_background("#0e1116", top="#26303f")
        # Each is best-effort: older VTK/headless builds may not support all.
        for setup in (
            lambda: pl.enable_anti_aliasing("ssaa"),
            lambda: pl.enable_ssao(radius=2.0, bias=0.01),
            lambda: pl.enable_lightkit(),
        ):
            try:
                setup()
            except Exception:
                pass

    # ------------------------------------------------------------------ build
    @property
    def _clip_ws(self) -> bool:
        return self.state.view_mode == "ws" and self._caches.ws_geometry is not None

    def _build_volume(self) -> None:
        if self._clip_ws:
            self._caches.volume = build_ws_block_volume(self.data, n_cells=1)
        else:
            self._caches.volume = build_crystal_volume(
                self.data, self.state.replication
            )

    def _build_crystal_context(self) -> None:
        # A large slab primitive cell is not useful context for a local WS
        # orbital (it is dominated by vacuum).  The WS boundary and local atom
        # cluster below replace it in expanded finite-volume mode.
        if not self.data.expanded_ws:
            edges = replicated_cell_edges(
                self.data.lattice, self.state.replication, origin=np.zeros(3)
            )
            self._add_named_actor(
                "cell_edges", edges, color="grey", line_width=1.5,
                opacity=0.6, visible=self.state.show_cells,
            )

        # WS polyhedron overlay (centred at ws_center).
        if self._caches.ws_geometry is not None:
            self._add_named_actor(
                "ws_cell", self._caches.ws_geometry.polyhedron,
                color="cyan", style="wireframe", line_width=2,
                visible=self.state.show_ws,
            )

        # WS / basis center marker.
        if self.data.ws_center_cart is not None:
            import pyvista as pv

            marker = pv.Sphere(
                radius=0.12, center=np.asarray(self.data.ws_center_cart, np.float64)
            )
            self._add_named_actor(
                "center_marker", marker, color="yellow", opacity=0.9,
                visible=self.state.show_center_marker,
            )

        self._build_atoms_and_bonds()

    def _build_atoms_and_bonds(self) -> None:
        """Build atoms + a fully-connected periodic bond network (halo-aware)."""
        if self.data.expanded_ws:
            assert self.data.ws_center_cart is not None
            draw_pos, draw_sym, bonds = build_local_bonded_crystal(
                self.data.atom_symbols,
                self.data.atoms_cart,
                self.data.lattice,
                self.data.ws_center_cart,
                radius=3.1,
                bond_scale=self.state.bond_scale,
            )
            atom_scale = 0.22
        else:
            draw_pos, draw_sym, bonds = build_bonded_crystal(
                self.data.atom_symbols,
                self.data.atoms_cart,
                self.data.lattice,
                self.state.replication,
                bond_scale=self.state.bond_scale,
            )
            atom_scale = 0.40
        atom_mesh, atom_rgb = build_atom_glyphs(draw_pos, draw_sym, radius_scale=atom_scale)
        if atom_mesh.n_points:
            self._add_named_actor(
                "atoms", atom_mesh, scalars=atom_rgb, rgb=True,
                visible=self.state.show_atoms, preference="cell",
                pbr=True, metallic=0.15, roughness=0.45,
            )
        bond_mesh = build_bond_glyphs(
            draw_pos, bonds, radius=0.05 if self.data.expanded_ws else 0.08
        )
        if bond_mesh.n_points:
            self._add_named_actor(
                "bonds", bond_mesh, color=(0.55, 0.55, 0.58),
                visible=self.state.show_bonds,
                pbr=True, metallic=0.2, roughness=0.5,
                opacity=0.62 if self.data.expanded_ws else 1.0,
            )

    def _focus_expanded_ws(self) -> None:
        """Start a slab WS viewer close enough to read the local orbital.

        The full WS prism includes the vacuum height and can be inspected with
        the regular ``0`` reset key.  The useful first view instead centres the
        orbital and its nearest coordination shell while retaining the local
        WS-boundary edges in frame.
        """
        surface = self._caches.current_surface
        if surface is None or surface.n_points == 0 or self.data.ws_center_cart is None:
            return
        bounds = np.asarray(surface.bounds, dtype=np.float64)
        orbital_extent = 0.5 * float(np.linalg.norm([
            bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]
        ]))
        extent = max(3.4, orbital_extent + 0.8)
        center = np.asarray(self.data.ws_center_cart, dtype=np.float64)
        axis = np.array([0.0, 0.0, 1.0])
        side = np.cross(axis, np.array([0.0, 1.0, 0.0]))
        view = axis * 0.60 + side
        view /= np.linalg.norm(view)
        self.plotter.camera.position = tuple(center + view * (4.0 * extent))
        self.plotter.camera.focal_point = tuple(center)
        self.plotter.camera.up = tuple(axis)
        try:
            self.plotter.renderer.reset_camera_clipping_range()
        except Exception:
            pass

    def _add_named_actor(self, name: str, mesh, *, visible: bool = True, **kwargs):
        if name in self.actors:
            try:
                self.plotter.remove_actor(self.actors[name], render=False)
            except Exception:
                pass
            self.actors.pop(name, None)
        actor = self.plotter.add_mesh(mesh, name=name, **kwargs)
        self.actors[name] = actor
        if not visible:
            self._set_visibility(name, False)
        return actor

    # ------------------------------------------------------------------ CNO
    def _refresh_after_cno_change(self) -> None:
        volume = self._caches.volume
        assert volume is not None
        self._caches.rho_max = set_active_cno(
            volume, self.data, self.state.selected_cno
        )
        self._caches.is_real = is_real_orbital(self.data.cno(self.state.selected_cno))
        # Auto coloring: signed amplitude for real CNOs, cyclic phase for complex.
        if self.state.auto_color:
            self.state.display_mode = "real" if self._caches.is_real else "phase"
        self._refresh_surface()

    def current_abs_isovalue(self) -> float:
        """Absolute isovalue = fraction x max(|psi|^2), comparable to a VESTA cube
        level (the stored CNOs satisfy sum|psi|^2 = 1, the cube normalization)."""
        return float(self.state.isovalue_fraction) * float(self._caches.rho_max)

    def _refresh_surface(self) -> None:
        volume = self._caches.volume
        assert volume is not None
        rho_iso = float(self.state.isovalue_fraction) * float(self._caches.rho_max)
        surface = contour_density(volume, rho_iso)
        if surface is not None and self._clip_ws and self._caches.ws_geometry is not None:
            surface = clip_surface_to_ws(surface, self._caches.ws_geometry)
        if surface is None or surface.n_points == 0:
            warnings.warn(
                f"Empty contour at iso={self.state.isovalue_fraction:.4f}; "
                "keeping previous surface."
            )
            return

        color_surface_by_phase(surface, offset=self.state.phase_offset)
        self._caches.current_surface = surface
        self._caches.last_good_surface = surface
        self._apply_display_mode(surface)

    def _scalar_style(self, surface, mode: str):
        """Return ``(scalars, cmap, clim, title)`` for a display mode."""
        if mode == "phase":
            surface.point_data["phase"] = np.mod(
                np.arctan2(surface["psi_imag"], surface["psi_real"])
                + self.state.phase_offset,
                2.0 * np.pi,
            )
            return ("phase", resolve_cmap(self.state.PHASE_CMAP),
                    self.state.PHASE_CLIM, "phase / rad")
        if mode == "density":
            return "rho", resolve_cmap(self.state.DENSITY_CMAP), None, "|psi|^2"
        if mode in ("real", "imaginary"):
            name = "psi_real" if mode == "real" else "psi_imag"
            arr = np.asarray(surface[name])
            m = float(np.max(np.abs(arr))) if arr.size else 1.0
            m = m if m > 0 else 1.0
            label = "Re psi" if mode == "real" else "Im psi"
            return name, resolve_cmap(self.state.SIGNED_CMAP), (-m, m), label
        raise ValueError(f"unknown display_mode {mode!r}")  # pragma: no cover

    def _apply_display_mode(self, surface) -> None:
        scalars, cmap, clim, title = self._scalar_style(surface, self.state.display_mode)
        surface.set_active_scalars(scalars)
        self._add_named_actor(
            "orbital_surface", surface,
            scalars=scalars, cmap=cmap, clim=clim,
            opacity=self.state.central_opacity,
            smooth_shading=self.state.smooth_shading,
            specular=0.25, specular_power=12, ambient=0.28, diffuse=0.75,
            scalar_bar_args={"title": title, "n_labels": 5},
        )

    # ------------------------------------------------------------------ slots
    def set_selected_cno(self, local_index: int) -> None:
        i = int(local_index)
        if 0 <= i < self.data.n_cnos and i != self.state.selected_cno:
            self.state.selected_cno = i
            self._refresh_after_cno_change()

    def set_isovalue(self, fraction: float) -> None:
        fr = max(1e-4, float(fraction))
        if fr != self.state.isovalue_fraction:
            self.state.isovalue_fraction = fr
            self._refresh_surface()

    def set_phase_offset(self, offset: float) -> None:
        off = float(offset)
        self.state.phase_offset = off
        surface = self._caches.current_surface
        if surface is None or self.state.display_mode != "phase":
            return
        surface.point_data["phase"][:] = np.mod(
            np.arctan2(surface["psi_imag"], surface["psi_real"]) + off, 2.0 * np.pi
        )
        self.plotter.render()

    def set_display_mode(self, mode: str) -> None:
        # An explicit mode choice (keyboard) pins coloring and disables auto.
        self.state.auto_color = False
        if mode == self.state.display_mode:
            return
        self.state.display_mode = mode  # type: ignore[assignment]
        if self._caches.current_surface is not None:
            self._apply_display_mode(self._caches.current_surface)
            self.plotter.render()

    def set_view_mode(self, mode: str) -> None:
        """Switch between the ``crystal`` and ``ws`` views (rebuilds the volume)."""
        if mode not in ("crystal", "ws") or mode == self.state.view_mode:
            return
        if self.data.expanded_ws:
            warnings.warn(
                "Expanded finite-volume CNOs are displayed only in their physical WS region."
            )
            return
        if mode == "ws" and self._caches.ws_geometry is None:
            warnings.warn("WS view unavailable: data is not WS-enabled.")
            return
        self.state.view_mode = mode  # type: ignore[assignment]
        self._build_volume()
        self._refresh_after_cno_change()
        self.plotter.render()

    def toggle_view_mode(self) -> None:
        self.set_view_mode("ws" if self.state.view_mode == "crystal" else "crystal")

    def set_central_opacity(self, opacity: float) -> None:
        self.state.central_opacity = float(opacity)
        actor = self.actors.get("orbital_surface")
        if actor is not None:
            actor.prop.opacity = float(opacity)
            self.plotter.render()

    def toggle(self, target: str) -> None:
        """Toggle visibility of a named group."""
        mapping = {
            "atoms": ("show_atoms", "atoms"),
            "bonds": ("show_bonds", "bonds"),
            "cells": ("show_cells", "cell_edges"),
            "ws": ("show_ws", "ws_cell"),
            "center": ("show_center_marker", "center_marker"),
        }
        if target in mapping:
            attr, actor_name = mapping[target]
            new = not getattr(self.state, attr)
            setattr(self.state, attr, new)
            self._set_visibility(actor_name, new)
        elif target == "axes":
            self.state.show_axes = not self.state.show_axes
            try:
                self.plotter.show_axes() if self.state.show_axes else self.plotter.hide_axes()
            except Exception:
                pass
        else:  # pragma: no cover
            return
        self.plotter.render()

    def _set_visibility(self, name: str, visible: bool) -> None:
        actor = self.actors.get(name)
        if actor is None:
            return
        try:
            actor.SetVisibility(bool(visible))
        except Exception:
            actor.visibility = bool(visible)

    # ------------------------------------------------------------------ I/O
    def screenshot(
        self, path: Optional[Path] = None, *, transparent: bool = False, scale: int = 1
    ) -> Path:
        if path is None:
            base = (
                self.state.material_label
                or self.data.material
                or (
                    self.data.source_path.stem.replace("_cno_visualizer_data", "")
                    if self.data.source_path is not None
                    else "cno"
                )
            )
            fname = (
                f"{base}_cno_{int(self.state.selected_cno):03d}"
                f"_{self.state.view_mode}_{self.state.display_mode}"
                f"_iso_{self.state.isovalue_fraction:.3f}.png"
            )
            outdir = self.state.screenshot_dir or Path.cwd() / "screenshots"
            outdir.mkdir(parents=True, exist_ok=True)
            path = outdir / fname
        path = Path(path)
        try:
            self.plotter.screenshot(
                str(path), transparent_background=transparent, scale=scale, return_img=False
            )
        except Exception as exc:
            raise RuntimeError(f"Screenshot to {path} failed: {exc}") from exc
        return path

    def render(self) -> None:
        self.plotter.render()

    def show(self) -> None:
        self.plotter.show()


__all__ = ["Viewer"]
