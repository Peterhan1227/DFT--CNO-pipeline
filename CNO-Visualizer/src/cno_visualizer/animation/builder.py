"""Build the static animation scene: a spherical crystal cluster + WS cell + axis.

A *spherical* cluster about the basis center (rather than a box) is used so that a
site-symmetry rotation about the center maps the cluster onto itself — the atoms
visibly land back on equivalent sites.  All geometry comes from the shared
:mod:`cno_visualizer.crystal` and :mod:`cno_visualizer.ws_geometry` helpers.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from cno_visualizer.crystal import (
    build_atom_glyphs,
    build_bond_glyphs,
    find_bonds,
)
from cno_visualizer.ws_geometry import ws_polyhedron


def atoms_in_sphere(
    symbols: Sequence,
    atoms_cart: np.ndarray,
    lattice: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> Tuple[np.ndarray, List[str]]:
    """Return all atom images within ``radius`` of ``center`` (Cartesian)."""
    lattice = np.asarray(lattice, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    syms = [str(s) for s in symbols]
    lengths = np.linalg.norm(lattice, axis=1)
    nmax = [int(np.ceil(radius / max(L, 1e-6))) + 1 for L in lengths]

    pos: List[np.ndarray] = []
    sym: List[str] = []
    for a in range(-nmax[0], nmax[0] + 1):
        for b in range(-nmax[1], nmax[1] + 1):
            for c in range(-nmax[2], nmax[2] + 1):
                shift = a * lattice[0] + b * lattice[1] + c * lattice[2]
                for s, p in zip(syms, atoms_cart):
                    q = p + shift
                    if np.linalg.norm(q - center) <= radius:
                        pos.append(q)
                        sym.append(s)
    if not pos:
        return np.zeros((0, 3), dtype=np.float64), []
    return np.asarray(pos, dtype=np.float64), sym


def _nice_step(length: float, target: int = 4) -> float:
    """A round tick spacing (1/2/2.5/5 × 10^k) giving ~``target`` ticks over ``length``."""
    raw = max(length / max(target, 1), 1e-9)
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0):
        if raw <= m * mag:
            return m * mag
    return 10.0 * mag


def build_coordinate_frame(
    plotter,
    center: np.ndarray,
    extent: float,
    *,
    tick_spacing: Optional[float] = None,
    show_labels: bool = True,
    show_numbers: bool = False,
    line_width: float = 3.0,
):
    """A Manim/3b1b-style 3-D xyz coordinate frame centered at ``center``.

    Three colored arrows (x red, y green, z blue) at the **same scale** as the
    structure, with tick marks and ``x/y/z`` labels (and optional numbers).  It is
    a fixed reference frame — the crystal rotates inside it.
    """
    import pyvista as pv

    c = np.asarray(center, dtype=np.float64).reshape(3)
    L = float(extent)
    if tick_spacing is None:
        tick_spacing = _nice_step(L)
    n_ticks = int(np.floor((L - 1e-9) / tick_spacing))

    dirs = [
        (np.array([1.0, 0.0, 0.0]), (0.90, 0.42, 0.42), "x", np.array([0.0, 1.0, 0.0])),
        (np.array([0.0, 1.0, 0.0]), (0.46, 0.72, 0.36), "y", np.array([0.0, 0.0, 1.0])),
        (np.array([0.0, 0.0, 1.0]), (0.45, 0.64, 0.93), "z", np.array([1.0, 0.0, 0.0])),
    ]
    actors = []
    tick_half = 0.06 * L / max(n_ticks, 1) + 0.06
    for d, col, name, perp in dirs:
        actors.append(plotter.add_mesh(
            pv.Line(c - L * d, c + L * d), color=col, line_width=line_width,
            name=f"frame_{name}",
        ))
        actors.append(plotter.add_mesh(
            pv.Cone(center=c + L * d, direction=d, height=0.45, radius=0.14, resolution=28),
            color=col, name=f"frame_{name}_tip",
        ))
        segs = []
        for k in range(1, n_ticks + 1):
            for s in (-1, 1):
                p = c + s * k * tick_spacing * d
                segs.append((p - tick_half * perp, p + tick_half * perp))
        if segs:
            pts = np.array([q for seg in segs for q in seg], dtype=np.float64)
            lines = np.hstack([[2, 2 * i, 2 * i + 1] for i in range(len(segs))]).astype(np.int64)
            actors.append(plotter.add_mesh(
                pv.PolyData(pts, lines=lines), color=col,
                line_width=max(1.0, line_width - 1.5), name=f"frame_{name}_ticks",
            ))
        if show_labels:
            try:
                actors.append(plotter.add_point_labels(
                    np.array([c + (L + 0.45) * d]), [name], font_size=22, text_color=col,
                    shape=None, show_points=False, always_visible=True, bold=True,
                    name=f"frame_{name}_lbl",
                ))
            except Exception:
                pass
        if show_numbers:
            npts, ntxt = [], []
            for k in range(1, n_ticks + 1):
                for s in (-1, 1):
                    val = s * k * tick_spacing
                    npts.append(c + val * d + 0.16 * perp)
                    ntxt.append(f"{val:g}")
            if npts:
                try:
                    actors.append(plotter.add_point_labels(
                        np.array(npts), ntxt, font_size=11, text_color=(0.8, 0.8, 0.82),
                        shape=None, show_points=False, always_visible=True,
                        name=f"frame_{name}_nums",
                    ))
                except Exception:
                    pass
    return actors


def apply_render_quality(plotter, background: str = "dark") -> None:
    """Match the viewer's look: gradient background, AA, ambient occlusion, lighting."""
    if background == "light":
        plotter.set_background("#e9edf2", top="#f8fafc")
    else:
        plotter.set_background("#0e1116", top="#26303f")
    for setup in (
        lambda: plotter.enable_anti_aliasing("ssaa"),
        lambda: plotter.enable_ssao(radius=2.0, bias=0.01),
        lambda: plotter.enable_lightkit(),
    ):
        try:
            setup()
        except Exception:
            pass


def build_scene(
    plotter,
    *,
    symbols: Sequence,
    atoms_cart: np.ndarray,
    lattice: np.ndarray,
    center: np.ndarray,
    radius: float = 6.0,
    bond_scale: float = 1.15,
    show_ws: bool = True,
    axis: Optional[np.ndarray] = None,
    show_axes: bool = True,
    axes_numbers: bool = False,
    axes_extent: Optional[float] = None,
    show_ghost: bool = True,
    ghost_opacity: float = 0.35,
    background: str = "dark",
):
    """Populate ``plotter`` with the static scene.

    Returns ``(actors, animated_names)`` where ``animated_names`` lists the actors
    that the symmetry operation should transform (the crystal cluster); the xyz
    coordinate frame, WS cell, axis, center marker, and the translucent *ghost*
    copy of the original lattice all stay fixed as the reference.
    """
    import pyvista as pv

    center = np.asarray(center, dtype=np.float64).reshape(3)
    actors: dict = {}

    pos, sym = atoms_in_sphere(symbols, atoms_cart, lattice, center, radius)
    bonds = find_bonds(pos, sym, bond_scale=bond_scale) if len(pos) else []

    atom_mesh, atom_rgb = build_atom_glyphs(pos, sym)
    bond_mesh = build_bond_glyphs(pos, bonds)

    # Ghost reference: a translucent grey copy of the original lattice that never
    # moves, so the rotated (colored) copy can be compared against it.  Drawn first
    # so the moving copy renders on top.
    if show_ghost:
        if atom_mesh.n_points:
            actors["atoms_ref"] = plotter.add_mesh(
                atom_mesh.copy(), color=(0.60, 0.65, 0.74), opacity=ghost_opacity,
                specular=0.1, name="atoms_ref",
            )
        if bond_mesh.n_points:
            actors["bonds_ref"] = plotter.add_mesh(
                bond_mesh.copy(), color=(0.50, 0.54, 0.62), opacity=ghost_opacity,
                name="bonds_ref",
            )

    # Moving copy: element colors, full opacity.
    if atom_mesh.n_points:
        actors["atoms"] = plotter.add_mesh(
            atom_mesh, scalars=atom_rgb, rgb=True, preference="cell",
            pbr=True, metallic=0.15, roughness=0.45, name="atoms",
        )
    if bond_mesh.n_points:
        actors["bonds"] = plotter.add_mesh(
            bond_mesh, color=(0.85, 0.62, 0.42),
            pbr=True, metallic=0.2, roughness=0.5, name="bonds",
        )

    if show_ghost:
        txt_color = "white" if background == "dark" else "black"
        try:
            actors["ghost_legend"] = plotter.add_text(
                "translucent grey = original lattice    |    colored = after operation",
                position="lower_left", font_size=9, color=txt_color, name="ghost_legend",
            )
        except Exception:
            pass

    if show_ws:
        try:
            geom = ws_polyhedron(lattice, center)
            actors["ws_cell"] = plotter.add_mesh(
                geom.polyhedron, color="cyan", style="wireframe",
                line_width=2, name="ws_cell",
            )
        except Exception:
            pass

    # Rotation axis: a thin rod through the center spanning the cluster.
    if axis is not None:
        ax = np.asarray(axis, dtype=np.float64).reshape(3)
        ax = ax / max(np.linalg.norm(ax), 1e-12)
        rod = pv.Cylinder(center=center, direction=ax, radius=0.05,
                          height=2.2 * radius, resolution=24)
        actors["axis"] = plotter.add_mesh(
            rod, color=(1.0, 0.85, 0.2), name="axis", opacity=0.9
        )

    marker = pv.Sphere(radius=0.16, center=center)
    actors["center"] = plotter.add_mesh(
        marker, color="yellow", name="center", pbr=True, metallic=0.1, roughness=0.3
    )

    if show_axes:
        ext = axes_extent if axes_extent is not None else (radius * 1.05 + 1.5)
        actors["frame"] = build_coordinate_frame(
            plotter, center, ext, show_numbers=axes_numbers
        )

    animated_names = [n for n in ("atoms", "bonds") if n in actors]
    return actors, animated_names


__all__ = ["atoms_in_sphere", "apply_render_quality", "build_scene"]
