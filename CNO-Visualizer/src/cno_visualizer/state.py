"""Viewer state dataclass — defaults and field documentation.

Holds purely-data view configuration; rendering decisions live in :mod:`scene`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple

DisplayMode = Literal["phase", "density", "real", "imaginary"]
Background = Literal["dark", "light"]
ViewMode = Literal["crystal", "ws"]


@dataclass
class ViewerState:
    selected_cno: int = 0
    isovalue_fraction: float = 0.05
    phase_offset: float = 0.0
    display_mode: DisplayMode = "phase"
    auto_color: bool = True          # pick signed (real CNO) vs phase (complex) automatically
    view_mode: ViewMode = "crystal"
    central_opacity: float = 1.0    # opacity of the orbital isosurface
    replication: Tuple[int, int, int] = (1, 1, 1)

    show_atoms: bool = True
    show_bonds: bool = True
    show_cells: bool = True
    show_ws: bool = True
    show_center_marker: bool = True
    show_axes: bool = True

    smooth_shading: bool = True
    background: Background = "dark"
    bond_scale: float = 1.15

    screenshot_dir: Optional[Path] = None
    camera_path: Optional[Path] = None
    off_screen: bool = False

    material_label: Optional[str] = None

    # Visual constants — kept here so callers don't reach into private modules.
    # PHASE_CMAP is resolved through cno_visualizer.colormaps (colorcet CET_* maps).
    PHASE_CMAP: str = field(default="CET_C7", repr=False)
    PHASE_CLIM: Tuple[float, float] = field(default=(0.0, 6.283185307179586), repr=False)
    SIGNED_CMAP: str = field(default="coolwarm", repr=False)   # diverging, for Re/Im
    DENSITY_CMAP: str = field(default="viridis", repr=False)

    def clamp_to(self, n_cnos: int) -> None:
        """Clamp ``selected_cno`` to ``[0, n_cnos)`` in-place."""
        if n_cnos <= 0:
            raise ValueError("n_cnos must be positive.")
        self.selected_cno = int(max(0, min(self.selected_cno, n_cnos - 1)))


__all__ = ["ViewerState", "DisplayMode", "Background"]
