"""SymmetryAnimator — smooth eased site-symmetry rotations rendered to MP4/GIF.

The crystal cluster (atoms + bonds) is rotated about the basis center while the
WS cell, rotation axis and center marker stay fixed.  Motion is driven by an
easing rate function (default ``smootherstep``) so each step gently accelerates
and decelerates.  ``n_steps`` applies the operation repeatedly (e.g. a 3-fold
shown three times returns the cluster to itself).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from cno_visualizer.animation.builder import apply_render_quality, build_scene
from cno_visualizer.animation.easing import smootherstep
from cno_visualizer.animation.operations import (
    GlideOp,
    RotationOp,
    ScrewOp,
    glide_matrix_4x4,
    rotation_matrix_4x4,
    screw_matrix_4x4,
)

AnyOp = Union[RotationOp, ScrewOp, GlideOp]


@dataclass
class AnimationSpec:
    op: AnyOp
    duration: float = 4.0                       # seconds per symmetry step
    fps: int = 30
    easing: Callable[[float], float] = field(default=smootherstep)
    hold_start: float = 0.6                      # static seconds before motion
    hold_step: float = 0.5                       # static seconds after each step
    hold_end: float = 1.0                        # static seconds at the end
    n_steps: int = 1                             # apply the op this many times
    camera_orbit_deg: float = 0.0                # optional cinematic camera drift


def _camera_axis(op: AnyOp) -> np.ndarray:
    """The vector to frame the camera around: rotation/screw axis, or glide-plane normal."""
    return op.normal if isinstance(op, GlideOp) else op.axis


def _set_user_matrix(actor, M: np.ndarray) -> None:
    try:
        actor.user_matrix = M
        return
    except Exception:
        pass
    try:
        from vtkmodules.vtkCommonMath import vtkMatrix4x4
    except Exception:  # pragma: no cover - very old vtk
        from vtk import vtkMatrix4x4  # type: ignore
    vm = vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vm.SetElement(i, j, float(M[i, j]))
    actor.SetUserMatrix(vm)


class SymmetryAnimator:
    """Build a fixed cluster + WS-cell scene and animate rotations about the center."""

    def __init__(
        self,
        data,
        *,
        center: Optional[Sequence[float]] = None,
        radius: float = 6.0,
        background: str = "dark",
        window_size: Sequence[int] = (1000, 780),
        off_screen: bool = True,
    ):
        import pyvista as pv

        self.data = data
        if center is not None:
            self.center = np.asarray(center, dtype=np.float64).reshape(3)
        elif getattr(data, "ws_center_cart", None) is not None:
            self.center = np.asarray(data.ws_center_cart, dtype=np.float64)
        else:
            self.center = np.zeros(3, dtype=np.float64)
        self.radius = float(radius)
        self.background = background
        self.plotter = pv.Plotter(off_screen=off_screen, window_size=tuple(window_size))
        apply_render_quality(self.plotter, background)
        self._actors: Optional[Dict[str, object]] = None
        self._animated: List[str] = []
        self._title_actor = None

    # ------------------------------------------------------------------ build
    def build(
        self,
        axis: Optional[np.ndarray] = None,
        show_ws: bool = True,
        show_axes: bool = True,
        axes_numbers: bool = False,
        show_ghost: bool = True,
    ) -> "SymmetryAnimator":
        self._actors, self._animated = build_scene(
            self.plotter,
            symbols=self.data.atom_symbols,
            atoms_cart=self.data.atoms_cart,
            lattice=self.data.lattice,
            center=self.center,
            radius=self.radius,
            show_ws=show_ws,
            axis=axis,
            show_axes=show_axes,
            axes_numbers=axes_numbers,
            show_ghost=show_ghost,
            background=self.background,
        )
        self._frame_camera(axis)
        return self

    def _frame_camera(self, axis: Optional[np.ndarray]) -> None:
        """Place the camera ~52° off the rotation axis so the rotation reads clearly.

        Looking straight down the axis hides the motion (an n-fold axis projects to
        an n-(or 2n-)gon that maps onto itself); an oblique view shows the 3-D
        rotation while keeping the axis roughly vertical in frame.
        """
        cam = self.plotter.camera
        cam.focal_point = tuple(self.center)
        if axis is not None:
            n = np.asarray(axis, dtype=np.float64)
            n = n / max(np.linalg.norm(n), 1e-12)
            ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            perp = np.cross(n, ref)
            perp /= max(np.linalg.norm(perp), 1e-12)
            theta = np.radians(52.0)
            view = n * np.cos(theta) + perp * np.sin(theta)
            cam.position = tuple(self.center + view)   # sets the view direction
            cam.up = tuple(perp)
        else:
            self.plotter.camera_position = "iso"
        self.plotter.reset_camera()                    # fit distance, keep direction
        self.plotter.camera.zoom(1.1)

    def _setup_op(self, op: AnyOp) -> None:
        """Cache the operation's fixed geometry (axis/normal/center/translation).

        ``total_progress`` (how many whole applications, possibly fractional) is the
        only thing that varies frame-to-frame; every op type maps it to a 4x4 matrix
        directly (no per-step compounding), so intermediate rounding never accumulates.
        """
        self._op = op
        self._op_center = np.asarray(op.center, dtype=np.float64)
        if isinstance(op, GlideOp):
            self._op_normal = np.asarray(op.normal, dtype=np.float64)
            self._op_translation = np.asarray(op.translation, dtype=np.float64)
        else:
            self._op_axis = np.asarray(op.axis, dtype=np.float64)
            self._op_translation = np.asarray(
                getattr(op, "translation", np.zeros(3)), dtype=np.float64
            )

    def _matrix_for_progress(self, total_progress: float) -> np.ndarray:
        op = self._op
        if isinstance(op, GlideOp):
            return glide_matrix_4x4(
                self._op_normal, total_progress, self._op_translation, self._op_center
            )
        angle = math.radians(op.angle_deg) * total_progress
        if isinstance(op, ScrewOp):
            return screw_matrix_4x4(
                self._op_axis, angle, self._op_translation * total_progress, self._op_center
            )
        return rotation_matrix_4x4(self._op_axis, angle, self._op_center)

    def _apply_progress(self, total_progress: float) -> None:
        M = self._matrix_for_progress(total_progress)
        for name in self._animated:
            actor = self._actors.get(name) if self._actors else None
            if actor is not None:
                _set_user_matrix(actor, M)

    def _title(self, text: str) -> None:
        try:
            if self._title_actor is not None:
                self.plotter.remove_actor(self._title_actor, render=False)
        except Exception:
            pass
        self._title_actor = self.plotter.add_text(
            text, position="upper_left", font_size=13,
            color="white" if self.background == "dark" else "black",
        )

    # ------------------------------------------------------------------ render
    def render(self, spec: AnimationSpec, output: str = "symmetry.mp4") -> str:
        if self._actors is None:
            self.build(axis=_camera_axis(spec.op))
        self._setup_op(spec.op)

        pl = self.plotter
        out = str(output)
        fps = int(spec.fps)
        if out.lower().endswith(".gif"):
            pl.open_gif(out, fps=fps)
        else:
            pl.open_movie(out, framerate=fps)

        self._title(spec.op.label)

        nf = max(1, int(round(spec.duration * fps)))
        hs = int(round(spec.hold_start * fps))
        hstep = int(round(spec.hold_step * fps))
        he = int(round(spec.hold_end * fps))
        n_steps = max(1, int(spec.n_steps))
        total_frames = hs + n_steps * (nf + hstep) + he
        orbit_per_frame = spec.camera_orbit_deg / max(1, total_frames)

        def emit() -> None:
            if spec.camera_orbit_deg:
                try:
                    pl.camera.Azimuth(orbit_per_frame)
                    pl.renderer.reset_camera_clipping_range()
                except Exception:
                    pass
            pl.write_frame()

        self._apply_progress(0.0)
        for _ in range(hs):
            emit()

        for step in range(n_steps):
            for f in range(1, nf + 1):
                self._apply_progress(step + spec.easing(f / nf))
                emit()
            self._apply_progress(step + 1)
            for _ in range(hstep):
                emit()

        for _ in range(he):
            emit()

        pl.close()
        return out

    # ------------------------------------------------------------------ play
    def play(self, spec: AnimationSpec) -> None:  # pragma: no cover - interactive
        """Best-effort interactive playback (on-screen only)."""
        if self._actors is None:
            self.build(axis=_camera_axis(spec.op))
        self._setup_op(spec.op)
        pl = self.plotter
        self._title(spec.op.label)
        pl.show(interactive_update=True, auto_close=False)
        nf = max(1, int(round(spec.duration * spec.fps)))
        for step in range(max(1, int(spec.n_steps))):
            for f in range(1, nf + 1):
                self._apply_progress(step + spec.easing(f / nf))
                pl.update()
                time.sleep(1.0 / spec.fps)
        pl.show()


__all__ = ["SymmetryAnimator", "AnimationSpec"]
