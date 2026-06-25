"""Symmetry-operation animations for the crystal lattice and Wigner–Seitz cell.

This subpackage is **deliberately isolated** from the interactive viewer: it only
reads the pure-geometry helpers in :mod:`cno_visualizer.crystal` and
:mod:`cno_visualizer.ws_geometry` and never imports the viewer (``scene``,
``controls``, ``state``).  Nothing here can affect the live viewer, and the viewer
does not depend on anything here.

It renders smooth (Manim-style eased) site-symmetry rotations of a spherical
crystal cluster about a chosen basis center, with the WS cell and rotation axis as
a fixed reference frame.  Output is an MP4 or GIF, or an interactive window.
"""

from cno_visualizer.animation.animator import AnimationSpec, SymmetryAnimator
from cno_visualizer.animation.operations import RotationOp, make_rotation

__all__ = ["SymmetryAnimator", "AnimationSpec", "RotationOp", "make_rotation"]
