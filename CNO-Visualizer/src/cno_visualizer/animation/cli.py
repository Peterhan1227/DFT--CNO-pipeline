"""``cno-anim`` — render a smooth site-symmetry rotation of the crystal + WS cell."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from cno_visualizer.animation.animator import AnimationSpec, SymmetryAnimator
from cno_visualizer.animation.easing import RATE_FUNCTIONS, get_rate_function
from cno_visualizer.animation.operations import make_rotation
from cno_visualizer.data import CNOData, CNODataError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cno-anim",
        description=(
            "Render a smooth (eased) site-symmetry rotation of the crystal lattice "
            "and Wigner-Seitz cell about a basis center. Output is MP4 or GIF."
        ),
    )
    p.add_argument("--data", required=True, type=Path, help="Path to cno_visualizer_data.npz")
    p.add_argument("--axis", default="111",
                   help="Rotation axis: 'z' or a crystallographic direction like '111' (default: 111)")
    p.add_argument("--nfold", type=int, default=3, help="Rotation order n -> 360/n deg (default: 3)")
    p.add_argument("--angle", type=float, default=None, help="Explicit rotation angle in deg (overrides --nfold)")
    p.add_argument("--center", default="ws",
                   help="'ws' (WS/basis center, default), 'origin', or 3 Cartesian numbers 'x y z'")
    p.add_argument("--center-frac", nargs=3, type=float, default=None, metavar=("U", "V", "W"),
                   help="Basis center in fractional coords (overrides --center)")
    p.add_argument("--radius", type=float, default=6.0, help="Cluster radius in Angstrom (default: 6)")
    p.add_argument("--steps", type=int, default=1, help="Apply the operation this many times in sequence")
    p.add_argument("--duration", type=float, default=4.0, help="Seconds per step (default: 4)")
    p.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    p.add_argument("--easing", default="smootherstep", choices=sorted(RATE_FUNCTIONS),
                   help="Rate function (default: smootherstep)")
    p.add_argument("--orbit", type=float, default=0.0, help="Cinematic camera azimuth drift in deg")
    p.add_argument("--background", choices=("dark", "light"), default="dark")
    p.add_argument("--no-ws", action="store_true", help="Hide the Wigner-Seitz cell")
    p.add_argument("--no-axes", action="store_true",
                   help="Hide the large xyz coordinate frame (shown by default)")
    p.add_argument("--axes-numbers", action="store_true",
                   help="Label the coordinate-frame tick marks with numbers")
    p.add_argument("--no-ghost", action="store_true",
                   help="Hide the translucent original-lattice reference copy (shown by default)")
    p.add_argument("--output", type=Path, default=Path("symmetry.mp4"),
                   help="Output .mp4 or .gif (default: symmetry.mp4)")
    p.add_argument("--show", action="store_true", help="Interactive playback instead of writing a file")
    return p


def _resolve_center(args, data: CNOData) -> np.ndarray:
    if args.center_frac is not None:
        return np.asarray(args.center_frac, dtype=np.float64) @ data.lattice
    s = str(args.center).strip().lower()
    if s == "ws":
        if data.ws_center_cart is not None:
            return np.asarray(data.ws_center_cart, dtype=np.float64)
        return np.zeros(3)
    if s == "origin":
        return np.zeros(3)
    nums = [float(x) for x in s.replace(",", " ").split()]
    if len(nums) != 3:
        raise SystemExit("error: --center must be 'ws', 'origin', or three numbers")
    return np.asarray(nums, dtype=np.float64)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        data = CNOData.from_npz(args.data)
    except CNODataError as exc:
        print(f"error: {exc}")
        return 2

    center = _resolve_center(args, data)
    op = make_rotation(
        center, axis=args.axis,
        n_fold=(None if args.angle is not None else args.nfold),
        angle_deg=args.angle, lattice=data.lattice,
    )
    anim = SymmetryAnimator(
        data, center=center, radius=args.radius,
        background=args.background, off_screen=not args.show,
    )
    anim.build(
        axis=op.axis, show_ws=not args.no_ws,
        show_axes=not args.no_axes, axes_numbers=args.axes_numbers,
        show_ghost=not args.no_ghost,
    )
    spec = AnimationSpec(
        op=op, duration=args.duration, fps=args.fps,
        easing=get_rate_function(args.easing),
        n_steps=max(1, args.steps), camera_orbit_deg=args.orbit,
    )

    if args.show:
        anim.play(spec)
        return 0
    try:
        out = anim.render(spec, str(args.output))
    except Exception as exc:
        print(f"error: failed to render animation: {exc}")
        return 1
    print(f"[cno-anim] wrote {out}  ({op.label}, {args.steps}x{op.angle_deg:.0f}deg, {args.easing})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "build_parser"]
