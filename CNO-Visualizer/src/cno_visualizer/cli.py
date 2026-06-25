"""Command-line entry point for ``cno-visualizer``."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from cno_visualizer.controls import attach_controls
from cno_visualizer.data import CNOData, CNODataError
from cno_visualizer.scene import Viewer
from cno_visualizer.state import ViewerState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cno-visualizer",
        description=(
            "Interactive viewer for complex Cell Natural Orbitals. "
            "Consumes a self-contained .npz produced by the exporter."
        ),
    )
    parser.add_argument(
        "--data", required=True, type=Path, help="Path to cno_visualizer_data.npz"
    )
    parser.add_argument("--cno", type=int, default=0, help="Local CNO row index to show")
    parser.add_argument(
        "--iso", type=float, default=0.05, help="Isovalue fraction of max |psi|^2"
    )
    parser.add_argument(
        "--display",
        choices=("phase", "density", "real", "imaginary"),
        default="phase",
        help="Initial surface coloring (default: phase)",
    )
    parser.add_argument(
        "--view",
        choices=("crystal", "ws"),
        default="crystal",
        help=(
            "crystal: periodic isosurface over --replicate cells (matches the cube); "
            "ws: one orbital clipped to the Wigner-Seitz cell. Default: crystal."
        ),
    )
    parser.add_argument(
        "--replicate",
        nargs=3,
        type=int,
        metavar=("NX", "NY", "NZ"),
        default=(1, 1, 1),
        help="Number of primitive cells to replicate along (a1, a2, a3)",
    )
    parser.add_argument("--no-atoms", action="store_true")
    parser.add_argument("--no-bonds", action="store_true")
    parser.add_argument("--no-cells", action="store_true")
    parser.add_argument("--no-ws", action="store_true")
    parser.add_argument(
        "--background",
        choices=("dark", "light"),
        default="dark",
        help="Background color",
    )
    parser.add_argument(
        "--screenshot-dir", type=Path, default=None, help="Output directory for screenshots"
    )
    parser.add_argument(
        "--camera", type=Path, default=None, help="JSON camera-state to load at startup"
    )
    parser.add_argument(
        "--off-screen",
        action="store_true",
        help="Build the scene without opening a window (headless smoke test).",
    )
    return parser


def _label_from_path(path: Path) -> str:
    stem = path.stem
    stem = stem.replace("_cno_visualizer_data", "").replace("cno_visualizer_data", "")
    stem = stem.strip("_") or "cno"
    return stem


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        data = CNOData.from_npz(args.data)
    except CNODataError as exc:
        print(f"error: {exc}")
        return 2

    state = ViewerState(
        selected_cno=max(0, min(int(args.cno), data.n_cnos - 1)),
        isovalue_fraction=float(args.iso),
        display_mode=args.display,
        view_mode=args.view,
        replication=tuple(max(1, int(n)) for n in args.replicate),  # type: ignore[arg-type]
        show_atoms=not args.no_atoms,
        show_bonds=not args.no_bonds,
        show_cells=not args.no_cells,
        show_ws=not args.no_ws,
        background=args.background,
        screenshot_dir=args.screenshot_dir,
        camera_path=args.camera,
        off_screen=bool(args.off_screen),
        material_label=data.material or _label_from_path(args.data),
    )

    viewer = Viewer(data, state)
    attach_controls(viewer)

    if state.off_screen:
        # Headless: render one frame and write a screenshot so callers have a
        # ground-truth artefact to inspect from the smoke-test harness.
        try:
            out = viewer.screenshot()
            print(f"[cno-visualizer] off-screen render saved -> {out}")
        except Exception as exc:
            print(f"[cno-visualizer] off-screen render failed: {exc}")
            return 1
        return 0

    viewer.show()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "build_parser"]
