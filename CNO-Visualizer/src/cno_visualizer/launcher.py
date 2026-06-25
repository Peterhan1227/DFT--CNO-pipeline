"""Clickable GUI launcher: pick a ``.npz`` with a file dialog, then open the viewer.

So the viewer can be started without typing a path in a terminal.  Either run
``cno-visualizer-gui`` or wire a Windows desktop shortcut to the bundled
``Launch CNO-Visualizer.bat``.  Any extra CLI flags are passed straight through,
and if ``--data`` is already supplied the file dialog is skipped.
"""

from __future__ import annotations

import sys
from typing import List, Optional, Sequence


def _pick_npz() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk.Tk()
    root.withdraw()
    root.update()
    path = filedialog.askopenfilename(
        title="Select cno_visualizer_data.npz",
        filetypes=[("CNO visualizer data", "*.npz"), ("All files", "*.*")],
    )
    try:
        root.destroy()
    except Exception:
        pass
    return path or None


def main(argv: Optional[Sequence[str]] = None) -> int:
    from cno_visualizer.cli import main as cli_main

    args: List[str] = list(sys.argv[1:] if argv is None else argv)
    if "--data" not in args:
        path = _pick_npz()
        if not path:
            print("No file selected.")
            return 0
        args = ["--data", path, *args]
    return cli_main(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
