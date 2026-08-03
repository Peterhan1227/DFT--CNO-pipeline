"""Compatibility entry point for CNO symmetry analysis/adaptation.

The old script rotated manually chosen groups with an unnormalised Euclidean
overlap matrix, and could therefore mix non-degenerate CNOs.  Use
``cno_symmetry.py --adapt`` after reviewing its read-only report instead.
"""

from cno_symmetry import main


if __name__ == "__main__":
    raise SystemExit(main())
