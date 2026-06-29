#!/usr/bin/env python3
"""
build_irrep_kpoints.py

Reads the 'Coordinates for DFT calculation:' section from IrRep output and
writes a VASP explicit-point KPOINTS file plus a JSON mapping file.

Usage:
    python build_irrep_kpoints.py high_symmetry_points.out
    python build_irrep_kpoints.py high_symmetry_points.out -o KPOINTS.irrep --force
"""

import argparse
import json
import math
import os
import re
import sys

SECTION_HEADER = "Coordinates for DFT calculation:"

# Matches optional whitespace, then label (any non-whitespace), then colon,
# then three floating-point numbers (supports plain and exponential notation).
KPOINT_RE = re.compile(
    r"^\s*(\S+)\s*:\s*"
    r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"
    r"\s+"
    r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"
    r"\s+"
    r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"
    r"\s*$"
)

# Lines that unambiguously mark the start of a new IrRep section.
SECTION_BOUNDARY_RE = re.compile(r"^\s*-{3,}")


def detect_encoding(filepath):
    """Detect file encoding from BOM or default to UTF-8."""
    with open(filepath, "rb") as f:
        raw = f.read(4)
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return "utf-16"
    if raw[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig"
    return "utf-8"


def parse_kpoints(input_path):
    """
    Return list of (label, [kx, ky, kz]) from the DFT calculation section.
    Raises SystemExit on any parsing error.
    """
    encoding = detect_encoding(input_path)
    try:
        with open(input_path, encoding=encoding, errors="replace") as f:
            lines = f.readlines()
    except OSError as exc:
        sys.exit(f"ERROR: Cannot read '{input_path}': {exc}")

    # Locate the DFT calculation section (use the last occurrence if somehow
    # the header appears more than once).
    section_start = None
    for i, line in enumerate(lines):
        if SECTION_HEADER in line:
            section_start = i

    if section_start is None:
        sys.exit(
            f"ERROR: Section '{SECTION_HEADER}' not found in '{input_path}'.\n"
            f"  Make sure the file was produced with 'print_hs_kpoints: true' in the config."
        )

    kpoints = []
    found_first = False

    for i in range(section_start + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            # Empty line: skip until first k-point; stop once we have at least one.
            if found_first:
                break
            continue

        if SECTION_BOUNDARY_RE.match(stripped):
            # Unambiguous section separator — end of this section.
            break

        m = KPOINT_RE.match(line)
        if m:
            label = m.group(1)
            try:
                coords = [float(m.group(2)), float(m.group(3)), float(m.group(4))]
            except ValueError:
                sys.exit(
                    f"ERROR: Could not convert coordinates on line {i + 1}:\n"
                    f"  {line.rstrip()!r}"
                )
            if not label:
                sys.exit(f"ERROR: Empty label on line {i + 1}: {line.rstrip()!r}")
            for j, c in enumerate(coords):
                if not math.isfinite(c):
                    sys.exit(
                        f"ERROR: Non-finite coordinate (component {j}) on line {i + 1}:\n"
                        f"  {line.rstrip()!r}"
                    )
            kpoints.append((label, coords))
            found_first = True
        else:
            # Non-empty line that does not parse — malformed, not silently ignored.
            sys.exit(
                f"ERROR: Malformed line {i + 1} inside the '{SECTION_HEADER}' section:\n"
                f"  {line.rstrip()!r}\n"
                f"  Expected format:  LABEL : kx ky kz"
            )

    return kpoints


def check_output_file(path, force):
    """Exit with an error if the file exists and --force is not set."""
    if os.path.exists(path) and not force:
        sys.exit(
            f"ERROR: Output file '{path}' already exists.\n"
            f"  Use --force to overwrite."
        )


def warn_duplicates(kpoints):
    """Print a warning (to stderr) for k-points with identical coordinates."""
    seen = {}
    for idx, (label, coords) in enumerate(kpoints):
        key = tuple(round(c, 9) for c in coords)
        if key in seen:
            prev_label, prev_vasp_idx = seen[key]
            print(
                f"WARNING: k-point #{idx + 1} ({label}) has coordinates identical "
                f"to k-point #{prev_vasp_idx} ({prev_label}): {coords}. "
                f"Both entries are preserved.",
                file=sys.stderr,
            )
        else:
            seen[key] = (label, idx + 1)


def write_kpoints(output_path, kpoints):
    """Write the VASP explicit-point KPOINTS file."""
    header = [
        "High-symmetry points generated from IrRep",
        str(len(kpoints)),
        "Reciprocal",
    ]
    data_lines = []
    for label, coords in kpoints:
        kx, ky, kz = coords
        data_lines.append(f"  {kx:.10f}  {ky:.10f}  {kz:.10f}  1.0  ! {label}")
    content = "\n".join(header + data_lines) + "\n"
    try:
        with open(output_path, "w") as f:
            f.write(content)
    except OSError as exc:
        sys.exit(f"ERROR: Cannot write KPOINTS file '{output_path}': {exc}")


def write_json(json_path, source_file, kpoints):
    """Write the JSON mapping file with VASP indices and IrRep command fragments."""
    indices = [str(i + 1) for i in range(len(kpoints))]
    labels = [label for label, _ in kpoints]
    data = {
        "source_file": source_file,
        "number_of_kpoints": len(kpoints),
        "kpoints": [
            {
                "vasp_index": i + 1,
                "label": label,
                "coordinates": coords,
            }
            for i, (label, coords) in enumerate(kpoints)
        ],
        "irrep_arguments": {
            "kpoints": ",".join(indices),
            "kpnames": ",".join(labels),
        },
    }
    try:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
    except OSError as exc:
        sys.exit(f"ERROR: Cannot write JSON file '{json_path}': {exc}")


def print_summary(kpoints, kpoints_path, json_path):
    labels = [label for label, _ in kpoints]
    indices = [str(i + 1) for i in range(len(kpoints))]
    print()
    print("=== IrRep KPOINTS builder ===")
    print(f"Extracted {len(kpoints)} k-point(s): {', '.join(labels)}")
    print(f"KPOINTS file : {kpoints_path}")
    print(f"JSON map     : {json_path}")
    print()
    print("IrRep command fragments:")
    print(f"  -kpoints={','.join(indices)}")
    print(f"  -kpnames={','.join(labels)}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract high-symmetry k-points from IrRep output "
            "('Coordinates for DFT calculation:' section) and write "
            "a VASP explicit-point KPOINTS file plus a JSON mapping file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python build_irrep_kpoints.py high_symmetry_points.out\n"
            "  python build_irrep_kpoints.py high_symmetry_points.out "
            "-o KPOINTS.irrep --force\n"
        ),
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="high_symmetry_points.out",
        metavar="INPUT",
        help=(
            "IrRep output file containing the high-symmetry k-point sections "
            "(default: high_symmetry_points.out)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default="KPOINTS",
        metavar="OUTPUT",
        help="VASP KPOINTS output file (default: KPOINTS)",
    )
    parser.add_argument(
        "--mapping",
        default="irrep_kpoint_map.json",
        metavar="JSON",
        help="JSON mapping/index file (default: irrep_kpoint_map.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files without prompting",
    )
    args = parser.parse_args()

    # --- Validate input ---
    if not os.path.isfile(args.input):
        sys.exit(f"ERROR: Input file '{args.input}' not found.")

    # --- Check outputs before doing any work ---
    check_output_file(args.output, args.force)
    check_output_file(args.mapping, args.force)

    # --- Parse ---
    kpoints = parse_kpoints(args.input)

    if not kpoints:
        sys.exit(
            f"ERROR: No k-points parsed from the '{SECTION_HEADER}' section "
            f"of '{args.input}'."
        )

    # --- Validate (belt-and-suspenders after parse) ---
    for idx, (label, coords) in enumerate(kpoints):
        if not label:
            sys.exit(f"ERROR: Empty label at k-point index {idx + 1}.")
        if len(coords) != 3:
            sys.exit(
                f"ERROR: k-point {label!r} (#{idx + 1}) has {len(coords)} "
                f"coordinates instead of 3."
            )
        for j, c in enumerate(coords):
            if not math.isfinite(c):
                sys.exit(
                    f"ERROR: k-point {label!r} (#{idx + 1}) has a non-finite "
                    f"coordinate at position {j}: {c}"
                )

    # --- Warn about coordinate duplicates (preserve both entries) ---
    warn_duplicates(kpoints)

    # --- Write outputs ---
    write_kpoints(args.output, kpoints)
    write_json(args.mapping, args.input, kpoints)

    # --- Summary ---
    print_summary(kpoints, args.output, args.mapping)


if __name__ == "__main__":
    main()
