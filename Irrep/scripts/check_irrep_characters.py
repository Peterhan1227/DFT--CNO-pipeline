#!/usr/bin/env python3
"""
check_irrep_characters.py  --  post-processing character consistency check

Reads a completed IrRep JSON output (Monty-serialised), independently
reconstructs the character vector for each degenerate block using raw
IrrepTable data and the patched transformation formula, and compares against
the DFT-derived characters already stored in the JSON.

Does NOT re-read WAVECAR.  Does NOT call get_irreps_from_table().

Patched formula applied here independently:

    chi_alpha_dftcell[isym] =
        raw_char[isym].conj()
        * sign
        * exp(-2 pi i  dt . k_table)

    dt = table_sym.t - json["translation refUC"]

Usage
-----
    python scripts/check_irrep_characters.py output/test_si_irrep.json
    python scripts/check_irrep_characters.py output/test_si_irrep.json --kpoint-index 2
    python scripts/check_irrep_characters.py output/test_si_irrep.json --tolerance 1e-5

Default tolerance is 1e-4. This is appropriate for DFT calculations where
numerical noise in the character traces is at the 1e-5 to 1e-4 level.
Use --tolerance 1e-6 only if your DFT characters are unusually clean.

The bug being guarded against (wrong sign in get_irreps_from_table) produces
residuals ~1 at ops 41/44 rather than ~1e-5, so any reasonable tolerance
distinguishes the correct from the buggy case.

Exit codes
----------
    0  all blocks pass
    1  any block fails (non-integer multiplicity or |chi_table - chi_DFT| > tol)
"""

import argparse
import sys
import numpy as np
from monty.serialization import loadfn
from irreptables.irreps import IrrepTable

# Ops with complex W-point characters; flagged in the printed table.
HIGHLIGHT_OPS = {41, 44}

COL_SYM  = 4
COL_CHI  = 28
COL_DIFF = 11


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Post-processing consistency check for IrRep JSON output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("json_file", help="Path to the IrRep JSON output file")
    p.add_argument(
        "--kpoint-index", type=int, default=None, metavar="N",
        help="0-based index of the k-point to check (default: auto-detect by W labels)"
    )
    p.add_argument(
        "--tolerance", type=float, default=1e-4, metavar="T",
        help=(
            "Absolute tolerance applied to both multiplicity integer check and "
            "|chi_table - chi_DFT| comparison (default: 1e-4). "
            "DFT numerical noise is typically 1e-5 to 1e-4; the bug being tested "
            "produces residuals ~1, so any value below 0.1 distinguishes them."
        )
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def collect_kpoints(data):
    """Return a flat list of all k-point dicts from the nested JSON structure."""
    kpoints = []
    for entry in data["characters and irreps"]:
        for kp in entry["subspace"]["k points"]:
            kpoints.append(kp)
    return kpoints


def find_w_kpoints(kpoints):
    """Return 0-based indices of k-points whose every assigned irrep label starts with 'W'."""
    hits = []
    for i, kp in enumerate(kpoints):
        labels = []
        for block_irreps in kp["irreps"]:
            labels.extend(block_irreps.keys())
        if labels and all(n.startswith("W") for n in labels):
            hits.append(i)
    return hits


def kpoint_summary(kp, idx):
    k      = np.asarray(kp["k"])
    k_ref  = np.asarray(kp["k_refUC"])
    labels = [list(b.keys())[0] for b in kp["irreps"] if b]
    return f"  [{idx}]  k={k.tolist()}  k_refUC={k_ref.tolist()}  irreps={labels}"


# ---------------------------------------------------------------------------
# Patched transformation formula (independent of get_irreps_from_table)
# ---------------------------------------------------------------------------

def build_table_chars(table, w_irreps, sym_indices, sg_symmetries_json):
    """
    For each W irrep alpha and each little-group symmetry index isym, compute:

        chi_alpha_dftcell[isym] =
            raw_char[isym].conj()
            * sign
            * exp(-2 pi i  dt . k_table)

    where  dt = table_sym.t - json "translation refUC"

    Note: irr.characters stores only non-zero entries; ops absent from the dict
    have character 0 (zero-trace ops of the representation).

    Returns: dict  irname -> {isym: complex}
    """
    chi = {}
    for irr in w_irreps:
        chi[irr.name] = {}
        k_table = np.asarray(irr.k)
        for isym in sym_indices:
            if isym not in irr.characters:
                # zero-character op — contributes 0, no entry needed
                continue
            table_sym = table.symmetries[isym - 1]   # 0-indexed; BCS index = position+1
            json_sym  = sg_symmetries_json[str(isym)]
            dt = (
                np.asarray(table_sym.t)
                - np.asarray(json_sym["translation refUC"])
            )
            sign = float(json_sym["sign"])
            chi[irr.name][isym] = (
                irr.characters[isym].conj()
                * sign
                * np.exp(-2j * np.pi * np.dot(dt, k_table))
            )
    return chi


# ---------------------------------------------------------------------------
# Per-k-point checker
# ---------------------------------------------------------------------------

def fmt_c(z):
    return f"{z.real:+.6f}{z.imag:+.6f}j"


def check_kpoint(kp, table, sg_json, tol):
    """
    Check one k-point.  Prints a per-block table.
    Returns True if all blocks pass, False otherwise.
    """
    k_refUC     = np.asarray(kp["k_refUC"])
    sym_indices = list(kp["symmetries"])          # ordered 1-based BCS indices
    n_syms      = len(sym_indices)
    characters  = kp["characters"]               # complex128 array, shape (n_blocks, n_syms)
    dimensions  = kp["dimensions"]
    irreps_json = kp["irreps"]                   # list of {irname: [re, im]} per block
    energies    = kp["energies_mean"]
    n_blocks    = len(irreps_json)

    # Identify W irreps in the table
    w_irreps = [irr for irr in table.irreps if irr.kpname == "W"]
    if not w_irreps:
        print("  ERROR: IrrepTable contains no W-point irreps.")
        return False

    # Verify k-vector consistency: table k == k_refUC mod Z^3
    k_table = np.asarray(w_irreps[0].k)
    diff     = k_table - k_refUC
    diff_mod = diff - np.round(diff)
    if not np.allclose(diff_mod, 0, atol=1e-4):
        print(
            f"  ERROR: table k={k_table.tolist()} does not agree mod Z^3 "
            f"with JSON k_refUC={k_refUC.tolist()}  (residual {diff_mod})"
        )
        return False
    print(f"  k_table={k_table.tolist()} agrees with k_refUC={k_refUC.tolist()} "
          f"mod Z^3  [OK]")

    sg_symmetries_json = sg_json["symmetries"]

    # Build patched table characters (independent of get_irreps_from_table)
    chi_alpha = build_table_chars(table, w_irreps, sym_indices, sg_symmetries_json)

    hline  = "  " + "-" * (COL_SYM + 2 + 2 * (COL_CHI + 2) + COL_DIFF + 16)
    header = (
        f"  {'sym':>{COL_SYM}}  "
        f"{'chi_DFT':<{COL_CHI}}  "
        f"{'chi_table':<{COL_CHI}}  "
        f"{'|difference|':>{COL_DIFF}}"
    )

    all_pass = True

    for ib in range(n_blocks):
        chi_dft     = np.array(characters[ib], dtype=complex)   # shape (n_syms,)
        block_dict  = irreps_json[ib]                            # {irname: [re, im]}
        e_mean      = float(energies[ib])
        dim         = int(dimensions[ib])

        # --- validate multiplicities ---
        multiplicities = {}
        mult_ok = True
        for irname, re_im in block_dict.items():
            re = float(re_im[0])
            im = float(re_im[1])

            if abs(im) > tol:
                print(f"  FAIL block {ib}: {irname} multiplicity has "
                      f"|imag part| = {abs(im):.3e} > tol ({tol:.1e})")
                mult_ok = False

            if re < -tol:
                print(f"  FAIL block {ib}: {irname} multiplicity is negative "
                      f"({re:.6f})")
                mult_ok = False

            residual = abs(re - round(re))
            if residual > tol:
                # This catches the bug: 0.5 → residual 0.5 >> tol
                print(f"  FAIL block {ib}: {irname} multiplicity {re:.6f} is "
                      f"not an integer (residual {residual:.3e} > tol {tol:.1e}; "
                      f"refusing to round — this may indicate the character-table bug)")
                mult_ok = False

            multiplicities[irname] = re

        if not mult_ok:
            all_pass = False

        # --- reconstruct chi_table = Σ_alpha n_alpha * chi_alpha_dftcell ---
        # Using exact multiplicity values (not rounded), as read from the JSON.
        # Ops absent from chi_alpha[irname] have table character 0.
        chi_table = np.zeros(n_syms, dtype=complex)
        for irname, n_alpha in multiplicities.items():
            if irname not in chi_alpha:
                print(f"  WARN block {ib}: assigned irrep {irname!r} not "
                      f"found in W-point IrrepTable")
                continue
            for j, isym in enumerate(sym_indices):
                chi_table[j] += n_alpha * chi_alpha[irname].get(isym, 0.0)

        # --- compare chi_table vs chi_DFT ---
        residuals = np.abs(chi_table - chi_dft)
        max_res   = float(residuals.max()) if n_syms else 0.0
        char_ok   = max_res <= tol
        block_ok  = mult_ok and char_ok
        if not block_ok:
            all_pass = False

        status   = "PASS" if block_ok else "FAIL"
        irr_str  = "  ".join(f"{k}({v:.4f})" for k, v in multiplicities.items())
        print(f"\n  Block {ib}  E={e_mean:+.4f} eV  dim={dim}  [{status}]")
        print(f"    Assigned: {irr_str}")
        print(f"    Max |chi_table - chi_DFT|: {max_res:.3e}   tolerance: {tol:.1e}")

        print(f"\n{header}")
        print(hline)
        for j, isym in enumerate(sym_indices):
            flag = "  <-- op 41/44" if isym in HIGHLIGHT_OPS else ""
            d    = chi_dft[j]
            t    = chi_table[j]
            diff = abs(d - t)
            print(
                f"  {isym:>{COL_SYM}}  "
                f"{fmt_c(d):<{COL_CHI}}  "
                f"{fmt_c(t):<{COL_CHI}}  "
                f"{diff:>{COL_DIFF}.3e}"
                f"{flag}"
            )

    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    tol  = args.tolerance

    print(f"Loading {args.json_file} ...")
    data = loadfn(args.json_file)

    sg_json   = data["spacegroup"]
    sg_number = str(sg_json["number"])
    spinor    = bool(sg_json["spinor"])
    magnetic  = bool(sg_json.get("magnetic", False))

    print(f"Space group: {sg_json.get('name', sg_number)} (#{sg_number})  "
          f"spinor={spinor}  magnetic={magnetic}")

    table   = IrrepTable(sg_number, spinor, magnetic=magnetic, v=0)
    kpoints = collect_kpoints(data)
    print(f"Found {len(kpoints)} k-point(s) in JSON.")

    # --- select k-point(s) to check ---
    if args.kpoint_index is not None:
        idx = args.kpoint_index
        if not (0 <= idx < len(kpoints)):
            print(f"ERROR: --kpoint-index {idx} out of range [0, {len(kpoints)-1}].")
            sys.exit(1)
        targets = [(idx, kpoints[idx])]
    else:
        w_indices = find_w_kpoints(kpoints)
        if len(w_indices) == 0:
            print("ERROR: No k-point with W-labeled irreps found.")
            print("Available k-points:")
            for i, kp in enumerate(kpoints):
                print(kpoint_summary(kp, i))
            sys.exit(1)
        if len(w_indices) > 1:
            print(f"ERROR: Multiple W k-points found at indices {w_indices}. "
                  "Use --kpoint-index N to select one.")
            for i in w_indices:
                print(kpoint_summary(kpoints[i], i))
            sys.exit(1)
        targets = [(w_indices[0], kpoints[w_indices[0]])]

    overall_pass = True
    for idx, kp in targets:
        k     = np.asarray(kp["k"])
        k_ref = np.asarray(kp["k_refUC"])
        print(f"\n{'='*74}")
        print(f"Checking k-point [{idx}]  "
              f"k_DFT={k.tolist()}  k_refUC={k_ref.tolist()}")
        print(f"Little-group op indices: {kp['symmetries']}")
        print(f"Tolerance: {tol:.1e}")

        ok = check_kpoint(kp, table, sg_json, tol)

        print(f"\n{'='*74}")
        print(f"K-point [{idx}]:  {'PASS' if ok else 'FAIL'}")
        if not ok:
            overall_pass = False

    print(f"\nOverall: {'PASS' if overall_pass else 'FAIL'}")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
