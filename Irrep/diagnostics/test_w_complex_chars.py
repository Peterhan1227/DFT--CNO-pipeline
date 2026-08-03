"""
test_w_complex_chars.py

Regression test: verify that the W-point of diamond Si (SG 227, Fd-3m) is
identified with integer multiplicities W1(1) and W2(1).

Before the patch in get_irreps_from_table() the bug returned W1(0.5)+W2(0.5)
for every twofold-degenerate block at W because the phase sign was wrong.

Run from the Irrep/ directory with:
    python diagnostics/test_w_complex_chars.py
"""

import os
import sys
import numpy as np

PKG = r"C:\Users\hanziruopeter\miniconda3\envs\irrep\Lib\site-packages"
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from irrep.bandstructure import BandStructure

IRREP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSCAR    = os.path.join(IRREP_DIR, "POSCAR")
WAVECAR   = os.path.join(IRREP_DIR, "WAVECAR_t")


def load_w_multiplicities():
    bs = BandStructure(
        fPOS=POSCAR,
        fWAV=WAVECAR,
        code="vasp",
        spinor=False,
        Ecut=300,
        IBstart=0,
        IBend=8,
        kplist=[2],              # W is the 3rd k-point (0-indexed = 2)
        search_cell=True,
        degen_thresh=7e-4,
        calculate_traces=True,
        irreps=True,
        verbosity=0,
        EF="0",
    )
    bs.identify_irreps(kpnames=["W"])
    w_kp = bs.kpoints[0]
    return w_kp.irreps   # list of dicts, one per degenerate block


def test_w_point_integer_multiplicities():
    irreps_per_block = load_w_multiplicities()

    print(f"W-point: {len(irreps_per_block)} degenerate block(s)")
    all_ok = True

    for ib, block_irreps in enumerate(irreps_per_block):
        print(f"  Block {ib}: {block_irreps}")
        for irname, m in block_irreps.items():
            m_real = float(m.real)
            m_imag = abs(float(m.imag))
            # multiplicity must be 1 (integer) with negligible imaginary part
            if abs(m_real - round(m_real)) > 1e-3 or m_imag > 1e-3:
                print(f"    FAIL: {irname} has non-integer multiplicity {m}")
                all_ok = False
            else:
                print(f"    OK  : {irname}({round(m_real)})")

    # also verify the specific physics: 4 doublets, each is purely W1 or W2
    w_names = {"W1", "W2"}
    for ib, block_irreps in enumerate(irreps_per_block):
        matched = {k: v for k, v in block_irreps.items() if abs(v.real) > 0.5}
        if len(matched) != 1:
            print(f"  FAIL Block {ib}: expected exactly one irrep, got {matched}")
            all_ok = False
        else:
            irname = list(matched.keys())[0]
            if irname not in w_names:
                print(f"  FAIL Block {ib}: unexpected irrep {irname}")
                all_ok = False

    if all_ok:
        print("\nPASS: All W-point blocks have integer multiplicity 1.")
    else:
        print("\nFAIL: W-point multiplicity bug present (check get_irreps_from_table patch).")
        sys.exit(1)


def test_gm_l_x_unchanged():
    """Verify GM, L, X give integer multiplicities (regression guard)."""
    bs = BandStructure(
        fPOS=POSCAR,
        fWAV=WAVECAR,
        code="vasp",
        spinor=False,
        Ecut=300,
        IBstart=0,
        IBend=8,
        kplist=[0, 1, 3],   # GM, L, X
        search_cell=True,
        degen_thresh=7e-4,
        calculate_traces=True,
        irreps=True,
        verbosity=0,
        EF="0",
    )
    bs.identify_irreps(kpnames=["GM", "L", "X"])

    print("\nGM / L / X integer-multiplicity check:")
    all_ok = True
    for kp, kpname in zip(bs.kpoints, ["GM", "L", "X"]):
        for ib, block_irreps in enumerate(kp.irreps):
            for irname, m in block_irreps.items():
                m_real = float(m.real)
                m_imag = abs(float(m.imag))
                if abs(m_real - round(m_real)) > 1e-2 or m_imag > 1e-2:
                    print(f"  FAIL {kpname} block {ib}: {irname}={m}")
                    all_ok = False
                else:
                    print(f"  OK   {kpname} block {ib}: {irname}({round(m_real)})")

    if all_ok:
        print("\nPASS: GM / L / X all have integer multiplicities.")
    else:
        print("\nFAIL: Non-integer multiplicity at GM / L / X (unexpected regression).")
        sys.exit(1)


if __name__ == "__main__":
    test_w_point_integer_multiplicities()
    test_gm_l_x_unchanged()
