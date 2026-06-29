"""
diagnose_w_irrep.py

Standalone diagnostic for the W-point complex-character bug in the
installed `irrep` package.  Run from the Irrep/ directory with:

    python diagnose_w_irrep.py

It loads the Si VASP calculation, finds the W k-point, and prints:
  1. Calculated char and char_refUC per degenerate block
  2. Raw W1/W2 characters from IrrepTable
  3. Transformed W1/W2 from get_irreps_from_table()
  4. dt and phase correction per W little-group operation
  5. Inner-product / multiplicity matrix
  6. Full symm matrices from symm_matrix() and their traces
  7. symm_eigenvalues() for the same operations and blocks
  8. Comparison of trace(symm_matrix) vs symm_eigenvalues
"""

import os
import sys
import numpy as np

# ── locate package so this script can be run without install-time hacks ──
PKG = r"C:\Users\hanziruopeter\miniconda3\envs\irrep\Lib\site-packages"
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from irrep.bandstructure import BandStructure
from irrep.gvectors import symm_eigenvalues, symm_matrix
from irreptables.irreps import IrrepTable

# ── helpers ──────────────────────────────────────────────────────────────

def fmt(x):
    if np.iscomplex(x) or isinstance(x, complex):
        r, i = x.real, x.imag
        return f"{r:+.4f}{i:+.4f}j"
    return f"{float(x):+.4f}"

def fmt_vec(v):
    return "[" + ", ".join(fmt(x) for x in v) + "]"

def sep(title=""):
    w = 70
    if title:
        pad = (w - len(title) - 2) // 2
        print("\n" + "─" * pad + " " + title + " " + "─" * (w - pad - len(title) - 2))
    else:
        print("\n" + "─" * w)


# ── main ─────────────────────────────────────────────────────────────────

def main():
    irrep_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    poscar  = os.path.join(irrep_dir, "POSCAR")
    wavecar = os.path.join(irrep_dir, "WAVECAR_t")

    sep("Loading BandStructure")
    print("POSCAR  :", poscar)
    print("WAVECAR :", wavecar)

    bs = BandStructure(
        fPOS=poscar,
        fWAV=wavecar,
        code="vasp",
        spinor=False,
        Ecut=300,
        IBstart=0,
        IBend=8,
        kplist=[0, 1, 2, 3],        # 0-indexed: GM, L, W, X
        search_cell=True,
        degen_thresh=7e-4,
        calculate_traces=True,
        irreps=True,
        verbosity=0,
    )

    sg = bs.spacegroup
    refUC   = sg.refUC
    shiftUC = sg.shiftUC
    print(f"\nrefUC  =\n{refUC}")
    print(f"shiftUC = {shiftUC}")

    # set_irreptables was already called inside BandStructure.__init__ → parse_files
    # (sg.u_symmetries_tables and sg.symmetries are already populated)
    sym_tables = sg.symmetries   # ordered list, sym.ind == 1-based BCS index

    # ── find the W k-point ────────────────────────────────────────────────
    sep("Locating W k-point")
    kpoints = bs.kpoints
    print("k-points loaded:")
    for kp in kpoints:
        print(f"  ik={kp.ik0}  k={np.round(kp.k, 5)}")

    W_kp = None
    for kp in kpoints:
        if np.allclose(kp.k % 1, np.array([0.25, 0.75, 0.50]) % 1, atol=1e-3):
            W_kp = kp
            break
        # also try (0.25,0.5,0.75) or other equivalent representations
        for perm in [[0.25,0.75,0.5],[0.25,0.5,0.75],[0.75,0.25,0.5],
                     [0.5,0.25,0.75],[0.75,0.5,0.25],[0.5,0.75,0.25]]:
            if np.allclose(kp.k % 1, np.array(perm) % 1, atol=1e-3):
                W_kp = kp
                break
        if W_kp is not None:
            break

    if W_kp is None:
        # fall back: use the 3rd k-point (index 2) which should be W
        W_kp = kpoints[2]
        print(f"WARNING: W not found by coordinate; using ik=3, k={W_kp.k}")
    else:
        print(f"\nFound W:  ik={W_kp.ik0}  k={W_kp.k}")

    # ── little group and traces were computed during BandStructure init ───
    sep("Little group / traces (computed during BandStructure init)")
    print(f"Little-group size: {len(W_kp.little_group)}")
    print("Op indices:", [s.ind for s in W_kp.little_group])

    k_refUC = np.dot(refUC.T, W_kp.k)
    print(f"k (DFT cell)  : {W_kp.k}")
    print(f"k_refUC       : {k_refUC}")
    print(f"Degeneracies  : {W_kp.degeneracies}")
    print(f"Block indices : {W_kp.block_indices}")

    # ── Section 1: char and char_refUC ───────────────────────────────────
    sep("1. char and char_refUC (per block × sym)")
    op_inds = [s.ind for s in W_kp.little_group]
    print(f"Symmetry op indices: {op_inds}")
    for ib, (b1, b2) in enumerate(W_kp.block_indices):
        print(f"\n  Block {ib}  (bands {b1}–{b2-1},  deg={b2-b1}):")
        print(f"    char       = {fmt_vec(W_kp.char[ib])}")
        print(f"    char_refUC = {fmt_vec(W_kp.char_refUC[ib])}")

    # ── Section 2: raw table characters ──────────────────────────────────
    sep("2. Raw W-point irrep characters from IrrepTable")
    table = IrrepTable(sg.number_str, sg.spinor, magnetic=sg.magnetic, v=0)
    w_irreps = [irr for irr in table.irreps if irr.kpname == "W"]
    print(f"Number of W irreps in table: {len(w_irreps)}")
    for irr in w_irreps:
        raw = [irr.characters.get(i+1, 0) for i in range(len(sym_tables))]
        # only show ops in the little group
        little_raw = [irr.characters.get(ind, "---") for ind in op_inds]
        print(f"\n  {irr.name}:  k_table={irr.k}")
        print(f"    raw chars at little-group ops: {little_raw}")

    # ── Section 3: transformed table characters ───────────────────────────
    sep("3. Transformed W-point chars from get_irreps_from_table()")
    irreptable = sg.get_irreps_from_table("W", W_kp.k, verbosity=0)
    print(f"Irreps returned: {list(irreptable.keys())}")
    for irname, tab in irreptable.items():
        lg_chars = np.array([tab[ind] for ind in op_inds])
        print(f"\n  {irname}:  {fmt_vec(lg_chars)}")

    # ── Section 4: dt and phase per op ───────────────────────────────────
    sep("4. dt and phase corrections per W little-group operation")
    print(f"{'Op':>4}  {'dt (refUC)':>40}  {'dt·k_table':>12}  {'phase(get_irreps)':>18}  {'dt·k_refUC':>12}  {'phase(calc_traces)':>18}")
    # re-load table to get irr.k
    irr_k_table = None
    for irr in table.irreps:
        if irr.kpname == "W":
            irr_k_table = irr.k
            break
    for sym1, sym2 in zip(sg.symmetries, table.symmetries):
        if sym1.ind not in op_inds:
            continue
        dt = sym2.t - sym1.translation_refUC(refUC, shiftUC)
        phase_get = sym1.sign * np.exp(2j * np.pi * dt.dot(irr_k_table))
        dtk_tbl   = dt.dot(irr_k_table)
        dtk_ref   = dt.dot(k_refUC)
        phase_cal = sym1.sign * np.exp(-2j * np.pi * dtk_ref)
        print(f"  {sym1.ind:>3}  {np.round(dt,4)}  {dtk_tbl:+12.6f}  {fmt(phase_get):>18}  {dtk_ref:+12.6f}  {fmt(phase_cal):>18}")

    # ── Section 5: multiplicity matrix ───────────────────────────────────
    sep("5. Multiplicity matrix  dot(ir_chars, ch.conj()) / |G|  (current code)")
    print(f"{'':10}", end="")
    for irname in irreptable:
        print(f"  {irname:>12}", end="")
    print()
    for ib in range(len(W_kp.block_indices)):
        ch = W_kp.char[ib]
        print(f"  Block {ib}  ", end="")
        for irname, tab in irreptable.items():
            ir_chars = np.array([tab[ind] for ind in op_inds])
            normalization = (np.abs(ir_chars) ** 2).sum()
            multipl = np.dot(ir_chars, ch.conj()) / normalization
            print(f"  {fmt(multipl):>12}", end="")
        print()

    sep("5b. Multiplicity matrix  dot(ir_chars.conj(), ch) / |G|  (proposed fix)")
    print(f"{'':10}", end="")
    for irname in irreptable:
        print(f"  {irname:>12}", end="")
    print()
    for ib in range(len(W_kp.block_indices)):
        ch = W_kp.char[ib]
        print(f"  Block {ib}  ", end="")
        for irname, tab in irreptable.items():
            ir_chars = np.array([tab[ind] for ind in op_inds])
            normalization = (np.abs(ir_chars) ** 2).sum()
            multipl = np.dot(ir_chars.conj(), ch) / normalization
            print(f"  {fmt(multipl):>12}", end="")
        print()

    sep("5c. What get_irreps_from_table() SHOULD return: conj(raw_char)*exp(-2j*pi*dt·k)")
    for irr in w_irreps:
        fixed_chars = []
        for sym1, sym2 in zip(sg.symmetries, table.symmetries):
            if sym1.ind not in op_inds:
                continue
            if sym1.ind not in irr.characters:
                continue
            dt = sym2.t - sym1.translation_refUC(refUC, shiftUC)
            fixed_char = irr.characters[sym1.ind].conj() * sym1.sign * np.exp(-2j * np.pi * dt.dot(irr_k_table))
            fixed_chars.append(fixed_char)
        fixed_arr = np.array(fixed_chars)
        print(f"\n  {irr.name} (fixed ir_chars): {fmt_vec(fixed_arr)}")

    sep("5d. Multiplicity with fixed ir_chars")
    print(f"{'':10}", end="")
    for irr in w_irreps:
        print(f"  {irr.name:>12}", end="")
    print()
    for ib in range(len(W_kp.block_indices)):
        ch = W_kp.char[ib]
        print(f"  Block {ib}  ", end="")
        for irr in w_irreps:
            fixed_chars = []
            for sym1, sym2 in zip(sg.symmetries, table.symmetries):
                if sym1.ind not in op_inds:
                    continue
                if sym1.ind not in irr.characters:
                    continue
                dt = sym2.t - sym1.translation_refUC(refUC, shiftUC)
                fc = irr.characters[sym1.ind].conj() * sym1.sign * np.exp(-2j * np.pi * dt.dot(irr_k_table))
                fixed_chars.append(fc)
            fc_arr = np.array(fixed_chars)
            normalization = (np.abs(fc_arr) ** 2).sum()
            multipl = np.dot(fc_arr, ch.conj()) / normalization
            print(f"  {fmt(multipl):>12}", end="")
        print()

    # ── Section 6: symm_matrix traces ────────────────────────────────────
    sep("6. symm_matrix() per degenerate block at W")
    for sym in W_kp.little_group:
        print(f"\n  Op ind={sym.ind}  R={sym.rotation.tolist()}  T={sym.translation.tolist()}")
        for ib, (b1, b2) in enumerate(W_kp.block_indices):
            if b2 - b1 < 2:
                continue  # skip non-degenerate
            wf_block = W_kp.WF[b1:b2]
            ig_block = W_kp.ig
            blocks = symm_matrix(
                K=W_kp.k,
                WF=W_kp.WF,
                igall=W_kp.ig,
                A=sym.rotation,
                S=sym.spinor_rotation,
                T=sym.translation,
                spinor=W_kp.spinor,
                block_ind=W_kp.block_indices,
                return_blocks=True,
            )
            M = blocks[ib]
            tr = np.trace(M)
            print(f"    Block {ib} ({b2-b1}×{b2-b1})  matrix:\n{M}")
            print(f"    trace = {fmt(tr)}")

    # ── Section 7: symm_eigenvalues ───────────────────────────────────────
    sep("7. symm_eigenvalues() at W for each block")
    print(f"{'Op':>4}", end="")
    for ib, (b1, b2) in enumerate(W_kp.block_indices):
        print(f"  {'Block'+str(ib)+' (eig)':>22}", end="")
    print()
    for sym in W_kp.little_group:
        eigs = symm_eigenvalues(
            K=W_kp.k,
            WF=W_kp.WF,
            igall=W_kp.ig,
            A=sym.rotation,
            S=sym.spinor_rotation,
            T=sym.translation,
            spinor=W_kp.spinor,
        )
        print(f"  {sym.ind:>3}", end="")
        for ib, (b1, b2) in enumerate(W_kp.block_indices):
            vals = eigs[b1:b2]
            print(f"  {fmt_vec(vals):>22}", end="")
        print()

    # ── Section 8: trace(symm_matrix) vs symm_eigenvalues comparison ─────
    sep("8. trace(symm_matrix) vs symm_eigenvalues vs char (comparison)")
    print(f"{'Op':>4}  {'sum(symm_eig)':>18}  {'char[block]':>18}  {'match?':>7}")
    for sym in W_kp.little_group:
        eigs = symm_eigenvalues(
            K=W_kp.k,
            WF=W_kp.WF,
            igall=W_kp.ig,
            A=sym.rotation,
            S=sym.spinor_rotation,
            T=sym.translation,
            spinor=W_kp.spinor,
        )
        for ib, (b1, b2) in enumerate(W_kp.block_indices):
            s = eigs[b1:b2].sum()
            c = W_kp.char[ib, [s2.ind for s2 in W_kp.little_group].index(sym.ind)]
            match = "OK" if abs(s - c) < 1e-4 else "MISMATCH"
            print(f"  {sym.ind:>3}  block={ib}  sum_eig={fmt(s)}  char={fmt(c)}  {match}")

    sep("Done")


if __name__ == "__main__":
    main()
