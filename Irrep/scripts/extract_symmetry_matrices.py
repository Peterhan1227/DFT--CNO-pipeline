#!/usr/bin/env python3
"""
extract_symmetry_matrices.py — Stage 2 of the Si IrRep–CNO project.

Loads DFT wavefunctions via the patched IrRep package, identifies irreps at
configured high-symmetry k-points, and exports full symmetry-operation matrices
in a chosen basis.

Basis modes
-----------
  raw          Retain the VASP eigenstate basis.
  diagonalize  Diagonalise one chosen symmetry matrix per degenerate energy
               block and apply the same U to every other operation in that
               block.  (A future 'commuting_set' mode can be added to
               build_basis() without redesign.)

Usage
-----
  python scripts/extract_symmetry_matrices.py [config.yml] [overrides]
  python scripts/extract_symmetry_matrices.py --dump-default-config
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

_PKG = r"C:\Users\hanziruopeter\miniconda3\envs\irrep\Lib\site-packages"
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from irrep.bandstructure import BandStructure

# ─────────────────────────────────────────────── default configuration ────

DEFAULT_CONFIG: dict = {
    "poscar": "POSCAR",
    "wavecar": "WAVECAR_t",
    "ecut": 300.0,
    "ibstart": 1,              # 1-based inclusive
    "ibend": 8,                # 1-based inclusive
    "kpoints": [1, 2, 3, 4],  # 1-based; must match kpnames length
    "kpnames": ["GM", "L", "W", "X"],
    "degen_thresh": 7e-4,
    "ef": "0",
    "basis": {
        "mode": "diagonalize",       # "raw" | "diagonalize"
        "primary_symop": "auto",     # int (1-based BCS) | "auto"
        "eigenvalue_atol": 1e-3,     # threshold for degenerate-eigenvalue clusters
    },
    "occupied_bands": 4,             # bands 1..occupied_bands form the subspace
    "output": {
        "npz":    "output/symmetry_matrices.npz",
        "json":   "output/symmetry_matrices.json",
        "report": "output/symmetry_matrices_report.txt",
    },
    "validation": {
        "unitary_atol": 1e-5,
        "trace_atol":   2e-3,  # DFT-computed traces have ~1e-3 numerical noise
    },
}


# ──────────────────────────────────────────────────────── data classes ────

@dataclass
class Validation:
    unitary_raw:       bool = True
    trace_match:       bool = True
    trace_preserved:   bool = True
    unitary_U:         bool = True
    primary_diagonal:  bool = True
    dimension_ok:      bool = True

    def all_pass(self) -> bool:
        return all(vars(self).values())

    def failures(self) -> list[str]:
        return [k for k, v in vars(self).items() if not v]


@dataclass
class SymopResult:
    """All data for one symmetry operation in one degenerate energy block."""
    ind:          int
    rotation:     np.ndarray      # (3,3)
    translation:  np.ndarray      # (3,)
    D_raw:        np.ndarray      # (dim, dim)
    D_adapted:    np.ndarray      # (dim, dim); equals D_raw in "raw" mode
    trace_raw:    complex
    trace_adapted: complex
    char_irrep:   complex         # from kp.char
    is_primary:   bool
    validation:   Validation


@dataclass
class BlockResult:
    """All data for one degenerate energy block at one k-point."""
    kp_label:               str
    kp_index_1:             int            # 1-based user-facing index
    kp_k:                   np.ndarray
    kp_k_refUC:             np.ndarray
    block_index:            int            # 0-based within k-point
    bands_1based:           tuple[int, int]  # (first, last) 1-based inclusive
    energies_raw:           np.ndarray
    energy_mean:            float
    dimension:              int
    irreps:                 dict[str, complex]  # {label: multiplicity}
    primary_symop_ind:      int            # 1-based BCS index
    primary_eigenvalues:    np.ndarray
    U:                      np.ndarray     # (dim, dim); identity in "raw" mode
    residual_degeneracies:  bool
    symops:                 list[SymopResult]


# ─────────────────────────────────────────────────── basis construction ───

def _count_distinct(vals: np.ndarray, atol: float) -> int:
    """Count eigenvalue clusters separated by more than atol in |e^{iφ}| metric."""
    if len(vals) == 0:
        return 0
    phases = np.sort(np.angle(vals))
    circle = np.exp(1j * phases)
    diffs  = np.abs(np.diff(circle))
    wrap   = abs(circle[0] - circle[-1])
    return max(1, int(np.sum(np.append(diffs, wrap) > atol)))


def _sort_by_phase(
    vals: np.ndarray, vecs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(np.angle(vals))
    return vals[idx], vecs[:, idx]


def _ortho_within_clusters(
    vals: np.ndarray, vecs: np.ndarray, atol: float
) -> np.ndarray:
    """QR re-orthonormalize eigenvectors that share a numerically degenerate eigenvalue."""
    V, d = vecs.copy(), len(vals)
    i = 0
    while i < d:
        j = i + 1
        while j < d and abs(vals[j] - vals[i]) < atol:
            j += 1
        if j - i > 1:
            Q, _ = np.linalg.qr(V[:, i:j])
            V[:, i:j] = Q
        i = j
    return V


def _fix_phases(V: np.ndarray) -> np.ndarray:
    """Make the largest-|component| of each column real and positive."""
    V = V.copy()
    for c in range(V.shape[1]):
        idx = np.argmax(np.abs(V[:, c]))
        V[:, c] *= np.exp(-1j * np.angle(V[idx, c]))
    return V


def _build_U_diagonalize(
    D_primary: np.ndarray, eigenvalue_atol: float
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Diagonalise D_primary.  Returns (U, eigenvalues, has_residual_degeneracy).
    U satisfies  U†·D_primary·U ≈ diag(eigenvalues).
    Residual degeneracy is True when the primary op has fewer distinct
    eigenvalues than the block dimension (U then does not fully resolve it).
    """
    evals, evecs = np.linalg.eig(D_primary)
    evals, evecs = _sort_by_phase(evals, evecs)
    evecs  = _ortho_within_clusters(evals, evecs, eigenvalue_atol)
    evecs  = _fix_phases(evecs)
    residual = (_count_distinct(evals, eigenvalue_atol) < len(evals))
    return evecs, evals, residual


def build_basis(
    mode: str,
    D_primary: np.ndarray,
    eigenvalue_atol: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Dispatcher for basis construction modes.  A future 'commuting_set' mode
    (simultaneous diagonalisation of all mutually commuting operations) can be
    added here without changing anything else.

    Returns (U, primary_eigenvalues, has_residual_degeneracy).
    D_adapted = U†·D_raw·U for every operation in the block.
    """
    if mode == "raw":
        d = D_primary.shape[0]
        return np.eye(d, dtype=complex), np.diag(D_primary).copy(), False
    if mode == "diagonalize":
        return _build_U_diagonalize(D_primary, eigenvalue_atol)
    raise ValueError(
        f"Unknown basis mode {mode!r}.  Supported: 'raw', 'diagonalize'.  "
        "A 'commuting_set' mode can be added to build_basis()."
    )


# ───────────────────────────────── auto-select primary symmetry operation ─

def _auto_primary(
    raw_matrices: dict[int, np.ndarray],
    little_group: list,
    eigenvalue_atol: float,
) -> int:
    """
    Return 1-based BCS index of the non-identity little-group operation with the
    most distinct eigenvalues in this block (best at resolving the representation).
    Falls back to the first operation if every operation is identity-like.
    """
    identity_ind: Optional[int] = None
    for sym in little_group:
        if np.allclose(sym.rotation, np.eye(3)) and np.allclose(sym.translation, 0):
            identity_ind = sym.ind
            break

    best_ind, best_n = None, -1
    for sym in little_group:
        if sym.ind == identity_ind:
            continue
        n = _count_distinct(np.linalg.eigvals(raw_matrices[sym.ind]), eigenvalue_atol)
        if n > best_n:
            best_n, best_ind = n, sym.ind

    return best_ind if best_ind is not None else little_group[0].ind


# ──────────────────────────────────────────── validation helpers ──────────

def _is_unitary(M: np.ndarray, atol: float) -> bool:
    return bool(np.allclose(M.conj().T @ M, np.eye(len(M)), atol=atol))


def _is_diagonal(M: np.ndarray, atol: float) -> bool:
    return bool(np.allclose(M - np.diag(np.diag(M)), 0, atol=atol))


# ────────────────────────────────────────────── core block processing ─────

def process_block(
    kp,
    block_idx: int,
    kp_label: str,
    kp_index_1: int,
    config: dict,
) -> BlockResult:
    """
    Compute, transform, and validate all symmetry matrices for one degenerate
    energy block.

    CRITICAL: every operation in the block is transformed by the SAME U that
    was derived from the single chosen primary operation.
    """
    b1 = int(kp.block_indices[block_idx, 0])
    b2 = int(kp.block_indices[block_idx, 1])
    dim = b2 - b1

    mode       = config["basis"]["mode"]
    primary_cfg = config["basis"]["primary_symop"]
    ev_atol    = float(config["basis"].get("eigenvalue_atol", 1e-3))
    u_atol     = float(config["validation"].get("unitary_atol", 1e-5))
    tr_atol    = float(config["validation"].get("trace_atol", 2e-3))

    # ── compute raw symmetry matrices ─────────────────────────────────────
    # kp.symm_matrix returns a list of matrices (one per entry in block_indices).
    # We always pass a single block so result[0] is our matrix.
    raw: dict[int, np.ndarray] = {}
    for sym in kp.little_group:
        result = kp.symm_matrix(
            other=kp,
            symop=sym,
            block_indices=[(b1, b2)],
            unitary=True,
        )
        raw[sym.ind] = np.asarray(result[0], dtype=complex)

    # ── determine primary operation ───────────────────────────────────────
    if mode == "raw":
        primary_ind = kp.little_group[0].ind
    elif primary_cfg == "auto":
        primary_ind = _auto_primary(raw, kp.little_group, ev_atol)
    else:
        primary_ind = int(primary_cfg)
        if primary_ind not in raw:
            raise ValueError(
                f"primary_symop={primary_ind} not in little group of "
                f"k-point {kp_label!r}.  Available BCS indices: "
                f"{[s.ind for s in kp.little_group]}"
            )

    # ── build ONE basis transformation for this block ─────────────────────
    U, primary_evals, residual = build_basis(mode, raw[primary_ind], ev_atol)
    U_ok = _is_unitary(U, u_atol)

    # ── transform ALL operations with the same U ──────────────────────────
    symop_results: list[SymopResult] = []
    for i_sym, sym in enumerate(kp.little_group):
        D_r = raw[sym.ind]
        D_a = U.conj().T @ D_r @ U

        tr_r = complex(np.trace(D_r))
        tr_a = complex(np.trace(D_a))
        chi  = complex(kp.char[block_idx, i_sym])  # char sorted by little_group order

        val = Validation(
            unitary_raw=_is_unitary(D_r, u_atol),
            trace_match=abs(tr_r - chi) < tr_atol * max(dim, 1),
            trace_preserved=abs(tr_r - tr_a) < u_atol * max(dim, 1),
            unitary_U=U_ok,
            primary_diagonal=(
                _is_diagonal(D_a, tr_atol) if sym.ind == primary_ind else True
            ),
            dimension_ok=(D_r.shape == (dim, dim)),
        )

        symop_results.append(SymopResult(
            ind=sym.ind,
            rotation=sym.rotation.copy(),
            translation=sym.translation.copy(),
            D_raw=D_r,
            D_adapted=D_a,
            trace_raw=tr_r,
            trace_adapted=tr_a,
            char_irrep=chi,
            is_primary=(sym.ind == primary_ind),
            validation=val,
        ))

    # ── parse irreps dict for this block ──────────────────────────────────
    irreps_dict: dict[str, complex] = {}
    if hasattr(kp, "irreps") and kp.irreps:
        raw_irreps = kp.irreps[block_idx]
        if isinstance(raw_irreps, dict):
            irreps_dict = {k: complex(v) for k, v in raw_irreps.items()}

    return BlockResult(
        kp_label=kp_label,
        kp_index_1=kp_index_1,
        kp_k=kp.k.copy(),
        kp_k_refUC=kp.k_refUC.copy(),
        block_index=block_idx,
        bands_1based=(b1 + 1, b2),  # 1-based inclusive: first=b1+1, last=b2
        energies_raw=kp.Energy_raw[b1:b2].copy(),
        energy_mean=float(kp.Energy_mean[block_idx]),
        dimension=dim,
        irreps=irreps_dict,
        primary_symop_ind=primary_ind,
        primary_eigenvalues=primary_evals,
        U=U,
        residual_degeneracies=residual,
        symops=symop_results,
    )


# ──────────────────────────────── occupied subspace assembly ──────────────

def assemble_occupied_subspace(
    kp,
    kp_label: str,
    kp_index_1: int,
    block_results: list[BlockResult],
    n_occupied: int,
) -> dict:
    """
    Assemble block-diagonal matrices for each little-group operation spanning
    the first n_occupied bands.

    Only blocks that lie entirely within bands 1..n_occupied are included.
    Blocks straddling the occupied/unoccupied boundary are skipped with a
    warning (this should not happen for physically consistent degeneracy
    thresholds in well-gapped materials).

    Returns a dict with keys:
        kp_label, kp_index_1, dimension, band_range_1based,
        little_group_inds, D_raw, D_adapted
    or an empty dict if no occupied blocks were found.
    """
    occ_blocks = [
        br for br in block_results
        if br.bands_1based[0] <= n_occupied and br.bands_1based[1] <= n_occupied
    ]
    partial = [
        br for br in block_results
        if br.bands_1based[0] <= n_occupied < br.bands_1based[1]
    ]
    if partial:
        for br in partial:
            print(f"  WARNING: block {br.block_index} at {kp_label} straddles the "
                  f"occupied boundary (bands {br.bands_1based[0]}–{br.bands_1based[1]}, "
                  f"n_occupied={n_occupied}); excluded from subspace assembly.")

    if not occ_blocks:
        return {}

    total_dim = sum(br.dimension for br in occ_blocks)
    all_inds  = [s.ind for s in kp.little_group]

    D_raw_bd:     dict[int, np.ndarray] = {}
    D_adapted_bd: dict[int, np.ndarray] = {}

    for ind in all_inds:
        M_r = np.zeros((total_dim, total_dim), dtype=complex)
        M_a = np.zeros((total_dim, total_dim), dtype=complex)
        offset = 0
        for br in occ_blocks:
            d = br.dimension
            sym_map = {s.ind: s for s in br.symops}
            if ind in sym_map:
                M_r[offset:offset+d, offset:offset+d] = sym_map[ind].D_raw
                M_a[offset:offset+d, offset:offset+d] = sym_map[ind].D_adapted
            offset += d
        D_raw_bd[ind]     = M_r
        D_adapted_bd[ind] = M_a

    return {
        "kp_label":         kp_label,
        "kp_index_1":       kp_index_1,
        "dimension":        total_dim,
        "band_range_1based": (occ_blocks[0].bands_1based[0],
                               occ_blocks[-1].bands_1based[1]),
        "little_group_inds": all_inds,
        "D_raw":             D_raw_bd,
        "D_adapted":         D_adapted_bd,
    }


# ────────────────────────────────────────────────────── export helpers ────

def _c2p(z: complex) -> list[float]:
    return [float(z.real), float(z.imag)]


def _mat2p(M: np.ndarray) -> list:
    return [[_c2p(z) for z in row] for row in M.tolist()]


def _arr2p(arr: np.ndarray) -> list:
    return [_c2p(z) for z in arr.tolist()]


def _val_to_dict(v: Validation) -> dict:
    d = vars(v).copy()
    d["all_pass"] = v.all_pass()
    return d


# ────────────────────────────────────────────────────────── npz export ────

def export_npz(
    block_results: list[BlockResult],
    occupied: list[dict],
    path: Path,
) -> None:
    arrays: dict[str, np.ndarray] = {}

    for br in block_results:
        pfx = f"kp{br.kp_index_1}_{br.kp_label}_blk{br.block_index}"
        arrays[f"{pfx}_U"]             = br.U
        arrays[f"{pfx}_primary_evals"] = br.primary_eigenvalues
        arrays[f"{pfx}_energies_raw"]  = br.energies_raw
        for sr in br.symops:
            sp = f"{pfx}_sym{sr.ind}"
            arrays[f"{sp}_D_raw"]     = sr.D_raw
            arrays[f"{sp}_D_adapted"] = sr.D_adapted

    for occ in occupied:
        if not occ:
            continue
        pfx = f"occ_kp{occ['kp_index_1']}_{occ['kp_label']}"
        for ind, M in occ["D_raw"].items():
            arrays[f"{pfx}_sym{ind}_D_raw"] = M
        for ind, M in occ["D_adapted"].items():
            arrays[f"{pfx}_sym{ind}_D_adapted"] = M

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **arrays)
    print(f"  saved {path}  ({len(arrays)} arrays)")


# ───────────────────────────────────────────────────────── json export ────

def export_json(
    block_results: list[BlockResult],
    occupied: list[dict],
    config: dict,
    path: Path,
) -> None:
    # Group block results by k-point
    kp_groups: dict[int, list[BlockResult]] = {}
    for br in block_results:
        kp_groups.setdefault(br.kp_index_1, []).append(br)

    kp_list = []
    for kp_idx1, blocks in sorted(kp_groups.items()):
        br0 = blocks[0]
        kp_entry: dict = {
            "label":       br0.kp_label,
            "index_1based": kp_idx1,
            "k":           br0.kp_k.tolist(),
            "k_refUC":     br0.kp_k_refUC.tolist(),
            "blocks":      [],
        }
        for br in sorted(blocks, key=lambda x: x.block_index):
            block_entry: dict = {
                "block_index":          br.block_index,
                "bands_1based":         list(br.bands_1based),
                "energies_raw":         br.energies_raw.tolist(),
                "energy_mean":          br.energy_mean,
                "dimension":            br.dimension,
                "irreps":               {k: _c2p(v) for k, v in br.irreps.items()},
                "primary_symop_ind":    br.primary_symop_ind,
                "primary_eigenvalues":  _arr2p(br.primary_eigenvalues),
                "U":                    _mat2p(br.U),
                "residual_degeneracies": br.residual_degeneracies,
                "symmetry_operations":  [],
            }
            for sr in br.symops:
                block_entry["symmetry_operations"].append({
                    "ind":           sr.ind,
                    "rotation":      sr.rotation.tolist(),
                    "translation":   sr.translation.tolist(),
                    "is_primary":    sr.is_primary,
                    "D_raw":         _mat2p(sr.D_raw),
                    "D_adapted":     _mat2p(sr.D_adapted),
                    "trace_raw":     _c2p(sr.trace_raw),
                    "trace_adapted": _c2p(sr.trace_adapted),
                    "char_irrep":    _c2p(sr.char_irrep),
                    "validation":    _val_to_dict(sr.validation),
                })
            kp_entry["blocks"].append(block_entry)
        kp_list.append(kp_entry)

    occ_list = []
    for occ in occupied:
        if not occ:
            continue
        occ_list.append({
            "kp_label":        occ["kp_label"],
            "kp_index_1":      occ["kp_index_1"],
            "dimension":       occ["dimension"],
            "band_range_1based": list(occ["band_range_1based"]),
            "little_group_inds": occ["little_group_inds"],
            "D_raw":    {str(k): _mat2p(M) for k, M in occ["D_raw"].items()},
            "D_adapted": {str(k): _mat2p(M) for k, M in occ["D_adapted"].items()},
        })

    out = {"config": config, "kpoints": kp_list, "occupied_subspace": occ_list}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"  saved {path}")


# ──────────────────────────────────────────────────────── text report ─────

def _fmt_z(z: complex) -> str:
    return f"{z.real:+.5f}{z.imag:+.5f}j"


def _fmt_matrix(M: np.ndarray, indent: int = 6) -> str:
    pad = " " * indent
    rows = []
    for row in M:
        rows.append(pad + "  ".join(f"{z.real:+.4f}{z.imag:+.4f}j" for z in row))
    return "\n".join(rows)


def export_report(
    block_results: list[BlockResult],
    occupied: list[dict],
    config: dict,
    path: Path,
) -> None:
    lines: list[str] = []
    W = 78

    def add(s: str = "") -> None:
        lines.append(s)

    add("=" * W)
    add("SYMMETRY MATRICES REPORT")
    add(f"basis mode   : {config['basis']['mode']}")
    add(f"primary_symop: {config['basis']['primary_symop']}")
    add(f"occupied bands: {config.get('occupied_bands', 4)}")
    add("=" * W)

    kp_groups: dict[int, list[BlockResult]] = {}
    for br in block_results:
        kp_groups.setdefault(br.kp_index_1, []).append(br)

    for kp_idx1, blocks in sorted(kp_groups.items()):
        br0 = blocks[0]
        op_inds = [sr.ind for sr in br0.symops]
        add()
        add("─" * W)
        add(f"K-POINT [{kp_idx1}]  {br0.kp_label}")
        add(f"  k (DFT cell) = {np.round(br0.kp_k, 5).tolist()}")
        add(f"  k (refUC)    = {np.round(br0.kp_k_refUC, 5).tolist()}")
        add(f"  little-group op indices: {op_inds}")
        add("─" * W)

        for br in sorted(blocks, key=lambda x: x.block_index):
            irrep_str = "  ".join(
                f"{k}({v.real:.3f})" for k, v in br.irreps.items()
            )
            n_fail = sum(1 for sr in br.symops if not sr.validation.all_pass())
            block_status = "ALL PASS" if n_fail == 0 else f"{n_fail} FAIL(s)"

            add()
            add(f"  BLOCK {br.block_index}  "
                f"bands {br.bands_1based[0]}–{br.bands_1based[1]}  "
                f"dim={br.dimension}  E_mean={br.energy_mean:+.4f} eV  "
                f"[{block_status}]")
            add(f"  Irreps: {irrep_str or '(none)'}")
            add(f"  Primary symop: {br.primary_symop_ind}  "
                f"residual degeneracy: {br.residual_degeneracies}")

            eval_str = "  ".join(_fmt_z(e) for e in br.primary_eigenvalues)
            add(f"  Primary eigenvalues: {eval_str}")
            add(f"  U (basis transform):")
            add(_fmt_matrix(br.U))

            for sr in br.symops:
                primary_tag = "  [PRIMARY]" if sr.is_primary else ""
                fail_str = (
                    ""
                    if sr.validation.all_pass()
                    else f"  FAILED: {sr.validation.failures()}"
                )
                add()
                add(f"    sym {sr.ind}{primary_tag}"
                    f"  R={sr.rotation.tolist()}"
                    f"  T={np.round(sr.translation, 4).tolist()}{fail_str}")
                add(f"      trace_raw={_fmt_z(sr.trace_raw)}  "
                    f"trace_adapted={_fmt_z(sr.trace_adapted)}  "
                    f"char_irrep={_fmt_z(sr.char_irrep)}")
                val_line = "  ".join(
                    f"{k}={'OK' if v else 'FAIL'}"
                    for k, v in vars(sr.validation).items()
                )
                add(f"      validation: {val_line}")
                add(f"      D_raw:")
                add(_fmt_matrix(sr.D_raw, indent=8))
                add(f"      D_adapted:")
                add(_fmt_matrix(sr.D_adapted, indent=8))

    # Occupied subspace
    for occ in occupied:
        if not occ:
            continue
        add()
        add("=" * W)
        add(f"OCCUPIED SUBSPACE  [{occ['kp_index_1']}] {occ['kp_label']}")
        add(f"  bands {occ['band_range_1based'][0]}–{occ['band_range_1based'][1]}"
            f"  total dim={occ['dimension']}")
        for ind in occ["little_group_inds"]:
            add()
            add(f"  sym {ind}  D_raw (block-diagonal {occ['dimension']}×{occ['dimension']}):")
            add(_fmt_matrix(occ["D_raw"][ind], indent=4))
            add(f"  sym {ind}  D_adapted:")
            add(_fmt_matrix(occ["D_adapted"][ind], indent=4))

    add()
    add("=" * W)
    add("END OF REPORT")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  saved {path}")


# ──────────────────────────────────────────────── configuration loading ───

def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[Path]) -> dict:
    if path is None:
        return _deep_merge({}, DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    return _deep_merge(DEFAULT_CONFIG, user)


def validate_config(cfg: dict) -> None:
    mode = cfg["basis"]["mode"]
    if mode not in {"raw", "diagonalize"}:
        raise ValueError(
            f"basis.mode={mode!r} unsupported.  Supported: 'raw', 'diagonalize'."
        )
    ib1, ib2 = cfg["ibstart"], cfg["ibend"]
    if ib1 < 1:
        raise ValueError(f"ibstart must be ≥ 1 (got {ib1})")
    if ib2 < ib1:
        raise ValueError(f"ibend ({ib2}) must be ≥ ibstart ({ib1})")
    kpts, knames = cfg["kpoints"], cfg["kpnames"]
    if len(kpts) != len(knames):
        raise ValueError(
            f"kpoints ({len(kpts)}) and kpnames ({len(knames)}) must have the "
            "same length."
        )
    n_occ = cfg.get("occupied_bands", 4)
    n_bands = ib2 - ib1 + 1
    if n_occ > n_bands:
        raise ValueError(
            f"occupied_bands={n_occ} exceeds loaded band count {n_bands} "
            f"(ibstart={ib1}, ibend={ib2})."
        )


# ─────────────────────────────────────────────────────────────── main ─────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config", nargs="?", type=Path, default=None,
        help="YAML config file (optional; all keys default to Si project values)",
    )
    parser.add_argument(
        "--dump-default-config", action="store_true",
        help="Print the default YAML config and exit",
    )
    parser.add_argument("--poscar",  type=str,   help="Override POSCAR path")
    parser.add_argument("--wavecar", type=str,   help="Override WAVECAR path")
    parser.add_argument("--ecut",    type=float, help="Override Ecut (eV)")
    parser.add_argument("--ibstart", type=int,   help="Override ibstart (1-based)")
    parser.add_argument("--ibend",   type=int,   help="Override ibend (1-based)")
    parser.add_argument(
        "--mode", choices=["raw", "diagonalize"], help="Override basis mode"
    )
    parser.add_argument(
        "--primary-symop",
        help="Override primary_symop: 1-based BCS integer or 'auto'",
    )
    args = parser.parse_args()

    if args.dump_default_config:
        print(yaml.dump(DEFAULT_CONFIG, sort_keys=False, default_flow_style=False))
        return

    cfg = load_config(args.config)
    if args.poscar:       cfg["poscar"]           = args.poscar
    if args.wavecar:      cfg["wavecar"]          = args.wavecar
    if args.ecut is not None: cfg["ecut"]         = args.ecut
    if args.ibstart is not None: cfg["ibstart"]   = args.ibstart
    if args.ibend is not None:   cfg["ibend"]     = args.ibend
    if args.mode:         cfg["basis"]["mode"]    = args.mode
    if args.primary_symop:
        ps = args.primary_symop
        cfg["basis"]["primary_symop"] = int(ps) if ps != "auto" else "auto"

    validate_config(cfg)

    IBstart_0 = cfg["ibstart"] - 1          # BandStructure uses 0-based
    IBend_0   = cfg["ibend"]                # BandStructure IBend is exclusive
    kplist_0  = [k - 1 for k in cfg["kpoints"]]
    kpnames   = list(cfg["kpnames"])
    n_occ     = int(cfg.get("occupied_bands", 4))

    print("\n── Loading BandStructure ──────────────────────────────────────────────")
    print(f"  POSCAR  : {cfg['poscar']}")
    print(f"  WAVECAR : {cfg['wavecar']}")
    print(f"  Ecut={cfg['ecut']} eV  bands [{cfg['ibstart']}, {cfg['ibend']}]  "
          f"kpoints {cfg['kpoints']} → {kpnames}")

    bs = BandStructure(
        fPOS=cfg["poscar"],
        fWAV=cfg["wavecar"],
        code="vasp",
        spinor=False,
        Ecut=float(cfg["ecut"]),
        IBstart=IBstart_0,
        IBend=IBend_0,
        kplist=kplist_0,
        search_cell=True,
        degen_thresh=float(cfg["degen_thresh"]),
        EF=str(cfg["ef"]),
        calculate_traces=True,
        save_wf=True,
        irreps=True,
        verbosity=0,
    )

    print("\n── Identifying irreps ─────────────────────────────────────────────────")
    bs.identify_irreps(kpnames=kpnames)

    mode = cfg["basis"]["mode"]
    print(f"\n── Computing symmetry matrices (mode={mode!r}) ─────────────────────")

    all_blocks:   list[BlockResult] = []
    occupied_subs: list[dict]       = []

    for kp, kpname, kp_idx1 in zip(bs.kpoints, kpnames, cfg["kpoints"]):
        n_blocks = len(kp.block_indices)
        op_inds  = [s.ind for s in kp.little_group]
        print(f"\n  [{kp_idx1}] {kpname}  k={np.round(kp.k, 4).tolist()}"
              f"  blocks={n_blocks}  |lg|={len(kp.little_group)}  ops={op_inds}")

        kp_blocks: list[BlockResult] = []
        for block_idx in range(n_blocks):
            br = process_block(kp, block_idx, kpname, kp_idx1, cfg)
            kp_blocks.append(br)
            all_blocks.append(br)

            irrep_str = "  ".join(
                f"{k}({v.real:.3f})" for k, v in br.irreps.items()
            )
            n_fail  = sum(1 for sr in br.symops if not sr.validation.all_pass())
            status  = "OK" if n_fail == 0 else f"{n_fail} FAIL(s)"
            print(f"    blk {block_idx}  bands {br.bands_1based[0]}-{br.bands_1based[1]}"
                  f"  dim={br.dimension}  E={br.energy_mean:+.4f} eV"
                  f"  primary={br.primary_symop_ind}"
                  f"  resid_degen={br.residual_degeneracies}"
                  f"  {irrep_str or '(none)'}  [{status}]")

        occ = assemble_occupied_subspace(kp, kpname, kp_idx1, kp_blocks, n_occ)
        occupied_subs.append(occ)
        if occ:
            print(f"    occupied subspace: dim={occ['dimension']}"
                  f"  bands {occ['band_range_1based'][0]}-{occ['band_range_1based'][1]}")

    # ── global validation summary ─────────────────────────────────────────
    n_total = sum(len(br.symops) for br in all_blocks)
    n_pass  = sum(
        1 for br in all_blocks for sr in br.symops if sr.validation.all_pass()
    )
    print(f"\n── Validation: {n_pass}/{n_total} checks passed ───────────────────────")
    if n_pass < n_total:
        for br in all_blocks:
            for sr in br.symops:
                if not sr.validation.all_pass():
                    print(f"  FAIL  {br.kp_label} blk{br.block_index} sym{sr.ind}: "
                          f"{sr.validation.failures()}")

    # ── export ────────────────────────────────────────────────────────────
    print("\n── Exporting ──────────────────────────────────────────────────────────")
    export_npz(all_blocks, occupied_subs, Path(cfg["output"]["npz"]))
    export_json(all_blocks, occupied_subs, cfg, Path(cfg["output"]["json"]))
    export_report(all_blocks, occupied_subs, cfg, Path(cfg["output"]["report"]))

    print("\n── Done ───────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
