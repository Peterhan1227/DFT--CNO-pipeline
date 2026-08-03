"""Trace CNO symmetry breaking to the pseudo-grid or PAW augmentation route.

It reconstructs the *pseudo-only* regional operator on exactly the same WS
samples as ``paw_regional_cno.py``:

    K_ps = sqrt(p) Psi^dagger Psi sqrt(p).

The natural orbitals of K_ps are then passed through the unified field
symmetry checker.  Comparing their block closure with the saved PAW-regional
CNOs isolates the source:

* pseudo-only CNOs also fail -> the discrete WS pseudo-grid is sufficient to
  cause the observed breaking;
* pseudo-only CNOs pass but saved PAW CNOs fail -> inspect Q_A / augmentation.

This is diagnostic only.  It reads WAVECAR/EIGENVAL and writes one JSON report
under ``symmetry/output``; it never changes the CNO output directory.

It also performs a *boundary-assignment control*: the same pseudo operator is
rebuilt after retaining only WS samples that are exact, complete cycles of
every point-group generator.  This is not a proposed production definition of
the regional projector (discarding a finite boundary layer would change its
quadrature).  Its sole purpose is to distinguish a physical/input failure
from the arbitrary representative chosen for a grid point lying on a
Wigner--Seitz boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from vaspwfc import vaspwfc

HERE = Path(__file__).resolve().parent
SYMMETRY_DIR = HERE.parent
PIPELINE_DIR = SYMMETRY_DIR.parent
PAW_DIR = PIPELINE_DIR / "paw_augmentation"
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(PIPELINE_DIR / "helper functions"))
sys.path.insert(0, str(PAW_DIR / "helper functions"))

import cno_symmetry as symmetry  # noqa: E402
from config import (EFERMI, FERMI_WINDOW_EV, ISPIN, MATERIAL, OUTPUT_SUBDIR,
                    RESTRICT_TO_FERMI_WINDOW, WS_CENTER, WS_CENTER_COORD_TYPE)  # noqa: E402
from beta_gauge_utils import read_eigenval_energies, read_eigenval_kweights  # noqa: E402
from realspace_beta import zero_pad_ifft  # noqa: E402
from ws_cell import read_poscar_structure  # noqa: E402


def _json(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    raise TypeError(type(value).__name__)


def _selection(wfc: vaspwfc, energies: np.ndarray | None, ik: int) -> tuple[np.ndarray, np.ndarray]:
    if RESTRICT_TO_FERMI_WINDOW:
        assert energies is not None
        bands = np.where(np.abs(energies[ik - 1] - EFERMI) <= FERMI_WINDOW_EV)[0] + 1
        return bands, np.ones(len(bands))
    occ = wfc._occs[ISPIN - 1, ik - 1, :]
    bands = np.where(occ > 1.0e-6)[0] + 1
    weight = occ[bands - 1].copy()
    if len(weight) and weight.max() > 1.5:
        weight /= 2.0
    return bands, weight


def _summarize(field: symmetry.FieldSymmetry, cutoff: float) -> list[dict]:
    checked = [int(i) for i in np.where(field.occ >= cutoff)[0]]
    initial = symmetry._clusters(checked, field.occ, 0.003, 0.004)
    recovery_args = argparse.Namespace(
        singleton_overlap=0.98,
        leakage_tol=0.05,
        unitary_tol=0.10,
        recovery_occ_atol=0.03,
        recovery_occ_rtol=0.03,
        recovery_coupling=0.20,
    )
    groups = symmetry._recover_singletons(initial, checked, field.occ, field, recovery_args)
    output = []
    for block in groups:
        metrics = field.block_metrics(block)
        output.append({
            "indices": block,
            "occupations": field.occ[block],
            "metrics": metrics,
        })
    return output


def _diagonalize_pseudo(psi: np.ndarray, p: np.ndarray, rows: np.ndarray,
                        sample_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize the pseudo regional Gram matrix using selected samples.

    ``x`` is evaluated on the complete stored WS grid even when ``rows`` is a
    diagnostic subset.  That lets the caller evaluate its symmetry action on
    precisely the same subset without changing the wavefunction definition.
    """
    sqrt_p = np.sqrt(p)
    gram = psi[rows].conj().T @ (sample_weights[rows, None] * psi[rows])
    k_ps = (sqrt_p[:, None] * gram) * sqrt_p[None, :]
    k_ps = 0.5 * (k_ps + k_ps.conj().T)
    eigvals, eigvecs = np.linalg.eigh(k_ps)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    selected = eigvals > 1.0e-6
    lam = eigvals[selected]
    y = sqrt_p[:, None] * eigvecs[:, selected] / np.sqrt(lam)[None, :]
    return lam, psi @ y


def _common_exact_cycles(field: symmetry.FieldSymmetry) -> np.ndarray:
    """Return samples closed under every exact generator action.

    The WS map has one representative per periodic grid point.  At a Voronoi
    boundary several representatives are equally valid, but the historical
    map picks one lexicographically.  ``FieldSymmetry`` already identifies
    complete cycles for an individual operation.  Their intersection is the
    unambiguous control set used only by this diagnostic.
    """
    rows = set(range(field.nr))
    source_maps: dict[str, dict[int, int]] = {}
    for name, op in field.ops.items():
        if op.method != "exact_lookup" or op.source_rows is None:
            raise ValueError(
                "Boundary-assignment control requires exact grid actions; "
                f"{name} uses {op.method}."
            )
        rows.intersection_update(int(i) for i in op.target_rows)
        source_maps[name] = {
            int(target): int(source)
            for target, source in zip(op.target_rows, op.source_rows)
        }
    out = np.array(sorted(rows), dtype=int)
    retained = set(int(i) for i in out)
    for name, mapping in source_maps.items():
        if any(mapping[int(row)] not in retained for row in out):
            raise RuntimeError(f"Common exact-cycle set is not closed under {name}")
    return out


def _closure_on_rows(field: symmetry.FieldSymmetry, x: np.ndarray,
                     rows: np.ndarray, n_orbitals: int = 12) -> dict:
    """Closure of singletons and adjacent pairs on a known closed sample set."""
    retained = set(int(i) for i in rows)
    result: dict[str, dict] = {}
    n = min(int(n_orbitals), x.shape[1])
    for name, op in field.ops.items():
        assert op.source_rows is not None
        mapping = {int(target): int(source)
                   for target, source in zip(op.target_rows, op.source_rows)}
        if any(int(row) not in mapping or mapping[int(row)] not in retained for row in rows):
            raise RuntimeError(f"Requested rows are not closed under {name}")
        sources = np.array([mapping[int(row)] for row in rows], dtype=int)
        sqrt_weight = np.sqrt(field.weights[rows])[:, None]
        target_values = x[rows, :n] * sqrt_weight
        transformed = x[sources, :n] * sqrt_weight
        singleton_leakage = []
        for i in range(n):
            overlap = symmetry._normalized_overlap(target_values[:, i], transformed[:, i])
            singleton_leakage.append(float(1.0 - overlap * overlap))
        adjacent_pair_leakage = []
        for i in range(n - 1):
            q, _ = np.linalg.qr(target_values[:, [i, i + 1]], mode="reduced")
            candidate = transformed[:, [i, i + 1]]
            residual = candidate - q @ (q.conj().T @ candidate)
            leak = np.sum(np.abs(residual) ** 2, axis=0) / np.sum(np.abs(candidate) ** 2, axis=0)
            adjacent_pair_leakage.append(leak)
        result[name] = {
            "singleton_leakage_first_n": np.array(singleton_leakage),
            "adjacent_pair_leakage_first_n_minus_1": adjacent_pair_leakage,
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", default=MATERIAL)
    parser.add_argument("--output-subdir", default=OUTPUT_SUBDIR)
    parser.add_argument("--occ-cutoff", type=float, default=0.03)
    args = parser.parse_args(argv)
    if args.material != MATERIAL:
        raise ValueError(
            "This trace follows config.py's band-selection semantics. Set config.py to the target "
            "material before using a material override. A different output_subdir is allowed."
        )

    data_dir = PIPELINE_DIR / "Data" / args.material
    output_dir = data_dir / "output" / args.output_subdir
    latvec, _, _, symbols, _, frac_coords, _ = read_poscar_structure(data_dir / "POSCAR")
    if (output_dir / "ws_center_frac_wrapped.npy").exists():
        center = np.load(output_dir / "ws_center_frac_wrapped.npy")
    elif WS_CENTER_COORD_TYPE == "fractional":
        center = np.asarray(WS_CENTER, dtype=float)
    else:
        raise FileNotFoundError("No saved fractional WS center")
    generators = symmetry.GENERATORS_BY_MATERIAL[args.material]
    symmetry._validate_atomic_site_symmetry(latvec, symbols, frac_coords, center, generators)

    field = symmetry.FieldSymmetry(output_dir, center, latvec, generators, 1, 0.5)
    saved_summary = _summarize(field, args.occ_cutoff)

    wfc = vaspwfc(str(data_dir / "WAVECAR"), lsorbit=False)
    source_grid_path = output_dir / "ws_source_fft_grid.npy"
    factor_path = output_dir / "ws_quadrature_factor.npy"
    source_grid = (np.load(source_grid_path).astype(int) if source_grid_path.exists()
                   else np.asarray(wfc._ngrid, dtype=int))
    factor = int(np.load(factor_path)) if factor_path.exists() else 1
    if tuple(wfc._ngrid) != tuple(source_grid) or tuple(field.nvec) != tuple(source_grid * factor):
        raise ValueError("WAVECAR/source grid/quadrature grid metadata disagree")
    weights = read_eigenval_kweights(data_dir / "EIGENVAL", wfc._nkpts, wfc._nbands)
    energies = (read_eigenval_energies(data_dir / "EIGENVAL", wfc._nkpts, wfc._nbands)
                if RESTRICT_TO_FERMI_WINDOW else None)

    all_states = []
    for ik in range(1, wfc._nkpts + 1):
        bands, occ = _selection(wfc, energies, ik)
        all_states.extend((ik, int(band), float(weights[ik - 1] * f)) for band, f in zip(bands, occ))
    p = np.array([item[2] for item in all_states])
    psi = np.empty((field.nr, len(all_states)), dtype=complex)
    base = np.load(output_dir / "ws_base_indices.npy").astype(np.int64)
    ny, nz = int(field.nvec[1]), int(field.nvec[2])
    base_flat = (base[:, 0] * ny + base[:, 1]) * nz + base[:, 2]

    per_k: dict[int, list[tuple[int, int]]] = {}
    for column, (ik, band, _) in enumerate(all_states):
        per_k.setdefault(ik, []).append((column, band))
    start = time.perf_counter()
    for done, (ik, pairs) in enumerate(per_k.items(), start=1):
        columns = [pair[0] for pair in pairs]
        bands = [pair[1] for pair in pairs]
        coefficients = np.stack([
            wfc.readBandCoeff(ispin=ISPIN, ikpt=ik, iband=band, norm=False)
            for band in bands
        ])
        u, padded_grid, _ = zero_pad_ifft(coefficients, wfc.gvectors(ik), factor, tuple(source_grid))
        if tuple(padded_grid) != tuple(field.nvec):
            raise RuntimeError("Pseudo trace padding grid disagrees with saved WS map")
        psi[:, columns] = (u[:, base_flat]
                           * np.exp(2j * np.pi * (field.positions @ wfc._kvecs[ik - 1]))[None, :]).T
        if done == 1 or done % 40 == 0 or done == len(per_k):
            print(f"  pseudo states: k {done}/{len(per_k)}")

    all_rows = np.arange(field.nr, dtype=int)
    lam, x_ps = _diagonalize_pseudo(psi, p, all_rows, field.weights)

    pseudo_field = symmetry.FieldSymmetry(output_dir, center, latvec, generators, 1, 0.5)
    pseudo_field.X = x_ps
    pseudo_field.occ = lam
    pseudo_summary = _summarize(pseudo_field, args.occ_cutoff)

    # This is a diagnosis of the WS-boundary representative convention, not
    # a replacement construction.  If the full-grid result fails but this
    # control restores exact multiplets/closure, the failure was introduced
    # by the finite-grid boundary assignment rather than PAW augmentation or
    # CNO-field interpolation.
    closed_rows = _common_exact_cycles(field)
    lam_closed, x_closed = _diagonalize_pseudo(psi, p, closed_rows, field.weights)
    boundary_control = {
        "purpose": (
            "Diagnostic only: discard points whose stored WS representative is not "
            "closed under every generator, then rebuild the identical pseudo operator."
        ),
        "n_full_ws_samples": field.nr,
        "n_common_exact_cycle_samples": len(closed_rows),
        "excluded_samples": field.nr - len(closed_rows),
        "excluded_fraction": 1.0 - len(closed_rows) / field.nr,
        "pseudo_cno_top20": lam_closed[:20],
        "spectral_gaps_top11": lam_closed[:11] - lam_closed[1:12],
        "closure_first12": _closure_on_rows(field, x_closed, closed_rows, n_orbitals=12),
    }
    elapsed = time.perf_counter() - start

    report = {
        "material": args.material,
        "output_subdir": args.output_subdir,
        "definition": "K_ps = sqrt(p) Psi^H Psi sqrt(p), no PAW augmentation",
        "n_states": len(all_states),
        "runtime_s": elapsed,
        "saved_paw_cno_top20": field.occ[:20],
        "pseudo_only_cno_top20": lam[:20],
        "saved_paw_blocks": saved_summary,
        "pseudo_only_blocks": pseudo_summary,
        "ws_boundary_assignment_control": boundary_control,
    }
    out_dir = SYMMETRY_DIR / "output"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"symmetry_source_trace_{args.material}_{args.output_subdir}.json"
    out.write_text(json.dumps(report, indent=2, default=_json), encoding="utf-8")
    print(f"Pseudo-only CNOs: {len(lam)} selected; trace={lam.sum():.8f}")
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
