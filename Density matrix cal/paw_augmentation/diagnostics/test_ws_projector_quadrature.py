"""Read-only tests for the weighted Wigner--Seitz pseudo projector.

This test is intentionally below ``paw_augmentation/diagnostics`` because it
tests the projector *construction*, before CNO fields are written.  It checks
both supported weighted maps:

* ``weighted_ties``: exact equal-distance boundary images receive 1/n weight;
* ``finite_volume``: sampling-lattice Voronoi voxels are clipped by the
  continuous W-centred WS polyhedron.

For WSe2 it additionally rebuilds the full pseudo-only spectrum and verifies
the [3,4] pair as a subspace.  It writes a new JSON report only; no CNO output
directory is read or modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAW_DIR = HERE.parent
PIPELINE_DIR = PAW_DIR.parent
sys.path[:0] = [str(PIPELINE_DIR), str(PIPELINE_DIR / "helper functions"),
                str(PAW_DIR), str(PAW_DIR / "helper functions"),
                str(PIPELINE_DIR / "symmetry")]

import cno_symmetry as symmetry  # noqa: E402
import config  # noqa: E402
import paw_regional_cno as regional  # noqa: E402
from ws_cell import read_poscar_structure  # noqa: E402


def _json(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    raise TypeError(type(value).__name__)


def _operation_maps(sys_, latvec, center_frac, generators):
    grid = np.array([sys_["Nx"], sys_["Ny"], sys_["Nz"]], dtype=int)
    actual = sys_["prim_indices"] + grid[None, :] * sys_["translations_all"]
    lookup = {tuple(row): i for i, row in enumerate(actual)}
    maps, report = {}, {}
    for name in generators:
        frac_matrix = np.linalg.inv(latvec.T) @ symmetry.R_CART[name] @ latvec.T
        frac_matrix = np.rint(frac_matrix).astype(int)
        mapped = np.rint(((sys_["r_ws_frac_cont"] - center_frac) @ frac_matrix.T + center_frac)
                          * grid[None, :]).astype(int)
        indices = np.array([lookup.get(tuple(row), -1) for row in mapped], dtype=int)
        missing = int(np.sum(indices < 0))
        max_weight_difference = (float(np.max(np.abs(sys_["sample_weights"] - sys_["sample_weights"][indices])))
                                 if missing == 0 else float("inf"))
        maps[name] = indices
        report[name] = {
            "missing_mapped_samples": missing,
            "max_weight_difference": max_weight_difference,
            "passed": bool(missing == 0 and max_weight_difference < 1.0e-8),
        }
    return maps, report


def _pseudo_spectrum(sys_):
    all_k = list(range(1, sys_["wfc"]._nkpts + 1))
    psi, _, p, states = regional.build_state_list_and_beta(sys_, [], all_k)
    sqrt_p = np.sqrt(p)
    gram = psi.conj().T @ (sys_["sample_weights"][:, None] * psi)
    k_ps = (sqrt_p[:, None] * gram) * sqrt_p[None, :]
    k_ps = 0.5 * (k_ps + k_ps.conj().T)
    occupations, vectors = np.linalg.eigh(k_ps)
    order = np.argsort(occupations)[::-1]
    occupations, vectors = occupations[order], vectors[:, order]
    selected = occupations > 1.0e-6
    occupations, vectors = occupations[selected], vectors[:, selected]
    fields = psi @ (sqrt_p[:, None] * vectors / np.sqrt(occupations)[None, :])
    return occupations, fields, len(states)


def _pair_leakage(fields, weights, mapping, pair):
    left = fields[:, pair] * np.sqrt(weights)[:, None]
    right = fields[mapping][:, pair] * np.sqrt(weights)[:, None]
    q, _ = np.linalg.qr(left, mode="reduced")
    residual = right - q @ (q.conj().T @ right)
    return np.sum(np.abs(residual) ** 2, axis=0) / np.sum(np.abs(right) ** 2, axis=0)


def run_one(method: str, factor: int):
    sys_ = regional.load_system(ws_quadrature=method, quadrature_factor=factor)
    data_dir = PIPELINE_DIR / "Data" / sys_["material"]
    latvec, _, _, symbols, _, frac, _ = read_poscar_structure(data_dir / "POSCAR")
    center = sys_["center_frac_wrapped"]
    generators = symmetry.GENERATORS_BY_MATERIAL[sys_["material"]]
    symmetry._validate_atomic_site_symmetry(latvec, symbols, frac, center, generators)
    maps, map_report = _operation_maps(sys_, latvec, center, generators)

    cell_volume = abs(float(np.linalg.det(latvec)))
    native_volume = cell_volume / sys_["native_nr"]
    result = {
        "method": method,
        "quadrature_factor": factor,
        "quadrature_grid": [sys_["Nx"], sys_["Ny"], sys_["Nz"]],
        "n_samples": sys_["Nr"],
        "weight_sum": float(sys_["sample_weights"].sum()),
        "integrated_volume": float(sys_["sample_weights"].sum() * native_volume),
        "exact_ws_volume": cell_volume,
        "volume_error": float(sys_["sample_weights"].sum() * native_volume - cell_volume),
        "n_fractional_weight_samples": int(np.sum((sys_["sample_weights"] > 1e-12)
                                                    & (sys_["sample_weights"] < 1.0 - 1e-12))),
        "map_symmetry": map_report,
    }
    occupations, fields, nstates = _pseudo_spectrum(sys_)
    result["n_states"] = nstates
    result["pseudo_occupations_top12"] = occupations[:12]
    result["pseudo_gaps_top11"] = occupations[:11] - occupations[1:12]
    if len(occupations) >= 7:
        blocks = {}
        for pair in ([1, 2], [3, 4], [5, 6]):
            blocks[str(pair)] = {
                "occupations": occupations[pair],
                "gap": float(occupations[pair[0]] - occupations[pair[1]]),
                "leakage": {name: _pair_leakage(fields, sys_["sample_weights"], mapping, pair)
                            for name, mapping in maps.items()},
            }
        result["pseudo_pair_subspaces"] = blocks
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["weighted_ties", "finite_volume"],
                        choices=("weighted_ties", "finite_volume"))
    parser.add_argument("--quadrature-factor", type=int, default=1)
    args = parser.parse_args(argv)
    if config.MATERIAL != "WSe2_mono":
        raise ValueError("This [3,4] regression test currently targets WSe2_mono; set config.py first.")
    report = {"material": config.MATERIAL,
              "results": [run_one(method, args.quadrature_factor) for method in args.methods]}
    out_dir = HERE / "output"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"test_ws_projector_quadrature_{config.MATERIAL}_f{args.quadrature_factor}.json"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic report: {out}")
    out.write_text(json.dumps(report, indent=2, default=_json), encoding="utf-8")
    for result in report["results"]:
        pair = result["pseudo_pair_subspaces"]["[3, 4]"]
        print(f"{result['method']}: samples={result['n_samples']} "
              f"volume_error={result['volume_error']:.3e} "
              f"[3,4] gap={pair['gap']:.3e}")
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
