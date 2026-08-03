"""Convergence study for the finite-volume regional PAW projector.

This is intentionally a diagnostic, not a production CNO writer.  It compares
``G_ps``, ``G_aug``, ``G_A``, occupations, and a *two-dimensional spectral
subspace projector* between zero-padded finite-volume quadrature meshes.
No existing CNO data are read or overwritten.

The default reduced k-point set is cheap and checks matrix convergence.  Use
``--full-spectrum`` only when memory is available: a factor-two WSe2 mesh
holds about eight times as many pseudo samples.  Only the full, symmetry-closed
k mesh gives a meaningful D3h [3,4] pair; this script refuses a pair-subspace
claim for the reduced, non-symmetry-closed validation subset.
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
                str(PAW_DIR), str(PAW_DIR / "helper functions")]

import config  # noqa: E402
import paw_regional_cno as regional  # noqa: E402


def _json(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _one_factor(factor, sites, full_spectrum):
    sys_ = regional.load_system(ws_quadrature="finite_volume", quadrature_factor=factor)
    kpoints = (list(range(1, sys_["wfc"]._nkpts + 1)) if full_spectrum
               else sys_["reduced_reference_kpoints"])
    psi, beta, p, states = regional.build_state_list_and_beta(sys_, sites, kpoints)
    gps, gaug, ga, ka = regional.build_G_A_K_A(
        psi, beta, sites, p, sample_weights=sys_["sample_weights"])
    occupation, vectors = np.linalg.eigh(ka)
    order = np.argsort(occupation)[::-1]
    occupation, vectors = occupation[order], vectors[:, order]
    return dict(
        factor=factor, n_samples=sys_["Nr"], quadrature_grid=[sys_["Nx"], sys_["Ny"], sys_["Nz"]],
        weight_sum=float(sys_["sample_weights"].sum()), n_states=len(states),
        gps=gps, gaug=gaug, ga=ga, occupation=occupation, vectors=vectors,
    )


def _relative_difference(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0e-30))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--full-spectrum", action="store_true")
    parser.add_argument("--pair-start", type=int, default=3,
                        help="Start index of the two-dimensional spectral cluster; meaningful only with --full-spectrum.")
    args = parser.parse_args(argv)
    factors = sorted(set(args.factors))
    if any(f < 1 for f in factors):
        raise ValueError("Factors must be positive integers")
    if not args.full_spectrum and args.pair_start != 3:
        raise ValueError("Pair subspace comparison requires --full-spectrum")

    # PAW augmentation is a continuous WS integral.  Build it once from the
    # factor-one geometry and reuse it at every pseudo quadrature factor.
    template = regional.load_system(ws_quadrature="finite_volume", quadrature_factor=1)
    sites = regional.precompute_sites_and_Q_A(template)
    rows = [_one_factor(factor, sites, args.full_spectrum) for factor in factors]
    reference = rows[-1]
    report_rows = []
    for row in rows:
        entry = {key: row[key] for key in ("factor", "n_samples", "quadrature_grid", "weight_sum", "n_states")}
        entry["occupations_top20"] = row["occupation"][:20]
        entry["relative_to_finest"] = {
            "G_ps": _relative_difference(row["gps"], reference["gps"]),
            "G_aug": _relative_difference(row["gaug"], reference["gaug"]),
            "G_A": _relative_difference(row["ga"], reference["ga"]),
            "occupations_top20": _relative_difference(row["occupation"][:20], reference["occupation"][:20]),
        }
        if args.full_spectrum:
            pair = [args.pair_start, args.pair_start + 1]
            projector = row["vectors"][:, pair] @ row["vectors"][:, pair].conj().T
            projector_ref = reference["vectors"][:, pair] @ reference["vectors"][:, pair].conj().T
            entry["pair_subspace"] = {
                "indices": pair,
                "occupations": row["occupation"][pair],
                "gap": float(row["occupation"][pair[0]] - row["occupation"][pair[1]]),
                "projector_difference_to_finest": float(np.linalg.norm(projector - projector_ref)),
            }
        report_rows.append(entry)

    report = dict(material=config.MATERIAL, full_spectrum=args.full_spectrum,
                  factors=factors, rows=report_rows,
                  note=("A two-dimensional pair projector, not individually labelled CNOs, is compared "
                        "when --full-spectrum is used."))
    out_dir = HERE / "output"
    out_dir.mkdir(exist_ok=True)
    tag = "full" if args.full_spectrum else "reduced"
    out = out_dir / f"regional_quadrature_convergence_{config.MATERIAL}_{tag}_f{'-'.join(map(str, factors))}.json"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic report: {out}")
    out.write_text(json.dumps(report, indent=2, default=_json), encoding="utf-8")
    for row in report_rows:
        print(f"f={row['factor']}: samples={row['n_samples']} "
              f"rel(G_ps)={row['relative_to_finest']['G_ps']:.3e} "
              f"rel(G_A)={row['relative_to_finest']['G_A']:.3e}")
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
