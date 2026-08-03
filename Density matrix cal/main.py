"""Build regional CNOs from the pseudo WAVECAR field or its PAW metric.

This is the public entry point for both production paths.  They share the
same band selection, Bloch-field evaluation, finite-volume Wigner--Seitz
quadrature and saved-output contract.  The sole physical difference is the
metric used for the regional Gram matrix:

* pseudo: ``G_A = <psi_tilde|P_A|psi_tilde>``;
* PAW:    the same pseudo term plus the regional PAW augmentation term.

Set ``USE_PAW_AUGMENTATION`` in ``config.py`` (or pass ``--augmentation
paw``) only when the chosen regional boundary intersects a PAW augmentation
sphere.  The inexpensive geometry pre-check is
``check_paw_augmentation_needed.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from vaspwfc import vaspwfc

import config

HERE = Path(__file__).resolve().parent
HELPERS = HERE / "helper functions"
PAW_DIR = HERE / "paw_augmentation"
sys.path.insert(0, str(HELPERS))

from plane_wave import bloch_fields_on_samples  # noqa: E402
from ws_cell import (  # noqa: E402
    build_ws_finite_volume_map,
    build_ws_grid_map,
    build_ws_weighted_tie_map,
    parse_ws_center,
    read_poscar_structure,
)


OCC_TOL = 1.0e-6


def _read_eigenval(path: Path, nkpts_expected: int, nbands_expected: int):
    """Read the BZ k-points, normalized weights, and band energies."""
    with path.open(encoding="utf-8") as handle:
        lines = handle.readlines()
    if len(lines) < 7:
        raise ValueError("EIGENVAL is too short")
    nkpts = int(lines[5].split()[1])
    nbands = int(lines[5].split()[2])
    if (nkpts, nbands) != (nkpts_expected, nbands_expected):
        raise ValueError(
            f"EIGENVAL has ({nkpts} k-points, {nbands} bands), but WAVECAR "
            f"has ({nkpts_expected}, {nbands_expected})."
        )
    kfrac = np.empty((nkpts, 3), dtype=float)
    kweights = np.empty(nkpts, dtype=float)
    energies = np.empty((nkpts, nbands), dtype=float)
    line = 6
    for ik in range(nkpts):
        while line < len(lines) and not lines[line].split():
            line += 1
        if line >= len(lines):
            raise ValueError("EIGENVAL ended before all k-points were read")
        fields = lines[line].split()
        kfrac[ik] = [float(value) for value in fields[:3]]
        kweights[ik] = float(fields[3])
        line += 1
        for ib in range(nbands):
            energies[ik, ib] = float(lines[line].split()[1])
            line += 1
    if not np.isfinite(kweights).all() or kweights.sum() <= 0.0:
        raise ValueError("EIGENVAL contains invalid k-point weights")
    return kfrac, kweights / kweights.sum(), energies


def _load_kpoint_data(wfc, data_dir: Path):
    """Load reliable k-point coordinates/weights, retaining old fallbacks."""
    eigenval = data_dir / "EIGENVAL"
    mismatch_note = None
    if eigenval.exists():
        try:
            kfrac, kweights, energies = _read_eigenval(
                eigenval, wfc._nkpts, wfc._nbands,
            )
            return kfrac, kweights, energies, "EIGENVAL", "EIGENVAL", mismatch_note
        except ValueError as exc:
            mismatch_note = str(exc)

    kfrac = None
    for attr in ("_kvecs", "_kpts", "kvecs", "kpts"):
        value = getattr(wfc, attr, None)
        if value is not None and np.asarray(value).shape == (wfc._nkpts, 3):
            kfrac = np.asarray(value, dtype=float)
            kcoord_source = f"vaspwfc.{attr}"
            break
    if kfrac is None:
        raise RuntimeError(
            "No fractional k-point coordinates found. Provide a matching EIGENVAL "
            "or use a vaspwfc build exposing _kvecs/_kpts."
        )

    for attr in ("_kweights", "_kwhts", "_weights", "kweights", "kwhts", "weights"):
        value = getattr(wfc, attr, None)
        if value is not None and np.asarray(value).shape == (wfc._nkpts,):
            weights = np.asarray(value, dtype=float)
            if np.isfinite(weights).all() and weights.sum() > 0.0:
                return kfrac, weights / weights.sum(), None, kcoord_source, f"vaspwfc.{attr}", mismatch_note

    weights = np.ones(wfc._nkpts, dtype=float) / wfc._nkpts
    return kfrac, weights, None, kcoord_source, "uniform fallback", mismatch_note


def _select_bands(wfc, ik: int, *, ispin: int, restrict_to_fermi_window: bool,
                  energies, efermi, fermi_window_ev):
    """The one band-selection rule used by the direct pseudo path."""
    if restrict_to_fermi_window:
        if energies is None:
            raise RuntimeError(
                "RESTRICT_TO_FERMI_WINDOW=True requires a matching EIGENVAL with band energies."
            )
        keep = np.abs(energies[ik - 1] - efermi) <= fermi_window_ev
        bands = np.where(keep)[0] + 1
        return bands, np.ones(len(bands), dtype=float)
    occupations = np.asarray(wfc._occs[ispin - 1, ik - 1], dtype=float)
    bands = np.where(occupations > OCC_TOL)[0] + 1
    occ = occupations[bands - 1].copy()
    # Per-spatial-orbital CNO occupations are conventionally in [0, 1].
    if len(occ) and occ.max() > 1.5:
        occ *= 0.5
    return bands, occ


def _regular_sample_map(grid_shape):
    """A full primitive-cell quadrature when no WS restriction is requested."""
    grid = np.asarray(grid_shape, dtype=int)
    indices = np.stack(np.meshgrid(*(np.arange(n) for n in grid), indexing="ij"), axis=-1)
    indices = indices.reshape(-1, 3)
    frac = indices / grid[None, :]
    return indices, np.zeros_like(indices), frac, np.ones(len(indices), dtype=float)


def _build_quadrature(latvec, source_grid, *, use_ws_cell: bool, center, center_coord_type,
                      nmax: int, method: str, factor: int):
    factor = int(factor)
    if factor < 1:
        raise ValueError("WS_QUADRATURE_FACTOR must be a positive integer")
    grid = np.asarray(source_grid, dtype=int) * factor
    if not use_ws_cell:
        base, translations, frac, weights = _regular_sample_map(grid)
        return dict(
            grid=grid, source_grid=np.asarray(source_grid, dtype=int), factor=factor,
            base_indices=base, translations=translations, points_frac_cont=frac,
            points_cart=frac @ latvec, weights=weights, method="regular_fft_grid",
            center_cart=None, center_frac_wrapped=None,
        )

    center_cart, _, center_frac_wrapped = parse_ws_center(center, center_coord_type, latvec)
    if method == "finite_volume":
        points_cart, points_frac, base, translations, weights = build_ws_finite_volume_map(
            latvec, grid, center_cart, nmax=nmax,
        )
    elif method == "weighted_ties":
        points_cart, points_frac, base, translations, weights, _ = build_ws_weighted_tie_map(
            latvec, grid, center_cart, nmax=nmax,
        )
    elif method == "legacy":
        points_cart, points_frac, base, translations = build_ws_grid_map(
            latvec, grid, center_cart, nmax=nmax,
        )
        weights = np.ones(len(base), dtype=float)
    else:
        raise ValueError("WS quadrature must be 'finite_volume', 'weighted_ties', or 'legacy'.")
    expected = int(np.prod(grid))
    if not np.isclose(weights.sum(), expected, rtol=0.0, atol=2.0e-7 * expected):
        raise RuntimeError("WS quadrature weights do not preserve one primitive-cell volume")
    return dict(
        grid=grid, source_grid=np.asarray(source_grid, dtype=int), factor=factor,
        base_indices=base, translations=translations, points_frac_cont=points_frac,
        points_cart=points_cart, weights=weights, method=method,
        center_cart=center_cart, center_frac_wrapped=center_frac_wrapped,
    )


def _build_pseudo_state_matrix(wfc, quadrature, kfrac_all, kweights, *, lsorbit: bool,
                               ispin: int, restrict_to_fermi_window: bool,
                               energies, efermi, fermi_window_ev):
    """Evaluate every included pseudo Bloch state on the shared WS samples."""
    by_k = []
    n_columns = 0
    for ik in range(1, wfc._nkpts + 1):
        bands, occupations = _select_bands(
            wfc, ik, ispin=ispin, restrict_to_fermi_window=restrict_to_fermi_window,
            energies=energies, efermi=efermi, fermi_window_ev=fermi_window_ev,
        )
        if len(bands):
            by_k.append((ik, bands, occupations))
            n_columns += len(bands) * (2 if lsorbit else 1)
    if n_columns == 0:
        raise RuntimeError("No WAVECAR bands were selected for the CNO density matrix")

    n_samples = len(quadrature["weights"])
    psi = np.empty((n_samples, n_columns), dtype=np.complex128)
    weights = np.empty(n_columns, dtype=float)
    states = []
    col = 0
    for count, (ik, bands, occupations) in enumerate(by_k, start=1):
        gvec = wfc.gvectors(ik)
        n_g = len(gvec)
        # ``norm=True`` is the direct-pseudo convention used by the original
        # main.py: every input state has unit norm in the pseudo metric.  The
        # PAW path deliberately uses raw coefficients instead, because its
        # augmentation term supplies the corresponding PAW norm.
        coeff = np.stack([
            wfc.readBandCoeff(ispin=ispin, ikpt=ik, iband=int(ib), norm=True)
            for ib in bands
        ])
        kfrac = kfrac_all[ik - 1]
        common = dict(
            gvec=gvec, source_grid=quadrature["source_grid"], grid_factor=quadrature["factor"],
            base_indices=quadrature["base_indices"],
            points_frac_cont=quadrature["points_frac_cont"], k_frac=kfrac,
        )
        state_weight = kweights[ik - 1] * occupations
        if lsorbit:
            if coeff.shape[1] != 2 * n_g:
                raise RuntimeError(
                    f"LSORBIT=True but k-point {ik} has {coeff.shape[1]} coefficients per band, "
                    f"expected 2*{n_g}."
                )
            for component, component_coeff in enumerate((coeff[:, :n_g], coeff[:, n_g:])):
                fields = bloch_fields_on_samples(component_coeff, **common)
                stop = col + len(bands)
                psi[:, col:stop] = fields.T
                weights[col:stop] = state_weight
                states.extend(dict(ik=ik, band=int(ib), p=float(p), spin_component=component)
                              for ib, p in zip(bands, state_weight))
                col = stop
        else:
            if coeff.shape[1] != n_g:
                raise RuntimeError(
                    f"Scalar WAVECAR coefficient count ({coeff.shape[1]}) does not match its G-vector count ({n_g})."
                )
            fields = bloch_fields_on_samples(coeff, **common)
            stop = col + len(bands)
            psi[:, col:stop] = fields.T
            weights[col:stop] = state_weight
            states.extend(dict(ik=ik, band=int(ib), p=float(p)) for ib, p in zip(bands, state_weight))
            col = stop
        if count == 1 or count % 40 == 0 or count == len(by_k):
            print(f"  k {count:4d}/{len(by_k)}  ik={ik:4d}  bands={len(bands)}")
    return psi, weights, states


def _solve_pseudo_cnos(psi, state_weights, sample_weights):
    """Diagonalize the weighted pseudo regional density operator in state space."""
    sample_weights = np.asarray(sample_weights, dtype=float)
    gram = psi.conj().T @ (sample_weights[:, None] * psi)
    gram = 0.5 * (gram + gram.conj().T)
    sqrt_p = np.sqrt(state_weights)
    kernel = (sqrt_p[:, None] * gram) * sqrt_p[None, :]
    kernel = 0.5 * (kernel + kernel.conj().T)
    eigvals, vectors = np.linalg.eigh(kernel)
    order = np.argsort(eigvals)[::-1]
    eigvals, vectors = eigvals[order], vectors[:, order]
    keep = eigvals > OCC_TOL
    if not np.any(keep):
        raise RuntimeError("No positive CNO occupation exceeded the storage tolerance")
    occupations = eigvals[keep]
    coefficients = sqrt_p[:, None] * vectors[:, keep] / np.sqrt(occupations)[None, :]
    cnos = psi @ coefficients
    weighted_overlap = cnos.conj().T @ (sample_weights[:, None] * cnos)
    return occupations, cnos, gram, kernel, weighted_overlap


def _write_pseudo_output(output_dir: Path, *, material: str, quadrature, cnos, occupations,
                         kpoint_weights, states, report, use_ws_cell: bool):
    """Write exactly the output contract consumed by all downstream tools."""
    np.save(output_dir / "cno_occupations.npy", occupations)
    np.save(output_dir / "cno_orbitals.npy", cnos)
    np.save(output_dir / "fft_grid_shape.npy", quadrature["grid"])
    np.save(output_dir / "kpoint_weights.npy", kpoint_weights)
    np.save(output_dir / "ws_enabled.npy", np.array(use_ws_cell))
    np.save(output_dir / "ws_points_cart.npy", quadrature["points_cart"])
    np.save(output_dir / "ws_points_frac_cont.npy", quadrature["points_frac_cont"])
    np.save(output_dir / "ws_base_indices.npy", quadrature["base_indices"])
    np.save(output_dir / "ws_translation_int.npy", quadrature["translations"])
    np.save(output_dir / "ws_quadrature_weights.npy", quadrature["weights"])
    np.save(output_dir / "ws_native_grid_count.npy", np.array(int(np.prod(quadrature["grid"])), dtype=int))
    np.save(output_dir / "ws_source_fft_grid.npy", quadrature["source_grid"])
    np.save(output_dir / "ws_quadrature_factor.npy", np.array(quadrature["factor"], dtype=int))
    if quadrature["center_cart"] is not None:
        np.save(output_dir / "ws_center_cart.npy", quadrature["center_cart"])
        np.save(output_dir / "ws_center_frac_wrapped.npy", quadrature["center_frac_wrapped"])
    np.savez(
        output_dir / "ws_quadrature_grid.npz",
        format_version=np.array(1, dtype=int),
        method=np.array(quadrature["method"]),
        sample_grid_shape=np.asarray(quadrature["grid"], dtype=int),
        source_fft_grid=np.asarray(quadrature["source_grid"], dtype=int),
        quadrature_factor=np.array(quadrature["factor"], dtype=int),
        native_grid_count=np.array(int(np.prod(quadrature["grid"])), dtype=int),
        base_indices=quadrature["base_indices"],
        translations=quadrature["translations"],
        points_frac_cont=quadrature["points_frac_cont"],
        points_cart=quadrature["points_cart"],
        weights=quadrature["weights"],
        center_cart=(quadrature["center_cart"] if quadrature["center_cart"] is not None
                     else np.full(3, np.nan)),
        center_frac_wrapped=(quadrature["center_frac_wrapped"] if quadrature["center_frac_wrapped"] is not None
                             else np.full(3, np.nan)),
    )
    with (output_dir / "cno_run_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with (output_dir / "cno_metadata.txt").open("w", encoding="utf-8") as handle:
        handle.write("=== Regional pseudo CNO metadata ===\n\n")
        for name, value in report.items():
            handle.write(f"{name}: {value}\n")
        handle.write(f"n_input_states: {len(states)}\n")


def _main_pseudo(*, material=None, output_subdir=None, ws_quadrature=None,
                 quadrature_factor=None, overwrite=False):
    material = material if material is not None else config.MATERIAL
    output_subdir = output_subdir if output_subdir is not None else config.OUTPUT_SUBDIR
    method = ws_quadrature if ws_quadrature is not None else getattr(config, "WS_QUADRATURE", "finite_volume")
    factor = (quadrature_factor if quadrature_factor is not None
              else getattr(config, "WS_QUADRATURE_FACTOR", 1))
    output_dir = HERE / "Data" / material / "output" / output_subdir
    protected = ("cno_occupations.npy", "cno_orbitals.npy", "ws_quadrature_grid.npz")
    if not overwrite and any((output_dir / name).exists() for name in protected):
        raise FileExistsError(
            f"Refusing to overwrite {output_dir}. Choose a new OUTPUT_SUBDIR or pass --overwrite explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = HERE / "Data" / material
    wfc = vaspwfc(str(data_dir / "WAVECAR"), lsorbit=config.LSORBIT)
    source_grid = np.asarray(wfc._ngrid, dtype=int)
    latvec, species, counts, _, _, frac_coords, _ = read_poscar_structure(data_dir / "POSCAR")
    volume = abs(float(np.linalg.det(latvec)))
    kfrac, kweights, energies, kcoord_source, kweight_source, mismatch_note = _load_kpoint_data(wfc, data_dir)
    if kweight_source == "uniform fallback":
        print("WARNING: uniform k-point weights are valid only for a fully unreduced mesh (ISYM=0).")

    quadrature = _build_quadrature(
        latvec, source_grid, use_ws_cell=config.USE_WS_CELL,
        center=config.WS_CENTER, center_coord_type=config.WS_CENTER_COORD_TYPE,
        nmax=config.WS_TRANSLATION_SEARCH_RANGE, method=method, factor=factor,
    )
    print(f"=== Regional pseudo CNO: {material} ===")
    print(f"output_dir: {output_dir}")
    print(f"WAVECAR: nkpts={wfc._nkpts} nbands={wfc._nbands} source_grid={tuple(source_grid)}")
    print(f"WS quadrature: {quadrature['method']} samples={len(quadrature['weights'])} "
          f"grid={tuple(quadrature['grid'])} factor={quadrature['factor']} "
          f"weight_sum={quadrature['weights'].sum():.10f}")
    if config.RESTRICT_TO_FERMI_WINDOW:
        if energies is None:
            raise RuntimeError("Fermi-window selection needs a matching EIGENVAL")
        selected = int(np.sum(np.abs(energies - config.EFERMI) <= config.FERMI_WINDOW_EV))
        print(f"Fermi window: {selected}/{energies.size} states in "
              f"[{config.EFERMI-config.FERMI_WINDOW_EV:.3f}, "
              f"{config.EFERMI+config.FERMI_WINDOW_EV:.3f}] eV")

    t0 = time.perf_counter()
    psi, state_weights, states = _build_pseudo_state_matrix(
        wfc, quadrature, kfrac, kweights, lsorbit=config.LSORBIT, ispin=config.ISPIN,
        restrict_to_fermi_window=config.RESTRICT_TO_FERMI_WINDOW, energies=energies,
        efermi=config.EFERMI, fermi_window_ev=config.FERMI_WINDOW_EV,
    )
    occupations, cnos, gram, kernel, overlap = _solve_pseudo_cnos(
        psi, state_weights, quadrature["weights"],
    )
    elapsed = time.perf_counter() - t0
    herm_gram = float(np.max(np.abs(gram - gram.conj().T)))
    herm_kernel = float(np.max(np.abs(kernel - kernel.conj().T)))
    ortho_error = float(np.max(np.abs(overlap - np.eye(len(occupations)))) )
    trace_expected = float(np.sum(state_weights))
    trace_kernel = float(np.trace(kernel).real)
    report = dict(
        format_version=1,
        material=material,
        calculation="regional_cno",
        metric="pseudo_only",
        paw_augmentation=False,
        input_state_normalization="vaspwfc pseudo norm",
        lsorbit=bool(config.LSORBIT),
        ispin=int(config.ISPIN),
        restrict_to_fermi_window=bool(config.RESTRICT_TO_FERMI_WINDOW),
        source_fft_grid=[int(x) for x in source_grid],
        sample_grid=[int(x) for x in quadrature["grid"]],
        ws_quadrature=quadrature["method"],
        quadrature_factor=int(quadrature["factor"]),
        n_samples=int(len(quadrature["weights"])),
        weight_sum=float(quadrature["weights"].sum()),
        n_input_states=int(len(states)),
        n_cnos=int(len(occupations)),
        kcoord_source=kcoord_source,
        kweight_source=kweight_source,
        eigenval_mismatch_note=mismatch_note,
        volume_Ang3=volume,
        trace_expected_input=trace_expected,
        trace_kernel=trace_kernel,
        trace_minus_expected=trace_kernel - trace_expected,
        gram_hermiticity_error=herm_gram,
        kernel_hermiticity_error=herm_kernel,
        cno_weighted_orthonormality_error=ortho_error,
        sum_cno_occupations=float(occupations.sum()),
        top_20_occupations=[float(x) for x in occupations[:20]],
        elapsed_s=elapsed,
    )
    _write_pseudo_output(
        output_dir, material=material, quadrature=quadrature, cnos=cnos,
        occupations=occupations, kpoint_weights=kweights, states=states,
        report=report, use_ws_cell=bool(config.USE_WS_CELL),
    )
    print(f"Top occupations: {[round(float(x), 7) for x in occupations[:10]]}")
    print(f"Tr(K)={trace_kernel:.8f}; input trace={trace_expected:.8f}; "
          f"weighted CNO orthonormality error={ortho_error:.3e}")
    print(f"Saved -> {output_dir}")
    return report


def main(*, material=None, output_subdir=None, ws_quadrature=None, quadrature_factor=None,
         use_paw_augmentation=None, overwrite=False):
    """Run the selected regional-CNO metric through the shared public entry point."""
    use_paw = (getattr(config, "USE_PAW_AUGMENTATION", False)
               if use_paw_augmentation is None else bool(use_paw_augmentation))
    if not use_paw:
        return _main_pseudo(
            material=material, output_subdir=output_subdir, ws_quadrature=ws_quadrature,
            quadrature_factor=quadrature_factor, overwrite=overwrite,
        )
    # The PAW module owns only the augmentation construction/validation.  It
    # writes the exact same weighted quadrature/CNO files as this pseudo path.
    sys.path.insert(0, str(PAW_DIR))
    from paw_regional_cno import main as paw_main  # noqa: WPS433
    return paw_main(
        material=material, output_subdir=output_subdir, ws_quadrature=(
            ws_quadrature if ws_quadrature is not None else getattr(config, "WS_QUADRATURE", "finite_volume")
        ),
        quadrature_factor=(quadrature_factor if quadrature_factor is not None
                           else getattr(config, "WS_QUADRATURE_FACTOR", 1)),
        overwrite=overwrite,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material")
    parser.add_argument("--output-subdir")
    parser.add_argument("--ws-quadrature", choices=("finite_volume", "weighted_ties", "legacy"))
    parser.add_argument("--quadrature-factor", type=int)
    parser.add_argument("--augmentation", choices=("pseudo", "paw"),
                        help="Override config.USE_PAW_AUGMENTATION for this run.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Permit replacing an existing output directory.")
    args = parser.parse_args()
    main(
        material=args.material, output_subdir=args.output_subdir,
        ws_quadrature=args.ws_quadrature, quadrature_factor=args.quadrature_factor,
        use_paw_augmentation=(None if args.augmentation is None else args.augmentation == "paw"),
        overwrite=args.overwrite,
    )
