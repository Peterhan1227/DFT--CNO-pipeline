"""Unified CNO point-group diagnosis and optional symmetry adaptation.

This is the supported replacement for ``check_point_group_symmetry.py`` and
``symmetry_adapt_cnos.py``.  It works from the saved CNO *fields* and therefore
does not assume that their ordinary sampled L2 Gram matrix is the identity.
It also supports a weighted, expanded WS sample map: a native FFT node may
have several periodic WS images with fractional finite-volume weights.

For every generator it uses the best available action on the WS samples:

* an exact lookup when the transformed point is on the stored grid and its
  actual WS representative exists;
* mask-normalised trilinear interpolation otherwise;
* removal (never zero filling) of samples where neither is trustworthy.

This lets an isolated CNO be tested with the normalized overlap
``|<phi|U_g phi>| / (||phi|| ||U_g phi||)``.  A block is tested by a QR
projection, so its closure test is valid even when the input CNO fields are
not Euclidean orthogonal.  The code deliberately does not claim that this is
a full PAW-metric proof; it is the correct field-level diagnostic available
from ``cno_orbitals.npy`` alone.

Default behaviour is read-only with respect to the CNO data.  ``--adapt``
writes a separate file and only rotates blocks that pass every generator.
The rotation itself is unitary in the original CNO-index space, which
preserves the normalization of a genuinely degenerate CNO subspace.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent
PIPELINE_DIR = HERE.parent
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(PIPELINE_DIR / "helper functions"))

from config import MATERIAL, OUTPUT_SUBDIR, WS_CENTER, WS_CENTER_COORD_TYPE  # noqa: E402
from ws_cell import read_poscar_structure  # noqa: E402


# Generators are sufficient: a subspace closed under all of them is closed
# under the full point group they generate.
GENERATORS_BY_MATERIAL = {
    "WSe2_mono": ("c3_z", "sigma_h", "sigma_v"),
    "Si": ("c3_111", "swap_xy", "inversion"),
}

R_CART = {
    "inversion": -np.eye(3),
    "c3_111": np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
    "swap_xy": np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    "c3_z": np.array([[-0.5, -np.sqrt(3.0) / 2.0, 0.0],
                       [np.sqrt(3.0) / 2.0, -0.5, 0.0],
                       [0.0, 0.0, 1.0]]),
    "sigma_h": np.diag([1.0, 1.0, -1.0]),
    # The W-centred vertical mirror of this POSCAR is the yz plane.  The
    # superficially plausible xz reflection diag(1,-1,1) maps the Bravais
    # lattice onto itself but not the WSe2 basis about q=(1/3,2/3,1/2).
    "sigma_v": np.diag([-1.0, 1.0, 1.0]),
}


def _json(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _unitary_polar(a: np.ndarray) -> np.ndarray:
    """Closest unitary matrix to ``a`` in Frobenius norm."""
    u, _, vh = np.linalg.svd(a, full_matrices=False)
    return u @ vh


def _normalized_overlap(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.vdot(a, a).real)
    nb = float(np.vdot(b, b).real)
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    return float(abs(np.vdot(a, b)) / np.sqrt(na * nb))


def _clusters(indices: list[int], occ: np.ndarray, atol: float, rtol: float) -> list[list[int]]:
    """Occupation clusters using a full-span criterion, never chained gaps."""
    out: list[list[int]] = []
    current: list[int] = []
    for i in indices:
        trial = current + [i]
        values = occ[trial]
        scale = max(float(np.mean(np.abs(values))), 1.0e-12)
        if current and float(values.max() - values.min()) > atol + rtol * scale:
            out.append(current)
            current = [i]
        else:
            current = trial
    if current:
        out.append(current)
    return out


def _validate_atomic_site_symmetry(latvec: np.ndarray, symbols: list[str],
                                   frac_coords: np.ndarray, center: np.ndarray,
                                   generators: tuple[str, ...]) -> None:
    """Reject a lattice operation that does not preserve the actual basis.

    Mapping the Bravais lattice onto itself is necessary but not sufficient:
    an incorrectly oriented mirror can pass the integer-matrix test while
    moving the atoms.  This guard is intentionally performed before CNO data
    is interpreted.
    """
    symbols_array = np.asarray(symbols)
    for name in generators:
        r_frac = np.linalg.inv(latvec.T) @ R_CART[name] @ latvec.T
        mapped = (frac_coords - center[None, :]) @ r_frac.T + center[None, :]
        worst = 0.0
        for i, point in enumerate(mapped):
            candidates = frac_coords[symbols_array == symbols[i]]
            delta = point[None, :] - candidates
            delta -= np.rint(delta)
            worst = max(worst, float(np.min(np.max(np.abs(delta), axis=1))))
        if worst > 1.0e-5:
            raise ValueError(
                f"{name} maps the Bravais lattice but not the atomic basis about "
                f"the selected center (max fractional mismatch {worst:.3e}).")


@dataclass
class OperationSamples:
    name: str
    r_inv: np.ndarray
    target_rows: np.ndarray
    source_rows: np.ndarray | None
    coords: np.ndarray | None
    denominator: np.ndarray | None
    method: str
    grid_max_offset: float
    exact_available_fraction: float
    interpolation_safe_fraction: float


class FieldSymmetry:
    """Coordinates, operation samplers, and field-level diagnostics."""

    def __init__(self, output_dir: Path, center: np.ndarray, latvec: np.ndarray,
                 generators: tuple[str, ...], interpolation_order: int,
                 min_interpolation_den: float):
        self.output_dir = output_dir
        self.center = center
        self.latvec = latvec
        self.generators = generators
        self.order = interpolation_order
        self.min_den = min_interpolation_den

        self.X = np.load(output_dir / "cno_orbitals.npy")
        self.occ = np.load(output_dir / "cno_occupations.npy")
        self.nvec = np.load(output_dir / "fft_grid_shape.npy").astype(int)
        self.native_nr = int(np.prod(self.nvec))
        self.nr = self.X.shape[0]

        base = np.load(output_dir / "ws_base_indices.npy").astype(int)
        translations = np.load(output_dir / "ws_translation_int.npy").astype(int)
        self.actual = base + self.nvec[None, :] * translations
        self.positions = np.load(output_dir / "ws_points_frac_cont.npy")
        if self.actual.shape != self.positions.shape:
            raise ValueError("WS position/index arrays have incompatible shapes")
        if self.actual.shape[0] != self.nr:
            raise ValueError(
                f"CNO rows {self.nr} != saved WS sample rows {self.actual.shape[0]}")
        err = float(np.max(np.abs(self.positions - self.actual / self.nvec[None, :])))
        if err > 1.0e-6:
            raise ValueError(f"WS positions disagree with integer indices by {err:.3e}")

        self.row_of_actual = {tuple(v): i for i, v in enumerate(self.actual)}
        if len(self.row_of_actual) != self.nr:
            raise ValueError("WS actual grid coordinates are not unique")

        weights_path = output_dir / "ws_quadrature_weights.npy"
        self.weights = (np.load(weights_path).astype(float) if weights_path.exists()
                        else np.ones(self.nr, dtype=float))
        if self.weights.shape != (self.nr,) or np.any(self.weights < 0.0):
            raise ValueError("WS quadrature weights must be a nonnegative length-Nr array")

        # The sparse WS field lives in this bounding box.  The box is enlarged
        # only enough for interpolation stencils; no periodic folding is used.
        padding = max(2, self.order + 1)
        lo = self.actual.min(axis=0)
        hi = self.actual.max(axis=0)
        self.offset = -lo + padding
        self.shape = tuple((hi - lo + 1 + 2 * padding).tolist())
        self.ijk = self.actual + self.offset[None, :]
        if np.any(self.ijk < 0) or np.any(self.ijk >= np.asarray(self.shape)):
            raise ValueError("WS images exceed the 3x interpolation embedding")
        self.mask = np.zeros(self.shape, dtype=float)
        self.mask[self.ijk[:, 0], self.ijk[:, 1], self.ijk[:, 2]] = 1.0

        self.ops = {name: self._make_operation(name) for name in generators}

    def _make_operation(self, name: str) -> OperationSamples:
        r_frac = np.linalg.inv(self.latvec.T) @ R_CART[name] @ self.latvec.T
        r_inv = np.rint(r_frac).astype(int)
        if not np.allclose(r_frac, r_inv, atol=1.0e-6):
            raise ValueError(f"{name} is not an integer fractional lattice operation:\n{r_frac}")

        source_frac = (self.positions - self.center[None, :]) @ r_inv.T + self.center[None, :]
        voxel = source_frac * self.nvec[None, :]
        rounded = np.rint(voxel).astype(int)
        grid_error = np.max(np.abs(voxel - rounded), axis=1)
        grid_exact = grid_error < 1.0e-8
        lookup = np.full(self.nr, -1, dtype=int)
        for row in np.where(grid_exact)[0]:
            lookup[row] = self.row_of_actual.get(tuple(rounded[row]), -1)
        direct = grid_exact & (lookup >= 0)

        # If every target is exactly represented, use only exact lookup.  If
        # the affine map is off-grid, interpolate on the sparse WS embedding;
        # low-denominator points are excluded, never interpreted as zeros.
        if bool(np.all(grid_exact)):
            # A direct lookup is exact pointwise, but a set that is not closed
            # under the operation has unequal source/target quadrature.  Keep
            # only complete finite cycles; U is then an exact permutation of
            # the retained samples, rather than a one-sided boundary map.
            order = 1
            product = np.eye(3, dtype=int)
            while True:
                product = product @ r_inv
                if np.array_equal(product, np.eye(3, dtype=int)):
                    break
                order += 1
                if order > 24:
                    raise RuntimeError(f"Could not determine finite order of {name}")
            cyclic = np.zeros(self.nr, dtype=bool)
            for start in np.where(direct)[0]:
                current = int(start)
                complete = True
                for _ in range(order):
                    current = int(lookup[current]) if current >= 0 else -1
                    if current < 0:
                        complete = False
                        break
                cyclic[start] = complete and current == start
            rows = np.where(cyclic)[0]
            return OperationSamples(
                name, r_inv, rows, lookup[rows], None, None, "exact_lookup",
                float(grid_error.max()), float(direct.mean()), float(cyclic.mean()),
            )

        coords = (voxel + self.offset[None, :]).T
        den = map_coordinates(self.mask, coords, order=1, mode="constant", cval=0.0)
        safe = den >= self.min_den
        rows = np.where(safe)[0]
        return OperationSamples(
            name, r_inv, rows, None, coords[:, rows], den[rows], "interpolation",
            float(grid_error.max()), float(direct.mean()), float(safe.mean()),
        )

    def transform(self, op: OperationSamples, values: np.ndarray) -> np.ndarray:
        """Apply ``U_g f(r)=f(g^-1 r)`` at this operation's trusted targets."""
        arr = np.asarray(values)
        if arr.ndim == 1:
            arr = arr[:, None]
            one_column = True
        elif arr.ndim == 2:
            one_column = False
        else:
            raise ValueError("values must have shape (Nr,) or (Nr,ncol)")
        if arr.shape[0] != self.nr:
            raise ValueError("field has incompatible number of grid points")
        if op.method == "exact_lookup":
            out = arr[op.source_rows]
            return out[:, 0] if one_column else out

        out = np.empty((len(op.target_rows), arr.shape[1]), dtype=complex)
        for col in range(arr.shape[1]):
            field = np.zeros(self.shape, dtype=complex)
            field[self.ijk[:, 0], self.ijk[:, 1], self.ijk[:, 2]] = arr[:, col]
            real = map_coordinates(field.real * self.mask, op.coords, order=self.order,
                                   mode="constant", cval=0.0, prefilter=self.order > 1)
            imag = map_coordinates(field.imag * self.mask, op.coords, order=self.order,
                                   mode="constant", cval=0.0, prefilter=self.order > 1)
            out[:, col] = (real + 1j * imag) / op.denominator
        return out[:, 0] if one_column else out

    def interpolation_control(self, op: OperationSamples) -> dict:
        """A radial Gaussian gives a geometry-only interpolation calibration."""
        displacement = (self.positions - self.center[None, :]) @ self.latvec
        radial = np.exp(-np.sum(displacement * displacement, axis=1) / (2.0 * 1.5**2)).astype(complex)
        source = self.transform(op, radial)
        target = radial[op.target_rows]
        corr = _normalized_overlap(target, source)
        residual = float(np.linalg.norm(source - target) / np.linalg.norm(target))
        return {"gaussian_normalized_overlap": corr, "gaussian_relative_residual": residual}

    def block_metrics(self, indices: list[int]) -> dict:
        """Closure metrics per generator using only the operation's trusted rows."""
        result: dict[str, dict] = {}
        for name, op in self.ops.items():
            sqrt_weight = np.sqrt(self.weights[op.target_rows])[:, None]
            phi = self.X[op.target_rows][:, indices] * sqrt_weight
            transformed = self.transform(op, self.X[:, indices]) * sqrt_weight
            q, r = np.linalg.qr(phi, mode="reduced")
            projection = q @ (q.conj().T @ transformed)
            norms2 = np.sum(np.abs(transformed)**2, axis=0)
            residual2 = np.sum(np.abs(transformed - projection)**2, axis=0)
            leakage = np.where(norms2 > 0.0, residual2 / norms2, np.nan)

            # UQ is U(phi R^-1); this is the representation in a sampled
            # orthonormal basis.  Its unitarity is a useful interpolation and
            # coverage diagnostic, while closure is the primary criterion.
            uq = np.linalg.solve(r.T, transformed.T).T
            d = q.conj().T @ uq
            unitary_error = float(np.linalg.norm(d.conj().T @ d - np.eye(len(indices))))
            singular_values = np.linalg.svd(d, compute_uv=False)
            entry = {
                "method": op.method,
                "n_samples": int(len(op.target_rows)),
                "sample_fraction": float(len(op.target_rows) / self.nr),
                "max_leakage": float(np.nanmax(leakage)),
                "per_cno_leakage": leakage,
                "unitarity_error": unitary_error,
                "singular_values": singular_values,
                "representation": d,
            }
            if len(indices) == 1:
                entry["normalized_overlap"] = _normalized_overlap(phi[:, 0], transformed[:, 0])
                entry["best_phase_residual"] = float(
                    np.sqrt(max(0.0, 2.0 - 2.0 * entry["normalized_overlap"]))
                )
            result[name] = entry
        return result

    def affinity(self, indices: list[int]) -> np.ndarray:
        """Largest normalized field overlap between any CNO and a transformed CNO."""
        n = len(indices)
        affinity = np.zeros((n, n), dtype=float)
        for op in self.ops.values():
            sqrt_weight = np.sqrt(self.weights[op.target_rows])[:, None]
            phi = self.X[op.target_rows][:, indices] * sqrt_weight
            transformed = self.transform(op, self.X[:, indices]) * sqrt_weight
            gram = phi.conj().T @ transformed
            norm_l = np.sqrt(np.sum(np.abs(phi)**2, axis=0))
            norm_r = np.sqrt(np.sum(np.abs(transformed)**2, axis=0))
            norm = norm_l[:, None] * norm_r[None, :]
            with np.errstate(invalid="ignore", divide="ignore"):
                affinity = np.maximum(affinity, np.abs(gram) / norm)
        return np.nan_to_num(affinity, nan=0.0)


def _pass_block(metrics: dict[str, dict], singleton_overlap: float,
                leakage_tol: float, unitary_tol: float) -> bool:
    for entry in metrics.values():
        if entry["max_leakage"] > leakage_tol:
            return False
        if entry["unitarity_error"] > unitary_tol:
            return False
        if "normalized_overlap" in entry and entry["normalized_overlap"] < singleton_overlap:
            return False
    return True


def _recover_singletons(blocks: list[list[int]], checked: list[int], occ: np.ndarray,
                        field: FieldSymmetry, args: argparse.Namespace) -> list[list[int]]:
    """Merge a failed singleton only with a nearby, symmetry-coupled CNO/block."""
    affinity = field.affinity(checked)
    pos = {idx: i for i, idx in enumerate(checked)}
    parent = {idx: idx for idx in checked}

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(i: int, j: int) -> None:
        ri, rj = root(i), root(j)
        if ri != rj:
            parent[rj] = ri

    for block in blocks:
        for i in block[1:]:
            join(block[0], i)

    for block in blocks:
        if len(block) != 1:
            continue
        i = block[0]
        metrics = field.block_metrics([i])
        if _pass_block(metrics, args.singleton_overlap, args.leakage_tol, args.unitary_tol):
            continue
        candidates = []
        for j in checked:
            if j == i:
                continue
            scale = max(0.5 * (abs(float(occ[i])) + abs(float(occ[j]))), 1.0e-12)
            close = abs(float(occ[i] - occ[j])) <= args.recovery_occ_atol + args.recovery_occ_rtol * scale
            coupling = max(float(affinity[pos[i], pos[j]]), float(affinity[pos[j], pos[i]]))
            if close and coupling >= args.recovery_coupling:
                candidates.append((abs(float(occ[i] - occ[j])), -coupling, j))
        if candidates:
            # One nearest strongly coupled block is the conservative recovery;
            # a later closure check still decides whether the expanded block is valid.
            candidates.sort()
            join(i, candidates[0][2])

    merged: dict[int, list[int]] = {}
    for i in checked:
        merged.setdefault(root(i), []).append(i)
    return sorted(merged.values(), key=lambda b: min(b))


def _closure_score(metrics: dict[str, dict], args: argparse.Namespace) -> float:
    """Dimensionless severity used only to diagnose a failed block."""
    values = []
    for entry in metrics.values():
        values.append(float(entry["max_leakage"]) / args.leakage_tol)
        values.append(float(entry["unitarity_error"]) / args.unitary_tol)
        if "normalized_overlap" in entry:
            values.append(max(0.0, args.singleton_overlap - entry["normalized_overlap"])
                          / max(1.0 - args.singleton_overlap, 1.0e-12))
    return max(values, default=0.0)


def _suggest_closure_expansion(block: list[int], checked: list[int], occ: np.ndarray,
                               field: FieldSymmetry, args: argparse.Namespace) -> dict:
    """Find a diagnostic (never adaptive) enlarged span for a failed block.

    If closure is recovered only by including CNOs whose occupations are far
    outside the original cluster, that is evidence of a symmetry-breaking
    density operator rather than a legitimate degenerate multiplet.  The
    suggested span is reported precisely to make that distinction explicit.
    """
    working = list(block)
    metrics = field.block_metrics(working)
    before = _closure_score(metrics, args)
    additions = []
    while len(working) < args.max_closure_size and not _pass_block(
            metrics, args.singleton_overlap, args.leakage_tol, args.unitary_tol):
        candidates = []
        for candidate in checked:
            if candidate in working:
                continue
            if min(abs(float(occ[candidate] - occ[i])) for i in working) > args.closure_search_occ_atol:
                continue
            trial = sorted(working + [candidate])
            trial_metrics = field.block_metrics(trial)
            score = _closure_score(trial_metrics, args)
            candidates.append((score, candidate, trial, trial_metrics))
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], abs(float(occ[item[1]] - np.mean(occ[working])))))
        score, candidate, trial, trial_metrics = candidates[0]
        if score >= before - 1.0e-3:
            break
        working, metrics, before = trial, trial_metrics, score
        additions.append(candidate)
    return {
        "indices": working,
        "added_indices": additions,
        "occupation_span": float(occ[working].max() - occ[working].min()),
        "passed_all_generators": _pass_block(
            metrics, args.singleton_overlap, args.leakage_tol, args.unitary_tol),
        "per_operation": metrics,
    }


def _adapt_block(field: FieldSymmetry, indices: list[int], op_name: str) -> tuple[np.ndarray, dict]:
    """Return a *unitary-in-index-space* rotation selected by one generator."""
    op = field.ops[op_name]
    phi = field.X[op.target_rows][:, indices]
    transformed = field.transform(op, field.X[:, indices])
    # Least-squares representation U Phi ~= Phi C.  Its polar factor is
    # unitary in the input CNO-index basis, so a degenerate PAW-normalized
    # block remains normalized under this rotation.
    c, *_ = np.linalg.lstsq(phi, transformed, rcond=None)
    c_unitary = _unitary_polar(c)
    eigenvalues, rotation = np.linalg.eig(c_unitary)
    order = np.argsort(np.angle(eigenvalues))
    eigenvalues = eigenvalues[order]
    rotation = rotation[:, order]
    return rotation, {
        "operation": op_name,
        "raw_representation": c,
        "polar_representation": c_unitary,
        "eigenvalues": eigenvalues,
        "rotation": rotation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", default=MATERIAL,
                        help="material directory under Data/ (default: config.py)")
    parser.add_argument("--output-subdir", default=OUTPUT_SUBDIR,
                        help="CNO output subdirectory (default: config.py)")
    parser.add_argument("--report-dir", default=None,
                        help="directory for the read-only report (default: symmetry/output)")
    parser.add_argument("--overwrite-report", action="store_true",
                        help="allow replacing an existing report with the same name")
    parser.add_argument("--occ-cutoff", type=float, default=0.03)
    parser.add_argument("--deg-atol", type=float, default=0.003,
                        help="initial full-span absolute occupation-cluster tolerance")
    parser.add_argument("--deg-rtol", type=float, default=0.004,
                        help="initial full-span relative occupation-cluster tolerance")
    parser.add_argument("--recovery-occ-atol", type=float, default=0.03)
    parser.add_argument("--recovery-occ-rtol", type=float, default=0.03)
    parser.add_argument("--recovery-coupling", type=float, default=0.20)
    parser.add_argument("--closure-search-occ-atol", type=float, default=0.25,
                        help="maximum occupation distance considered for a diagnostic closure expansion")
    parser.add_argument("--max-closure-size", type=int, default=8)
    parser.add_argument("--singleton-overlap", type=float, default=0.98)
    parser.add_argument("--leakage-tol", type=float, default=0.05)
    parser.add_argument("--unitary-tol", type=float, default=0.10)
    parser.add_argument("--interp-order", type=int, choices=(1, 3), default=1)
    parser.add_argument("--min-interp-den", type=float, default=0.50)
    parser.add_argument("--adapt", action="store_true",
                        help="write a separately symmetry-adapted field for verified multi-CNO blocks")
    parser.add_argument("--adapt-operation", default=None,
                        help="generator used to diagonalize verified blocks (default: first material generator)")
    parser.add_argument("--allow-low-confidence-adaptation", action="store_true")
    args = parser.parse_args(argv)

    material = args.material
    output_subdir = args.output_subdir
    output_dir = PIPELINE_DIR / "Data" / material / "output" / output_subdir
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    if material not in GENERATORS_BY_MATERIAL:
        raise ValueError(f"No generator list is configured for {material!r}")
    generators = GENERATORS_BY_MATERIAL[material]
    if args.adapt:
        if args.adapt_operation is None:
            args.adapt_operation = generators[0]
        if args.adapt_operation not in generators:
            raise ValueError(f"--adapt-operation must be one of {generators}")

    latvec, _, _, atom_symbols, _, frac_coords, _ = read_poscar_structure(
        PIPELINE_DIR / "Data" / material / "POSCAR"
    )
    saved_center = output_dir / "ws_center_frac_wrapped.npy"
    if saved_center.exists():
        # The saved output is authoritative when reviewing another material or
        # a historical run; it avoids changing config.py merely to inspect it.
        center = np.load(saved_center).astype(float)
    elif material == MATERIAL and WS_CENTER_COORD_TYPE == "fractional":
        center = np.asarray(WS_CENTER, dtype=float)
    else:
        raise FileNotFoundError(
            f"No saved WS center in {output_dir}; supply a run that contains ws_center_frac_wrapped.npy."
        )
    _validate_atomic_site_symmetry(latvec, atom_symbols, frac_coords, center, generators)
    field = FieldSymmetry(output_dir, center, latvec, generators,
                          args.interp_order, args.min_interp_den)

    checked = [int(i) for i in np.where(field.occ >= args.occ_cutoff)[0]]
    initial = _clusters(checked, field.occ, args.deg_atol, args.deg_rtol)
    recovered = _recover_singletons(initial, checked, field.occ, field, args)

    controls = {name: field.interpolation_control(op) for name, op in field.ops.items()}
    op_info = {
        name: {
            "method": op.method,
            "r_inv": op.r_inv,
            "grid_max_offset_voxel": op.grid_max_offset,
            "exact_available_fraction": op.exact_available_fraction,
            "trusted_sample_fraction": op.interpolation_safe_fraction,
            **controls[name],
        }
        for name, op in field.ops.items()
    }

    report_blocks = []
    for block in recovered:
        metrics = field.block_metrics(block)
        passed = _pass_block(metrics, args.singleton_overlap, args.leakage_tol, args.unitary_tol)
        report_blocks.append({
            "indices": block,
            "occupations": field.occ[block],
            "kind": "singleton" if len(block) == 1 else "candidate_degenerate_block",
            "passed_all_generators": passed,
            "per_operation": metrics,
        })

    # Do not silently redefine a failed block as degenerate.  Instead, record
    # the smallest nearby span that improves closure, if any; this is the
    # useful diagnostic for cases such as a nominal [3,4] doublet leaking into
    # CNOs with clearly different occupations.
    for block in report_blocks:
        if not block["passed_all_generators"]:
            block["closure_expansion"] = _suggest_closure_expansion(
                block["indices"], checked, field.occ, field, args
            )

    report = {
        "material": material,
        "output_subdir": output_subdir,
        "center_fractional": center,
        "generators": generators,
        "settings": vars(args),
        "n_total_cnos": int(len(field.occ)),
        "checked_indices": checked,
        "initial_occupation_blocks": initial,
        "recovered_blocks": recovered,
        "operations": op_info,
        "blocks": report_blocks,
        "interpretation": (
            "Singleton overlap is normalized. Block leakage is the QR projection residual. "
            "Exact-lookup operations use only grid points with an actual WS counterpart; "
            "interpolated operations exclude denominator-poor samples."
        ),
    }

    out_dir = Path(args.report_dir) if args.report_dir else HERE / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"cno_symmetry_{material}_{output_subdir}"
    report_path = out_dir / f"{tag}.json"
    if report_path.exists() and not args.overwrite_report:
        raise FileExistsError(
            f"Refusing to overwrite existing symmetry report {report_path}. "
            "Choose --report-dir or pass --overwrite-report explicitly.")
    report_path.write_text(json.dumps(report, indent=2, default=_json), encoding="utf-8")

    print(f"=== Unified CNO symmetry: {material} / {output_subdir} ===")
    print(f"Checking {len(checked)} CNOs with occupation >= {args.occ_cutoff:g}")
    print("\nOperation calibration:")
    for name, info in op_info.items():
        print(f"  {name:9s} {info['method']:12s} trusted={info['trusted_sample_fraction']:.3f} "
              f"Gaussian overlap={info['gaussian_normalized_overlap']:.6f} "
              f"residual={info['gaussian_relative_residual']:.3e}")
    print("\nBlocks:")
    for block in report_blocks:
        label = "PASS" if block["passed_all_generators"] else "FAIL"
        occ_text = ", ".join(f"{x:.6f}" for x in block["occupations"])
        print(f"  {label:4s} {block['indices']}  occ=[{occ_text}]")
        for name, metric in block["per_operation"].items():
            extra = (f"overlap={metric['normalized_overlap']:.6f}"
                     if "normalized_overlap" in metric else
                     f"leak={metric['max_leakage']:.3e}")
            print(f"       {name:9s} {extra}  unitary_err={metric['unitarity_error']:.3e}")
        expansion = block.get("closure_expansion")
        if expansion and expansion["added_indices"]:
            status = "closes" if expansion["passed_all_generators"] else "still fails"
            print(f"       diagnostic expansion -> {expansion['indices']} ({status}; "
                  f"occupation span={expansion['occupation_span']:.6f})")
    print(f"\nReport: {report_path}")

    if args.adapt:
        low_conf = [name for name, value in controls.items()
                    if value["gaussian_normalized_overlap"] < 0.995]
        if low_conf and not args.allow_low_confidence_adaptation:
            raise RuntimeError(
                "Refusing adaptation: interpolation control is below 0.995 for "
                f"{low_conf}. Re-run with --allow-low-confidence-adaptation only after review."
            )
        adapted = field.X.copy()
        adaptation = []
        for block in report_blocks:
            if len(block["indices"]) < 2 or not block["passed_all_generators"]:
                continue
            rotation, detail = _adapt_block(field, block["indices"], args.adapt_operation)
            adapted[:, block["indices"]] = field.X[:, block["indices"]] @ rotation
            adaptation.append({"indices": block["indices"], **detail})
        if not adaptation:
            print("No multi-CNO block passed every generator; no adapted field was written.")
            return 0
        adapted_path = output_dir / f"cnos_sym_adapted_{args.adapt_operation}.npy"
        if adapted_path.exists() and not args.overwrite_report:
            raise FileExistsError(
                f"Refusing to overwrite existing adapted field {adapted_path}. "
                "Pass --overwrite-report only after confirming that replacement is intended.")
        np.save(adapted_path, adapted)
        adaptation_path = out_dir / f"{tag}_adapt_{args.adapt_operation}.json"
        if adaptation_path.exists() and not args.overwrite_report:
            raise FileExistsError(
                f"Refusing to overwrite existing adaptation report {adaptation_path}. "
                "Pass --overwrite-report only after confirming that replacement is intended.")
        adaptation_path.write_text(json.dumps(adaptation, indent=2, default=_json), encoding="utf-8")
        print(f"Adapted fields: {adapted_path}")
        print(f"Adaptation data: {adaptation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
