"""Saved real-space quadrature contract for regional-CNO post-processing.

Regional finite-volume CNOs are stored on a *weighted, expanded* WS sample
map.  It is not in general possible to recover that map from ``fft_grid``:
boundary voxels may have several periodic images, each with a fractional
weight.  This module is the one supported way for DOS/fatband-style tools to
evaluate a WAVECAR state on precisely those saved sample rows.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SavedCNOQuadrature:
    """The fixed quadrature geometry associated with one CNO output folder."""

    sample_grid_shape: tuple[int, int, int]
    source_fft_grid: tuple[int, int, int]
    quadrature_factor: int
    native_grid_count: int
    base_indices: np.ndarray
    translations: np.ndarray
    points_frac_cont: np.ndarray
    points_cart: np.ndarray | None
    weights: np.ndarray
    method: str

    @property
    def n_samples(self) -> int:
        return int(len(self.weights))

    @property
    def expanded(self) -> bool:
        """Whether rows cannot be represented by one primitive FFT array."""
        return self.n_samples != int(np.prod(self.sample_grid_shape))

    @classmethod
    def load(cls, output_dir: str | Path) -> "SavedCNOQuadrature":
        """Load the saved map, with a strict backward-compatible fallback.

        New finite-volume outputs contain ``ws_quadrature_grid.npz``.  Older
        outputs retain their individual ``ws_*.npy`` files and have implicit
        unit weights.  Neither branch rebuilds a WS representative map.
        """
        output_dir = Path(output_dir)
        contract = output_dir / "ws_quadrature_grid.npz"
        if contract.exists():
            with np.load(contract, allow_pickle=False) as saved:
                version = int(np.asarray(saved["format_version"]).item())
                if version != 1:
                    raise ValueError(f"Unsupported ws_quadrature_grid format {version}")
                sample_grid = tuple(int(x) for x in saved["sample_grid_shape"])
                source_grid = tuple(int(x) for x in saved["source_fft_grid"])
                factor = int(np.asarray(saved["quadrature_factor"]).item())
                native_count = int(np.asarray(saved["native_grid_count"]).item())
                method = str(np.asarray(saved["method"]).item())
                points_cart = np.asarray(saved["points_cart"], dtype=float)
                result = cls(
                    sample_grid_shape=sample_grid,
                    source_fft_grid=source_grid,
                    quadrature_factor=factor,
                    native_grid_count=native_count,
                    base_indices=np.asarray(saved["base_indices"], dtype=int),
                    translations=np.asarray(saved["translations"], dtype=int),
                    points_frac_cont=np.asarray(saved["points_frac_cont"], dtype=float),
                    points_cart=points_cart,
                    weights=np.asarray(saved["weights"], dtype=float),
                    method=method,
                )
        else:
            sample_grid = tuple(int(x) for x in np.load(output_dir / "fft_grid_shape.npy"))
            source_path = output_dir / "ws_source_fft_grid.npy"
            source_grid = (tuple(int(x) for x in np.load(source_path)) if source_path.exists()
                           else sample_grid)
            factor_path = output_dir / "ws_quadrature_factor.npy"
            factor = int(np.asarray(np.load(factor_path)).item()) if factor_path.exists() else 1
            base_path = output_dir / "ws_base_indices.npy"
            pos_path = output_dir / "ws_points_frac_cont.npy"
            if base_path.exists() and pos_path.exists():
                base = np.asarray(np.load(base_path), dtype=int)
                points = np.asarray(np.load(pos_path), dtype=float)
                trans_path = output_dir / "ws_translation_int.npy"
                translations = (np.asarray(np.load(trans_path), dtype=int) if trans_path.exists()
                                else np.rint(points - base / np.asarray(sample_grid)).astype(int))
                weight_path = output_dir / "ws_quadrature_weights.npy"
                weights = (np.asarray(np.load(weight_path), dtype=float) if weight_path.exists()
                           else np.ones(len(base), dtype=float))
                cart_path = output_dir / "ws_points_cart.npy"
                points_cart = np.asarray(np.load(cart_path), dtype=float) if cart_path.exists() else None
                method = "legacy_saved_ws_map"
                report_path = output_dir / "paw_regional_report.json"
                if report_path.exists():
                    try:
                        report = json.loads(report_path.read_text(encoding="utf-8"))
                        method = str(report.get("ws_quadrature", {}).get("method", method))
                    except (OSError, ValueError, TypeError):
                        pass
            else:
                # Non-WS legacy output: its stored field really is a regular
                # primitive FFT array, so construct only that obvious ordering.
                base = np.stack(np.meshgrid(*(np.arange(n) for n in sample_grid), indexing="ij"),
                                axis=-1).reshape(-1, 3)
                points = base / np.asarray(sample_grid, dtype=float)[None, :]
                translations = np.zeros_like(base)
                weights = np.ones(len(base), dtype=float)
                points_cart = None
                method = "regular_fft_grid"
            result = cls(
                sample_grid_shape=sample_grid,
                source_fft_grid=source_grid,
                quadrature_factor=factor,
                native_grid_count=int(np.prod(sample_grid)),
                base_indices=base,
                translations=translations,
                points_frac_cont=points,
                points_cart=points_cart,
                weights=weights,
                method=method,
            )
        result._validate()
        return result

    def _validate(self) -> None:
        n = self.n_samples
        if (self.base_indices.shape != (n, 3)
                or self.translations.shape != (n, 3)
                or self.points_frac_cont.shape != (n, 3)):
            raise ValueError("Saved CNO quadrature arrays do not share one sample dimension")
        if self.points_cart is not None and self.points_cart.shape != (n, 3):
            raise ValueError("Saved CNO Cartesian positions have the wrong shape")
        if self.weights.shape != (n,) or np.any(~np.isfinite(self.weights)) or np.any(self.weights < 0):
            raise ValueError("Saved CNO quadrature weights must be finite and nonnegative")
        grid = np.asarray(self.sample_grid_shape, dtype=int)
        if np.any(grid <= 0) or np.any(np.asarray(self.source_fft_grid) <= 0):
            raise ValueError("Saved CNO grid shapes must be positive")
        if np.any(self.base_indices < 0) or np.any(self.base_indices >= grid[None, :]):
            raise ValueError("Saved CNO base indices lie outside the sample grid")
        expected = self.base_indices / grid[None, :] + self.translations
        if np.max(np.abs(self.points_frac_cont - expected), initial=0.0) > 1.0e-7:
            raise ValueError("Saved CNO positions disagree with base indices/translations")
        if not np.isclose(self.weights.sum(), self.native_grid_count,
                          rtol=0.0, atol=2.0e-7 * max(1, self.native_grid_count)):
            raise ValueError("Saved CNO quadrature weights do not preserve the regional volume")

    def validate_cno_rows(self, cno_fields: np.ndarray) -> None:
        if cno_fields.ndim != 2 or cno_fields.shape[0] != self.n_samples:
            raise ValueError(
                f"CNO field has shape {cno_fields.shape}; saved quadrature requires "
                f"({self.n_samples}, n_cno).")

    def validate_source_fft_grid(self, wavecar_grid) -> None:
        wavecar_grid = tuple(int(x) for x in wavecar_grid)
        if wavecar_grid != self.source_fft_grid:
            raise ValueError(
                f"WAVECAR FFT grid {wavecar_grid} does not match the saved CNO source "
                f"FFT grid {self.source_fft_grid}.  The plane-wave bases differ.")

    def normalized_weighted_bra(self, cno_fields: np.ndarray) -> np.ndarray:
        """Return rows ``<CNO_i| W`` with unit pseudo-field regional norm.

        This preserves the old DOS/fatband convention (individual
        pseudo-field normalization), but replaces its implicit unit weights by
        the exact saved finite-volume weights.
        """
        self.validate_cno_rows(cno_fields)
        norms2 = np.sum(self.weights[:, None] * np.abs(cno_fields) ** 2, axis=0)
        if np.any(~np.isfinite(norms2)) or np.any(norms2 <= 0.0):
            raise ValueError("A selected CNO has zero or non-finite weighted pseudo norm")
        return (cno_fields / np.sqrt(norms2)[None, :]).conj().T * self.weights[None, :]

    def bloch_field_from_coeff(self, coeff: np.ndarray, gvec: np.ndarray,
                               k_frac: np.ndarray) -> np.ndarray:
        """Evaluate one plane-wave Bloch field on the saved sample rows.

        The IFFT is zero-padded directly to ``sample_grid_shape``.  This is
        exact band-limited evaluation of the supplied plane-wave state, not
        interpolation of a saved CNO field.
        """
        coeff = np.asarray(coeff)
        gvec = np.asarray(gvec, dtype=int)
        if coeff.ndim != 1 or gvec.shape != (len(coeff), 3):
            raise ValueError("Plane-wave coefficients/G-vectors have incompatible shapes")
        nx, ny, nz = self.sample_grid_shape
        grid = np.zeros((nx, ny, nz), dtype=np.complex128)
        grid[gvec[:, 0] % nx, gvec[:, 1] % ny, gvec[:, 2] % nz] = coeff
        periodic = np.fft.ifftn(grid) * np.sqrt(nx * ny * nz)
        base = self.base_indices
        values = periodic[base[:, 0], base[:, 1], base[:, 2]]
        return values * np.exp(2j * np.pi * (self.points_frac_cont @ np.asarray(k_frac)))


def write_saved_quadrature_contract(output_dir: str | Path) -> Path:
    """Create ``ws_quadrature_grid.npz`` from an existing saved map.

    This migration helper is deliberately geometry-free: it only consolidates
    arrays that already belong to the CNO result, never recomputes or changes
    any CNO, WS weight, or representative position.
    """
    output_dir = Path(output_dir)
    quadrature = SavedCNOQuadrature.load(output_dir)
    if quadrature.points_cart is None:
        raise ValueError(
            "Cannot consolidate this old output: it has no saved Cartesian sample positions.")
    path = output_dir / "ws_quadrature_grid.npz"
    np.savez(
        path,
        format_version=np.array(1, dtype=int),
        method=np.array(quadrature.method),
        sample_grid_shape=np.asarray(quadrature.sample_grid_shape, dtype=int),
        source_fft_grid=np.asarray(quadrature.source_fft_grid, dtype=int),
        quadrature_factor=np.array(quadrature.quadrature_factor, dtype=int),
        native_grid_count=np.array(quadrature.native_grid_count, dtype=int),
        base_indices=quadrature.base_indices,
        translations=quadrature.translations,
        points_frac_cont=quadrature.points_frac_cont,
        points_cart=quadrature.points_cart,
        weights=quadrature.weights,
    )
    return path
