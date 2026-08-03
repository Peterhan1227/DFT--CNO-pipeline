"""Shared plane-wave reconstruction used by both CNO construction paths.

The direct-pseudo and PAW builders must evaluate precisely the same
pseudo-wavefunction.  Keeping the zero-padded IFFT here makes that common
step explicit: PAW only adds its regional augmentation metric afterwards.
"""
from __future__ import annotations

import numpy as np


def zero_pad_ifft(coefficients, gvec, grid_factor, source_grid):
    """Evaluate plane-wave coefficients on an integer-refined FFT mesh.

    This is band-limited plane-wave evaluation, not interpolation of a saved
    CNO.  With ``grid_factor=1`` it is the ordinary normalized inverse FFT.

    Parameters
    ----------
    coefficients
        ``(n_band, n_G)`` raw WAVECAR coefficients.
    gvec
        Integer reciprocal-grid vectors matching the coefficient columns.
    grid_factor
        Positive integer factor applied to every source FFT dimension.
    source_grid
        Native WAVECAR FFT-grid shape.
    """
    factor = int(grid_factor)
    if factor != grid_factor or factor < 1:
        raise ValueError("grid_factor must be a positive integer")
    nx0, ny0, nz0 = (int(v) for v in source_grid)
    nx, ny, nz = nx0 * factor, ny0 * factor, nz0 * factor
    coeff = np.asarray(coefficients)
    gvec = np.asarray(gvec, dtype=int)
    if coeff.ndim != 2 or gvec.shape != (coeff.shape[1], 3):
        raise ValueError("coefficients and G vectors have incompatible shapes")
    grid = np.zeros((coeff.shape[0], nx, ny, nz), dtype=np.complex128)
    grid[:, gvec[:, 0] % nx, gvec[:, 1] % ny, gvec[:, 2] % nz] = coeff
    values = np.fft.ifftn(grid, axes=(1, 2, 3)) * np.sqrt(nx * ny * nz)
    return values.reshape(coeff.shape[0], -1), (nx, ny, nz), nx * ny * nz


def bloch_fields_on_samples(coefficients, gvec, *, source_grid, grid_factor,
                            base_indices, points_frac_cont, k_frac):
    """Return full Bloch fields on saved regional-quadrature sample rows."""
    periodic_flat, grid, _ = zero_pad_ifft(coefficients, gvec, grid_factor, source_grid)
    base = np.asarray(base_indices, dtype=int)
    if base.ndim != 2 or base.shape[1] != 3:
        raise ValueError("base_indices must have shape (n_samples, 3)")
    if np.any(base < 0) or np.any(base >= np.asarray(grid)[None, :]):
        raise ValueError("base_indices lie outside the reconstructed FFT mesh")
    periodic = periodic_flat.reshape((periodic_flat.shape[0],) + tuple(grid))
    sample = periodic[:, base[:, 0], base[:, 1], base[:, 2]]
    phase = np.exp(2j * np.pi * (np.asarray(points_frac_cont) @ np.asarray(k_frac)))
    return sample * phase[None, :]
