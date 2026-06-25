"""Colormap resolution and real/complex orbital detection.

`twilight_shifted` is muddy for phase; the perceptually-uniform cyclic maps from
``colorcet`` (the ``CET_C*`` family) read far better.  This module resolves a
colormap *name* to something ``pyvista.add_mesh(cmap=...)`` accepts, importing
``colorcet`` lazily and falling back to a matplotlib cyclic map if it is absent.
"""

from __future__ import annotations

import numpy as np


def resolve_cmap(name):
    """Return a colormap usable by ``add_mesh(cmap=...)``.

    ``CET_*`` names are resolved through ``colorcet`` (a ``matplotlib.Colormap``
    object); plain matplotlib names are returned unchanged.  If ``colorcet`` is
    not installed, ``CET_*`` falls back to ``twilight_shifted``.
    """
    if isinstance(name, str) and name.upper().startswith("CET_"):
        try:
            import colorcet as cc

            return cc.cm[name.upper()]
        except Exception:
            import matplotlib

            return matplotlib.colormaps["twilight_shifted"]
    return name


def is_real_orbital(psi: np.ndarray, tol: float = 1e-3) -> bool:
    """True if ``psi`` is (numerically) real up to a global phase.

    A Γ-point / real-Hamiltonian calculation yields real eigenvectors; for those
    the cyclic phase map shows ~one color, so the viewer defaults to a signed
    diverging map instead.  The test is gauge-aware: ``psi`` is first rotated so
    its largest-magnitude component is real, then the residual imaginary part is
    compared to the maximum magnitude.
    """
    psi = np.asarray(psi)
    if psi.size == 0:
        return True
    mag = np.abs(psi)
    mx = float(mag.max())
    if mx == 0.0:
        return True
    # Remove the global gauge: align to the dominant component's phase.
    anchor = psi[int(np.argmax(mag))]
    aligned = psi * np.conj(anchor / abs(anchor))
    return float(np.max(np.abs(aligned.imag))) / mx < tol


__all__ = ["resolve_cmap", "is_real_orbital"]
