"""Phase utilities — wrap to ``[0, 2π)``, gauge reference, no PyVista deps."""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi
_ZERO_EPS = 1e-10


def wrap(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) into ``[0, 2π)``."""
    return np.mod(angle, TWO_PI)


def phase_from_ri(
    real: np.ndarray,
    imag: np.ndarray,
    offset: float = 0.0,
) -> np.ndarray:
    """Compute wrapped phase from interpolated real and imaginary parts.

    Phase angles are wrapped after the offset is applied; never interpolate phase
    angles directly — interpolate the complex components instead.
    """
    real = np.asarray(real)
    imag = np.asarray(imag)
    return np.mod(np.arctan2(imag, real) + float(offset), TWO_PI)


def set_phase_reference(psi_value: complex) -> float:
    """Return the gauge offset that maps the phase of ``psi_value`` to zero.

    ``psi_value`` must have non-negligible magnitude — otherwise the phase is
    numerically ill-defined and the reference choice is meaningless.
    """
    z = complex(psi_value)
    if abs(z) < _ZERO_EPS:
        raise ValueError(
            f"Cannot set phase reference at |psi| = {abs(z):.3e} ~ 0; "
            "pick a point with non-negligible amplitude."
        )
    return float(wrap(-np.angle(z)))


__all__ = ["wrap", "phase_from_ri", "set_phase_reference", "TWO_PI"]
