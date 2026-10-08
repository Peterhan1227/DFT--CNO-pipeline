"""Shared inclusive energy-window selection for regional CNO calculations."""
from __future__ import annotations

import numpy as np


def resolve_fermi_window_range(config) -> tuple[float, float]:
    """Return the configured ``[E-E_F]`` interval in eV.

    ``FERMI_WINDOW_RANGE_EV`` is the current explicit configuration.  The
    symmetric ``FERMI_WINDOW_EV`` setting is retained only so archived scripts
    and old local configs still have a sensible fallback.
    """
    value = getattr(config, "FERMI_WINDOW_RANGE_EV", None)
    if value is None:
        half_width = float(config.FERMI_WINDOW_EV)
        value = (-half_width, half_width)
    if len(value) != 2:
        raise ValueError("FERMI_WINDOW_RANGE_EV must contain exactly [lower, upper] in eV")
    lower, upper = (float(x) for x in value)
    if not (np.isfinite(lower) and np.isfinite(upper) and lower <= upper):
        raise ValueError("FERMI_WINDOW_RANGE_EV must be finite and satisfy lower <= upper")
    return lower, upper


def in_fermi_window(energies, efermi: float, relative_range) -> np.ndarray:
    """Inclusive mask for energies whose ``E - E_F`` lies in ``[lower, upper]``."""
    lower, upper = relative_range
    relative_energy = np.asarray(energies, dtype=float) - float(efermi)
    return (relative_energy >= lower) & (relative_energy <= upper)
