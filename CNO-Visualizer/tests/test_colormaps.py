"""Real/complex detection and colormap resolution."""

from __future__ import annotations

import numpy as np

from cno_visualizer.colormaps import is_real_orbital, resolve_cmap


def test_is_real_detects_real_up_to_global_phase():
    x = np.linspace(-2, 2, 60)
    real = np.exp(-(x**2)).astype(complex)
    assert is_real_orbital(real)
    # A global gauge rotation must not change the verdict.
    assert is_real_orbital(real * np.exp(1j * 0.7))


def test_is_real_detects_complex():
    x = np.linspace(-2, 2, 60)
    psi = np.exp(1j * 2.0 * x) * np.exp(-(x**2))  # genuine phase winding
    assert not is_real_orbital(psi)


def test_resolve_cmap_handles_cet_and_plain():
    cm = resolve_cmap("CET_C7")
    assert callable(cm) or hasattr(cm, "__call__")   # a matplotlib Colormap object
    # Plain matplotlib names pass through unchanged for add_mesh.
    assert resolve_cmap("coolwarm") == "coolwarm"
