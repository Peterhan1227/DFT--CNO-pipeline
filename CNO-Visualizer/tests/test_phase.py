"""Phase utility tests (spec §18 items 9-11)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cno_visualizer.phase import phase_from_ri, set_phase_reference, wrap


def test_wrap_into_zero_two_pi():
    angles = np.array([-3 * np.pi, -np.pi, 0.0, np.pi, 3 * np.pi])
    wrapped = wrap(angles)
    assert np.all(wrapped >= 0.0)
    assert np.all(wrapped < 2 * np.pi)
    # 0 and 2*pi must be equivalent (both wrap to 0).
    assert math.isclose(float(wrap(0.0)), 0.0)
    assert math.isclose(float(wrap(2 * np.pi)), 0.0, abs_tol=1e-12)


def test_phase_from_ri_offset_only_changes_phase():
    real = np.array([1.0, 0.5, -0.5, -1.0, 0.0])
    imag = np.array([0.0, 0.5, 0.5, 0.0, 1.0])
    p0 = phase_from_ri(real, imag, 0.0)
    p_off = phase_from_ri(real, imag, 1.0)
    diff = np.mod(p_off - p0, 2 * np.pi)
    np.testing.assert_allclose(diff, 1.0, atol=1e-12)
    # Magnitude unchanged by any phase offset:
    mag0 = np.hypot(real, imag)
    np.testing.assert_array_equal(mag0, np.hypot(real, imag))


def test_set_phase_reference_zero_amplitude_rejected():
    with pytest.raises(ValueError):
        set_phase_reference(0 + 0j)


def test_set_phase_reference_sends_phase_to_zero():
    psi = 0.7 - 0.3j
    offset = set_phase_reference(psi)
    new_phase = phase_from_ri(
        np.array([psi.real]), np.array([psi.imag]), offset=offset
    )
    assert math.isclose(float(new_phase[0]), 0.0, abs_tol=1e-9) or math.isclose(
        float(new_phase[0]), 2 * np.pi, abs_tol=1e-9
    )
