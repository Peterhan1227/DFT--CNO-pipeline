"""Animation easing + rotation-operation math (no rendering required)."""

from __future__ import annotations

import numpy as np
import pytest

from cno_visualizer.animation.easing import (
    get_rate_function,
    linear,
    smootherstep,
    there_and_back,
)
from cno_visualizer.animation.operations import (
    axis_to_cartesian,
    make_rotation,
    rotation_matrix_4x4,
)


def test_easing_endpoints_and_monotonic():
    for fn in (linear, smootherstep):
        assert fn(0.0) == pytest.approx(0.0)
        assert fn(1.0) == pytest.approx(1.0)
        xs = np.linspace(0, 1, 50)
        ys = np.array([fn(x) for x in xs])
        assert np.all(np.diff(ys) >= -1e-12)  # non-decreasing


def test_smootherstep_zero_velocity_at_ends():
    eps = 1e-4
    # derivative ~ 0 at both ends (the gentle accelerate/decelerate)
    assert abs(smootherstep(eps) - smootherstep(0)) / eps < 1e-2
    assert abs(smootherstep(1) - smootherstep(1 - eps)) / eps < 1e-2


def test_there_and_back_returns():
    assert there_and_back(0.0) == pytest.approx(0.0)
    assert there_and_back(0.5) == pytest.approx(1.0)
    assert there_and_back(1.0) == pytest.approx(0.0, abs=1e-9)


def test_rotation_matrix_is_rigid_and_fixes_center():
    center = np.array([1.0, -2.0, 0.5])
    M = rotation_matrix_4x4(np.array([0.0, 0.0, 1.0]), np.pi / 3, center)
    R = M[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)
    # center is fixed
    ch = np.r_[center, 1.0]
    np.testing.assert_allclose((M @ ch)[:3], center, atol=1e-12)


def test_nfold_rotation_returns_to_identity():
    center = np.zeros(3)
    op = make_rotation(center, axis="z", n_fold=3)
    M = rotation_matrix_4x4(op.axis, np.deg2rad(op.angle_deg), center)
    M3 = M @ M @ M
    np.testing.assert_allclose(M3, np.eye(4), atol=1e-9)


def test_axis_crystallographic_direction_uses_lattice():
    lattice = np.diag([2.0, 3.0, 4.0])
    ax = axis_to_cartesian("001", lattice)
    np.testing.assert_allclose(ax, [0.0, 0.0, 1.0], atol=1e-12)
    ax2 = axis_to_cartesian("100", lattice)
    np.testing.assert_allclose(ax2, [1.0, 0.0, 0.0], atol=1e-12)


def test_get_rate_function_unknown_raises():
    with pytest.raises(ValueError):
        get_rate_function("nope")


def test_nice_step_is_round_and_reasonable():
    from cno_visualizer.animation.builder import _nice_step

    for length in (1.0, 3.7, 6.2, 12.0, 48.0):
        step = _nice_step(length, target=4)
        assert step > 0
        n = length / step
        assert 1 <= n <= 12  # a sensible number of ticks
        # mantissa is one of the round choices
        mant = step / (10 ** np.floor(np.log10(step)))
        assert np.isclose(mant, round(mant, 3)) and mant in (1.0, 2.0, 2.5, 5.0)


def test_build_coordinate_frame_runs_offscreen():
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    from cno_visualizer.animation.builder import build_coordinate_frame

    pl = pv.Plotter(off_screen=True)
    actors = build_coordinate_frame(pl, np.zeros(3), 5.0, show_numbers=False)
    assert len(actors) >= 6  # at least a line + cone per axis
    pl.close()
