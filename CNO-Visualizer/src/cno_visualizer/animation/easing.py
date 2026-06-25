"""Rate functions (easing) — the smooth accelerate/decelerate that Manim uses.

A rate function maps normalized time ``t in [0, 1]`` to an eased progress in
``[0, 1]``.  ``smootherstep`` (zero first *and* second derivative at both ends)
is the default — it is the "gradually speed up, then gently arrive" motion the
user asked for, and matches the feel of Manim's ``smooth``.
"""

from __future__ import annotations

import math
from typing import Callable


def _clip01(t: float) -> float:
    return 0.0 if t < 0.0 else 1.0 if t > 1.0 else t


def linear(t: float) -> float:
    return _clip01(t)


def smoothstep(t: float) -> float:
    """Cubic S-curve ``3t^2 - 2t^3`` (zero first derivative at the ends)."""
    t = _clip01(t)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(t: float) -> float:
    """Quintic S-curve ``6t^5 - 15t^4 + 10t^3`` (zero 1st *and* 2nd derivative)."""
    t = _clip01(t)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def ease_in_out_sine(t: float) -> float:
    t = _clip01(t)
    return 0.5 * (1.0 - math.cos(math.pi * t))


def rush_into(t: float) -> float:
    """Slow start, fast finish (ease-in only)."""
    return 2.0 * smootherstep(_clip01(t) / 2.0)


def rush_from(t: float) -> float:
    """Fast start, slow finish (ease-out only)."""
    return 2.0 * smootherstep(_clip01(t) / 2.0 + 0.5) - 1.0


def there_and_back(t: float) -> float:
    """Go to the destination and smoothly return (0 -> 1 -> 0)."""
    t = _clip01(t)
    return smootherstep(2.0 * t) if t < 0.5 else smootherstep(2.0 - 2.0 * t)


RATE_FUNCTIONS: dict[str, Callable[[float], float]] = {
    "linear": linear,
    "smoothstep": smoothstep,
    "smootherstep": smootherstep,
    "ease_in_out_sine": ease_in_out_sine,
    "rush_into": rush_into,
    "rush_from": rush_from,
    "there_and_back": there_and_back,
}


def get_rate_function(name: str) -> Callable[[float], float]:
    key = str(name).strip().lower()
    if key not in RATE_FUNCTIONS:
        raise ValueError(
            f"unknown easing {name!r}; choose from {sorted(RATE_FUNCTIONS)}"
        )
    return RATE_FUNCTIONS[key]


__all__ = [
    "linear", "smoothstep", "smootherstep", "ease_in_out_sine",
    "rush_into", "rush_from", "there_and_back",
    "RATE_FUNCTIONS", "get_rate_function",
]
