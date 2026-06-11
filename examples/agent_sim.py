"""Hero demo: a cheap-tick stochastic sim that occasionally blows up to NaN.

Demonstrates the rewind workflow: record millions of ticks in sqrt(T) memory,
scrub to the exact tick a NaN first appears, inspect the state and the RNG draw
that caused it, then branch a counterfactual that clamps and continues.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Rare tail events amplify the state hard; a handful in a row overflow a float
# to +inf, which the next op turns into NaN. Tuned so a NaN reliably occurs
# within a few thousand cheap ticks (the bug rewind helps you localize).
_TAIL_THRESHOLD = 2.0
_TAIL_FACTOR = 1e50


@dataclass
class NaNState:
    value: float


def sim_step(s: NaNState, rng) -> NaNState:
    v = s.value
    shock = rng.normalvariate(0.0, 1.0)
    v = v * (1.0 + 0.5 * shock)
    if abs(shock) > _TAIL_THRESHOLD:        # rare tail event amplifies hard
        v = v * _TAIL_FACTOR
    if math.isinf(v):                       # once it overflows -> NaN downstream
        v = v - v                           # inf - inf -> NaN
    return NaNState(value=v)
