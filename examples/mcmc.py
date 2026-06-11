"""MCMC as ONE example step_fn for rewind (it is not the identity of the library).

A random-walk Metropolis sampler of the standard normal, written as a pure
step_fn(state, rng) -> state. Freeze any adaptation before recording.
"""
from __future__ import annotations

import math


def _log_prob(x: float) -> float:
    return -0.5 * x * x


def metropolis_step(x: float, rng, step_scale: float = 1.0) -> float:
    proposal = x + rng.normalvariate(0.0, step_scale)  # python backend API
    if math.log(rng.random()) < _log_prob(proposal) - _log_prob(x):
        return proposal
    return x
