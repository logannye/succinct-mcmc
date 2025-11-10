"""
Hamiltonian Monte Carlo (HMC) transition kernel (simplified, educational stub).

Purpose:
- Provide a reference HMC-style TransitionKernel that is compatible with
  SuccinctChain's deterministic replay requirements.
- Demonstrate how gradient-based proposals fit into the succinct MCMC framework.

Design constraints for succinct MCMC:
- step(x, rng) must be deterministic given:
    - current state x
    - RNG stream for this step
- No hidden global state or side effects.

Notes:
- This stub intentionally avoids framework-specific tensors (PyTorch/JAX/etc.).
- A full implementation would:
    - accept vector states,
    - use leapfrog integrator,
    - support mass matrices / step size adaptation,
    - integrate with backend autodiff.
"""

from __future__ import annotations

import math
from typing import Callable, Any, Sequence

from .base import TransitionKernel


def _as_sequence(state: Any) -> tuple[list[float], Callable[[list[float]], Any]]:
    if isinstance(state, list):
        return list(state), lambda seq: list(seq)
    if isinstance(state, tuple):
        return list(state), lambda seq: tuple(seq)
    return [float(state)], lambda seq: seq[0]


def _to_vector(values: Any) -> list[float]:
    if isinstance(values, (list, tuple)):
        return [float(v) for v in values]
    return [float(values)]


class SimpleHMC(TransitionKernel):
    """
    Minimal HMC kernel supporting scalar or vector states.
    """

    def __init__(
        self,
        log_prob_fn: Callable[[Any], float],
        grad_log_prob_fn: Callable[[Any], Any],
        step_size: float = 0.1,
        num_leapfrog_steps: int = 10,
    ):
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if num_leapfrog_steps <= 0:
            raise ValueError("num_leapfrog_steps must be positive")
        self.log_prob_fn = log_prob_fn
        self.grad_log_prob_fn = grad_log_prob_fn
        self.step_size = step_size
        self.num_leapfrog_steps = num_leapfrog_steps

    def step(self, x: Any, rng: Any) -> Any:
        position_vec, pack = _as_sequence(x)
        momentum = [rng.normalvariate(0.0, 1.0) for _ in position_vec]

        current_lp = self.log_prob_fn(x)
        current_H = -current_lp + 0.5 * sum(p * p for p in momentum)

        prop_position = position_vec[:]
        prop_momentum = momentum[:]

        grad = _to_vector(self.grad_log_prob_fn(pack(prop_position)))
        prop_momentum = [p + 0.5 * self.step_size * g for p, g in zip(prop_momentum, grad)]

        for i in range(self.num_leapfrog_steps):
            prop_position = [q + self.step_size * p for q, p in zip(prop_position, prop_momentum)]
            grad = _to_vector(self.grad_log_prob_fn(pack(prop_position)))
            if i != self.num_leapfrog_steps - 1:
                prop_momentum = [p + self.step_size * g for p, g in zip(prop_momentum, grad)]

        prop_momentum = [p + 0.5 * self.step_size * g for p, g in zip(prop_momentum, grad)]
        reversed_momentum = [-p for p in prop_momentum]

        proposed_state = pack(prop_position)
        prop_lp = self.log_prob_fn(proposed_state)
        prop_H = -prop_lp + 0.5 * sum(p * p for p in reversed_momentum)

        log_alpha = current_H - prop_H
        if log_alpha >= 0 or math.log(rng.random()) < log_alpha:
            return proposed_state
        return x
