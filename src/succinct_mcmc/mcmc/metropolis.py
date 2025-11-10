"""
Metropolis(-Hastings) transition kernels.

Provides reusable kernels:
- GaussianRandomWalkMH for scalar/vector states.
- AdaptiveGaussianRandomWalkMH with simple Robbins-Monro adaptation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Callable, Iterable

from .base import TransitionKernel


def _as_sequence(state: Any) -> tuple[list[float], Callable[[Iterable[float]], Any]]:
    """Convert scalar or vector states into list form with a packer."""
    if isinstance(state, list):
        return list(state), lambda seq: list(seq)
    if isinstance(state, tuple):
        return list(state), lambda seq: tuple(seq)
    return [float(state)], lambda seq: seq[0]


class GaussianRandomWalkMH(TransitionKernel):
    """Metropolis-Hastings with symmetric Gaussian random-walk proposals."""

    def __init__(
        self,
        log_prob_fn: Callable[[Any], float],
        step_scale: float | Sequence[float] = 1.0,
    ) -> None:
        self.log_prob_fn = log_prob_fn
        self._set_step_scale(step_scale)
        self._last_accept: bool = False

    @property
    def step_scale(self) -> float | Sequence[float]:
        return self._step_scale_raw

    @property
    def last_accept(self) -> bool:
        return self._last_accept

    def _set_step_scale(self, value: float | Sequence[float]) -> None:
        if isinstance(value, Sequence) and not isinstance(value, (list, tuple)):
            value = list(value)
        if isinstance(value, Sequence):
            if not value:
                raise ValueError("step_scale sequence must be non-empty")
            scales = [float(v) for v in value]
            if any(s <= 0 for s in scales):
                raise ValueError("step_scale entries must be positive")
            self._step_scale_raw: float | Sequence[float] = scales
        else:
            scale = float(value)
            if scale <= 0:
                raise ValueError("step_scale must be positive")
            self._step_scale_raw = scale

    def _scales_for_dim(self, dim: int) -> list[float]:
        raw = self._step_scale_raw
        if isinstance(raw, Sequence):
            if len(raw) != dim:
                raise ValueError(
                    f"step_scale length {len(raw)} does not match state dimension {dim}"
                )
            return list(raw)
        return [raw] * dim

    def _propose(self, x: Any, rng: Any) -> Any:
        vec, pack = _as_sequence(x)
        scales = self._scales_for_dim(len(vec))
        proposal = [xi + scale * rng.normalvariate(0.0, 1.0) for xi, scale in zip(vec, scales)]
        return pack(proposal)

    def step(self, x: Any, rng: Any) -> Any:
        current_lp = self.log_prob_fn(x)
        proposal = self._propose(x, rng)
        proposal_lp = self.log_prob_fn(proposal)

        log_alpha = proposal_lp - current_lp
        if log_alpha >= 0 or math.log(rng.random()) < log_alpha:
            self._last_accept = True
            return proposal

        self._last_accept = False
        return x


class AdaptiveGaussianRandomWalkMH(GaussianRandomWalkMH):
    """Gaussian random walk with simple step-scale adaptation."""

    def __init__(
        self,
        log_prob_fn: Callable[[Any], float],
        step_scale: float = 1.0,
        target_accept: float = 0.234,
        adaptation_rate: float = 0.05,
        min_scale: float = 1e-3,
        max_scale: float = 10.0,
    ) -> None:
        if isinstance(step_scale, Sequence):
            raise ValueError("AdaptiveGaussianRandomWalkMH requires scalar step_scale")
        super().__init__(log_prob_fn, step_scale=step_scale)
        if not (0.0 < target_accept < 1.0):
            raise ValueError("target_accept must lie in (0, 1)")
        if adaptation_rate <= 0:
            raise ValueError("adaptation_rate must be positive")
        self.target_accept = target_accept
        self.adaptation_rate = adaptation_rate
        self.min_scale = min_scale
        self.max_scale = max_scale
        self._accept_stat = target_accept

    def step(self, x: Any, rng: Any) -> Any:
        state = super().step(x, rng)
        accept_indicator = 1.0 if self.last_accept else 0.0
        self._accept_stat = (
            self._accept_stat + self.adaptation_rate * (accept_indicator - self._accept_stat)
        )
        adjustment = math.exp(self.adaptation_rate * (accept_indicator - self.target_accept))
        new_scale = float(self.step_scale) * adjustment
        new_scale = max(self.min_scale, min(self.max_scale, new_scale))
        self._set_step_scale(new_scale)
        return state

    @property
    def acceptance_statistic(self) -> float:
        return self._accept_stat
