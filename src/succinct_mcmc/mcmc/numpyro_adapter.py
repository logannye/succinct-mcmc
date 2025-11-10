"""
NumPyro adapter for SuccinctChain.
"""

from __future__ import annotations

from typing import Any, Callable

from .base import TransitionKernel


class NumPyroStepKernel(TransitionKernel):
    """
    Adapter wrapping a NumPyro transition (e.g., HMC/NUTS) for deterministic replay.
    """

    def __init__(
        self,
        factory: Callable[[Any], Any],
        extract: Callable[[Any], Any],
    ):
        self._factory = factory
        self._extract = extract

    def step(self, x: Any, rng: Any) -> Any:
        kernel = self._factory(rng)
        new_state = kernel(x)
        return self._extract(new_state)

