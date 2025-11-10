"""
PyMC adapter for SuccinctChain.

Wraps a PyMC step kernel (RandomStep) so it can be executed inside SuccinctChain.
"""

from __future__ import annotations

from typing import Any, Callable

from .base import TransitionKernel


class PyMCStepKernel(TransitionKernel):
    """
    Adapter around a PyMC `RandomStep` object.

    PyMC steps expose `step(point, state=None)` using PyMC's RandomStream. To keep
    replay deterministic, we require a function that, given a PyMC RNG state,
    produces a fresh step kernel with deterministic randomness.
    """

    def __init__(
        self,
        factory: Callable[[Any], Any],
        extract: Callable[[Any], Any],
    ):
        """
        Args:
            factory: Function accepting a Python RNG (random.Random-compatible)
                and returning a PyMC step kernel bound to that RNG.
            extract: Callable converting PyMC step returns into a serializable state.
        """
        self._factory = factory
        self._extract = extract

    def step(self, x: Any, rng: Any) -> Any:
        stepper = self._factory(rng)
        new_point = stepper.step(x)
        return self._extract(new_point)

