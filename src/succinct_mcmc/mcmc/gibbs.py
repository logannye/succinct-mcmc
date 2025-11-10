"""
Generic Gibbs-style transition kernel.

Purpose:
- Provide a template for Gibbs or block-Gibbs updates.
- Show how "update one coordinate at a time using its conditional" can fit into
  the deterministic step(x, rng) interface.

This is a stub to be filled with concrete models (e.g. Gaussian conditionals),
but it documents the expected structure.
"""

from typing import Protocol, Any
from .base import TransitionKernel


class GibbsUpdater(Protocol):
    """
    Protocol for a single-variable (or block) conditional updater.

    Given:
        - current full state,
        - RNG,
    it returns an updated full state with one component resampled from
    its conditional distribution.
    """

    def update(self, state: Any, rng: Any) -> Any:  # pragma: no cover - interface
        ...


class GenericGibbsKernel(TransitionKernel):
    """
    Compose multiple conditional updaters into one Gibbs sweep.

    Args:
        updaters: list of GibbsUpdater objects, each responsible for
                  updating part of the state.

    On each step():
        - Apply all updaters in sequence to form one full Gibbs sweep.
    """

    def __init__(self, updaters: list[GibbsUpdater]):
        self.updaters = list(updaters)

    def step(self, x: Any, rng: Any) -> Any:
        state = x
        for updater in self.updaters:
            state = updater.update(state, rng)
        return state
