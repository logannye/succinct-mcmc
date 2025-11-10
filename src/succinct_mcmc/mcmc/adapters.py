"""
Adapters for external MCMC libraries.

Purpose:
- Let users plug in existing step functions (e.g. from PyMC, NumPyro, hand-written)
  into SuccinctChain *without* modifying their modeling code.
- Enforce the determinism contract required for succinct replay.

The key pattern:
    - Wrap any `step_fn(state, rng)` into a TransitionKernel.
    - Optionally capture configuration / model info for artifact metadata.
"""

from typing import Callable, Any
from .base import TransitionKernel


class StepFunctionKernel(TransitionKernel):
    """
    Wrap a plain (state, rng) -> new_state function as a TransitionKernel.

    This is the simplest adapter. It assumes:
        - `step_fn` is purely functional (no hidden global mutable state).
        - All randomness comes from the provided `rng`.
    """

    def __init__(self, step_fn: Callable[[Any, Any], Any], name: str | None = None):
        self.step_fn = step_fn
        self.name = name or getattr(step_fn, "__name__", "step_fn")

    def step(self, x: Any, rng: Any) -> Any:
        return self.step_fn(x, rng)
