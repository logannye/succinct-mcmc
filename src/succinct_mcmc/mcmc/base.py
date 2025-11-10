"""
Abstract interface for MCMC transition kernels.

SuccinctChain is agnostic to MCMC flavor; it only needs a "step" operation:

    x_{t+1} = step(x_t, rng)

This module defines:
- TransitionKernel protocol / base class.
- Minimal contracts for determinism and replayability.
"""

from typing import Protocol, Any


class TransitionKernel(Protocol):
    """
    Protocol for MCMC transition kernels.

    Requirements:
        - Stateless or purely parameterized; no hidden global mutable state.
        - Deterministic given (current_state, rng_draws).
    """

    def step(self, x: Any, rng: Any) -> Any:
        """
        Compute one MCMC transition.

        Args:
            x: current state.
            rng: random generator with .random(), .normalvariate(), etc.

        Returns:
            next state.
        """
        ...
