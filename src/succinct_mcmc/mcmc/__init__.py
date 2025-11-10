"""
MCMC transition kernels and adapters.

This subpackage provides:
- TransitionKernel protocol defining the step interface.
- Reference implementations:
    - GaussianRandomWalkMH (symmetric MH proposals),
    - AdaptiveGaussianRandomWalkMH (Robbins-Monro adaptation),
    - SimpleHMC (scalar/vector leapfrog),
    - GenericGibbsKernel for block updates.
- StepFunctionKernel for wrapping arbitrary (state, rng) -> state functions.
"""

from .base import TransitionKernel
from .metropolis import AdaptiveGaussianRandomWalkMH, GaussianRandomWalkMH
from .hmc import SimpleHMC
from .gibbs import GenericGibbsKernel
from .adapters import StepFunctionKernel
from .pymc_adapter import PyMCStepKernel
from .numpyro_adapter import NumPyroStepKernel

__all__ = [
    "TransitionKernel",
    "GaussianRandomWalkMH",
    "AdaptiveGaussianRandomWalkMH",
    "SimpleHMC",
    "GenericGibbsKernel",
    "StepFunctionKernel",
    "PyMCStepKernel",
    "NumPyroStepKernel",
]
