"""
Example: High-dimensional HMC with Succinct MCMC (conceptual).

This is a scaffold for:
- using SimpleHMC (or a real HMC) on R^d,
- running long chains with succinct storage.

A real version would:
- Represent state as a vector (numpy, torch, etc.),
- Implement vectorized grad_log_prob_fn,
- Possibly subclass/extend SimpleHMC.
"""

import math
from typing import List

from succinct_mcmc.mcmc import SimpleHMC
from succinct_mcmc.trace import SuccinctChain
from succinct_mcmc.diagnostics import mean


def log_prob_iso_gaussian(vec: List[float]) -> float:
    return -0.5 * sum(x * x for x in vec)


def grad_log_prob_iso_gaussian(vec: List[float]) -> List[float]:
    # Gradient of -0.5 * ||x||^2 is -x
    return [-x for x in vec]


def main():
    dim = 5
    initial_state = [0.0] * dim

    # For real HMC in high-d, you'd tune step_size & L carefully.
    kernel = SimpleHMC(
        log_prob_fn=lambda x1d: log_prob_iso_gaussian([x1d]),
        grad_log_prob_fn=lambda x1d: -x1d,
        step_size=0.1,
        num_leapfrog_steps=5,
    )
    # NOTE: This SimpleHMC is scalar; this example is illustrative only.

    chain = SuccinctChain(kernel, initial_state=0.0, num_steps=100_000)
    chain.run()

    est_mean = mean(chain, lambda x: x)
    print("Estimated mean (1D slice):", est_mean)


if __name__ == "__main__":
    main()
