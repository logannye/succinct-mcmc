"""
Example: Succinct MCMC for a 1D standard normal target.

Demonstrates:
- Wrapping a simple Metropolis-Hastings kernel.
- Running a long succinct chain.
- Computing a mean from the succinct representation.
"""

import math

from succinct_mcmc.mcmc import GaussianRandomWalkMH
from succinct_mcmc.trace import SuccinctChain
from succinct_mcmc.diagnostics import mean


def log_prob(x: float) -> float:
    # Standard normal: log N(0,1)
    return -0.5 * x * x


def main():
    num_steps = 10**6

    kernel = GaussianRandomWalkMH(log_prob_fn=log_prob, step_scale=1.0)
    chain = SuccinctChain(
        kernel=kernel,
        initial_state=0.0,
        num_steps=num_steps,
        master_seed=42,
    )

    chain.run()

    est_mean = mean(chain, lambda x: x)
    print(f"Estimated mean: {est_mean:.4f} (true = 0.0)")
    # For quick sanity, we might also print a few random positions
    print("Sample X_100:", chain.get(100))


if __name__ == "__main__":
    main()
