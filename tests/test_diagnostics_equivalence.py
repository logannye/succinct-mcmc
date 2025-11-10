"""
Tests: diagnostics on succinct vs dense.

Goal:
- For a simple chain, compare succinct-based mean to dense mean.
"""

from succinct_mcmc.mcmc import GaussianRandomWalkMH
from succinct_mcmc.trace import SuccinctChain
from succinct_mcmc.diagnostics import mean


def test_mean_matches_dense():
    num_steps = 5000

    def log_prob(x: float) -> float:
        return -0.5 * x * x

    kernel = GaussianRandomWalkMH(log_prob_fn=log_prob, step_scale=1.0)

    # Succinct
    chain = SuccinctChain(kernel, initial_state=0.0, num_steps=num_steps, master_seed=999)
    chain.run()
    mu_succinct = mean(chain, lambda x: x)

    # Dense reference using same kernel behavior
    import random
    rng = random.Random(999)
    xs = [0.0]
    x = 0.0
    for _ in range(1, num_steps):
        x = kernel.step(x, rng)
        xs.append(x)
    mu_dense = sum(xs) / len(xs)

    assert abs(mu_succinct - mu_dense) < 1e-6
