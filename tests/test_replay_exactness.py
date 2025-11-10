"""
Tests: replay correctness.

Goal:
- Ensure that SuccinctChain.get(t) is consistent with a dense reference chain
  for simple kernels.
"""

import math

from succinct_mcmc.mcmc import GaussianRandomWalkMH
from succinct_mcmc.trace import SuccinctChain


def dense_chain(kernel, x0, num_steps, seed=0):
    import random
    rng = random.Random(seed)
    xs = [x0]
    x = x0
    for _ in range(1, num_steps):
        x = kernel.step(x, rng)
        xs.append(x)
    return xs


def test_replay_matches_dense():
    num_steps = 1000

    def log_prob(x: float) -> float:
        return -0.5 * x * x

    kernel = GaussianRandomWalkMH(log_prob_fn=log_prob, step_scale=1.0)

    # Dense reference
    ref = dense_chain(kernel, x0=0.0, num_steps=num_steps, seed=123)

    # Succinct chain uses same master_seed and kernel behavior
    chain = SuccinctChain(kernel, initial_state=0.0, num_steps=num_steps, master_seed=123)
    chain.run()

    for t in range(num_steps):
        assert math.isclose(chain.get(t), ref[t], rel_tol=1e-12, abs_tol=1e-12)
