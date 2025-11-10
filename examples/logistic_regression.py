"""
Example: Bayesian logistic regression with Succinct MCMC.

This is a sketch for how succinct chains can be used in a real model.

In a full implementation:
- You'd use a vector-valued state (e.g., tuple of coefficients),
- Possibly wrap NumPy / JAX / torch operations,
- And use an HMC or MH kernel over R^d.

Here we:
- Build a tiny synthetic logistic regression dataset,
- Use a simple random-walk MH kernel on parameter vector,
- Run a succinct chain and compute posterior mean of coefficients.
"""

import math
import random
from typing import List, Tuple

from succinct_mcmc.mcmc import StepFunctionKernel
from succinct_mcmc.trace import SuccinctChain
from succinct_mcmc.diagnostics import mean


def make_synthetic_data(n: int = 200) -> Tuple[List[float], List[int]]:
    xs = []
    ys = []
    for _ in range(n):
        x = random.uniform(-3, 3)
        # True parameter ~ 1.5
        p = 1 / (1 + math.exp(-(1.5 * x)))
        y = 1 if random.random() < p else 0
        xs.append(x)
        ys.append(y)
    return xs, ys


def log_prob(theta: float, xs: List[float], ys: List[int]) -> float:
    # Prior: N(0, 10^2)
    lp = -0.5 * (theta / 10.0) ** 2
    # Likelihood:
    for x, y in zip(xs, ys):
        z = theta * x
        if z >= 0:
            p = 1 / (1 + math.exp(-z))
        else:
            ez = math.exp(z)
            p = ez / (1 + ez)
        lp += y * math.log(p + 1e-12) + (1 - y) * math.log(1 - p + 1e-12)
    return lp


def rw_step_factory(xs, ys, step_scale: float = 0.1):
    def step(theta: float, rng) -> float:
        cur_lp = log_prob(theta, xs, ys)
        proposal = theta + step_scale * rng.normalvariate(0.0, 1.0)
        prop_lp = log_prob(proposal, xs, ys)
        log_alpha = prop_lp - cur_lp
        if log_alpha >= 0 or math.log(rng.random()) < log_alpha:
            return proposal
        return theta

    return step


def main():
    xs, ys = make_synthetic_data()
    step_fn = rw_step_factory(xs, ys)
    kernel = StepFunctionKernel(step_fn, name="logistic_rw")

    chain = SuccinctChain(
        kernel=kernel,
        initial_state=0.0,
        num_steps=200_000,
        master_seed=123,
    )
    chain.run()

    theta_mean = mean(chain, lambda th: th)
    print(f"Posterior mean(theta) ≈ {theta_mean:.3f} (true ≈ 1.5)")


if __name__ == "__main__":
    main()
