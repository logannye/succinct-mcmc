"""
Tests for R-hat and ESS diagnostics.
"""

import math
import random

import pytest

from succinct_mcmc.mcmc import StepFunctionKernel
from succinct_mcmc.trace import MultiChain, SuccinctChain
from succinct_mcmc.diagnostics import (
    rank_normalized_split_ess,
    rank_normalized_split_rhat,
    split_ess,
    split_rhat,
    multivariate_split_ess,
)


def make_random_walk_chain(seed: int, *, num_steps: int = 400) -> SuccinctChain:
    def step(x, rng):
        return x + rng.normalvariate(0.0, 1.0)

    kernel = StepFunctionKernel(step)
    chain = SuccinctChain(kernel, initial_state=0.0, num_steps=num_steps, master_seed=seed)
    chain.run()
    return chain


def test_split_rhat_close_to_one_for_mixed_chains():
    chains = [make_random_walk_chain(seed) for seed in range(4)]
    mc = MultiChain(chains)
    rhat_value = split_rhat(mc, skip_warmup=True)
    assert math.isclose(rhat_value, 1.0, rel_tol=1e-1)


def test_rank_normalized_rhat_matches_classic_for_gaussian():
    chains = [make_random_walk_chain(seed) for seed in range(4)]
    mc = MultiChain(chains)
    rn = rank_normalized_split_rhat(mc, skip_warmup=True)
    classic = split_rhat(mc, skip_warmup=True)
    assert math.isfinite(rn)
    assert abs(rn - classic) < 0.05


def test_split_ess_positive():
    chains = [make_random_walk_chain(seed) for seed in range(4)]
    mc = MultiChain(chains)
    ess_value = split_ess(mc, skip_warmup=True)
    assert ess_value > 0


def test_rank_normalized_ess_positive():
    chains = [make_random_walk_chain(seed) for seed in range(4)]
    mc = MultiChain(chains)
    ess_value = rank_normalized_split_ess(mc, skip_warmup=True)
    assert ess_value > 0


def test_multivariate_ess_min_dimension():
    chains = [make_random_walk_chain(seed) for seed in range(4)]
    mc = MultiChain(chains)
    result = multivariate_split_ess(mc, lambda x: (x, x * x))
    assert result["overall"] == min(result["per_dimension"])
    assert result["overall"] > 0


def test_split_rhat_window_matches_subset():
    chains = [make_random_walk_chain(seed) for seed in range(4)]
    mc = MultiChain(chains)
    full = split_rhat(mc, skip_warmup=True)
    windowed = split_rhat(mc, skip_warmup=True, max_samples=200)
    assert math.isfinite(windowed)
    assert windowed > 0
    assert abs(windowed - full) < 0.2


def test_insufficient_samples_raise_error():
    chains = [make_random_walk_chain(seed, num_steps=4) for seed in range(2)]
    mc = MultiChain(chains)
    with pytest.raises(ValueError):
        split_rhat(mc, skip_warmup=False)
