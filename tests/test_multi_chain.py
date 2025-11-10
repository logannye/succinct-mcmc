"""
Tests: MultiChain behavior.

Goal:
- Ensure MultiChain enforces consistent lengths.
- Ensure basic iteration works.
"""

import pytest

from succinct_mcmc.trace import SuccinctChain, MultiChain
from succinct_mcmc.mcmc import StepFunctionKernel


def dummy_step(x, rng):
    return x + 1


def make_chain(T, seed):
    kernel = StepFunctionKernel(dummy_step)
    c = SuccinctChain(kernel, initial_state=0, num_steps=T, master_seed=seed)
    c.run()
    return c


def test_multichain_lengths_agree():
    c1 = make_chain(100, 1)
    c2 = make_chain(100, 2)
    mc = MultiChain([c1, c2])
    assert mc.num_chains == 2
    assert mc.num_steps == 100


def test_multichain_mismatched_lengths_raises():
    c1 = make_chain(100, 1)
    c2 = make_chain(50, 2)
    with pytest.raises(ValueError):
        MultiChain([c1, c2])
