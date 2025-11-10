"""
Tests for parallel expectation utilities.
"""

import math

import pytest

from succinct_mcmc.mcmc import StepFunctionKernel
from succinct_mcmc.trace import SuccinctChain
from succinct_mcmc.utils.parallel import parallel_expectation


def make_chain(num_steps: int = 200, warmup_steps: int = 0) -> SuccinctChain:
    def step(x, rng):
        return x + 1

    kernel = StepFunctionKernel(step)
    chain = SuccinctChain(
        kernel,
        initial_state=0,
        num_steps=num_steps,
        master_seed=7,
        warmup_steps=warmup_steps,
    )
    chain.run()
    return chain


def test_parallel_expectation_matches_serial():
    chain = make_chain()
    serial = chain.expectation(lambda x: x, skip_warmup=False)
    parallel = parallel_expectation(
        chain,
        lambda x: x,
        skip_warmup=False,
        chunk_size=25,
        max_workers=4,
    )
    assert math.isclose(parallel, serial, rel_tol=1e-12)


def test_parallel_expectation_respects_warmup():
    chain = make_chain(warmup_steps=40)
    serial = chain.expectation(lambda x: x, skip_warmup=True)
    parallel = parallel_expectation(
        chain,
        lambda x: x,
        skip_warmup=True,
        chunk_size=20,
        max_workers=2,
    )
    assert math.isclose(parallel, serial, rel_tol=1e-12)


def test_parallel_expectation_no_samples_error():
    chain = make_chain(num_steps=20, warmup_steps=20)
    with pytest.raises(ValueError):
        parallel_expectation(chain, lambda x: x, skip_warmup=True)

