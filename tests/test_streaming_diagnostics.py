"""
Tests for streaming statistics helpers.
"""

import math
import random

import pytest

from succinct_mcmc.mcmc import StepFunctionKernel
from succinct_mcmc.trace import SuccinctChain
from succinct_mcmc.diagnostics.summary import mean, quantiles, summary, covariance


def make_linear_chain(num_steps: int = 200):
    kernel = StepFunctionKernel(lambda x, rng: x + 1)
    chain = SuccinctChain(kernel, initial_state=0, num_steps=num_steps, master_seed=5)
    chain.run()
    return chain


def test_mean_matches_closed_form():
    chain = make_linear_chain()
    est_mean = mean(chain, lambda x: x, skip_warmup=False)
    true_mean = (chain.num_steps - 1) / 2
    assert math.isclose(est_mean, true_mean, rel_tol=1e-9)


def test_summary_variance():
    chain = make_linear_chain()
    stats = summary(chain, lambda x: x, skip_warmup=False)
    assert math.isclose(stats["mean"], (chain.num_steps - 1) / 2, rel_tol=1e-9)
    expected_variance = sum((i - stats["mean"]) ** 2 for i in range(chain.num_steps)) / (
        chain.num_steps - 1
    )
    assert math.isclose(stats["variance"], expected_variance, rel_tol=1e-9)


def test_quantiles_interpolation():
    chain = make_linear_chain(num_steps=11)
    qs = {0.0, 0.25, 0.5, 0.75, 1.0}
    result = quantiles(chain, lambda x: x, qs, skip_warmup=False)
    for q in qs:
        assert q in result
    assert result[0.0] == 0
    assert result[1.0] == 10
    assert result[0.5] == 5


def test_quantiles_invalid_input():
    chain = make_linear_chain()
    with pytest.raises(ValueError):
        quantiles(chain, lambda x: x, [-0.1], skip_warmup=False)


def test_covariance_matches():
    chain = make_linear_chain()
    cov = covariance(chain, lambda x: [x, x])
    expected_mean = (chain.num_steps - 1) / 2
    expected_var = sum((i - expected_mean) ** 2 for i in range(chain.num_steps)) / (chain.num_steps - 1)
    assert math.isclose(cov[0][0], expected_var, rel_tol=1e-9)
    assert math.isclose(cov[0][1], expected_var, rel_tol=1e-9)
