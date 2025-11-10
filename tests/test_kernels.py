"""
Tests for the expanded MCMC kernel library.
"""

import random

import pytest

from succinct_mcmc.mcmc import (
    AdaptiveGaussianRandomWalkMH,
    GaussianRandomWalkMH,
    SimpleHMC,
)


def std_normal_log_prob(x):
    if isinstance(x, (list, tuple)):
        return -0.5 * sum(xi * xi for xi in x)
    return -0.5 * x * x


def test_gaussian_random_walk_vector_state_is_deterministic():
    kernel = GaussianRandomWalkMH(std_normal_log_prob, step_scale=[0.5, 0.25])
    state = (0.0, 0.0)

    rng1 = random.Random(123)
    rng2 = random.Random(123)

    proposed1 = kernel.step(state, rng1)
    proposed2 = kernel.step(state, rng2)

    assert isinstance(proposed1, tuple)
    assert proposed1 == proposed2
    assert kernel.step_scale == [0.5, 0.25]


def test_gaussian_random_walk_invalid_scale_length_raises():
    kernel = GaussianRandomWalkMH(std_normal_log_prob, step_scale=[1.0, 1.0])
    with pytest.raises(ValueError):
        kernel.step((0.0, 0.0, 0.0), random.Random(1))


def test_adaptive_random_walk_reduces_large_step():
    kernel = AdaptiveGaussianRandomWalkMH(
        std_normal_log_prob,
        step_scale=5.0,
        target_accept=0.3,
        adaptation_rate=0.1,
    )
    rng = random.Random(99)
    state = 0.0
    initial_scale = kernel.step_scale

    for _ in range(50):
        state = kernel.step(state, rng)

    assert float(kernel.step_scale) < initial_scale
    assert 0.0 < kernel.acceptance_statistic < 1.0


def test_simple_hmc_vector_state_runs():
    def grad(vec):
        if isinstance(vec, (list, tuple)):
            return [-v for v in vec]
        return -vec

    kernel = SimpleHMC(std_normal_log_prob, grad, step_size=0.1, num_leapfrog_steps=5)
    state = [0.2, -0.1]

    rng1 = random.Random(7)
    rng2 = random.Random(7)

    proposed1 = kernel.step(state, rng1)
    proposed2 = kernel.step(state, rng2)

    assert isinstance(proposed1, list)
    assert proposed1 == proposed2
    assert len(proposed1) == 2


def test_simple_hmc_rejects_invalid_params():
    with pytest.raises(ValueError):
        SimpleHMC(std_normal_log_prob, lambda x: x, step_size=-0.1)
    with pytest.raises(ValueError):
        SimpleHMC(std_normal_log_prob, lambda x: x, step_size=0.1, num_leapfrog_steps=0)
