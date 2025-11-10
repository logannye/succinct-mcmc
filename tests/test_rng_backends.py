"""
Tests: RNG backend registry and determinism guarantees.
"""

import math
import uuid

import pytest

from succinct_mcmc.core.rng import (
    RNGBackend,
    RNGFactoryError,
    available_rng_backends,
    make_block_rng,
    register_rng_backend,
)
from succinct_mcmc.mcmc import StepFunctionKernel
from succinct_mcmc.trace import SuccinctChain


def test_python_backend_is_deterministic():
    rng1 = make_block_rng(123, backend="python")
    rng2 = make_block_rng(123, backend="python")

    draws1 = [rng1.random() for _ in range(3)]
    draws2 = [rng2.random() for _ in range(3)]
    assert draws1 == draws2

    normals1 = [rng1.normalvariate(0.0, 1.0) for _ in range(2)]
    normals2 = [rng2.normalvariate(0.0, 1.0) for _ in range(2)]
    assert normals1 == normals2


def test_register_custom_backend_and_use():
    class DummyRNG:
        def __init__(self, seed: int):
            self.seed = seed

        def random(self):
            return float(self.seed)

    name = f"dummy-{uuid.uuid4()}"

    register_rng_backend(
        RNGBackend(
            name=name,
            factory=lambda seed: DummyRNG(seed),
            provides="Dummy deterministic RNG",
        ),
        overwrite=True,
    )

    rng = make_block_rng(7, backend=name)
    assert math.isclose(rng.random(), 7.0)


def test_invalid_backend_raises_error():
    with pytest.raises(RNGFactoryError):
        make_block_rng(0, backend="does-not-exist")


def test_chain_rejects_unknown_backend():
    kernel = StepFunctionKernel(lambda x, rng: x)
    with pytest.raises(ValueError):
        SuccinctChain(kernel, initial_state=0, num_steps=10, rng_backend="invalid")


@pytest.mark.skipif("numpy" not in available_rng_backends(), reason="NumPy backend unavailable")
def test_numpy_backend_consistency():
    rng1 = make_block_rng(42, backend="numpy")
    rng2 = make_block_rng(42, backend="numpy")

    draws1 = rng1.random(5)
    draws2 = rng2.random(5)
    assert (draws1 == draws2).all()

    normal1 = rng1.normal(loc=0.0, scale=1.0, size=3)
    normal2 = rng2.normal(loc=0.0, scale=1.0, size=3)
    assert (normal1 == normal2).all()
