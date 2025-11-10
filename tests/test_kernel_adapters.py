"""
Tests for external kernel adapters.
"""

from succinct_mcmc.mcmc import StepFunctionKernel, PyMCStepKernel, NumPyroStepKernel


def test_step_function_kernel():
    kernel = StepFunctionKernel(lambda x, rng: x + 1)
    assert kernel.step(0, None) == 1


def test_pymc_adapter_factory():
    def factory(rng):
        def step(x):
            return x + rng.random()

        class Dummy:
            def __init__(self, fn):
                self.fn = fn

            def step(self, point):
                return self.fn(point)

        return Dummy(step)

    adapter = PyMCStepKernel(factory, lambda x: x)
    out = adapter.step(0.0, __import__("random").Random(0))
    assert isinstance(out, float)


def test_numpyro_adapter_factory():
    def factory(rng):
        def step(x):
            return x + 1

        return step

    adapter = NumPyroStepKernel(factory, lambda x: x)
    out = adapter.step(2, None)
    assert out == 3

