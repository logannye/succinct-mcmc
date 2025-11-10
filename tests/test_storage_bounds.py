"""
Tests: storage scaling.

Goal:
- Check that the number of anchors / blocks is O(√T) for default config.
- This is a structural sanity check, not a formal proof.
"""

from succinct_mcmc.trace import SuccinctChain
from succinct_mcmc.mcmc import StepFunctionKernel


def dummy_step(x, rng):
    return x  # deterministic no-op


def test_anchor_count_scales_sublinearly():
    T = 10_000
    kernel = StepFunctionKernel(dummy_step)
    chain = SuccinctChain(kernel, initial_state=0, num_steps=T, master_seed=1)
    chain.run()

    num_blocks = len(chain.blocks)
    assert num_blocks * num_blocks <= T * 4  # rough: num_blocks ~ sqrt(T)
    assert len(chain.anchor_store) == num_blocks  # one anchor per block start
