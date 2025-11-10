"""
Tests for iterator utilities over succinct chains.
"""

from succinct_mcmc.mcmc import StepFunctionKernel
from succinct_mcmc.trace import SuccinctChain
from succinct_mcmc.trace.iterator import block_iter, iter_range, iter_states


def make_chain(num_steps: int = 100, warmup: int = 0) -> SuccinctChain:
    kernel = StepFunctionKernel(lambda x, rng: x + 1)
    chain = SuccinctChain(
        kernel,
        initial_state=0,
        num_steps=num_steps,
        master_seed=1,
        warmup_steps=warmup,
    )
    chain.run()
    return chain


def test_iter_states_matches_chain_iter():
    chain = make_chain()
    assert list(iter_states(chain)) == list(chain.iter())


def test_iter_states_respects_warmup():
    chain = make_chain(warmup=5)
    states = list(iter_states(chain, skip_warmup=True))
    assert states[0] == 5


def test_iter_range_subset():
    chain = make_chain()
    subset = list(iter_range(chain, 10, 15))
    assert subset == list(range(10, 15))


def test_iter_range_respects_warmup():
    chain = make_chain(warmup=10)
    subset = list(iter_range(chain, 0, 12, skip_warmup=True))
    assert subset == list(range(10, 12))


def test_block_iter_groups_states():
    chain = make_chain(20)
    blocks = list(block_iter(chain))
    assert blocks
    total = sum(len(states) for _, states in blocks)
    assert total == chain.num_steps


def test_block_iter_respects_warmup():
    chain = make_chain(20, warmup=5)
    blocks = list(block_iter(chain, skip_warmup=True))
    total = sum(len(states) for _, states in blocks)
    assert total == chain.num_steps - chain.warmup_steps
