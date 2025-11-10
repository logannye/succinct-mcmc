"""
Tests: resume and warmup functionality for SuccinctChain.
"""

from succinct_mcmc.mcmc import StepFunctionKernel
from succinct_mcmc.trace import SuccinctChain
from succinct_mcmc.core.anchors import AnchorStore
from succinct_mcmc.io.storage_backends import FilePerAnchorStorage


def increment_step(x, rng):
    # Deterministic increment, ignores RNG for simplicity.
    return x + 1


def test_checkpoint_resume_matches_full_run():
    num_steps = 400
    kernel = StepFunctionKernel(increment_step)

    baseline = SuccinctChain(kernel, initial_state=0, num_steps=num_steps, master_seed=7)
    baseline.run()

    partial = SuccinctChain(kernel, initial_state=0, num_steps=num_steps, master_seed=7)
    partial.run(until_step=partial.block_size)
    checkpoint = partial.checkpoint()

    resumed = SuccinctChain(
        kernel,
        initial_state=0,
        num_steps=num_steps,
        master_seed=7,
        resume_from=checkpoint,
    )
    resumed.run()

    for t in range(num_steps):
        assert resumed.get(t) == baseline.get(t)


def test_iter_skip_warmup_trims_prefix():
    num_steps = 100
    warmup = 10
    kernel = StepFunctionKernel(increment_step)
    chain = SuccinctChain(
        kernel,
        initial_state=0,
        num_steps=num_steps,
        master_seed=11,
        warmup_steps=warmup,
    )
    chain.run()

    full_states = list(chain.iter())
    trimmed_states = list(chain.iter(skip_warmup=True))

    assert len(full_states) == num_steps
    assert len(trimmed_states) == num_steps - warmup
    assert trimmed_states[0] == warmup

    # Expectation defaults to skipping warmup.
    tail_mean = chain.expectation(lambda x: x)
    manual_tail_mean = sum(trimmed_states) / len(trimmed_states)
    assert tail_mean == manual_tail_mean


def test_checkpoint_with_file_storage(tmp_path):
    storage = AnchorStore(FilePerAnchorStorage(tmp_path))

    def increment_step(x, rng):
        return x + 1

    kernel = StepFunctionKernel(increment_step)
    chain = SuccinctChain(
        kernel,
        initial_state=0,
        num_steps=50,
        master_seed=3,
        anchor_store=storage,
    )
    chain.run()

    checkpoint = chain.checkpoint()

    resumed_store = AnchorStore(FilePerAnchorStorage(tmp_path))

    resumed = SuccinctChain(
        kernel,
        initial_state=0,
        num_steps=50,
        master_seed=3,
        anchor_store=resumed_store,
        resume_from=checkpoint,
    )
    resumed.run()

    for t in range(50):
        assert resumed.get(t) == chain.get(t)

