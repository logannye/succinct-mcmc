"""
Deterministic replay of chain segments.

Given:
- A transition function step_fn(x, rng) -> x_next
- Block metadata (anchor state + RNG seed)
- A target index t within that block

We can reconstruct X_t by:
- Re-initializing RNG from block seed,
- Iterating step_fn from the anchor until we reach t.

This file will later contain optimized routines for:
- Replaying entire blocks.
- Bulk evaluation of functions over a block (for diagnostics).
- Multi-level replay using tree layouts.
"""

from typing import Any, Callable, Iterable, Iterator

from .rng import make_block_rng
from .block import BlockMeta
from .anchors import AnchorStore


StepFn = Callable[[Any, Any], Any]


def replay_block(
    step_fn: StepFn,
    anchor_store: AnchorStore,
    block: BlockMeta,
    *,
    rng_backend: str = "python",
) -> Iterable[Any]:
    """
    Replay all states in a block, starting from its anchor.

    Yields:
        States X_{start_step}, X_{start_step+1}, ..., X_{end_step-1}.

    Note:
        This basic version assumes:
        - The anchor_store holds state at start_step.
        - The PRNG uses only the derived block seed.
    """
    rng = make_block_rng(block.seed, backend=rng_backend)
    state = anchor_store.get_anchor(block.index)

    # Yield anchor state (start_step)
    yield state

    # Replay subsequent steps
    for _ in range(block.start_step + 1, block.end_step):
        state = step_fn(state, rng)
        yield state


def stream_block(
    step_fn: StepFn,
    anchor_store: AnchorStore,
    block: BlockMeta,
    *,
    rng_backend: str = "python",
) -> Iterator[Any]:
    """Return iterator for block states including anchor."""

    rng = make_block_rng(block.seed, backend=rng_backend)
    state = anchor_store.get_anchor(block.index)
    yield state
    for _ in range(block.start_step + 1, block.end_step):
        state = step_fn(state, rng)
        yield state


def block_apply(
    step_fn: StepFn,
    anchor_store: AnchorStore,
    block: BlockMeta,
    fn: Callable[[int, Any], None],
    *,
    rng_backend: str = "python",
) -> None:
    """Apply function to each state in block with logical index."""
    rng = make_block_rng(block.seed, backend=rng_backend)
    state = anchor_store.get_anchor(block.index)
    fn(block.start_step, state)
    for step in range(block.start_step + 1, block.end_step):
        state = step_fn(state, rng)
        fn(step, state)


def replay_point(
    step_fn: StepFn,
    anchor_store: AnchorStore,
    block: BlockMeta,
    target_step: int,
    *,
    rng_backend: str = "python",
) -> Any:
    """
    Replay up to a specific step within a block and return that state.

    Complexity:
        O(block_length) in this flat layout.
        Later variants may reduce this using tree layouts.
    """
    if not (block.start_step <= target_step < block.end_step):
        raise ValueError("target_step not inside given block")

    rng = make_block_rng(block.seed, backend=rng_backend)
    state = anchor_store.get_anchor(block.index)

    current_step = block.start_step
    while current_step < target_step:
        state = step_fn(state, rng)
        current_step += 1

    return state
