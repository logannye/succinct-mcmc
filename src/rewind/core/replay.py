"""Deterministic block replay from an anchor + the block's derived seed."""
from __future__ import annotations

from typing import Any, Callable, Iterator

from .blocks import Block
from .seeds import make_rng

StepFn = Callable[[Any, Any], Any]


def replay_to(step_fn: StepFn, anchor_state: Any, block: Block, target: int, *,
              seed: int, backend: str = "python") -> Any:
    """Return X_target by replaying from anchor_state (== X_block.start)."""
    if not (block.start <= target <= block.stop):
        raise ValueError(f"target {target} outside block [{block.start}, {block.stop}]")
    rng = make_rng(seed, backend=backend)
    state = anchor_state
    for _ in range(block.start, target):
        state = step_fn(state, rng)
    return state


def iter_block(step_fn: StepFn, anchor_state: Any, block: Block, *,
               seed: int, backend: str = "python") -> Iterator[Any]:
    """Yield X_block.start .. X_block.stop-1 (anchor first)."""
    rng = make_rng(seed, backend=backend)
    state = anchor_state
    yield state
    for _ in range(block.start + 1, block.stop):
        state = step_fn(state, rng)
        yield state
