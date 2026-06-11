"""record(): the forward sweep that stores one anchor per block."""
from __future__ import annotations

from typing import Any, Callable, Optional

from ..errors import NondeterministicStepError
from ..stats.streaming import StreamingStats
from .anchors import AnchorStore
from .blocks import default_block_size, make_blocks
from .replay import replay_to
from .run import Run, _states_equal
from .seeds import derive_seed, make_rng


def _default_step_id(step_fn: Callable) -> str:
    mod = getattr(step_fn, "__module__", "?")
    qual = getattr(step_fn, "__qualname__", repr(step_fn))
    return f"{mod}:{qual}"


def record(step_fn: Callable[[Any, Any], Any], init_state: Any, n_steps: int,
           seed: int, *, block_size: Optional[int] = None, backend: str = "python",
           observe: Optional[dict] = None, self_check: bool = False,
           step_id: Optional[str] = None) -> Run:
    """Record a length-n_steps run, storing O(sqrt(n_steps)) anchors."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    bs = block_size or default_block_size(n_steps)
    blocks = make_blocks(n_steps, bs)
    store = AnchorStore()
    store.set(0, init_state)

    accumulators = {name: StreamingStats() for name in (observe or {})}
    observe = observe or {}

    state = init_state  # == X_block.start at the top of each iteration
    for block in blocks:
        rng = make_rng(derive_seed(seed, block.index), backend=backend)
        for name, fn in observe.items():
            accumulators[name].update(fn(state))           # observe X_block.start
        for _ in range(block.start + 1, block.stop):
            state = step_fn(state, rng)
            for name, fn in observe.items():
                accumulators[name].update(fn(state))       # observe interior states
        if block.stop < n_steps:                            # compute next anchor X_block.stop
            state = step_fn(state, rng)
            store.set(block.index + 1, state)

    observed = {name: acc.summary() for name, acc in accumulators.items()}
    run = Run(step_fn=step_fn, init_state=init_state, n_steps=n_steps, block_size=bs,
              seed=seed, backend=backend, store=store, observed=observed,
              observe=observe, step_id=step_id or _default_step_id(step_fn))

    if self_check:
        for block in blocks[:-1]:
            recomputed = replay_to(step_fn, store.get(block.index), block, block.stop,
                                   seed=derive_seed(seed, block.index), backend=backend)
            if not _states_equal(recomputed, store.get(block.index + 1)):
                raise NondeterministicStepError(block_index=block.index)
    return run
