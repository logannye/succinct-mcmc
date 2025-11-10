"""
Iterator utilities for succinct chains.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, Optional

from .succinct_chain import SuccinctChain
from ..core.replay import block_apply


def iter_states(chain: SuccinctChain, *, skip_warmup: bool = False) -> Iterator[Any]:
    yield from chain.iter(skip_warmup=skip_warmup)


def iter_range(
    chain: SuccinctChain,
    start: int,
    stop: Optional[int] = None,
    *,
    skip_warmup: bool = False,
) -> Iterator[Any]:
    if start < 0 or start >= chain.num_steps:
        raise IndexError("start out of range")
    stop = chain.num_steps if stop is None else stop
    if stop < start or stop > chain.num_steps:
        raise IndexError("stop out of range")

    warmup_cutoff = chain.warmup_steps if skip_warmup else 0
    start = max(start, warmup_cutoff)

    if start >= stop:
        return iter(())

    step_fn = chain.kernel.step
    block = chain._find_block_for_step(start)

    buffer: list[Any] = []

    def collect(step: int, state: Any) -> None:
        if step < start or step >= stop:
            return
        buffer.append(state)

    while True:
        buffer.clear()
        block_apply(
            step_fn,
            chain.anchor_store,
            block,
            collect,
            rng_backend=chain.rng_backend,
        )
        for value in buffer:
            yield value
        next_index = block.index + 1
        if next_index >= len(chain.blocks):
            break
        block = chain.blocks[next_index]
        if block.start_step >= stop:
            break


def block_iter(chain: SuccinctChain, *, skip_warmup: bool = False) -> Iterator[tuple[int, list[Any]]]:
    warmup_cutoff = chain.warmup_steps if skip_warmup else 0
    step_fn = chain.kernel.step

    for block in chain.blocks:
        buffer: list[Any] = []

        def collect(step: int, state: Any) -> None:
            if step < warmup_cutoff:
                return
            buffer.append(state)

        block_apply(
            step_fn,
            chain.anchor_store,
            block,
            collect,
            rng_backend=chain.rng_backend,
        )
        if buffer:
            yield block.index, buffer
