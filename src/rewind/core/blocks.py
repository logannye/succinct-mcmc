"""Flat √T block layout over the index range [0, n_steps)."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Block:
    index: int
    start: int  # inclusive
    stop: int   # exclusive


def default_block_size(n_steps: int) -> int:
    """ceil(sqrt(n_steps)), computed exactly via integer sqrt."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    r = math.isqrt(n_steps)
    return r if r * r == n_steps else r + 1


def make_blocks(n_steps: int, block_size: int) -> list[Block]:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    blocks: list[Block] = []
    index, start = 0, 0
    while start < n_steps:
        stop = min(start + block_size, n_steps)
        blocks.append(Block(index=index, start=start, stop=stop))
        index += 1
        start = stop
    return blocks


def block_index_for_step(t: int, block_size: int) -> int:
    return t // block_size
