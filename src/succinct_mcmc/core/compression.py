"""
Compression layout strategies.

Given:
- total steps T
- block size b

We define how to:
- Partition steps into blocks.
- Assign seeds/metadata.
- Optionally build multi-level (tree) layouts for faster random access.

Initially we support a flat layout: blocks [0..b), [b..2b), ....

Later we may add:
- tree-based layouts,
- adaptive rebucketing,
- memory-aware planning.
"""

from typing import List, Set

from .block import BlockMeta
from .rng import derive_block_seed


def make_flat_blocks(num_steps: int, block_size: int, master_seed: int) -> List[BlockMeta]:
    """
    Create a flat list of BlockMeta for a simple partition of [0, num_steps).

    This is the default √T-space scheme:
    - Number of blocks K ≈ num_steps / block_size = O(√T).
    """
    blocks: List[BlockMeta] = []
    index = 0
    start = 0

    while start < num_steps:
        end = min(start + block_size, num_steps)
        seed = derive_block_seed(master_seed, index)
        blocks.append(BlockMeta(index=index, start_step=start, end_step=end, seed=seed))
        index += 1
        start = end

    return blocks


def compute_tree_anchor_steps(
    num_steps: int,
    base_block_size: int,
    branching_factor: int = 4,
    levels: int = 2,
) -> List[int]:
    """
    Compute additional anchor steps for tree-style replay.

    For each level, form segments whose length grows by ``branching_factor`` and
    store anchors at their midpoints. This adds O(√T / branching_factor) anchors,
    improving random-access latency without sacrificing succinctness.
    """
    if branching_factor < 2 or levels <= 0:
        return []

    steps: Set[int] = set()
    segment = base_block_size
    for _ in range(levels):
        segment *= branching_factor
        if segment <= base_block_size or segment >= num_steps:
            continue
        half = max(1, segment // 2)
        for start in range(0, num_steps, segment):
            mid = start + half
            if 0 < mid < num_steps:
                steps.add(mid)

    return sorted(steps)
