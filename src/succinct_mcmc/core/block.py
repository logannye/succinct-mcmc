"""
Block metadata definitions.

A "block" is a contiguous segment of the chain, e.g. steps [j*b, (j+1)*b).

We don't store all states in a block; we store:
- An anchor state at block start (or end).
- A block-level seed for RNG.
- Optional integrity hashes or cached statistics.

This module defines the small, serializable metadata objects that describe blocks.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class BlockMeta:
    """
    Metadata for a single block.

    Attributes:
        index: Block index (0-based).
        start_step: Global step index of the first state in this block.
        end_step: Global step index one past the last state in this block.
        seed: RNG seed used to regenerate this block's transitions.
        checksum: Optional integrity hash over the block (to detect corruption).
    """
    index: int
    start_step: int
    end_step: int
    seed: int
    checksum: Optional[str] = None

    # Later:
    # - Support extra fields (e.g. partial summaries, acceptance stats).
    # - Methods to compute/verify checksum.
