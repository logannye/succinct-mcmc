"""
Core primitives for Succinct MCMC.

This subpackage contains:
- CompressionConfig & block size policies.
- BlockMeta and anchor storage abstractions.
- RNG utilities for deterministic block-level replay.
- Layout and replay helpers.

Most users won't import from here directly; they will go through:
    succinct_mcmc.SuccinctChain
"""

from .config import CompressionConfig, choose_block_size
from .block import BlockMeta
from .anchors import AnchorStore
from .compression import make_flat_blocks

__all__ = [
    "CompressionConfig",
    "choose_block_size",
    "BlockMeta",
    "AnchorStore",
    "make_flat_blocks",
]
