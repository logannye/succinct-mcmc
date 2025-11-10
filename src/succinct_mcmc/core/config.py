"""
Core configuration utilities.

This module defines configuration objects and helper functions that control:

- Block sizing policies (e.g. b = ceil(sqrt(T)) or tuned by memory budget).
- Layout strategies (flat anchors vs multi-level / tree).
- Time/space trade-off knobs and safety limits.

Eventually this will:
- Provide a Config dataclass used by SuccinctChain.
- Expose presets (e.g. "sqrt", "memory_bound", "fast_random_access").
"""

from dataclasses import dataclass
from math import ceil
from typing import Optional


@dataclass
class CompressionConfig:
    """
    Configuration for succinct trajectory compression.

    Attributes:
        num_steps: Total planned chain length T.
        block_size: Primary block size b; if None, will be derived (e.g. sqrt(T)).
        max_memory_bytes: Optional hard memory budget to adapt block_size.
        use_tree_layout: If True, enable multi-level checkpoint structure later.
        tree_branching_factor: branching factor for tree layout (>=2).
        tree_levels: maximum number of tree levels beyond base blocks.
    """
    num_steps: int
    block_size: Optional[int] = None
    max_memory_bytes: Optional[int] = None
    use_tree_layout: bool = False
    tree_branching_factor: int = 4
    tree_levels: int = 2


def choose_block_size(cfg: CompressionConfig) -> int:
    """
    Compute the block size b given config.

    Initial behavior:
    - If cfg.block_size is set, trust it.
    - Else default to ceil(sqrt(T)).

    Later:
    - Incorporate memory budget.
    - Tune for runtime characteristics.
    """
    if cfg.block_size is not None:
        return cfg.block_size
    if cfg.num_steps <= 0:
        raise ValueError("num_steps must be positive")
    return ceil(cfg.num_steps ** 0.5)
