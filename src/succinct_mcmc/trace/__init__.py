"""
Succinct trajectory abstractions.

Exports:
- SuccinctChain: √T-space MCMC chain.
- MultiChain: container for multiple succinct chains (for diagnostics).
"""

from .succinct_chain import ChainCheckpoint, ReplayPolicy, SuccinctChain
from .multi_chain import MultiChain

__all__ = ["SuccinctChain", "MultiChain", "ReplayPolicy", "ChainCheckpoint"]
