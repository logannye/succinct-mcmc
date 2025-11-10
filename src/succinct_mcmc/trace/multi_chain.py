"""
Support for multiple succinct chains (for convergence diagnostics).

Purpose:
- Manage an ensemble of SuccinctChain instances.
- Provide a clean interface for multi-chain statistics like R-hat and ESS.
- Keep this logic separate from single-chain mechanics.

This module will be used heavily by diagnostics/rhat.py and diagnostics/ess.py.
"""

from dataclasses import dataclass
from typing import List
from .succinct_chain import SuccinctChain


@dataclass
class MultiChain:
    """
    Container for multiple SuccinctChain objects.

    Responsibilities:
        - Ensure they are logically comparable (same num_steps, etc.).
        - Provide iterators/aggregates for diagnostics modules.
    """
    chains: List[SuccinctChain]

    def __post_init__(self) -> None:
        if not self.chains:
            raise ValueError("MultiChain requires at least one chain")

        n0 = self.chains[0].num_steps
        for c in self.chains[1:]:
            if c.num_steps != n0:
                # In a more general implementation we could support different lengths,
                # but standard R-hat/ESS assume aligned draws.
                raise ValueError("All chains must have the same num_steps")

    @property
    def num_chains(self) -> int:
        return len(self.chains)

    @property
    def num_steps(self) -> int:
        return self.chains[0].num_steps

    def iter_chain(self, i: int):
        """
        Iterate over the i-th chain.
        """
        return self.chains[i].iter()
