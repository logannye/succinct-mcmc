"""
SuccinctArtifact: portable representation of a succinct MCMC run.

Contains enough information to:
- Reconstruct the SuccinctChain layout.
- Replay any part of the trajectory.
- Verify integrity (checksums).

This is what gets written to disk or shared.
"""

from dataclasses import dataclass
from typing import Any, List, Dict


@dataclass
class SuccinctArtifact:
    """
    Serializable descriptor of a succinct chain.

    Fields:
        version: Library/artifact version for compatibility.
        num_steps: Total logical steps T.
        block_size: Block size b used.
        master_seed: Master RNG seed.
        block_seeds: Per-block seeds (optional if derivable).
        anchors: Serialized anchors (or a handle/path thereto).
        kernel_metadata: Dict to reconstruct transition kernel.
        extra: Optional for diagnostics, checksums, etc.
    """
    version: str
    num_steps: int
    block_size: int
    master_seed: int
    block_seeds: List[int]
    anchors: Any
    kernel_metadata: Dict[str, Any]
    extra: Dict[str, Any] | None = None
