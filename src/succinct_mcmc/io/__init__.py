"""
Serialization and storage utilities for succinct artifacts.

Exports:
- SuccinctArtifact dataclass.
- save_artifact, load_artifact helpers.
- Basic storage backends (in-memory, file-per-anchor).
"""

from .artifact import SuccinctArtifact
from .serialize import save_artifact, load_artifact
from .storage_backends import (
    InMemoryStorage,
    FilePerAnchorStorage,
    NumpyMemmapStorage,
    AnchorStorageBackend,
)

__all__ = [
    "SuccinctArtifact",
    "save_artifact",
    "load_artifact",
    "InMemoryStorage",
    "FilePerAnchorStorage",
    "NumpyMemmapStorage",
    "AnchorStorageBackend",
]
