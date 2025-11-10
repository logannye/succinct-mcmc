"""
Anchor state storage.
"""

from __future__ import annotations

from typing import Any, Optional

from ..io.storage_backends import AnchorStorageBackend, InMemoryStorage


class AnchorStore:
    """Configurable storage for block anchors."""

    def __init__(self, backend: Optional[AnchorStorageBackend] = None):
        self._backend: AnchorStorageBackend = backend or InMemoryStorage()

    def set_anchor(self, block_index: int, state: Any) -> None:
        self._backend.put(block_index, state)

    def has_anchor(self, block_index: int) -> bool:
        return self._backend.contains(block_index)

    def get_anchor(self, block_index: int) -> Any:
        return self._backend.get(block_index)

    def __len__(self) -> int:
        return len(self._backend)

    def snapshot(self) -> Any:
        return self._backend.snapshot()

    def restore(self, snapshot: Any) -> None:
        self._backend.restore(snapshot)

    @property
    def backend(self) -> AnchorStorageBackend:
        return self._backend
