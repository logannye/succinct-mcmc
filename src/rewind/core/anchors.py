"""In-memory anchor store: {block_index: state}.

v1 ships only the in-memory backend; persistence is handled by the .replay
artifact (rewind.io.artifact). File/memmap backends are a documented fast-follow.
"""
from __future__ import annotations

from typing import Any, Iterable


class AnchorStore:
    def __init__(self) -> None:
        self._data: dict[int, Any] = {}

    def set(self, index: int, state: Any) -> None:
        self._data[index] = state

    def get(self, index: int) -> Any:
        return self._data[index]

    def has(self, index: int) -> bool:
        return index in self._data

    def __len__(self) -> int:
        return len(self._data)

    def items(self) -> Iterable[tuple[int, Any]]:
        return self._data.items()

    def as_dict(self) -> dict[int, Any]:
        return dict(self._data)
