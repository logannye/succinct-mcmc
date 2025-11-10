"""
Storage backends for anchors and metadata.

Purpose:
- Allow Succinct MCMC to store anchors and block metadata:
    - purely in memory,
    - in memory-mapped files,
    - or other mediums in future (e.g., remote object stores).

This abstraction makes it easy to scale to very large chains without
changing the core algorithms.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    np = None


class AnchorStorageBackend:
    def put(self, key: int, value: Any) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def get(self, key: int) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def contains(self, key: int) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def __len__(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def snapshot(self) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def restore(self, snapshot: Any) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class InMemoryStorage(AnchorStorageBackend):
    """Dictionary-backed storage."""

    def __init__(self):
        self._data: Dict[int, Any] = {}

    def put(self, key: int, value: Any) -> None:
        self._data[key] = value

    def get(self, key: int) -> Any:
        return self._data[key]

    def contains(self, key: int) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def snapshot(self) -> Dict[int, Any]:
        return dict(self._data)

    def restore(self, snapshot: Dict[int, Any]) -> None:
        self._data = dict(snapshot)


class FilePerAnchorStorage(AnchorStorageBackend):
    """Store anchors as individual pickle files."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: int) -> Path:
        return self.dir / f"anchor_{key}.pkl"

    def put(self, key: int, value: Any) -> None:
        with self._path(key).open("wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get(self, key: int) -> Any:
        with self._path(key).open("rb") as f:
            return pickle.load(f)

    def contains(self, key: int) -> bool:
        return self._path(key).exists()

    def __len__(self) -> int:
        return len(list(self.dir.glob("anchor_*.pkl")))

    def snapshot(self) -> Dict[str, Any]:
        return {"type": "file_per_anchor", "directory": str(self.dir)}

    def restore(self, snapshot: Dict[str, Any]) -> None:
        if snapshot.get("type") != "file_per_anchor":
            raise ValueError("Snapshot type mismatch for FilePerAnchorStorage")
        self.dir = Path(snapshot["directory"])
        self.dir.mkdir(parents=True, exist_ok=True)


class NumpyMemmapStorage(AnchorStorageBackend):
    """Chunked anchor storage using NumPy memmap files."""

    def __init__(
        self,
        directory: str | Path,
        per_anchor_shape: Sequence[int],
        *,
        dtype: str = "float64",
        capacity: int = 10_000,
    ):
        if np is None:
            raise ImportError("NumPy is required for NumpyMemmapStorage")
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.per_anchor_shape = tuple(per_anchor_shape)
        self.dtype = dtype
        self.capacity = capacity
        self._file = self.dir / "anchors.dat"
        self._meta = self.dir / "meta.json"
        if not self._file.exists():
            self._initialize_storage()
        self._load_metadata()
        total = self._total_entries
        self._memmap = np.memmap(self._file, mode="r+", dtype=self.dtype, shape=(total,))
        self._count = 0

    def _initialize_storage(self) -> None:
        meta = {
            "per_anchor_shape": self.per_anchor_shape,
            "dtype": self.dtype,
            "capacity": self.capacity,
        }
        self._meta.write_text(json.dumps(meta))
        Path(self._file).touch()

    def _load_metadata(self) -> None:
        meta = json.loads(self._meta.read_text())
        if tuple(meta["per_anchor_shape"]) != self.per_anchor_shape or meta["dtype"] != self.dtype:
            raise ValueError("Memmap metadata mismatch")
        self.capacity = int(meta.get("capacity", self.capacity))
        self._per_anchor_size = int(np.prod(self.per_anchor_shape))
        self._total_entries = self.capacity * self._per_anchor_size
        if self._file.stat().st_size == 0:
            mm = np.memmap(self._file, mode="w+", dtype=self.dtype, shape=(self._total_entries,))
            mm[:] = 0
            mm.flush()

    def put(self, key: int, value: Any) -> None:
        arr = np.asarray(value, dtype=self.dtype)
        if arr.shape != self.per_anchor_shape:
            raise ValueError("Anchor array shape mismatch")
        start = key * self._per_anchor_size
        end = start + self._per_anchor_size
        if end > self._total_entries:
            raise IndexError("Memmap capacity exceeded")
        self._memmap[start:end] = arr.ravel()
        self._memmap.flush()
        self._count = max(self._count, key + 1)

    def get(self, key: int) -> np.ndarray:
        start = key * self._per_anchor_size
        end = start + self._per_anchor_size
        data = np.array(self._memmap[start:end], copy=True)
        return data.reshape(self.per_anchor_shape)

    def contains(self, key: int) -> bool:
        return key < self._count

    def __len__(self) -> int:
        return self._count

    def snapshot(self) -> Dict[str, Any]:
        return {
            "type": "numpy_memmap",
            "directory": str(self.dir),
            "per_anchor_shape": self.per_anchor_shape,
            "dtype": self.dtype,
            "capacity": self.capacity,
            "count": self._count,
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        if snapshot.get("type") != "numpy_memmap":
            raise ValueError("Snapshot type mismatch for NumpyMemmapStorage")
        self.dir = Path(snapshot["directory"])
        self.per_anchor_shape = tuple(snapshot["per_anchor_shape"])
        self.dtype = snapshot["dtype"]
        self.capacity = int(snapshot.get("capacity", self.capacity))
        self._file = self.dir / "anchors.dat"
        self._meta = self.dir / "meta.json"
        self._load_metadata()
        self._memmap = np.memmap(self._file, mode="r+", dtype=self.dtype, shape=(self._total_entries,))
        self._count = snapshot.get("count", 0)
