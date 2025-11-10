"""
Tests for anchor storage backends.
"""

import pytest

from succinct_mcmc.core.anchors import AnchorStore
from succinct_mcmc.io import storage_backends
from succinct_mcmc.io.storage_backends import (
    FilePerAnchorStorage,
    InMemoryStorage,
    NumpyMemmapStorage,
)


def test_inmemory_snapshot_restore():
    storage = InMemoryStorage()
    storage.put(0, {"x": 1})
    storage.put(1, {"x": 2})

    snapshot = storage.snapshot()

    restored = InMemoryStorage()
    restored.restore(snapshot)

    assert restored.get(0)["x"] == 1
    assert restored.get(1)["x"] == 2


def test_file_per_anchor_roundtrip(tmp_path):
    storage = FilePerAnchorStorage(tmp_path)
    storage.put(0, {"v": 42})
    assert storage.contains(0)

    snap = storage.snapshot()
    storage.restore(snap)
    assert storage.get(0)["v"] == 42


@pytest.mark.skipif(storage_backends.np is None, reason="NumPy backend unavailable")
def test_numpy_memmap_storage(tmp_path):
    np = storage_backends.np
    backend = NumpyMemmapStorage(tmp_path, per_anchor_shape=(3,), capacity=10)
    backend.put(0, [1.0, 2.0, 3.0])
    backend.put(1, [4.0, 5.0, 6.0])

    snap = backend.snapshot()

    new_backend = NumpyMemmapStorage(tmp_path, per_anchor_shape=(3,), capacity=10)
    new_backend.restore(snap)
    assert new_backend.contains(1)
    assert np.allclose(new_backend.get(1), np.array([4.0, 5.0, 6.0]))


def test_anchor_store_with_backend(tmp_path):
    backend = FilePerAnchorStorage(tmp_path)
    store = AnchorStore(backend)
    store.set_anchor(5, {"a": 7})
    assert store.has_anchor(5)
    assert store.get_anchor(5)["a"] == 7
