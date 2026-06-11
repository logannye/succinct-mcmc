"""Deterministic per-block seed derivation and RNG backends.

Randomness is *block-local*: block j draws from an RNG seeded by
derive_seed(master_seed, j). Seeds are small ints (re-derivable, never
snapshotted), so anchors stay tiny and any block replays independently.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any

_MASK64 = (1 << 64) - 1


def _hash_to_int(label: str) -> int:
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _MASK64


def derive_seed(master_seed: int, block_index: int) -> int:
    """Seed for block ``block_index`` of the run rooted at ``master_seed``."""
    return _hash_to_int(f"block:{master_seed}:{block_index}")


def derive_branch_seed(master_seed: int, t: int) -> int:
    """Default seed for a branch forked at tick ``t`` (distinct namespace)."""
    return _hash_to_int(f"branch:{master_seed}:{t}")


def make_rng(seed: int, *, backend: str = "python") -> Any:
    """Construct a fresh RNG of ``backend`` from an integer ``seed``."""
    if backend == "python":
        return random.Random(seed)
    if backend == "numpy":
        import numpy as np

        return np.random.Generator(np.random.PCG64(seed))
    raise ValueError(
        f"Unknown RNG backend {backend!r}. Available: {', '.join(available_backends())}."
    )


def available_backends() -> tuple[str, ...]:
    backends = ["python"]
    try:
        import numpy  # noqa: F401

        backends.append("numpy")
    except ImportError:
        pass
    return tuple(backends)
