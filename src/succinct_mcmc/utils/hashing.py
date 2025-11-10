"""
Hashing helpers.

Purpose:
- Provide stable, deterministic hashes for:
    - anchor states,
    - block metadata,
    - artifacts.

Used for:
- Integrity checks,
- Reproducibility,
- Optional tamper detection.
"""

import hashlib
from typing import Any


def hash_bytes(data: bytes) -> str:
    """
    Return hex-encoded SHA256 of given bytes.
    """
    return hashlib.sha256(data).hexdigest()


def hash_repr(obj: Any) -> str:
    """
    Hash the repr() of an object.

    This is a simple baseline; full implementation should:
    - Use a stable serialization format for complex states.
    """
    return hash_bytes(repr(obj).encode("utf-8"))
