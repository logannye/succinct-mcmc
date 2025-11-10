"""
Deterministic block-level RNG utilities.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional


class RNGFactoryError(ValueError):
    """Raised when attempting to use an unknown or invalid RNG backend."""


RNGFactory = Callable[[Any], Any]


@dataclass(frozen=True)
class RNGBackend:
    """
    Descriptor for an RNG backend.

    Attributes:
        name: Identifier used in configuration.
        factory: Callable returning an RNG object when given a seed.
        provides: Optional human-readable description.
    """

    name: str
    factory: RNGFactory
    provides: Optional[str] = None


_registered_backends: Dict[str, RNGBackend] = {}


def register_rng_backend(backend: RNGBackend, *, overwrite: bool = False) -> None:
    """
    Register a new RNG backend.

    Args:
        backend: RNGBackend descriptor.
        overwrite: Allow replacing an existing backend with the same name.
    """
    if not overwrite and backend.name in _registered_backends:
        raise RNGFactoryError(
            f"Backend '{backend.name}' already registered. "
            "Set overwrite=True to replace it."
        )
    _registered_backends[backend.name] = backend


def available_rng_backends() -> Iterable[str]:
    """
    Return the names of currently registered RNG backends.
    """
    return tuple(_registered_backends.keys())


def derive_block_seed(master_seed: int, block_index: int) -> int:
    """
    Deterministically derive a per-block seed from a master seed.

    This allows:
    - No need to store all U_t.
    - Recompute randomness inside block j by re-seeding with this value.
    """
    data = f"{master_seed}:{block_index}".encode("utf-8")
    h = hashlib.sha256(data).hexdigest()
    # Convert hex digest to int; trim to 64 bits for typical PRNG compatibility.
    return int(h, 16) & ((1 << 64) - 1)


def make_block_rng(seed: Any, backend: str = "python") -> Any:
    """
    Return a block-local RNG object for the given backend.

    Args:
        seed: Deterministic seed for the block.
        backend: Name of the registered backend to create.
    """
    if backend not in _registered_backends:
        raise RNGFactoryError(
            f"Unknown RNG backend '{backend}'. "
            f"Available backends: {', '.join(available_rng_backends()) or 'none'}"
        )
    return _registered_backends[backend].factory(seed)


def _register_default_backends() -> None:
    """
    Register built-in RNG backends.
    """

    def python_random(seed: Any):
        import random
        rng = random.Random()
        if isinstance(seed, tuple):
            rng.setstate(seed)
        else:
            rng.seed(seed)
        return rng

    register_rng_backend(
        RNGBackend(
            name="python",
            factory=python_random,
            provides="Python random.Random",
        ),
        overwrite=True,
    )

    try:
        import numpy as np

        class _NumpyRandomWrapper:
            def __init__(self, generator: Any):
                self._generator = generator

            def random(self, *args, **kwargs):
                return self._generator.random(*args, **kwargs)

            def normalvariate(self, mu: float, sigma: float) -> float:
                return float(self._generator.normal(loc=mu, scale=sigma))

            def gauss(self, mu: float, sigma: float) -> float:
                return float(self._generator.normal(loc=mu, scale=sigma))

            def __getattr__(self, name: str) -> Any:
                return getattr(self._generator, name)

        def numpy_generator(seed: Any):
            if isinstance(seed, tuple):
                raise RNGFactoryError("NumPy backend does not support tuple RNG state seeds")
            bitgen = np.random.PCG64(seed)
            generator = np.random.Generator(bitgen)
            return _NumpyRandomWrapper(generator)

        register_rng_backend(
            RNGBackend(
                name="numpy",
                factory=numpy_generator,
                provides="NumPy PCG64 Generator",
            ),
            overwrite=True,
        )
    except ImportError:
        # NumPy is optional; skip registration when unavailable.
        pass


_register_default_backends()
