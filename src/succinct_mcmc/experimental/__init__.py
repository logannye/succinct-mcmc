"""
Experimental algorithms and strategies.

Content here is:
- Not part of the stable public API.
- Intended for research exploration:
    - tree-based checkpoint layouts,
    - parallel replay,
    - alternative compression strategies.
"""

from .compression_strategies import sqrt_strategy, memory_bound_strategy

__all__ = [
    "sqrt_strategy",
    "memory_bound_strategy",
]
