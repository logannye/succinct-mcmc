"""
Lightweight profiling & instrumentation helpers.

Purpose:
- Measure time and (approx) memory usage of succinct vs dense runs.
- Provide hooks for experiments without pulling in heavy dependencies.

This is intentionally minimal; users can replace with their own tooling.
"""

import time
from contextlib import contextmanager
from typing import Callable, Any


@contextmanager
def timed(section: str = ""):
    """
    Context manager to time a code block.

    Usage:
        with timed("run_succinct_chain"):
            chain.run()
    """
    start = time.time()
    yield
    end = time.time()
    elapsed = end - start
    label = f"[{section}] " if section else ""
    print(f"{label}elapsed {elapsed:.3f}s")
    

def benchmark(fn: Callable[[], Any], repeats: int = 3) -> float:
    """
    Run `fn` multiple times and return the best (min) runtime.

    Useful for quick-and-dirty comparisons in examples/experiments.
    """
    best = float("inf")
    for _ in range(repeats):
        start = time.time()
        fn()
        end = time.time()
        best = min(best, end - start)
    return best
