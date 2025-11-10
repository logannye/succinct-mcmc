"""
Experimental: parallel replay utilities.

Purpose:
- Explore parallelization over blocks for:
    - computing expectations,
    - diagnostics,
    - regenerating subsets of the chain.
- Demonstrate how succinct structure enables embarrassingly parallel workloads.

This module is intentionally experimental and may depend on multiprocessing/threading.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Any, Iterable, List

from ..trace.succinct_chain import SuccinctChain


def parallel_expectation(
    chain: SuccinctChain,
    f: Callable[[Any], float],
    num_workers: int = 4,
) -> float:
    """
    Experimental: compute E[f(X)] using block-wise parallel replay.

    Sketch:
        - Split logical index range into chunks.
        - For each chunk, spawn a worker that:
            - reconstructs needed states (via chain.get or future block-wise APIs),
            - accumulates partial sum.
        - Combine results.

    Note:
        This is a placeholder; a production implementation would:
        - avoid heavy IPC,
        - use block metadata directly instead of chain.get().
    """
    N = chain.num_steps
    if N == 0:
        raise ValueError("Empty chain")

    # Simple chunking
    chunk_size = max(1, N // (num_workers * 4))

    def worker(start: int, end: int) -> float:
        s = 0.0
        for t in range(start, end):
            x = chain.get(t)
            s += f(x)
        return s

    futures = []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for start in range(0, N, chunk_size):
            end = min(N, start + chunk_size)
            futures.append(ex.submit(worker, start, end))

        total = 0.0
        for fut in as_completed(futures):
            total += fut.result()

    return total / N
