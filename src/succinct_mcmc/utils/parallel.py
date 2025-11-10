"""
Parallel utilities for block-wise replay and analytics.

These helpers coordinate concurrent evaluation over SuccinctChain without
mutating the underlying RNG state.
"""

from __future__ import annotations

from concurrent.futures import Executor, ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, Literal, Optional, Sequence, Tuple

from ..trace.succinct_chain import SuccinctChain
from ..trace.iterator import iter_range

ExecutorKind = Literal["thread", "process"]


def _select_executor(kind: ExecutorKind) -> Callable[..., Executor]:
    if kind == "thread":
        return ThreadPoolExecutor
    if kind == "process":
        return ProcessPoolExecutor
    raise ValueError(f"Unknown executor kind '{kind}'")


def _compute_ranges(start: int, stop: int, chunk_size: int) -> Sequence[Tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    ranges = []
    current = start
    while current < stop:
        end = min(stop, current + chunk_size)
        ranges.append((current, end))
        current = end
    return ranges


def parallel_expectation(
    chain: SuccinctChain,
    f: Callable[[object], float],
    *,
    skip_warmup: bool = True,
    chunk_size: Optional[int] = None,
    max_workers: Optional[int] = None,
    executor: ExecutorKind = "thread",
) -> float:
    """
    Compute the empirical mean of ``f`` over a SuccinctChain using parallel replay.

    Args:
        chain: The succinct chain to scan.
        f: Callable applied to each state.
        skip_warmup: When True, skip the configured warmup prefix.
        chunk_size: Number of logical steps per chunk; defaults to block_size.
        max_workers: Maximum concurrent workers.
        executor: ``"thread"`` (default) or ``"process"``.

    Returns:
        Empirical mean of ``f`` across the logical chain.
    """
    start = chain.warmup_steps if skip_warmup else 0
    stop = chain.num_steps
    if start >= stop:
        raise ValueError("No samples remain after warmup exclusion")

    chunk = chunk_size or max(1, chain.block_size)
    ranges = _compute_ranges(start, stop, chunk)
    if not ranges:
        raise ValueError("No ranges to evaluate")

    ExecutorCls = _select_executor(executor)
    total = 0.0
    count = 0

    def worker(args: Tuple[int, int]) -> Tuple[float, int]:
        s, e = args
        subtotal = 0.0
        local_count = 0
        for val in iter_range(chain, s, e, skip_warmup=False):
            subtotal += f(val)
            local_count += 1
        return subtotal, local_count

    with ExecutorCls(max_workers=max_workers) as pool:
        futures = [pool.submit(worker, r) for r in ranges]
        for fut in as_completed(futures):
            subtotal, local_count = fut.result()
            total += subtotal
            count += local_count

    if count == 0:
        raise ValueError("No samples processed during parallel expectation")
    return total / count

