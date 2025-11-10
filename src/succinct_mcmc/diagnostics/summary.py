"""
High-level summary statistics over SuccinctChain / MultiChain.

This module provides streaming-friendly estimators that operate directly on
succinct chains, avoiding dense storage. Warmup handling and chunked iteration
support allow diagnostics to scale to very long runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence, List

from ..trace.succinct_chain import SuccinctChain


@dataclass
class StreamingStats:
    """Online mean/variance calculator using Welford's algorithm."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return self.variance ** 0.5


def _iter_chain(
    chain: SuccinctChain,
    *,
    f: Callable[[Any], float],
    skip_warmup: bool,
    thin: int,
) -> Iterable[float]:
    step = 0
    for state in chain.iter(skip_warmup=skip_warmup):
        if step % thin == 0:
            yield f(state)
        step += 1


def mean(
    chain: SuccinctChain,
    f: Callable[[Any], float],
    *,
    skip_warmup: bool = True,
    thin: int = 1,
) -> float:
    stats = StreamingStats()
    for value in _iter_chain(chain, f=f, skip_warmup=skip_warmup, thin=thin):
        stats.update(value)
    if stats.count == 0:
        raise ValueError("Empty chain")
    return stats.mean


def summary(
    chain: SuccinctChain,
    f: Callable[[Any], float],
    *,
    skip_warmup: bool = True,
    thin: int = 1,
    compute_variance: bool = True,
) -> dict[str, float]:
    stats = StreamingStats()
    for value in _iter_chain(chain, f=f, skip_warmup=skip_warmup, thin=thin):
        stats.update(value)

    if stats.count == 0:
        raise ValueError("Empty chain")

    out = {"mean": stats.mean}
    if compute_variance:
        out["variance"] = stats.variance
        out["std"] = stats.std
    return out


@dataclass
class StreamingVectorStats:
    dim: int
    count: int = 0
    mean: Optional[List[float]] = None
    m2: Optional[List[List[float]]] = None

    def update(self, vec: Sequence[float]) -> None:
        if len(vec) != self.dim:
            raise ValueError("Vector dimension mismatch")
        if self.count == 0:
            self.mean = [float(v) for v in vec]
            self.m2 = [[0.0 for _ in range(self.dim)] for _ in range(self.dim)]
            self.count = 1
            return

        assert self.mean is not None and self.m2 is not None
        self.count += 1
        delta = [float(v) - m for v, m in zip(vec, self.mean)]
        factor = 1.0 / self.count
        new_mean = [m + d * factor for m, d in zip(self.mean, delta)]
        for i in range(self.dim):
            for j in range(self.dim):
                self.m2[i][j] += delta[i] * (float(vec[j]) - new_mean[j])
        self.mean = new_mean

    def covariance(self) -> List[List[float]]:
        if self.count < 2 or self.m2 is None:
            raise ValueError("Not enough samples to compute covariance")
        return [[cell / (self.count - 1) for cell in row] for row in self.m2]


def covariance(
    chain: SuccinctChain,
    f: Callable[[Any], Sequence[float]],
    *,
    skip_warmup: bool = True,
) -> List[List[float]]:
    """
    Compute covariance matrix of f(X_t) over the chain.
    """
    stats: Optional[StreamingVectorStats] = None
    for state in chain.iter(skip_warmup=skip_warmup):
        values = list(f(state))
        if stats is None:
            stats = StreamingVectorStats(dim=len(values))
        stats.update(values)
    if stats is None:
        raise ValueError("Chain has zero length")
    return stats.covariance()

def quantiles(
    chain: SuccinctChain,
    f: Callable[[Any], float],
    qs: Iterable[float],
    *,
    skip_warmup: bool = True,
    thin: int = 1,
) -> dict[float, float]:
    values = list(_iter_chain(chain, f=f, skip_warmup=skip_warmup, thin=thin))
    if not values:
        raise ValueError("Empty chain")
    values.sort()
    results: dict[float, float] = {}
    n = len(values)
    for q in qs:
        if not (0.0 <= q <= 1.0):
            raise ValueError("quantile must be in [0, 1]")
        if q == 1.0:
            results[q] = float(values[-1])
            continue
        index = q * (n - 1)
        lo = int(index)
        hi = min(lo + 1, n - 1)
        weight = index - lo
        results[q] = (1 - weight) * values[lo] + weight * values[hi]
    return results
