"""Online (streaming) statistics — exact whole-run analytics in O(1)/O(d^2) state.

Ported from the proven succinct_mcmc Welford implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class StreamingStats:
    count: int = 0
    mean: float = 0.0
    _m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self._m2 += delta * (value - self.mean)

    @property
    def variance(self) -> float:
        return self._m2 / (self.count - 1) if self.count >= 2 else 0.0

    @property
    def std(self) -> float:
        return self.variance ** 0.5

    def summary(self) -> dict:
        return {"count": self.count, "mean": self.mean,
                "variance": self.variance, "std": self.std}


@dataclass
class StreamingVectorStats:
    dim: int
    count: int = 0
    mean: list[float] = field(default_factory=list)
    _m2: list[list[float]] = field(default_factory=list)

    def update(self, vec: Sequence[float]) -> None:
        if len(vec) != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {len(vec)}")
        if self.count == 0:
            self.mean = [float(v) for v in vec]
            self._m2 = [[0.0] * self.dim for _ in range(self.dim)]
            self.count = 1
            return
        self.count += 1
        delta = [float(v) - m for v, m in zip(vec, self.mean)]
        self.mean = [m + d / self.count for m, d in zip(self.mean, delta)]
        for i in range(self.dim):
            for j in range(self.dim):
                self._m2[i][j] += delta[i] * (float(vec[j]) - self.mean[j])

    def covariance(self) -> list[list[float]]:
        if self.count < 2:
            raise ValueError("need at least two samples for covariance")
        n = self.count - 1
        return [[cell / n for cell in row] for row in self._m2]

    def summary(self) -> dict:
        return {"count": self.count, "mean": list(self.mean),
                "covariance": self.covariance() if self.count >= 2 else None}
