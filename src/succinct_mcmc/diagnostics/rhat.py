"""
R-hat (Gelman-Rubin) diagnostics for Succinct MCMC.

Provides both classic split R-hat and rank-normalized split R-hat estimators.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

from .summary import StreamingStats
from ..trace.multi_chain import MultiChain


def _iter_chain(
    chain,
    f: Callable[[Any], float],
    *,
    skip_warmup: bool,
    thin: int,
) -> Iterable[float]:
    step = 0
    for state in chain.iter(skip_warmup=skip_warmup):
        if step % thin == 0:
            yield f(state)
        step += 1


def _split_chains_values(
    multichain: MultiChain,
    f: Callable[[Any], float],
    *,
    skip_warmup: bool,
    thin: int,
    max_samples: Optional[int],
) -> list[list[float]]:
    values: list[list[float]] = []
    for chain in multichain.chains:
        samples = list(_iter_chain(chain, f=f, skip_warmup=skip_warmup, thin=thin))
        if len(samples) < 5:
            raise ValueError("Each chain must have at least 4 post-warmup samples for split R-hat")
        if max_samples is not None and max_samples > 0:
            samples = samples[-max_samples:]
        if len(samples) < 5:
            raise ValueError("Not enough samples after applying window/max_samples")
        samples = [samples[i + 1] - samples[i] for i in range(len(samples) - 1)]
        mean_val = sum(samples) / len(samples)
        samples = [s - mean_val for s in samples]
        mid = len(samples) // 2
        values.append(samples[:mid])
        values.append(samples[mid:])
    return values


def _variance(values: list[float]) -> float:
    stats = StreamingStats()
    for v in values:
        stats.update(v)
    if stats.count < 2:
        raise ValueError("Need at least two samples to compute variance")
    return stats.variance


def _rank_transform(values: list[list[float]]) -> list[list[float]]:
    flat = [v for chain_vals in values for v in chain_vals]
    sorted_vals = sorted((v, i) for i, v in enumerate(flat))
    ranks = [0.0] * len(flat)
    running = 0
    while running < len(sorted_vals):
        j = running
        while j < len(sorted_vals) and sorted_vals[j][0] == sorted_vals[running][0]:
            j += 1
        avg_rank = (running + j - 1) / 2.0
        for k in range(running, j):
            _, idx = sorted_vals[k]
            ranks[idx] = avg_rank
        running = j

    result: list[list[float]] = []
    offset = 0
    for chain_vals in values:
        length = len(chain_vals)
        result.append([ranks[offset + i] / (len(flat) - 1) for i in range(length)])
        offset += length
    return result


def _split_rhat_from_values(values: list[list[float]]) -> float:
    m = len(values)
    n = len(values[0])
    chain_means = [sum(chain_vals) / n for chain_vals in values]
    grand_mean = sum(chain_means) / m

    B = n * sum((mean - grand_mean) ** 2 for mean in chain_means) / (m - 1)
    W = sum(_variance(chain_vals) for chain_vals in values) / m

    if W == 0:
        return 1.0

    var_hat = ((n - 1) / n) * W + (B / n)
    return (var_hat / W) ** 0.5


def split_rhat(
    multichain: MultiChain,
    f: Callable[[Any], float] = lambda x: x,
    *,
    skip_warmup: bool = True,
    thin: int = 1,
    max_samples: Optional[int] = None,
) -> float:
    values = _split_chains_values(
        multichain,
        f,
        skip_warmup=skip_warmup,
        thin=thin,
        max_samples=max_samples,
    )
    return _split_rhat_from_values(values)


def rank_normalized_split_rhat(
    multichain: MultiChain,
    f: Callable[[Any], float] = lambda x: x,
    *,
    skip_warmup: bool = True,
    thin: int = 1,
    max_samples: Optional[int] = None,
) -> float:
    values = _split_chains_values(
        multichain,
        f,
        skip_warmup=skip_warmup,
        thin=thin,
        max_samples=max_samples,
    )
    ranked = _rank_transform(values)
    return _split_rhat_from_values(ranked)
