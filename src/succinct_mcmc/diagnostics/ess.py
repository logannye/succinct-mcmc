"""
Effective Sample Size (ESS) for Succinct MCMC.

Provides split-R and rank-normalized ESS estimates following Stan's reference
implementations, operating directly on succinct chains with warmup skipping and
thinning support.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, Sequence, Dict

from .acf import autocorrelation
from .rhat import _iter_chain, _split_chains_values, _rank_transform
from ..trace.multi_chain import MultiChain


def _autocovariance(values: List[float]) -> List[float]:
    mean = sum(values) / len(values)
    return [
        sum((values[t] - mean) * (values[t + lag] - mean) for t in range(len(values) - lag))
        / (len(values) - lag)
        for lag in range(len(values))
    ]


def _positive_sequence(autocov: List[float]) -> float:
    tau = 0.0
    for lag in range(1, len(autocov) - 1, 2):
        pair_sum = autocov[lag] + autocov[lag + 1]
        if pair_sum <= 0:
            break
        tau += 2.0 * pair_sum
    return tau


def _split_chain_variance(values: List[float]) -> float:
    acov = _autocovariance(values)
    return acov[0]


def _split_chain_ess(values: List[float]) -> float:
    n = len(values)
    if n < 4:
        return float(n)

    acov = _autocovariance(values)
    tau = 1.0 + _positive_sequence(acov)
    return max(float(n) / tau, 1.0)


def split_ess(
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
    per_chain = [_split_chain_ess(chain_vals) for chain_vals in values]
    return float(sum(per_chain))


def rank_normalized_split_ess(
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
    per_chain = [_split_chain_ess(chain_vals) for chain_vals in ranked]
    return float(sum(per_chain))


def multivariate_split_ess(
    multichain: MultiChain,
    f: Callable[[Any], Sequence[float]],
    *,
    skip_warmup: bool = True,
    thin: int = 1,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute multivariate ESS by evaluating ESS per dimension and returning the
    minimum (following Stan's recommendation of using the smallest univariate ESS).
    """
    chain_iter = multichain.chains[0].iter(skip_warmup=skip_warmup)
    try:
        first_vec = list(f(next(chain_iter)))
    except StopIteration as exc:
        raise ValueError("No samples available for multivariate ESS") from exc
    if not first_vec:
        raise ValueError("Function f must return a non-empty sequence")

    per_dimension = []
    dim = len(first_vec)
    for d in range(dim):
        values = _split_chains_values(
            multichain,
            lambda state, d=d: f(state)[d],
            skip_warmup=skip_warmup,
            thin=thin,
            max_samples=max_samples,
        )
        per_dimension.append(float(sum(_split_chain_ess(chain_vals) for chain_vals in values)))

    overall = min(per_dimension)
    return {"overall": overall, "per_dimension": per_dimension}
