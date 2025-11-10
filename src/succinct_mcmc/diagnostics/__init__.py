"""
Diagnostics over succinct chains.

High-level entrypoints for:
- Posterior summaries,
- R-hat,
- ESS,
- Autocorrelation helpers.

These functions are designed to operate over SuccinctChain / MultiChain
without requiring dense in-memory arrays.
"""

from .summary import mean, summary, quantiles, covariance
from .rhat import rank_normalized_split_rhat, split_rhat
from .ess import rank_normalized_split_ess, split_ess, multivariate_split_ess
from .acf import autocorrelation

__all__ = [
    "mean",
    "summary",
    "quantiles",
    "covariance",
    "split_rhat",
    "rank_normalized_split_rhat",
    "split_ess",
    "rank_normalized_split_ess",
    "multivariate_split_ess",
    "autocorrelation",
]
