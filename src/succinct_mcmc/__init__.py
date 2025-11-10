"""
succinct_mcmc

Public API for √T-space MCMC trajectory compression.
"""

from __future__ import annotations

try:  # pragma: no cover - best effort version
    from importlib.metadata import version as _pkg_version
except ImportError:  # pragma: no cover
    _pkg_version = lambda _: "0.0.0"

from .trace import ChainCheckpoint, MultiChain, ReplayPolicy, SuccinctChain
from .core.anchors import AnchorStore
from .core.config import CompressionConfig
from .mcmc import (
    AdaptiveGaussianRandomWalkMH,
    GenericGibbsKernel,
    GaussianRandomWalkMH,
    SimpleHMC,
    StepFunctionKernel,
    TransitionKernel,
)
from .diagnostics import (
    autocorrelation,
    mean,
    quantiles,
    rank_normalized_split_ess,
    rank_normalized_split_rhat,
    split_ess,
    split_rhat,
    summary,
)
from .io import (
    AnchorStorageBackend,
    FilePerAnchorStorage,
    InMemoryStorage,
    NumpyMemmapStorage,
    SuccinctArtifact,
    load_artifact,
    save_artifact,
)

try:  # pragma: no cover - dev installs
    __version__ = _pkg_version("succinct-mcmc")
except Exception:  # pragma: no cover
    __version__ = "0.1.0-dev"

__all__ = [
    "__version__",
    "SuccinctChain",
    "MultiChain",
    "ChainCheckpoint",
    "ReplayPolicy",
    "AnchorStore",
    "CompressionConfig",
    "TransitionKernel",
    "GaussianRandomWalkMH",
    "AdaptiveGaussianRandomWalkMH",
    "SimpleHMC",
    "GenericGibbsKernel",
    "StepFunctionKernel",
    "mean",
    "summary",
    "quantiles",
    "split_rhat",
    "rank_normalized_split_rhat",
    "split_ess",
    "rank_normalized_split_ess",
    "autocorrelation",
    "SuccinctArtifact",
    "save_artifact",
    "load_artifact",
    "InMemoryStorage",
    "FilePerAnchorStorage",
    "NumpyMemmapStorage",
    "AnchorStorageBackend",
]
