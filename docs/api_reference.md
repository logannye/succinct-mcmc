# API Reference

## Top-Level (`succinct_mcmc`)

### Core
- `SuccinctChain` — √T-space chain runner.
  ```python
  from succinct_mcmc import SuccinctChain, StepFunctionKernel

  def step(x, rng):
      return x + rng.normalvariate(0.0, 1.0)

  chain = SuccinctChain(
      StepFunctionKernel(step),
      initial_state=0.0,
      num_steps=10_000,
      warmup_steps=1_000,
      rng_backend="python",
  )
  chain.run()
  ```
- `MultiChain` — collection of succinct chains for diagnostics.
- `ChainCheckpoint` — checkpoint snapshot for resumable runs (`SuccinctChain.checkpoint()` / `resume_from=`).
- `ReplayPolicy` — configure replay strategy (currently `flat`).
- `CompressionConfig` — block sizing and layout options.
- `AnchorStore` — configurable anchor storage wrapper.

### Kernels
- `TransitionKernel` — protocol for `(state, rng) -> state` transitions.
- `GaussianRandomWalkMH` / `AdaptiveGaussianRandomWalkMH` — Metropolis kernels supporting scalar & vector states.
- `SimpleHMC` — leapfrog-based Hamiltonian Monte Carlo (scalar/vector).
- `GenericGibbsKernel` — compose conditional updaters.
- `StepFunctionKernel` — wrap plain step functions.

### Diagnostics
- `mean`, `summary`, `quantiles`, `covariance` — streaming statistics utilities with `skip_warmup`/`thin` arguments.
- `split_rhat`, `rank_normalized_split_rhat` — convergence diagnostics with optional thinning/window (`max_samples`).
- `split_ess`, `rank_normalized_split_ess`, `multivariate_split_ess` — effective sample size estimates (scalar & multivariate).
- `autocorrelation` — autocorrelation helper for dense sequences.

### Iterators
- `iterator.iter_states(chain, skip_warmup=True)` — stream states once without redundant replay.
- `iterator.iter_range(chain, start, stop, skip_warmup=True)` — subset iteration.
- `iterator.block_iter(chain, skip_warmup=True)` — iterate over blocks and their states for batching.

### I/O & Storage
- `SuccinctArtifact` — serialized run descriptor.
- `save_artifact(path)`, `load_artifact(path)` — JSON metadata round-trip.
- `InMemoryStorage`, `FilePerAnchorStorage`, `NumpyMemmapStorage` — anchor backends (memmap requires NumPy).
- `AnchorStorageBackend` — interface for custom storage implementations.

- ### Utilities
- `utils.parallel.parallel_expectation` — block-wise parallel expectation helper (thread/process).
- `__version__` — package version string.

## `succinct_mcmc.trace`
- Exposes `SuccinctChain`, `MultiChain`, `ChainCheckpoint`, `ReplayPolicy`, and iterator utilities described above.

## `succinct_mcmc.mcmc`
- Kernel implementations listed above; module also exposes helpers for building custom Gibbs updaters.

## `succinct_mcmc.diagnostics`
- Module-level functions mirror the top-level exports; additional internals handle rank transforms and streaming autocovariance/variance.

## `succinct_mcmc.io`
- Artifact dataclasses, serialization utilities, storage backends, and helper interfaces for persisting succinct chains.
