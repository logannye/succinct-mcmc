# Integration Guide

This guide summarizes how to plug Succinct MCMC into existing probabilistic workflows.

## 1. Provide a Deterministic Transition Kernel
Succinct storage relies on deterministic replay. Implement a function of the form `step(state, rng) -> new_state` where *all* randomness flows through the supplied RNG. Wrap it with `succinct_mcmc.StepFunctionKernel`, or subclass one of the provided kernels.

```python
from succinct_mcmc import StepFunctionKernel

kernel = StepFunctionKernel(my_step_fn, name="custom_rw")
```

### Notes
- `state` can be any Python object (floats, dicts, NumPy arrays, etc.).
- The RNG object exposes Python's `random.Random` methods; use `rng.random()`, `rng.normalvariate(...)`, etc. NumPy-backed RNGs are available via `rng_backend="numpy"`.

## 2. Configure the Succinct Chain
```python
from succinct_mcmc import SuccinctChain, CompressionConfig, FilePerAnchorStorage

chain = SuccinctChain(
    kernel,
    initial_state=my_state,
    num_steps=5_000_000,
    warmup_steps=100_000,
    config=CompressionConfig(num_steps=5_000_000),
    anchor_store=FilePerAnchorStorage("anchors/"),
)
chain.run()
```

- `warmup_steps` marks samples skipped by default when computing diagnostics or iterating with `skip_warmup=True`.
- Swap storage backends depending on capacity: `InMemoryStorage` (default), `FilePerAnchorStorage`, or `NumpyMemmapStorage` for large vector states.

## 3. Checkpoints & Resuming
Capture progress mid-run:
```python
checkpoint = chain.checkpoint()
```
Resume later by passing `resume_from=checkpoint` to a new `SuccinctChain` with identical configuration (including storage backend and RNG backend).

## 4. Iterator Utilities
Use `succinct_mcmc.trace.iterator` for efficient replay:
- `iter_states(chain, skip_warmup=True)` — iterate without re-running completed blocks.
- `iter_range(chain, start, stop)` — partial replay.
- `block_iter(chain)` — process chunks for batching or parallel post-processing.

## 5. Diagnostics
```python
from succinct_mcmc import split_rhat, split_ess
from succinct_mcmc.trace import MultiChain

multi = MultiChain([chain1, chain2, chain3, chain4])
print(split_rhat(multi, f=lambda x: x[0]))
print(split_ess(multi, f=lambda x: x[0]))
```
Rank-normalized variants (`rank_normalized_split_rhat`, `rank_normalized_split_ess`) are available for heavy-tailed targets.

## 6. Persist Artifacts
```python
from succinct_mcmc import save_artifact

artifact = chain.checkpoint()  # or use future chain.to_artifact()
save_artifact(artifact, "run.json")
```

Artifacts capture seeds, anchors, metadata, and storage configuration—ideal for audit and reproducibility.

## 7. Extending Storage
Implement `succinct_mcmc.io.AnchorStorageBackend` to plug remote stores or compressed formats:
```python
from succinct_mcmc.io import AnchorStorageBackend

class S3Storage(AnchorStorageBackend):
    ...
```

Register your backend and supply it via `AnchorStore(custom_backend)`.
