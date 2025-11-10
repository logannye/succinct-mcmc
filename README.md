# Succinct MCMC

**Succinct MCMC** is a library for running **very long Markov chain Monte Carlo (MCMC)** simulations while storing only a **√T-sized representation** of the full trajectory.

## Installation
```bash
pip install succinct-mcmc
# optional NumPy-backed RNG + memmap storage
pip install "succinct-mcmc[numpy]"
```
For development and tests:
```bash
pip install "succinct-mcmc[dev]"
```

## Command-Line Interface
After installation the `succinct-mcmc` CLI is available:
```bash
succinct-mcmc artifact-info path/to/artifact.json
succinct-mcmc artifact-info artifact.json --field num_steps --compact
```

It lets you inspect succinct artifacts without writing custom scripts.

---

It lets you:

- Run chains with **10⁶–10⁹+ iterations** without massive memory usage.
- Preserve a **logically complete chain** (no thinning, no lossy truncation).
- Regenerate any sample `X_t` on demand.
- Compute expectations, R-hat, ESS, and other diagnostics **over the full chain** using a compact artifact.
- Serialize/share a succinct, auditable representation of the run.

All of this is built on a precise, deterministic replay scheme inspired by **height-compression / time–space tradeoff** ideas from modern complexity theory and checkpointing.

---

## Why this exists

Most MCMC stacks force one of three painful choices:
- Keep a dense chain in memory (huge, often impossible).
- Maintain only running summaries (no ability to replay or audit individual draws).
- Thin aggressively (throwing away correlation structure and wasting compute).

**Succinct MCMC** answers the question:

> “Can we behave as if we kept the entire trajectory,  
> without paying O(T) memory?”

For Markov chains, yes: we exploit their **height-compressible** structure (state depends only on the previous state + randomness) to store anchors + deterministic RNG state and replay anything on demand.

---

## Core idea (the efficiency breakthrough)

We represent a length-`T` chain using **O(√T)** “anchors” plus replay metadata.

### 1. Block decomposition

Choose a block size:

- `b ≈ ceil(sqrt(T))`
- Number of blocks: `K ≈ T / b ≈ sqrt(T)`

Partition the chain:

- Block 0: steps `[0, ..., b)`
- Block 1: steps `[b, ..., 2b)`
- ...
- Block `K-1`: last segment

### 2. Anchors + deterministic RNG (and tree layouts)

For each block we store an anchor state and the RNG state derived from a single master seed. Optional tree anchors add midpoints across larger segments so random access becomes O(log T) instead of O(√T), still with tiny storage.

### 3. Space bound

- Anchors: `K` states → `O(√T · d)` memory.
- Seeds + metadata: `O(√T)`.

Compare:

- Naive dense storage: `O(T · d)`
- Succinct MCMC: `O(√T · d)`

For `T = 10⁸`, this is a **10⁴× reduction** in stored states.

### 4. Access & analytics: act as if you stored everything

Because anchors + RNG state define the entire trajectory:

- Reconstruct any sample `X_t` via replay from the nearest anchor (≤ log T steps with tree layout).
- Stream expectations, (co)variance, multivariate ESS, split / rank-normalized R-hat, etc., over the *true* chain.
- Parallelize block replay to accelerate long diagnostics while maintaining determinism.
- Serialize succinct, auditable artifacts that support exact replay and resuming.

---

## What this repo provides today

**Runtime & storage**
- `SuccinctChain` with block + tree anchors, deterministic replay, checkpoint/resume, and warmup-aware iterators.
- Configurable compression via `CompressionConfig` (tree branching factor/levels; memory-aware strategies on the roadmap).
- Storage backends: in-memory, file-per-anchor, NumPy memmap (optional); checkpoints capture RNG state so replay/resume are exact.

**Diagnostics & analytics**
- Streaming mean/variance/quantiles/covariance.
- Split & rank-normalized R-hat/ESS with thinning/windowing controls.
- Multivariate ESS, autocorrelation utilities, and block-wise parallel expectation helpers.

**Parallel utilities**
- `succinct_mcmc.utils.parallel_expectation` (threads or processes) for analytics over very long chains.

**Kernels & adapters**
- MH (scalar/vector, adaptive), Simple HMC, Gibbs scaffolding, `StepFunctionKernel`.
- Adapters for PyMC and NumPyro so external kernels can run under SuccinctChain deterministically.

**CLI & docs**
- `succinct-mcmc artifact-info` for artifact inspection (more commands planned).
- Installation, quickstart, integration guide, and examples under `docs/` and `examples/`.

---

## Quick usage snippets

```python
# Tree layout (logarithmic random access)
from succinct_mcmc.core import CompressionConfig
cfg = CompressionConfig(
    num_steps=10_000,
    use_tree_layout=True,
    tree_branching_factor=4,
    tree_levels=2,
)
chain = SuccinctChain(kernel, initial_state=x0, num_steps=10_000, config=cfg)

# Parallel expectation over long chains
from succinct_mcmc.utils import parallel_expectation
mean = parallel_expectation(chain, lambda state: state[0], skip_warmup=True, executor="thread")

# Multivariate ESS
from succinct_mcmc.diagnostics import multivariate_split_ess
result = multivariate_split_ess(mc, lambda state: (state[0], state[1]))
print(result["overall"], result["per_dimension"])

# PyMC adapter scaffolding (illustrative)
from succinct_mcmc.mcmc import PyMCStepKernel
adapter = PyMCStepKernel(
    factory=lambda rng: pymc_step_using_rng(rng),  # provide RNG to PyMC step
    extract=lambda new_point: dict(new_point),     # convert PyMC point to serializable state
)
succinct_chain = SuccinctChain(adapter, initial_state=initial_point, num_steps=50_000)
```

---

## Repository structure

Key pieces (simplified):

```text
src/succinct_mcmc/
  __init__.py               # public API entrypoint
  core/                     # height-compression engine (blocks, RNG, replay)
  mcmc/                     # transition kernels + adapters
  trace/                    # SuccinctChain + MultiChain abstractions
  diagnostics/              # R-hat, ESS, ACF, summaries on succinct traces
  io/                       # SuccinctArtifact + serialization
  experimental/             # tree layouts, parallel replay, strategies
  utils/                    # logging, progress, hashing, profiling

examples/                   # end-to-end usage demos
tests/                      # correctness & regression tests
docs/                       # design + API documentation
````

---

## How it works (conceptual walkthrough)

### 1. Deterministic transition

You provide a kernel that looks like:

```python
def step_fn(state, rng) -> state:
    ...
```

Succinct MCMC wraps this in a `TransitionKernel`:

* All randomness must come from the passed `rng`.
* Given the same `(state, rng-sequence)`, you get the same next state.

This is crucial: it lets us **replay**.

### 2. Running a chain succinctly

Conceptually:

```python
from succinct_mcmc.trace import SuccinctChain
from succinct_mcmc.mcmc import StepFunctionKernel

kernel = StepFunctionKernel(step_fn)
chain = SuccinctChain(
    kernel=kernel,
    initial_state=x0,
    num_steps=T,
    # config / master_seed optional
)
chain.run()
```

During `run()`:

* Library chooses `block_size = ceil(sqrt(T))` (or via config).
* Partitions `[0..T)` into blocks.
* For each block:

  * Stores anchor at block start.
  * Derives a block seed from a master seed.
  * Uses that seed to drive `step()` calls within the block.
* Only anchors & seeds remain after completion.

### 3. Random access

To get `X_t`:

1. Identify block `j` such that `t ∈ [start_j, end_j)`.
2. Load anchor `X_{start_j}`.
3. Rebuild PRNG with that block’s seed.
4. Replay steps until `t`.

In code:

```python
x_t = chain.get(t)
```

### 4. Expectations and summaries

To compute `E[f(X)]`:

* Iterate logically over all `t`:

  * Reconstruct in blocks (implementation optimized to avoid redoing work),
  * Accumulate `f(X_t)` on the fly.

```python
from succinct_mcmc.diagnostics import summary

mean = summary.mean(chain, lambda x: x)  # or pick a param from x
```

No dense `(T, d)` array is ever required.

### 5. Multi-chain diagnostics

To compute R-hat / ESS:

```python
from succinct_mcmc.trace import MultiChain
from succinct_mcmc.diagnostics import rhat, ess

chains = [chain1, chain2, chain3, chain4]
mc = MultiChain(chains)
rhat_val = rhat.rhat(mc, f=lambda x: x[0])     # e.g., first coordinate
ess_val = ess.ess(mc, f=lambda x: x[0])
```

Under the hood:

* Each diagnostic module streams over each chain via succinct replay.
* Memory stays proportional to anchors + small working buffers.

---

## When are the efficiency gains meaningful?

Succinct MCMC is especially valuable when:

1. **T is very large**
   You want 10⁶–10⁹ iterations (or beyond).
2. **State dimension d is nontrivial**
   Many parameters or latent variables.
3. **You care about the entire trajectory**
   For new functionals, tail probabilities, convergence checks, audits.
4. **Memory is the bottleneck, not flops**
   Offline inference, long runs on CPU/GPU where extra compute is acceptable.
5. **You need reproducibility & auditability**
   Ability to regenerate `X_t` exactly is highly desirable.

Domains:

* Bayesian deep learning & probabilistic programming.
* Hierarchical models in epidemiology, ecology, economics.
* Risk, reliability, stress testing.
* Pharma & clinical trial modeling (regulatory-grade reproducibility).
* Any workflow where today you “thin just to make it fit” and would rather keep full-fidelity chains.

---

## Limitations & requirements

To take full advantage:

* Your kernel must be **deterministic given (state, rng)**.
* No hidden global mutable state.
* All randomness must be driven by the library’s RNG plumbing
  (so replay is bit-for-bit reproducible).

Trade-offs:

* Random access is **O(b)** with a flat scheme (b ≈ √T).
* Bulk operations over the entire chain are **O(T)** (as they must be),
  but done in a streaming / replay style.
* There is some CPU overhead due to replay vs dense storage—intentional trade: CPU is cheap, memory is not.

---

## Development & Testing

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python3 -m pytest
```

CI, coverage, and contribution guidelines are being formalized; the commands above provide reproducible validation today.

---

## Getting started (once implemented as a package)

Install (hypothetical):

```bash
pip install succinct-mcmc
```

Minimal usage:

```python
from succinct_mcmc.mcmc import StepFunctionKernel
from succinct_mcmc.trace import SuccinctChain

def log_prob(x):
    return -0.5 * (x**2)  # standard normal

def rw_step(x, rng):
    proposal = x + rng.normalvariate(0.0, 1.0)
    lp_cur = log_prob(x)
    lp_prop = log_prob(proposal)
    import math
    if math.log(rng.random()) < (lp_prop - lp_cur):
        return proposal
    return x

kernel = StepFunctionKernel(rw_step)
chain = SuccinctChain(kernel, initial_state=0.0, num_steps=10**7)
chain.run()

# Query samples
x_100 = chain.get(100)

# Compute mean
from succinct_mcmc.diagnostics import summary
mu = summary.mean(chain, lambda x: x)
print("Estimated mean:", mu)
```

Multi-chain + R-hat:

```python
from succinct_mcmc.trace import MultiChain
from succinct_mcmc.diagnostics import rhat

chains = []
for s in [1, 2, 3, 4]:
    c = SuccinctChain(kernel, initial_state=0.0, num_steps=10**6, master_seed=s)
    c.run()
    chains.append(c)

mc = MultiChain(chains)
print("R-hat:", rhat.rhat(mc, f=lambda x: x))
```

Export succinct artifact:

```python
from succinct_mcmc.io.serialize import save_artifact
from succinct_mcmc.io.artifact import SuccinctArtifact

# (In a full implementation, SuccinctChain would expose a to_artifact() helper.)
artifact = SuccinctArtifact(
    version="0.1.0",
    num_steps=chain.metadata.num_steps,
    block_size=chain.metadata.block_size,
    master_seed=chain.metadata.master_seed,
    block_seeds=[b.seed for b in chain.blocks],
    anchors=None,            # placeholder; real code: encoded anchors
    kernel_metadata={"type": "rw", "details": "..."},
    extra={},
)

save_artifact(artifact, "chain_artifact.json")
```

---

## Coming soon

* Remote/object-store storage backends with optional compression/encryption.
* Expanded CLI (run orchestration, diagnostics, artifact conversion) and richer artifact tooling.
* End-to-end integrations for PyMC/NumPyro/Stan including guided notebooks.
* Comprehensive benchmarking suite comparing dense, thinned, and succinct runs; progress logging improvements.
* CI/CD pipeline, coverage reporting, contribution guide, and release automation.

---

Succinct MCMC already runs long chains with √T storage, deterministic replay, and streaming diagnostics. The items above will polish the developer experience and integrations as we march toward a production release.
