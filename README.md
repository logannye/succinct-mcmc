# rewind

**Time-travel for long, seeded simulations.** `rewind` records a multi-million-step stochastic
run in a tiny artifact, then lets you jump to *any* step, fork an alternate timeline from it, and
compute exact whole-run statistics — without storing the full trajectory or re-running from the
start.

Think "Git for a long stochastic run": a kilobyte you can scrub, branch, share, and verify.

---

## The problem it solves

You run a long stochastic loop — an agent-based simulation, a physics or epidemiology model, a
seeded training/rollout loop. Hours in, something goes wrong at step 8,400,000: a NaN, a rare
event, a divergence. Now what?

- **Storing every step** is too big (millions of states × your state size).
- **Re-running from step 0** to inspect one step is a full linear scan, every time you look.
- **Keeping only summary metrics** throws away the states you actually need to debug.

`rewind` exploits the one thing these loops have in common: each step is a deterministic function
of the previous state and some seeded randomness. So it stores only **√T checkpoints** (≈1,000 for
a million steps) plus the seed, and **regenerates any step on demand** by replaying from the nearest
checkpoint. You get random access to the whole run for a fraction of the memory.

## Who it's for

Engineers and researchers running **long, seed-deterministic, cheap-per-step** stochastic loops:

- **Simulation** — agent-based models, traffic/economic/epidemiology sims, game/netcode replay,
  robotics and digital-twin loops.
- **ML/RL infrastructure** — debugging loss-spikes/NaNs in cheap-step training or rollout loops,
  regenerating the exact batch/RNG behind a failure, forking "what-if" runs from a shared prefix.
- **Scientific computing** — any long Monte-Carlo / SDE / sampler run where you want exact,
  reproducible random access and a small, verifiable record of the run.

If your steps are cheap to recompute and your randomness flows through a seed, `rewind` fits. (See
[When it fits](#when-it-fits) for where it doesn't.)

## Install

```bash
pip install rewind-engine            # the import package is `rewind`
pip install "rewind-engine[numpy]"   # optional: numpy RNG backend + compact array checkpoints
```

## Quickstart

Write your loop as a pure `step(state, rng) -> state` (all randomness from `rng`), then record it:

```python
import rewind

def step(x, rng):
    return x + (1 if rng.random() < 0.5 else -1)   # a 1-D random walk

run = rewind.record(step, init_state=0, n_steps=1_000_000, seed=42)

x     = run.get(842_137)                                 # regenerate any step, exactly
fork  = run.branch(842_137, mutate=lambda s: s + 100)    # fork a counterfactual timeline
stats = run.stats(lambda s: s)                           # exact whole-run stats, one pass

run.save("run.replay")                                   # a tiny (~KB) shareable artifact
loaded = rewind.load("run.replay", step_fn=step, allow_pickle=True)  # see Security
loaded.verify()                                          # prove bit-exact replay on this machine
```

A 1,000,000-step run records in a fraction of a second, stores **1,000 checkpoints**, and saves to
a **~4 KB** `.replay` file you can commit to a repo or attach to a bug report.

## Example: find and fix a late-run NaN

`rewind` turns "where did this blow up?" into a binary search over a run you never stored.

```python
import math, rewind
from dataclasses import dataclass

@dataclass
class State:
    value: float

def step(s, rng):
    shock = rng.normalvariate(0, 1)
    v = s.value * (1.0 + 0.5 * shock)
    if abs(shock) > 2.0:               # a rare tail event amplifies hard
        v *= 1e50
    if math.isinf(v):
        v = v - v                      # inf -> NaN
    return State(v)

run = rewind.record(step, init_state=State(1.0), n_steps=1_000_000, seed=7)

# scrub to the first step that went bad — each get() replays only ~√T steps
bad = next(t for t in range(run.n_steps) if math.isnan(run.get(t).value))
print("first NaN at step", bad, "— state just before:", run.get(bad - 1).value)

# fork an alternate timeline from just before the blow-up, with a fix applied
safe = run.branch(bad - 1, mutate=lambda s: State(min(s.value, 1e6)))
print("clamped fork stays finite:", not math.isnan(safe.get(100).value))
```

## API at a glance

| Call | What it does |
| --- | --- |
| `rewind.record(step_fn, init_state, n_steps, seed, *, block_size=None, backend="python", observe=None, self_check=False)` | Record the run, storing O(√T) checkpoints. `observe={name: fn}` computes exact streaming stats inline (no extra pass). `self_check=True` verifies determinism while recording. |
| `run.get(t)` | Regenerate the exact state at step `t`. |
| `run.branch(t, mutate=None, seed=None, n_steps=None)` | Fork a new run forward from step `t` (optionally mutate the state). Forks link back via `parent_hash` + `branch_point`. |
| `run.stats(f=identity)` | Exact mean/variance/covariance of `f(state)` over the whole run, in one streaming pass. |
| `run.save(path)` / `rewind.load(path, step_fn, *, allow_pickle=False)` | Write / read the `.replay` artifact (lossless, content-hashed). |
| `run.verify(full=False)` | Re-derive checkpoints and confirm bit-exact replay on this machine. |

RNG backends: `"python"` (`random.Random`) and `"numpy"` (`Generator(PCG64)`). Pick one and write
`step_fn` against its API.

## Command line

Inspect, regenerate, and verify shared `.replay` files from the shell:

```bash
rewind info   run.replay                  # metadata only (no step code needed)
rewind verify run.replay  --allow-pickle  # confirm bit-exact replay here
rewind get    run.replay 8400000 --allow-pickle
```

`get`/`verify` replay the run, so they import the run's `step_fn` from the recorded
`step_id` (`"module:function"`). Closures/lambdas have no import path — use the library API for those.

## When it fits

`rewind` is built for loops that are:

- **deterministic given a seed** — all randomness through `rng`, no uncontrolled I/O or clock/network,
- **cheap per step** — regenerating a step replays up to √T steps and a full `stats` pass is O(T);
  trivial for microsecond ticks, not for an expensive log-prob / GPU / LLM step,
- **modest in state size** — storage is O(√T · state size); great for sim/optimizer state, not for
  raw multi-million-parameter network weights.

It is **not** a fit for nondeterministic GPU/LLM stacks, and it is not a compliance/attestation
tool. The boundary is a feature: `verify()` and `self_check` *refuse to certify* a run that isn't
bit-exact, rather than silently producing wrong replays.

## Security

A `.replay` artifact stores numeric (numpy) checkpoints safely. When your state is an arbitrary
Python object, checkpoints are stored with `pickle`, and **loading an untrusted artifact executes
code** — the same risk as `torch.load`. `rewind.load` therefore refuses pickle-encoded artifacts by
default; pass `allow_pickle=True` only for artifacts you trust. Metadata (JSON) and numpy `.npz`
checkpoints are always safe.

## How it works

A length-`T` run is split into `b ≈ √T`-sized blocks. `rewind` stores one checkpoint per block plus
a per-block seed *derived* from your master seed, so randomness is block-local and any block replays
independently. `get(t)` loads the nearest checkpoint and replays forward to `t`; `verify` re-derives
each checkpoint and checks it matches. It's the gradient-checkpointing (REVOLVE/treeverse)
time–space tradeoff applied to a whole stochastic run.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -q
```

The full design spec and implementation plan live under `docs/superpowers/`.
