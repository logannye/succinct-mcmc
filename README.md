# rewind — Git for long stochastic runs

**rewind** records any seeded, cheap-step stochastic loop in **O(√T) memory**, then lets you
**scrub to any tick**, **branch a counterfactual**, **stream exact whole-run statistics**, and
**verify bit-exact replay** — all from a tiny, shareable artifact. Instead of storing a
length-`T` trajectory, it keeps one *anchor* state per `√T`-sized block plus a derivable seed,
and regenerates any state `X_t` on demand by replaying from the nearest anchor.

It's the gradient-checkpointing (REVOLVE/treeverse) time–space tradeoff applied to a whole run:
trade a bounded recompute for sublinear memory, and get a run you can rewind.

```bash
pip install rewind-engine          # import package is `rewind`
pip install "rewind-engine[numpy]" # optional numpy RNG backend + compact array anchors
```

## Quickstart

```python
import rewind

def step(x, rng):
    return x + (1 if rng.random() < 0.5 else -1)

run = rewind.record(step, init_state=0, n_steps=1_000_000, seed=42)

x     = run.get(842_137)                          # regenerate the exact state at any tick
fork  = run.branch(842_137, mutate=lambda s: s + 100)   # fork a counterfactual timeline
stats = run.stats(lambda s: s)                    # exact whole-run stats, one streaming pass

run.save("run.replay")                            # a tiny, shareable artifact (~KB)
loaded = rewind.load("run.replay", step_fn=step)
loaded.verify()                                   # prove bit-exact replay on this machine
```

A 1,000,000-step run records in a fraction of a second, stores **1,000 anchors**, and saves to a
**~4 KB** `.replay` file you can commit.

## The model

You provide a pure transition `step_fn(state, rng) -> state`. All randomness must come from the
supplied `rng`; there must be no hidden global/external state. `rewind` drives the run with
**block-local** randomness — block `j` draws from an RNG derived from `(seed, j)` — so anchors
carry only a derivable seed and any block replays independently.

| Verb | What it does |
| --- | --- |
| `record(step_fn, init_state, n_steps, seed, *, block_size=None, backend="python", observe=None, self_check=False)` | Sweep the run forward, storing O(√T) anchors. `observe={name: fn}` accumulates exact streaming stats inline (no extra pass). |
| `run.get(t)` | Regenerate `X_t` exactly (O(√T) replay from the nearest anchor). |
| `run.branch(t, mutate=None, seed=None, n_steps=None)` | Fork a new `Run` forward from `X_t` (optionally mutated). Records `parent_hash` + `branch_point`, so forks form a provenance DAG. |
| `run.stats(f=identity)` | Exact mean/variance/covariance over the whole run in one streaming pass. |
| `run.save(path)` / `rewind.load(path, step_fn)` | Write / read the `.replay` artifact (lossless, content-hashed). |
| `run.verify(full=False)` | Re-derive anchors and check bit-exact replay on this machine. |

Backends: `"python"` (`random.Random`) and `"numpy"` (`Generator(PCG64)`). Pick one and write your
`step_fn` against its API.

## CLI

```bash
rewind info   run.replay     # metadata only (no step_fn needed)
rewind get    run.replay 8400000
rewind verify run.replay
```

`get`/`verify` replay, so they import the run's `step_fn` from the artifact's `step_id`
(`"module:function"`). Closures/lambdas have no import path — use the library API for those.

## When it fits (the honest regime)

rewind shines when your loop is:

- **deterministic given a seed** — all randomness routed through `rng`, no uncontrolled I/O,
- **cheap per step** — regenerating `X_t` replays up to √T steps, and a full `stats` pass is O(T);
  for microsecond ticks that's nothing, for an expensive log-prob / GPU / LLM step it isn't,
- **modest-dimensional in state** — storage is O(√T·d); great for sim/optimizer-input state, not
  for raw multi-million-parameter NN weights.

It is **not** for nondeterministic GPU/LLM stacks, and it does not sell itself as compliance
attestation — the boundary is the strength: rewind *enforces* determinism (`verify`/`self_check`
refuse to certify a run that isn't bit-exact) rather than quietly breaking it.

## The honest boundary

A `.replay` artifact replays **alongside the same `step_fn` code** — a closure can't be
serialized, so the artifact stores a `step_id` and verifies reproduction; the code doesn't travel
inside the file. `verify()` is what makes an artifact trustworthy on a given machine.

> **Security:** when state isn't a numpy array it's stored via `pickle`. Loading an **untrusted**
> `.replay` executes arbitrary code (same risk as `torch.load`). Treat `.replay` files like code;
> only load ones you trust. (The numpy/JSON paths are safe.)

## What it is not

- **not an array store** (zarr/HDF5 save bytes but can't regenerate state #8,400,000 or fork from it),
- **not a full-tape recorder** (`rr`/Antithesis capture O(T) I/O with no compact shippable artifact),
- **not an experiment tracker** (those log metrics, not regenerable processes).

rewind gives you the one thing they can't: random-access, rewind-and-branch, and one-pass analytics
over a run you could never afford to keep — from a file small enough to commit.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -q
```

Design spec and implementation plan live under `docs/superpowers/`.
