# Design Spec — `rewind`: a replayable-run / time-travel engine for seeded stochastic loops

**Date:** 2026-06-11
**Status:** Approved (brainstorming) → ready for implementation plan
**Author:** Logan Nye
**Name:** `rewind` (confirmed)
**Supersedes positioning of:** `succinct-mcmc` (the √T-MCMC framing is retired; MCMC becomes one example)

---

## 1. Problem & context

`succinct-mcmc` is built on one genuinely elegant idea: an MCMC step is `X_{t+1} = F(X_t, U_t)` — deterministic given the previous state plus seeded randomness — so a length-`T` chain can be represented in **O(√T)** memory by storing one "anchor" state per `√T`-sized block plus the per-block randomness, and **any** sample `X_t` regenerated on demand by replaying from the nearest anchor. This is the gradient-checkpointing (REVOLVE/treeverse) time–space tradeoff applied to a stochastic trajectory.

A comprehensive review (full code read + empirical probes + adversarial use-case scoring) established:

- **What works:** the deterministic block-anchor replay core is correct (`get(t)` is bit-exact; anchors = ⌈√T⌉ to within 1%; streaming Welford mean/var/cov is sound; checkpoint/resume is exact).
- **What's broken/oversold:** R-hat & ESS are statistically invalid (they first-difference the chain → R-hat ≈ 1 even for divergent chains; ESS over-reports by exactly the autocorrelation time τ and is O(T²)); the √T *memory* win is mostly fictional at low dimension because each anchor stores a 24 KB Mersenne-Twister snapshot (scalar chains are *larger* than a dense array until T > ~4.2M); the shareable artifact roundtrip is unimplemented (no `to_artifact`, JSON corrupts the RNG tuples); the numpy backend is write-only; the PyMC/NumPyro adapters can't satisfy the determinism contract.
- **The fatal economic objection to the original pitch:** a full-chain diagnostic replays ~1.0×T expensive log-prob evals (a full re-run per query), and real Bayesian workflows are ESS-capped, so ArviZ+zarr already store/diagnose the draws people keep far better. Memory-scaling-for-MCMC is a dead wedge.

An adversarial idea-space exploration (7 domain lenses → cluster → champion/skeptic/judge with market research) found that the highest-value, most-achievable use case is **not** the original one and **not** the seductive regulated "audit receipt" (that's the runner-up — it dies on FIT, since auditors want a signed Merkle root over retained outputs that sigstore/in-toto already provide, and on C3, since its real workloads aren't bit-reproducible). The winner is the one use case where the √T-replay core is **load-bearing** and the three hard constraints are satisfied **by construction**:

> **A "Git for long stochastic runs": an interactive time-travel / rewind-and-branch engine for seeded, cheap-step stochastic loops.**

### Decisions locked during brainstorming

| Decision | Choice |
|---|---|
| **Direction** | Time-travel debugger ("Git for long stochastic runs"); provenance/`verify` rides free as a feature, not the headline |
| **Goal** | A useful OSS tool others adopt — optimize for fit, reach, clean wedge |
| **Beachhead** | Generic seeded simulation (domain-agnostic `step_fn`); MCMC is one example |
| **RNG model** | Per-block derived seeds (small seeds, numpy/Python/JAX-friendly, independent/parallel block replay) |
| **v1 scope** | `get(t)` + small replayable artifact (core) **plus** `branch(t)`, `stats()`, `verify`, and a CLI |
| **Packaging** | Reframe in place (rename `succinct_mcmc` → `rewind`, port the proven internals, delete the broken/dead weight) |

### The three hard constraints this design respects

- **C1 (replay cost):** regenerating `X_t` costs replaying up to `b ≈ √T` transition steps. The engine wins when steps are **cheap** (the selected domain) or replay is occasional. Documented honestly; not hidden.
- **C2 (dimension cost):** storage is O(√T · d). Fits modest-d sim/optimizer-input state; not for huge-d (e.g. raw NN weights).
- **C3 (determinism):** all randomness must be seed-routed, no external nondeterminism. **Enforced as a precondition** via `verify` / `self_check`, not promised and quietly broken.

### Non-goals (v1)

- MCMC convergence diagnostics (R-hat/ESS/ACF) — dropped from core; may return as an optional `[mcmc]` extra later.
- Wrapping external samplers (PyMC/NumPyro/Stan) — those manage their own RNG and can't satisfy the contract.
- Expensive-tick domains (LLM inference, GPU Monte-Carlo, full-weight training replay) — explicitly out of the supported regime.
- Cryptographic attestation / regulated-compliance product — the provenance DAG metadata is present so it's *possible* later, but it is not a v1 product.

---

## 2. Core model & API surface

A **`Run`** is defined by:
- `step_fn(state, rng) -> state` — pure given `(state, rng draws)`; no hidden global/external state,
- `init_state` — `X_0`,
- `n_steps` — `T`,
- `seed` — `master_seed`,
- `block_size` — default `⌈√T⌉`.

### Per-block-derived-seed semantics (the key idea)

Block `j` gets a fresh seed `seed_j = H(master_seed, j)` (counter-based, e.g. a hash or `numpy.random.SeedSequence(master_seed).spawn`). At each block boundary the engine stores the anchor `A_j = X_{j·b}` (O(√T) anchors). **Randomness is block-local by design** — within a block the stream runs from `seed_j`; at the boundary it resets. Consequences:
- anchors carry only a few bytes of derivable seed (no 24 KB snapshot),
- any block replays **independently** (true parallel scrub),
- the chain is well-defined: `get(t)` loads `A_{t//b}`, builds `rng(seed_{t//b})`, steps forward to `t`.

This is the deliberate fix for the current code's incoherence (it computed per-block seeds, then overwrote them with continuous-stream snapshots).

### v1 API

```python
import rewind

run = rewind.record(
    step_fn,
    init_state=x0,
    n_steps=10_000_000,
    seed=42,
    observe={"mean_x": lambda s: s.x},   # inline streaming stats, no extra pass
)

x    = run.get(8_400_000)                 # bit-exact random-access scrub (O(√T); O(log T) w/ tree)
fork = run.branch(8_399_000,              # counterfactual timeline from a tick
                  mutate=lambda s: s.with_(dt=0.5))
e    = run.stats(lambda s: s.energy)      # exact whole-run stats, one streaming pass
run.save("run.replay")                    # tiny artifact
run.verify()                              # bit-exact self-check on this machine
```

- **`get(t)`** — exact random-access regeneration. O(√T) worst case; O(log T) with tree anchors.
- **`branch(t, mutate=None, seed=None)`** — materialize `X_t` (optionally mutate state/params), return a **new `Run`** that continues forward under its own seed (when `seed=None`, derived deterministically from `parent_hash + branch_point`, so a branch is itself reproducible). Records `parent_hash` + `branch_point` → forks form a provenance DAG.
- **`stats(f=identity)`** — exact streaming mean/var/cov over the full run (one replay pass for new functionals). Functionals passed to `observe=` at `record` time are accumulated **inline** (zero extra replay).
- **`save(path)` / `rewind.load(path)`** — the `.replay` artifact roundtrip.
- **`verify(full=False)`** — re-derive anchors and check against stored anchors + content hash.

---

## 3. The `.replay` artifact, state model & provenance DAG

The current artifact roundtrip is the critical missing piece; this is the one genuinely new build.

### Container format

`.replay` = metadata + binary anchor blob + content hash:
- **metadata** (msgpack or JSON): format version, `master_seed`, `n_steps`, `block_size`, seed-derivation rule, `step_id` + optional source hash, `observe` results, and for forks `parent_hash` + `branch_point`.
- **anchors:** `.npz` when state is an ndarray / PyTree of arrays (compact, fast); pickle fallback for arbitrary Python objects. Seeds are **not stored** by default — re-derived from `master_seed`; only anchors cost space.
- **content hash:** a Merkle-style root over `(metadata ‖ anchors)` → powers `verify` and makes the artifact tamper-evident (the provenance feature that rides free).

This directly fixes the review findings: no `json.dumps` tuple corruption, anchors actually present, a real `save`/`load`.

### State model — opaque with a fast path

- **Default:** any picklable object (zero ceremony for sim authors with dict/object state).
- **Fast path:** ndarray / PyTree of arrays auto-detected → compact binary (`.npz`).
- **Extension:** optional `codec=(encode, decode)` for large/custom states.
- v1 ships pickle-default + ndarray fast path; `codec` is the documented hook.

### Honesty boundary (documented up front)

The artifact replays **alongside the same `step_fn` code** — a closure can't be serialized, so the artifact stores a `step_id` + optional source hash and *verifies reproduction*, rather than pretending the code travels in the file. `verify()` is what makes an artifact trustworthy on a given machine.

### Provenance DAG

Each `branch` records `parent_hash` + `branch_point`, so forked timelines form a hash-linked DAG traceable to the root run + tick. (Same rooted-provenance pattern as the Rosalind `chain verify`; present as metadata in v1, not built into a separate product.)

---

## 4. Package structure & keep/drop

Reframe in place: rename `src/succinct_mcmc/` → `src/rewind/`, port the proven internals, delete the broken/dead weight, demote MCMC to one example.

```
src/rewind/
  core/
    record.py     # record() → Run: forward sweep that stores anchors
    run.py        # Run: get / branch / stats / verify / save / metadata
    blocks.py     # block layout            ← ported from core/compression.py
    seeds.py      # per-block seed derivation + RNG backends (python, numpy; jax later)
    anchors.py    # anchor store + storage backends   ← ported (in-mem, file, npz-memmap)
    replay.py     # replay_point/replay_block ← ported, FIXED to per-block seeds
    tree.py       # log-depth tree anchors    ← ported + FIXED (off-grid midpoints)
  stats/
    streaming.py  # Welford mean/var/cov      ← ported from diagnostics/summary.py
  io/
    artifact.py   # .replay container: save/load/hash/provenance  (NEW)
  cli.py          # replay get / branch / stats / verify <artifact>
examples/
  agent_sim.py    # HERO DEMO: ~10M-tick sim that NaNs at 8.4M → scrub, root-cause, branch
  mcmc.py         # MCMC (MH/HMC) as ONE example step_fn
tests/            # keep replay-exactness/storage/resume; ADD roundtrip/branch/verify/stats
```

**KEEP & port (proven):** block-anchor replay engine, anchor store + storage backends, Welford streaming stats, checkpoint/resume, tree-layout concept, MH/HMC kernels (→ `examples/mcmc.py`), `StepFunctionKernel` (→ canonical `step_fn` adapter).

**DELETE (broken/dead/oversold):**
- `diagnostics/rhat.py`, `ess.py`, `acf.py` — statistically-wrong O(T²) R-hat/ESS (out of scope; maybe a later `[mcmc]` extra).
- the 24 KB continuous-stream RNG snapshot path → replaced by `seeds.py`.
- `mcmc/pymc_adapter.py`, `numpyro_adapter.py` — non-functional scaffolding.
- `experimental/` (tree_index stub, broken process-pool `parallel_replay`).
- broken `numpy` backend → reborn correctly in `seeds.py`.
- `utils/hashing.hash_repr` (collision-prone) → replaced by canonical-bytes hashing in `io/artifact.py`.

---

## 5. Error handling & determinism enforcement

The determinism contract is the product's integrity, so it is enforced, not trusted:

- **`record(..., self_check=False)`** by default for speed; **`self_check=True`** re-derives each anchor by an independent replay during recording and raises `NondeterministicStepError(first_diverging_block)` on mismatch — catches a non-pure `step_fn` (hidden global state, unseeded randomness, wall-clock/IO) early.
- **`verify(full=False)`** re-derives a sample of anchors (or all) from the artifact, checks against stored anchors + content hash. Answers "does this replay bit-exactly *on this machine*?" → turns C3 into a gate. On mismatch → `ArtifactIntegrityError` (corruption) or `NondeterministicReplayError` (platform/code drift), with the first failing tick.
- **Stateful-kernel guard:** the silent adaptive-kernel trap is structurally impossible (state lives only in the threaded `state` value); `examples/mcmc.py` documents "freeze adaptation before recording," and `self_check=True` catches a violation.
- Other clear errors: `get(t)` out of range → `IndexError`; branch from a partial/unrecorded run → explicit error; unpicklable/oversized state → error pointing to `codec=`; artifact version/hash mismatch on load → `ArtifactIntegrityError`.

---

## 6. Testing strategy & performance bar

**Tests (the ones that should have existed):**
- **Replay exactness:** `get(t) == full_sweep[t]` for all `t`, across block *and* tree-anchor boundaries, on **both** `python` and `numpy` backends, incl. non-divisible T / partial final block.
- **Artifact roundtrip:** `record → save → load → get(t)` bit-identical (absent today).
- **Branch:** fork at `t` → prefix matches parent, suffix diverges as specified, provenance metadata correct.
- **Stats:** streaming mean/var/cov match a dense numpy reference to float-eps; `observe=` inline == post-hoc `stats(f)`.
- **Verify:** detects a deliberately corrupted artifact *and* a deliberately nondeterministic `step_fn`.
- **Property-based (Hypothesis):** random `step_fn` / T / block-size → exactness + roundtrip invariants.

**Performance bar (hero demo):** T = 10⁷ cheap-tick sim → artifact ~tens of KB, `get(t)` in milliseconds, full `stats` pass in seconds. Pure-Python for cheap ticks; numpy backend for vectorized state. Documented honestly: for *expensive* ticks, replay cost is the known √T tradeoff — name the regime, don't hide it.

---

## 7. The honest one-paragraph positioning

> **`rewind` — Git for long stochastic runs.** A tiny, exactly-replayable artifact that lets you scrub to any tick of a multi-million-step seeded run, regenerate the precise state and RNG draw behind a late-run NaN or rare event, fork a counterfactual branch forward, and stream exact whole-run statistics — all without recording the full tape or re-running from zero. It is for engineers running long, deterministic-given-seed simulations and cheap-step loops, where the trajectory is too big to store but you still need to land on the one tick that broke. The boundary is the strength: it works *because* your randomness is seed-routed and your steps are cheap, so it enforces that as a precondition and refuses to certify runs that aren't bit-exact — which is exactly why it doesn't chase nondeterministic GPU/LLM stacks or sell itself as compliance attestation. Not an array store (zarr saves bytes but can't regenerate state #8.4M or fork from it), not a full-tape recorder (`rr`/Antithesis capture O(T) I/O with no compact shippable artifact), not an experiment tracker (those log metrics, not regenerable processes). It gives you the one thing they can't: random-access, rewind-and-branch, and one-pass analytics over a run you could never afford to keep — from a file small enough to commit.
