"""
SuccinctChain: √T-space MCMC trajectory representation.

This class:

- Runs an MCMC chain of length T.
- Stores only O(√T) "anchor" states (one per block) + block metadata.
- Uses deterministic per-block RNG seeding so that each block is exactly
  replayable from its anchor and seed.
- Supports:
    - get(t): random access to X_t via local replay,
    - iter(): streaming all X_0..X_{T-1} via block-wise replay,
    - expectation(f): simple streaming expectations over the logical chain.

Key invariants:

1. The chain is defined by:
    - initial_state,
    - a TransitionKernel `kernel` (deterministic given state + rng),
    - num_steps T,
    - a master_seed,
    - a fixed layout of blocks (BlockMeta) with per-block seeds.

2. run():
    - Computes anchors for all blocks exactly once.
    - Uses per-block seeds for transitions INSIDE that block.
    - Does NOT retain all intermediate states.

3. get(t):
    - Looks up the block containing t.
    - Replays from that block's anchor with that block's seed.
    - Returns X_t exactly as defined by the run() procedure.

4. iter():
    - Reconstructs the full sequence by replaying blocks in order.
    - Never uses dense O(T) storage.

This file is the core "user-facing" entry point.
"""

import random
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..core.config import CompressionConfig, choose_block_size
from ..core.anchors import AnchorStore
from ..core.compression import make_flat_blocks, compute_tree_anchor_steps
from ..core.block import BlockMeta
from ..core.rng import make_block_rng, available_rng_backends, RNGFactoryError
from ..mcmc.base import TransitionKernel


@dataclass
class ReplayPolicy:
    """
    Configuration for how replay operations should be executed.

    Currently supports only the flat √T layout but provides an extension point
    for future strategies (tree-based, caching, etc.).
    """

    strategy: str = "flat"

    def validate(self) -> None:
        if self.strategy != "flat":
            raise ValueError(f"Unsupported replay strategy '{self.strategy}'")


@dataclass
class ChainMetadata:
    """
    Lightweight metadata about a succinct chain.

    Useful for:
        - logging,
        - artifact creation,
        - debugging.
    """
    num_steps: int
    block_size: int
    num_blocks: int
    master_seed: int
    rng_backend: str


@dataclass
class ChainCheckpoint:
    """
    Lightweight checkpoint for resuming succinct chain execution.

    Attributes:
        num_steps: Total logical steps of the original chain.
        block_size: Block size used for compression.
        next_block_index: Index of the next block that still needs simulation.
        anchor_snapshot: Backend-provided snapshot of anchors.
        anchor_rng_states: RNG state captured for each stored anchor.
        ran: Whether the original chain had completed execution.
    """

    num_steps: int
    block_size: int
    next_block_index: int
    anchor_snapshot: Any
    anchor_rng_states: Dict[int, Any]
    ran: bool = False
    rng_backend: str = "python"


class SuccinctChain:
    """
    Succinct MCMC chain with √T-space storage.

    Parameters:
        kernel: TransitionKernel implementing step(x, rng) -> x_next.
        initial_state: Initial MCMC state X_0.
        num_steps: Total number of MCMC steps T (produces states X_0..X_{T-1}).
        config: Optional CompressionConfig; if omitted, uses sqrt(T) block size.
        master_seed: Global seed used to derive per-block seeds.

    Usage (intended full implementation):

        from succinct_mcmc.mcmc import GaussianRandomWalkMH
        from succinct_mcmc.trace import SuccinctChain

        kernel = GaussianRandomWalkMH(log_prob_fn)
        chain = SuccinctChain(kernel, initial_state=0.0, num_steps=10**7, master_seed=42)
        chain.run()

        x_100 = chain.get(100)
        mu = chain.expectation(lambda x: x)

    Notes on determinism:

    - All randomness flows through make_block_rng(block.seed).
    - TransitionKernel.step MUST be deterministic given (state, rng).
    - Given the same code + seeds, get(t) is fully reproducible.
    """

    def __init__(
        self,
        kernel: TransitionKernel,
        initial_state: Any,
        num_steps: int,
        config: Optional[CompressionConfig] = None,
        master_seed: int = 12345,
        warmup_steps: int = 0,
        replay_policy: Optional[ReplayPolicy] = None,
        resume_from: Optional[ChainCheckpoint] = None,
        anchor_store: Optional[AnchorStore] = None,
        rng_backend: str = "python",
    ):
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if warmup_steps > num_steps:
            raise ValueError("warmup_steps cannot exceed num_steps")

        self.kernel = kernel
        self.initial_state = initial_state
        self.num_steps = num_steps
        self.master_seed = master_seed
        self.rng_backend = rng_backend

        try:
            make_block_rng(0, backend=self.rng_backend)
        except RNGFactoryError as exc:
            available = ", ".join(available_rng_backends()) or "none"
            raise ValueError(
                f"Unknown rng_backend '{self.rng_backend}'. Available: {available}."
            ) from exc

        # Choose block size (default: ceil(sqrt(T))) and layout.
        self.config = config or CompressionConfig(num_steps=num_steps)
        self.block_size = choose_block_size(self.config)

        # Anchor store: holds O(√T) states, one per block start.
        self.anchor_store = anchor_store or AnchorStore()

        # Flat list of blocks covering [0, num_steps).
        # Each BlockMeta has:
        #   index, start_step, end_step, seed
        self.blocks: List[BlockMeta] = make_flat_blocks(
            num_steps=self.num_steps,
            block_size=self.block_size,
            master_seed=self.master_seed,
        )
        self._block_start_steps: List[int] = [block.start_step for block in self.blocks]

        # Anchor bookkeeping (block starts + optional tree anchors).
        self._anchor_rng_states: Dict[int, Any] = {}
        self._anchor_step_to_index: Dict[int, int] = {
            block.start_step: block.index for block in self.blocks
        }
        self._extra_anchor_indices: Dict[int, int] = {}
        if self.config.use_tree_layout:
            extra_steps = compute_tree_anchor_steps(
                num_steps=self.num_steps,
                base_block_size=self.block_size,
                branching_factor=self.config.tree_branching_factor,
                levels=self.config.tree_levels,
            )
            base_index = len(self.blocks)
            for step in extra_steps:
                if step in self._anchor_step_to_index or step >= self.num_steps:
                    continue
                anchor_index = base_index + len(self._extra_anchor_indices)
                self._extra_anchor_indices[step] = anchor_index
                self._anchor_step_to_index[step] = anchor_index
        self._anchor_steps_sorted: List[int] = sorted(self._anchor_step_to_index.keys())

        self.replay_policy = (replay_policy or ReplayPolicy())
        self.replay_policy.validate()

        self.warmup_steps = min(warmup_steps, self.num_steps)

        self.metadata = ChainMetadata(
            num_steps=self.num_steps,
            block_size=self.block_size,
            num_blocks=len(self.blocks),
            master_seed=self.master_seed,
            rng_backend=self.rng_backend,
        )

        # Indicates whether run() has been executed.
        self._ran: bool = False
        self._next_block_index: int = 0

        if resume_from is not None:
            self._restore_from_checkpoint(resume_from)

        if not self.anchor_store.has_anchor(0):
            self.anchor_store.set_anchor(0, self.initial_state)

    # -------------------------------------------------------------------------
    # Core execution
    # -------------------------------------------------------------------------

    def run(self, until_step: Optional[int] = None) -> None:
        """
        Execute the succinct MCMC run and populate anchors.

        Parameters:
            until_step: If provided, execute blocks up to (and including) the
                block that ends at `until_step`. `until_step` must coincide with
                a block boundary (or be equal to num_steps). This enables
                checkpoint-friendly, incremental execution without recomputing
                completed blocks.

        Notes:
            - run() is idempotent for completed blocks.
            - Calling run() multiple times with increasing `until_step` values
              allows resume-style execution.
        """
        if until_step is not None:
            if until_step <= 0 or until_step > self.num_steps:
                raise ValueError("until_step must lie in (0, num_steps]")
            boundary_block = self._find_block_for_step(until_step - 1)
            if until_step != boundary_block.end_step and until_step != self.num_steps:
                raise ValueError(
                    "until_step must align with a block boundary or equal num_steps"
                )

        if self._next_block_index >= len(self.blocks):
            # Already fully executed.
            self._ran = True
            return

        target_index = len(self.blocks) - 1
        if until_step is not None:
            target_index = self._find_block_for_step(until_step - 1).index

        master_rng = random.Random(self.master_seed)

        if self._next_block_index == 0:
            self.blocks[0].seed = master_rng.getstate()
            self._anchor_rng_states.setdefault(0, self.blocks[0].seed)
            if not self.anchor_store.has_anchor(0):
                self.anchor_store.set_anchor(0, self.initial_state)
        else:
            master_rng.setstate(self.blocks[self._next_block_index].seed)

        current_block = self.blocks[self._next_block_index]
        state = self.anchor_store.get_anchor(current_block.index)
        current_step = current_block.start_step
        self._anchor_rng_states.setdefault(current_block.index, master_rng.getstate())
        self._maybe_record_extra_anchor(current_step, state, master_rng)

        for block_index in range(self._next_block_index, target_index + 1):
            block = self.blocks[block_index]
            if block_index != self._next_block_index:
                state = self.anchor_store.get_anchor(block.index)
                current_step = block.start_step
                self._anchor_rng_states.setdefault(block.index, master_rng.getstate())
                self._maybe_record_extra_anchor(current_step, state, master_rng)

            for step in range(block.start_step + 1, block.end_step):
                state = self.kernel.step(state, master_rng)
                current_step = step
                self._maybe_record_extra_anchor(current_step, state, master_rng)

            next_block_index = block.index + 1
            if next_block_index < len(self.blocks):
                state = self.kernel.step(state, master_rng)
                current_step = self.blocks[next_block_index].start_step
                self.anchor_store.set_anchor(next_block_index, state)
                seed = master_rng.getstate()
                self.blocks[next_block_index].seed = seed
                self._anchor_rng_states[next_block_index] = seed
                self._maybe_record_extra_anchor(current_step, state, master_rng)

            self._next_block_index = next_block_index

        self._ran = self._next_block_index >= len(self.blocks)

    def _maybe_record_extra_anchor(
        self, step: int, state: Any, rng: random.Random
    ) -> None:
        anchor_index = self._extra_anchor_indices.get(step)
        if anchor_index is None:
            return
        if anchor_index in self._anchor_rng_states:
            return
        self.anchor_store.set_anchor(anchor_index, state)
        self._anchor_rng_states[anchor_index] = rng.getstate()

    # -------------------------------------------------------------------------
    # Random access
    # -------------------------------------------------------------------------

    def _find_block_for_step(self, t: int) -> BlockMeta:
        """
        Locate the BlockMeta that contains logical step t.

        For flat layout:
            - blocks are disjoint and contiguous,
            - a simple linear scan is O(√T) and fine.
        Later:
            - can be upgraded to binary search or indexed structure.
        """
        idx = bisect_right(self._block_start_steps, t) - 1
        if idx < 0 or idx >= len(self.blocks):
            raise RuntimeError(f"No block covers step {t}")
        block = self.blocks[idx]
        if not (block.start_step <= t < block.end_step):
            raise RuntimeError(f"Computed block {idx} does not cover step {t}")
        return block

    def get(self, t: int) -> Any:
        """
        Return state X_t via local replay.

        Preconditions:
            - run() has been called successfully.
            - 0 <= t < num_steps.

        Process:
            - Find block B containing t.
            - Use B.seed + anchor[B.index] to replay up to t.
            - This uses the same kernel + RNG scheme as in run(), so it is exact.
        """
        if not self._ran:
            raise RuntimeError("Chain not yet run. Call run() first.")
        if not (0 <= t < self.num_steps):
            raise IndexError("t out of range")

        anchor_pos = bisect_right(self._anchor_steps_sorted, t) - 1
        if anchor_pos < 0:
            raise RuntimeError("No anchor available to reconstruct requested step")
        anchor_step = self._anchor_steps_sorted[anchor_pos]
        anchor_index = self._anchor_step_to_index[anchor_step]
        try:
            state = self.anchor_store.get_anchor(anchor_index)
        except KeyError as exc:
            raise RuntimeError(f"Missing anchor state at step {anchor_step}") from exc

        seed = self._anchor_rng_states.get(anchor_index)
        if seed is None:
            raise RuntimeError(f"Missing RNG state for anchor index {anchor_index}")
        rng = make_block_rng(seed, backend=self.rng_backend)

        current_step = anchor_step
        while current_step < t:
            state = self.kernel.step(state, rng)
            current_step += 1
        return state

    # -------------------------------------------------------------------------
    # Iteration / streaming
    # -------------------------------------------------------------------------

    def iter(self, *, skip_warmup: bool = False) -> Iterable[Any]:
        """
        Stream all states X_0, X_1, ..., X_{T-1}.

        Implementation:

        - Replays blocks in order.
        - For block 0:
            * yield anchor[0] (X_0),
            * step inside block and yield each state.
        - For block j > 0:
            * start from anchor[j] (which is X_{start_step}),
            * DO NOT yield it immediately (it equals last state of previous block),
            * step inside block and yield X_{start_step+1}..X_{end_step-1}.

        This:
        - avoids dense storage,
        - avoids N^2 recomputation (we only simulate each block once per full iter()),
        - yields each logical state exactly once.
        """
        if not self._ran:
            raise RuntimeError("Chain not yet run. Call run() first.")

        warmup_cutoff = self.warmup_steps if skip_warmup else 0

        for block in self.blocks:
            state = self.anchor_store.get_anchor(block.index)
            current_step = block.start_step
            if current_step >= warmup_cutoff:
                yield state

            rng = make_block_rng(block.seed, backend=self.rng_backend)
            for step in range(block.start_step + 1, block.end_step):
                state = self.kernel.step(state, rng)
                current_step = step
                if current_step >= warmup_cutoff:
                    yield state

    # -------------------------------------------------------------------------
    # Simple analytics
    # -------------------------------------------------------------------------

    def expectation(self, f: Callable[[Any], float], *, skip_warmup: bool = True) -> float:
        """
        Compute the empirical mean of f(X_t) over the full logical chain.

        This is a simple, generic helper implemented via iter().
        For more advanced streaming statistics, see diagnostics.summary.
        """
        total = 0.0
        n = 0
        for x in self.iter(skip_warmup=skip_warmup):
            total += f(x)
            n += 1
        if n == 0:
            raise ValueError("Chain has zero length")
        return total / n

    # -------------------------------------------------------------------------
    # Checkpointing utilities
    # -------------------------------------------------------------------------

    def checkpoint(self) -> ChainCheckpoint:
        """
        Generate a checkpoint capturing computed anchors and execution progress.

        The checkpoint can be serialized externally and later passed to a new
        SuccinctChain via the resume_from parameter.
        """
        return ChainCheckpoint(
            num_steps=self.num_steps,
            block_size=self.block_size,
            next_block_index=self._next_block_index,
            anchor_snapshot=self.anchor_store.snapshot(),
            anchor_rng_states=dict(self._anchor_rng_states),
            ran=self._ran,
            rng_backend=self.rng_backend,
        )

    def _restore_from_checkpoint(self, checkpoint: ChainCheckpoint) -> None:
        """
        Internal helper to restore chain state from a checkpoint.
        """
        if checkpoint.num_steps != self.num_steps:
            raise ValueError(
                "Checkpoint num_steps does not match chain configuration"
            )
        if checkpoint.block_size != self.block_size:
            raise ValueError(
                "Checkpoint block_size does not match chain configuration"
            )

        if checkpoint.rng_backend != self.rng_backend:
            raise ValueError(
                "Checkpoint rng_backend does not match chain configuration"
            )

        self.anchor_store.restore(checkpoint.anchor_snapshot)
        self._anchor_rng_states = dict(checkpoint.anchor_rng_states)
        for index, seed in self._anchor_rng_states.items():
            if index < len(self.blocks):
                self.blocks[index].seed = seed
        self._next_block_index = checkpoint.next_block_index
        self._ran = checkpoint.ran
