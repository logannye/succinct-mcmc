"""The Run object: the user-facing handle over a recorded trajectory."""
from __future__ import annotations

from typing import Any, Callable, Iterator, Optional

from ..errors import NondeterministicReplayError
from .anchors import AnchorStore
from .blocks import Block, block_index_for_step, make_blocks
from .replay import iter_block, replay_to
from .seeds import derive_branch_seed, derive_seed


def _states_equal(a: Any, b: Any) -> bool:
    try:
        import numpy as np

        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return np.array_equal(a, b)
    except ImportError:
        pass
    return bool(a == b)


def _is_vector(v: Any) -> bool:
    return hasattr(v, "__len__") and not isinstance(v, (str, bytes))


class Run:
    def __init__(self, *, step_fn: Callable[[Any, Any], Any], init_state: Any,
                 n_steps: int, block_size: int, seed: int, backend: str,
                 store: AnchorStore, observed: Optional[dict] = None,
                 step_id: Optional[str] = None, parent_hash: Optional[str] = None,
                 branch_point: Optional[int] = None,
                 observe: Optional[dict] = None):
        self.step_fn = step_fn
        self.init_state = init_state
        self.n_steps = n_steps
        self.block_size = block_size
        self.seed = seed
        self.backend = backend
        self.store = store
        self.observed = observed or {}
        self.observe = observe or {}
        self.step_id = step_id
        self.parent_hash = parent_hash
        self.branch_point = branch_point
        self.blocks: list[Block] = make_blocks(n_steps, block_size)

    # -- random access -----------------------------------------------------
    def get(self, t: int) -> Any:
        if not (0 <= t < self.n_steps):
            raise IndexError(f"t={t} out of range [0, {self.n_steps})")
        j = block_index_for_step(t, self.block_size)
        block = self.blocks[j]
        return replay_to(self.step_fn, self.store.get(j), block, t,
                         seed=derive_seed(self.seed, j), backend=self.backend)

    def iter(self) -> Iterator[Any]:
        for block in self.blocks:
            yield from iter_block(self.step_fn, self.store.get(block.index), block,
                                  seed=derive_seed(self.seed, block.index),
                                  backend=self.backend)

    # -- analytics ---------------------------------------------------------
    def stats(self, f: Callable[[Any], Any] = lambda s: s) -> dict:
        from ..stats.streaming import StreamingStats, StreamingVectorStats

        acc = None
        for state in self.iter():
            v = f(state)
            if acc is None:
                acc = StreamingVectorStats(len(v)) if _is_vector(v) else StreamingStats()
            acc.update(list(v) if _is_vector(v) else v)
        if acc is None:
            raise ValueError("empty run")
        return acc.summary()

    # -- integrity ---------------------------------------------------------
    def verify(self, full: bool = False) -> bool:
        n_blocks = len(self.blocks)
        indices = range(n_blocks - 1) if full else self._sample_block_indices(n_blocks - 1)
        for j in indices:
            block = self.blocks[j]
            recomputed = replay_to(self.step_fn, self.store.get(j), block, block.stop,
                                   seed=derive_seed(self.seed, j), backend=self.backend)
            if not _states_equal(recomputed, self.store.get(j + 1)):
                raise NondeterministicReplayError(block_index=j)
        return True

    @staticmethod
    def _sample_block_indices(upper: int) -> list[int]:
        if upper <= 0:
            return []
        if upper <= 16:
            return list(range(upper))
        step = max(1, upper // 16)
        return list(range(0, upper, step))

    # -- branching ---------------------------------------------------------
    def branch(self, t: int, mutate: Optional[Callable[[Any], Any]] = None,
               seed: Optional[int] = None, n_steps: Optional[int] = None) -> "Run":
        from .record import record

        x = self.get(t)
        if mutate is not None:
            x = mutate(x)
        child_n = n_steps if n_steps is not None else (self.n_steps - t)
        if child_n <= 0:
            raise ValueError("branch length must be positive (t too close to end?)")
        child_seed = seed if seed is not None else derive_branch_seed(self.seed, t)
        child = record(self.step_fn, x, child_n, child_seed,
                       block_size=None, backend=self.backend, observe=self.observe,
                       step_id=self.step_id)
        child.parent_hash = self.content_hash()
        child.branch_point = t
        return child

    # -- persistence -------------------------------------------------------
    def content_hash(self) -> str:
        from ..io.artifact import content_hash

        return content_hash(self)

    def metadata(self) -> dict:
        from ..io.artifact import metadata

        return metadata(self)

    def save(self, path) -> None:
        from ..io.artifact import save

        save(self, path)
