import itertools
import pytest
import rewind
from rewind.errors import NondeterministicReplayError, NondeterministicStepError


def walk(x, rng):
    return x + (1 if rng.random() < 0.5 else -1)


def test_verify_passes_for_deterministic_run():
    run = rewind.record(walk, init_state=0, n_steps=400, seed=2)
    assert run.verify(full=True) is True
    assert run.verify(full=False) is True


def test_self_check_passes_for_pure_step():
    rewind.record(walk, init_state=0, n_steps=120, seed=2, self_check=True)  # no raise


def test_self_check_detects_hidden_state():
    counter = itertools.count()

    def impure(x, rng):
        # draws from a hidden global generator, not rng -> non-reproducible
        return x + next(counter)

    with pytest.raises(NondeterministicStepError):
        rewind.record(impure, init_state=0, n_steps=120, seed=2, self_check=True)


def test_verify_detects_corrupted_anchor():
    run = rewind.record(walk, init_state=0, n_steps=400, seed=2)
    run.store.set(2, 999999)  # corrupt one anchor
    with pytest.raises(NondeterministicReplayError):
        run.verify(full=True)
