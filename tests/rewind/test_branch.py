import pytest
import rewind


def walk(x, rng):
    return x + (1 if rng.random() < 0.5 else -1)


def test_branch_starts_at_fork_state():
    run = rewind.record(walk, init_state=0, n_steps=100, seed=4)
    fork = run.branch(40)
    assert fork.get(0) == run.get(40)
    assert fork.branch_point == 40
    assert fork.n_steps == 60               # remaining horizon by default


def test_branch_mutate_applies_at_fork():
    run = rewind.record(walk, init_state=0, n_steps=100, seed=4)
    fork = run.branch(40, mutate=lambda x: x + 1000)
    assert fork.get(0) == run.get(40) + 1000


def test_branch_diverges_from_parent_suffix():
    run = rewind.record(walk, init_state=0, n_steps=100, seed=4)
    fork = run.branch(40)
    parent_suffix = [run.get(40 + k) for k in range(1, 20)]
    fork_suffix = [fork.get(k) for k in range(1, 20)]
    assert parent_suffix != fork_suffix     # different (branch) seed -> different path


def test_branch_explicit_seed_is_reproducible():
    run = rewind.record(walk, init_state=0, n_steps=100, seed=4)
    a = [run.branch(30, seed=777).get(k) for k in range(10)]
    b = [run.branch(30, seed=777).get(k) for k in range(10)]
    assert a == b


def test_branch_too_close_to_end_raises():
    run = rewind.record(walk, init_state=0, n_steps=10, seed=4)
    with pytest.raises(ValueError):
        run.branch(9, n_steps=0)
