import pytest
import rewind


def make_walk():
    def step(x, rng):
        return x + (1 if rng.random() < 0.5 else -1)
    return step


@pytest.mark.parametrize("n_steps", [1, 2, 7, 16, 25, 100, 101])
def test_get_matches_full_iter(n_steps):
    run = rewind.record(make_walk(), init_state=0, n_steps=n_steps, seed=42)
    seq = list(run.iter())
    assert len(seq) == n_steps
    for t in range(n_steps):
        assert run.get(t) == seq[t]


def test_get_out_of_range_raises():
    run = rewind.record(make_walk(), init_state=0, n_steps=10, seed=1)
    with pytest.raises(IndexError):
        run.get(10)
    with pytest.raises(IndexError):
        run.get(-1)


def test_anchor_count_is_sqrt_t():
    run = rewind.record(make_walk(), init_state=0, n_steps=10_000, seed=1)
    assert len(run.store) == 100          # ceil(sqrt(10000)) blocks


def test_same_seed_identical_different_seed_differs():
    a = list(rewind.record(make_walk(), 0, 60, seed=7).iter())
    b = list(rewind.record(make_walk(), 0, 60, seed=7).iter())
    c = list(rewind.record(make_walk(), 0, 60, seed=8).iter())
    assert a == b and a != c


def test_numpy_backend_records_and_replays():
    pytest.importorskip("numpy")

    def step(x, rng):
        return x + rng.normal()

    run = rewind.record(step, init_state=0.0, n_steps=50, seed=3, backend="numpy")
    seq = list(run.iter())
    assert run.get(37) == seq[37]
