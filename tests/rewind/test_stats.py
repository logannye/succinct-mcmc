import pytest
import rewind


def walk(x, rng):
    return x + (1.0 if rng.random() < 0.5 else -1.0)


def test_stats_scalar_matches_dense_reference():
    pytest.importorskip("numpy")
    import numpy as np

    run = rewind.record(walk, init_state=0.0, n_steps=500, seed=11)
    dense = np.array(list(run.iter()))
    s = run.stats(lambda x: x)
    assert abs(s["mean"] - dense.mean()) < 1e-9
    assert abs(s["variance"] - dense.var(ddof=1)) < 1e-9


def test_stats_vector():
    run = rewind.record(walk, init_state=0.0, n_steps=200, seed=5)
    s = run.stats(lambda x: (x, x * x))
    assert s["count"] == 200
    assert len(s["mean"]) == 2
    assert len(s["covariance"]) == 2


def test_inline_observe_matches_posthoc_stats():
    run = rewind.record(walk, init_state=0.0, n_steps=300, seed=9,
                        observe={"x": lambda s: s})
    posthoc = run.stats(lambda s: s)
    assert run.observed["x"]["count"] == posthoc["count"] == 300
    assert abs(run.observed["x"]["mean"] - posthoc["mean"]) < 1e-12
    assert abs(run.observed["x"]["variance"] - posthoc["variance"]) < 1e-12
