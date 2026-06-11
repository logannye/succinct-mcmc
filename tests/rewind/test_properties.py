from hypothesis import given, settings, strategies as st
import rewind


def _walk(x, rng):
    return x + (1 if rng.random() < 0.5 else -1)


@settings(max_examples=50, deadline=None)
@given(n_steps=st.integers(min_value=1, max_value=300),
       seed=st.integers(min_value=0, max_value=10_000),
       block_size=st.one_of(st.none(), st.integers(min_value=1, max_value=40)))
def test_get_equals_iter_for_random_configs(n_steps, seed, block_size):
    run = rewind.record(_walk, init_state=0, n_steps=n_steps, seed=seed,
                        block_size=block_size)
    seq = list(run.iter())
    assert len(seq) == n_steps
    for t in (0, n_steps // 2, n_steps - 1):
        assert run.get(t) == seq[t]


@settings(max_examples=30, deadline=None)
@given(n_steps=st.integers(min_value=4, max_value=200),
       seed=st.integers(min_value=0, max_value=10_000))
def test_roundtrip_preserves_get(tmp_path_factory, n_steps, seed):
    run = rewind.record(_walk, init_state=0, n_steps=n_steps, seed=seed)
    path = tmp_path_factory.mktemp("rt") / "r.replay"
    run.save(path)
    loaded = rewind.load(path, step_fn=_walk, allow_pickle=True)
    assert loaded.get(n_steps - 1) == run.get(n_steps - 1)
    assert loaded.verify(full=True)
