import pytest
import rewind
from rewind.errors import UnsafeArtifactError


def walk(x, rng):
    return x + (1 if rng.random() < 0.5 else -1)


def test_pickle_artifact_refused_by_default(tmp_path):
    run = rewind.record(walk, init_state=0, n_steps=40, seed=5)  # int state -> pickle encoding
    path = tmp_path / "p.replay"
    run.save(path)
    with pytest.raises(UnsafeArtifactError):
        rewind.load(path, step_fn=walk)                 # allow_pickle defaults to False


def test_pickle_artifact_loads_with_opt_in(tmp_path):
    run = rewind.record(walk, init_state=0, n_steps=40, seed=5)
    path = tmp_path / "p.replay"
    run.save(path)
    loaded = rewind.load(path, step_fn=walk, allow_pickle=True)
    assert loaded.get(20) == run.get(20)


def test_numpy_artifact_loads_without_opt_in(tmp_path):
    np = pytest.importorskip("numpy")

    def vstep(x, rng):
        return x + rng.normal(size=x.shape)

    run = rewind.record(vstep, init_state=np.zeros(3), n_steps=40, seed=2, backend="numpy")
    path = tmp_path / "v.replay"
    run.save(path)
    loaded = rewind.load(path, step_fn=vstep, backend="numpy")   # npz is safe, no opt-in
    assert np.array_equal(loaded.get(20), run.get(20))
