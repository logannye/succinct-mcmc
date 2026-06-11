import pytest
import rewind
from rewind.errors import ArtifactIntegrityError


def walk(x, rng):
    return x + (1 if rng.random() < 0.5 else -1)


def test_roundtrip_get_is_bit_identical(tmp_path):
    run = rewind.record(walk, init_state=0, n_steps=64, seed=5, block_size=8,
                        observe={"x": lambda s: s})
    path = tmp_path / "run.replay"
    run.save(path)
    loaded = rewind.load(path, step_fn=walk, allow_pickle=True)  # trusted, self-generated
    for t in (0, 1, 7, 8, 13, 31, 32, 50, 63):
        assert loaded.get(t) == run.get(t)
    assert loaded.observed["x"]["count"] == 64


def test_load_meta_needs_no_step_fn(tmp_path):
    run = rewind.record(walk, init_state=0, n_steps=40, seed=5)
    path = tmp_path / "run.replay"
    run.save(path)
    meta = rewind.io.artifact.load_meta(path)
    assert meta["n_steps"] == 40
    assert meta["name"] == "rewind"
    assert "content_hash" in meta


def test_corrupted_artifact_detected(tmp_path):
    import zipfile, json
    run = rewind.record(walk, init_state=0, n_steps=40, seed=5)
    path = tmp_path / "run.replay"
    run.save(path)
    # rewrite meta with a wrong n_steps but keep stale content_hash
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        data = {n: zf.read(n) for n in names}
    meta = json.loads(data["meta.json"])
    meta["n_steps"] = 999999
    data["meta.json"] = json.dumps(meta).encode()
    with zipfile.ZipFile(path, "w") as zf:
        for n, b in data.items():
            zf.writestr(n, b)
    with pytest.raises(ArtifactIntegrityError):
        rewind.load(path, step_fn=walk, allow_pickle=True)  # opt in so the hash check is reached


def test_numpy_anchor_roundtrip(tmp_path):
    np = pytest.importorskip("numpy")

    def vstep(x, rng):
        return x + rng.normal(size=x.shape)

    run = rewind.record(vstep, init_state=np.zeros(3), n_steps=50, seed=2,
                        backend="numpy")
    path = tmp_path / "v.replay"
    run.save(path)
    loaded = rewind.load(path, step_fn=vstep, backend="numpy")
    assert np.array_equal(loaded.get(37), run.get(37))
