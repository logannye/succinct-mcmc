import pytest
from rewind.core.seeds import derive_seed, derive_branch_seed, make_rng, available_backends


def test_derive_seed_is_deterministic_and_index_sensitive():
    assert derive_seed(42, 0) == derive_seed(42, 0)
    assert derive_seed(42, 0) != derive_seed(42, 1)
    assert derive_seed(1, 0) != derive_seed(2, 0)
    assert 0 <= derive_seed(42, 7) < 2**64


def test_derive_branch_seed_distinct_from_block_seed():
    assert derive_branch_seed(42, 5) == derive_branch_seed(42, 5)
    assert derive_branch_seed(42, 5) != derive_seed(42, 5)


def test_python_backend_reproducible():
    a = make_rng(derive_seed(42, 0), backend="python")
    b = make_rng(derive_seed(42, 0), backend="python")
    assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]


def test_numpy_backend_reproducible_and_accepts_int_seed():
    np = pytest.importorskip("numpy")
    a = make_rng(derive_seed(42, 0), backend="numpy")
    b = make_rng(derive_seed(42, 0), backend="numpy")
    assert list(a.normal(size=5)) == list(b.normal(size=5))


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        make_rng(123, backend="nope")


def test_available_backends_includes_python():
    assert "python" in available_backends()
