import pytest
from rewind.stats.streaming import StreamingStats, StreamingVectorStats


def test_scalar_matches_reference():
    import statistics
    xs = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    s = StreamingStats()
    for x in xs:
        s.update(x)
    assert s.count == len(xs)
    assert abs(s.mean - statistics.fmean(xs)) < 1e-12
    assert abs(s.variance - statistics.variance(xs)) < 1e-12
    summary = s.summary()
    assert set(summary) == {"count", "mean", "variance", "std"}


def test_scalar_variance_needs_two_points():
    s = StreamingStats()
    s.update(1.0)
    assert s.variance == 0.0


def test_vector_covariance_matches_numpy():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    data = rng.normal(size=(200, 3))
    s = StreamingVectorStats(dim=3)
    for row in data:
        s.update(row.tolist())
    cov = np.array(s.covariance())
    ref = np.cov(data, rowvar=False, bias=False)
    assert np.allclose(cov, ref, atol=1e-10)
    assert s.summary()["count"] == 200
