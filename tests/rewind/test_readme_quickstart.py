def test_readme_quickstart_runs():
    # Mirrors the README "Quickstart" block exactly; if the README changes, update both.
    import rewind

    def step(x, rng):
        return x + (1 if rng.random() < 0.5 else -1)

    run = rewind.record(step, init_state=0, n_steps=1_000_000, seed=42)
    x = run.get(842_137)
    fork = run.branch(842_137, mutate=lambda s: s + 100)
    stats = run.stats(lambda s: s)
    assert run.get(842_137) == x
    assert fork.get(0) == x + 100
    assert stats["count"] == 1_000_000
