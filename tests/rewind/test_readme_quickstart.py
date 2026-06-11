def test_readme_quickstart_runs(tmp_path):
    # Mirrors the README "Quickstart" block. If the README changes, update both.
    import rewind

    def step(x, rng):
        return x + (1 if rng.random() < 0.5 else -1)

    run = rewind.record(step, init_state=0, n_steps=1_000_000, seed=42)
    x = run.get(842_137)                                    # regenerate any tick, exactly
    fork = run.branch(842_137, mutate=lambda s: s + 100)    # fork a counterfactual timeline
    stats = run.stats(lambda s: s)                          # exact whole-run stats, one pass

    path = tmp_path / "run.replay"
    run.save(path)                                          # tiny shareable artifact
    loaded = rewind.load(path, step_fn=step, allow_pickle=True)
    assert loaded.verify()                                  # bit-exact replay on this machine

    assert run.get(842_137) == x
    assert fork.get(0) == x + 100
    assert stats["count"] == 1_000_000
    assert loaded.get(842_137) == x
