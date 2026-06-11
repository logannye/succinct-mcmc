import pytest
import rewind
from rewind.cli import main


# Module-level so step_id is importable as "tests.rewind.test_cli:cli_walk"
def cli_walk(x, rng):
    return x + (1 if rng.random() < 0.5 else -1)


def _make(tmp_path):
    run = rewind.record(cli_walk, init_state=0, n_steps=40, seed=5,
                        step_id="tests.rewind.test_cli:cli_walk")
    path = tmp_path / "run.replay"
    run.save(path)
    return path, run


def test_cli_info_needs_no_step(tmp_path, capsys):
    path, _ = _make(tmp_path)
    rc = main(["info", str(path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "n_steps" in out and "40" in out


def test_cli_get_imports_step_and_matches(tmp_path, capsys):
    path, run = _make(tmp_path)
    rc = main(["get", str(path), "17"])
    out = capsys.readouterr().out
    assert rc == 0
    assert str(run.get(17)) in out


def test_cli_verify_ok(tmp_path, capsys):
    path, _ = _make(tmp_path)
    rc = main(["verify", str(path)])
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_cli_get_without_importable_step_errors(tmp_path, capsys):
    run = rewind.record(lambda x, rng: x + 1, init_state=0, n_steps=20, seed=1)
    path = tmp_path / "lam.replay"
    run.save(path)
    rc = main(["get", str(path), "5"])
    assert rc != 0
    assert "import" in capsys.readouterr().err.lower()
