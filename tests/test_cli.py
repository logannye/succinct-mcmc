"""
Tests for the succinct-mcmc CLI.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from succinct_mcmc.cli import artifact_info, main
from succinct_mcmc.io import SuccinctArtifact, save_artifact


def make_artifact(tmp_path: Path) -> Path:
    artifact = SuccinctArtifact(
        version="0.1.0",
        num_steps=100,
        block_size=10,
        master_seed=42,
        block_seeds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        anchors={},
        kernel_metadata={"name": "dummy"},
        extra={"note": "unit test"},
    )
    path = tmp_path / "artifact.json"
    save_artifact(artifact, path)
    return path


def test_artifact_info_returns_json(tmp_path):
    path = make_artifact(tmp_path)
    output = artifact_info(path)
    assert "\"num_steps\": 100" in output


def test_cli_main_artifact_info(tmp_path, capsys):
    path = make_artifact(tmp_path)
    exit_code = main(["artifact-info", str(path), "--field", "num_steps", "--compact"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "{\"num_steps\": 100}"
