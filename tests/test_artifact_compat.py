"""
Tests: artifact serialization round-trip.

Goal:
- Ensure SuccinctArtifact can be serialized and deserialized with current schema.
"""

import json
from pathlib import Path

from succinct_mcmc.io import SuccinctArtifact, save_artifact, load_artifact


def test_artifact_roundtrip(tmp_path: Path):
    artifact = SuccinctArtifact(
        version="0.0-test",
        num_steps=100,
        block_size=10,
        master_seed=42,
        block_seeds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        anchors={"0": [0.0]},  # placeholder
        kernel_metadata={"name": "dummy"},
        extra={"note": "test"},
    )

    path = tmp_path / "art.json"
    save_artifact(artifact, path)

    # Check it's valid JSON
    json.loads(path.read_text())

    loaded = load_artifact(path)
    assert loaded.version == artifact.version
    assert loaded.num_steps == artifact.num_steps
    assert loaded.block_size == artifact.block_size
    assert loaded.master_seed == artifact.master_seed
