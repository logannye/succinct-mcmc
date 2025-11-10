"""
Serialization helpers for SuccinctArtifact.

Responsibilities:
- Convert SuccinctArtifact <-> JSON + binary blobs.
- Hide file format details from users.

Initially:
- Simple JSON for metadata,
- Numpy .npy or similar for anchors.

Later:
- Backwards-compatible schema evolution,
- Checksums and signatures.
"""

import json
from pathlib import Path
from typing import Union

from .artifact import SuccinctArtifact


def save_artifact(artifact: SuccinctArtifact, path: Union[str, Path]) -> None:
    """
    Write artifact metadata to a JSON file.

    For now:
    - Assume anchors, etc. are JSON-serializable or small.
    - Later split large arrays into separate files.
    """
    path = Path(path)
    data = artifact.__dict__
    path.write_text(json.dumps(data, indent=2))


def load_artifact(path: Union[str, Path]) -> SuccinctArtifact:
    """
    Load artifact from JSON.

    Later:
    - Validate schema and version,
    - Reconstruct complex anchor storage layouts.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    return SuccinctArtifact(**data)
