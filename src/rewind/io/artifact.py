"""The .replay artifact: a zip container with lossless, content-hashed anchors.

Layout (zip):
  meta.json        -- run metadata + content_hash
  anchors.npz      -- when every anchor is an ndarray (numpy fast path)
  anchors.pkl      -- otherwise (pickle fallback for arbitrary state)

The step_fn is NOT stored (a closure cannot be serialized). meta records a
step_id so the CLI can import it when it is an importable "module:function";
otherwise replay requires the caller to pass step_fn to load().

SECURITY: when anchors are arbitrary Python objects they are stored via
``pickle`` (the ``anchors.pkl`` fallback). Loading such an artifact from an
UNTRUSTED source executes arbitrary code during unpickling — the same risk as
``torch.load`` / ``joblib.load``. The JSON metadata and the numpy ``anchors.npz``
fast path (loaded with allow_pickle=False) are safe; only the pickle fallback is
dangerous. Treat .replay files like code: only load ones you trust. A future
release should gate the pickle path behind an explicit ``allow_pickle=True``.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import zipfile
from pathlib import Path
from typing import Any, Optional

FORMAT_VERSION = 1


def _anchor_encoding(store) -> str:
    try:
        import numpy as np
    except ImportError:
        return "pickle"
    items = list(store.items())
    if items and all(isinstance(s, np.ndarray) for _, s in items):
        return "npz"
    return "pickle"


def _encode_anchors(store, encoding: str) -> tuple[str, bytes]:
    if encoding == "npz":
        import io as _io

        import numpy as np

        buf = _io.BytesIO()
        np.savez(buf, **{str(i): s for i, s in store.items()})
        return "anchors.npz", buf.getvalue()
    return "anchors.pkl", pickle.dumps(store.as_dict(), protocol=pickle.HIGHEST_PROTOCOL)


def _decode_anchors(encoding: str, blob: bytes) -> dict:
    if encoding == "npz":
        import io as _io

        import numpy as np

        with np.load(_io.BytesIO(blob), allow_pickle=False) as npz:  # safe: no code exec
            return {int(k): npz[k] for k in npz.files}
    return pickle.loads(blob)


def _canonical_state_bytes(state: Any) -> bytes:
    """A deterministic byte image of one anchor state, for content hashing.

    Hashing the .npz/.pkl *container* is unsafe: numpy's savez embeds
    timestamps, so re-encoding identical data yields different bytes. We hash
    the array/object values directly instead (fixed pickle protocol for
    cross-version stability of plain Python state).
    """
    try:
        import numpy as np

        if isinstance(state, np.ndarray):
            return b"ndarray|%s|%s|%s" % (
                str(state.dtype).encode(), str(state.shape).encode(), state.tobytes())
    except ImportError:
        pass
    return b"pickle|" + pickle.dumps(state, protocol=4)


def _canonical_anchor_bytes(store) -> bytes:
    parts = [str(i).encode() + b"=" + _canonical_state_bytes(store.get(i))
             for i in sorted(store.as_dict())]
    return b"\x00".join(parts)


def metadata(run) -> dict:
    """Metadata dict EXCLUDING content_hash (used to compute it)."""
    return {
        "name": "rewind",
        "format_version": FORMAT_VERSION,
        "master_seed": run.seed,
        "n_steps": run.n_steps,
        "block_size": run.block_size,
        "backend": run.backend,
        "seed_derivation": "blake2b-v1",
        "step_id": run.step_id,
        "observed": run.observed,
        "parent_hash": run.parent_hash,
        "branch_point": run.branch_point,
        "anchor_encoding": _anchor_encoding(run.store),
    }


def content_hash(run) -> str:
    meta = metadata(run)
    h = hashlib.blake2b(digest_size=32)
    h.update(json.dumps(meta, sort_keys=True).encode("utf-8"))
    h.update(_canonical_anchor_bytes(run.store))   # canonical, container-independent
    return h.hexdigest()


def save(run, path) -> None:
    path = Path(path)
    meta = metadata(run)
    encoding = meta["anchor_encoding"]
    anchor_name, anchor_bytes = _encode_anchors(run.store, encoding)
    meta["content_hash"] = content_hash(run)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps(meta, sort_keys=True))
        zf.writestr(anchor_name, anchor_bytes)


def load_meta(path) -> dict:
    with zipfile.ZipFile(Path(path)) as zf:
        return json.loads(zf.read("meta.json"))


def load(path, step_fn, *, backend: Optional[str] = None):
    from ..core.anchors import AnchorStore
    from ..core.run import Run
    from ..errors import ArtifactIntegrityError

    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        meta = json.loads(zf.read("meta.json"))
        encoding = meta["anchor_encoding"]
        anchor_name = "anchors.npz" if encoding == "npz" else "anchors.pkl"
        anchor_blob = zf.read(anchor_name)

    stored_hash = meta.pop("content_hash", None)
    if meta.get("name") != "rewind" or meta.get("format_version") != FORMAT_VERSION:
        raise ArtifactIntegrityError("not a rewind v1 artifact")

    # rebuild store, recompute hash from canonical anchor bytes, compare
    anchors = _decode_anchors(encoding, anchor_blob)
    store = AnchorStore()
    for i, s in anchors.items():
        store.set(int(i), s)

    h = hashlib.blake2b(digest_size=32)
    h.update(json.dumps(meta, sort_keys=True).encode("utf-8"))
    h.update(_canonical_anchor_bytes(store))
    if stored_hash != h.hexdigest():
        raise ArtifactIntegrityError("content hash mismatch (artifact corrupted or altered)")

    return Run(
        step_fn=step_fn,
        init_state=store.get(0),
        n_steps=meta["n_steps"],
        block_size=meta["block_size"],
        seed=meta["master_seed"],
        backend=backend or meta["backend"],
        store=store,
        observed=meta.get("observed", {}),
        step_id=meta.get("step_id"),
        parent_hash=meta.get("parent_hash"),
        branch_point=meta.get("branch_point"),
    )
