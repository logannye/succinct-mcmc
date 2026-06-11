"""rewind CLI: inspect, regenerate, and verify .replay artifacts.

`info` reads metadata only. `get`/`verify` replay, so they need the run's
step_fn; they import it from the artifact's step_id ("module:function").
Closures/lambdas have no import path and require the library API instead.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Callable, Optional

from .io.artifact import load, load_meta


def _import_step(step_id: Optional[str]) -> Callable:
    if not step_id or ":" not in step_id or "<" in step_id:
        raise ValueError(
            f"step_id {step_id!r} is not importable; replay needs an importable "
            f"'module:function' step. Use the library API with your step_fn."
        )
    module_name, _, qual = step_id.partition(":")
    obj = importlib.import_module(module_name)
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj


def _cmd_info(args) -> int:
    print(json.dumps(load_meta(args.artifact), indent=2, sort_keys=True))
    return 0


def _cmd_get(args) -> int:
    meta = load_meta(args.artifact)
    step_fn = _import_step(meta.get("step_id"))
    run = load(args.artifact, step_fn=step_fn)
    print(repr(run.get(args.t)))
    return 0


def _cmd_verify(args) -> int:
    meta = load_meta(args.artifact)
    step_fn = _import_step(meta.get("step_id"))
    run = load(args.artifact, step_fn=step_fn)
    run.verify(full=True)
    print("OK: artifact replays bit-for-bit on this machine")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="rewind", description="inspect .replay artifacts")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_info = sub.add_parser("info", help="print artifact metadata (no step_fn needed)")
    p_info.add_argument("artifact")
    p_info.set_defaults(func=_cmd_info)

    p_get = sub.add_parser("get", help="regenerate state at tick t")
    p_get.add_argument("artifact")
    p_get.add_argument("t", type=int)
    p_get.set_defaults(func=_cmd_get)

    p_verify = sub.add_parser("verify", help="check bit-exact replay on this machine")
    p_verify.add_argument("artifact")
    p_verify.set_defaults(func=_cmd_verify)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, ImportError, AttributeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
