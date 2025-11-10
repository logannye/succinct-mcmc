"""Command-line utilities for Succinct MCMC."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .io import load_artifact


def _artifact_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _artifact_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _artifact_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_artifact_to_dict(v) for v in obj]
    return obj


def artifact_info(path: Path, *, fields: Optional[Iterable[str]] = None, pretty: bool = True) -> str:
    artifact = load_artifact(path)
    data = _artifact_to_dict(artifact)
    if fields:
        data = {field: data.get(field) for field in fields}
    indent = 2 if pretty else None
    return json.dumps(data, indent=indent)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="succinct-mcmc", description="Succinct MCMC command-line tools")
    subparsers = parser.add_subparsers(dest="command")

    info_parser = subparsers.add_parser("artifact-info", help="Inspect a succinct artifact JSON")
    info_parser.add_argument("path", type=Path, help="Path to artifact JSON file")
    info_parser.add_argument("--field", action="append", dest="fields", help="Specific fields to display")
    info_parser.add_argument("--compact", action="store_true", help="Output compact JSON (default pretty)")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "artifact-info":
        try:
            output = artifact_info(args.path, fields=args.fields, pretty=not args.compact)
        except FileNotFoundError:
            parser.error(f"Artifact not found: {args.path}")
            return 2
        print(output)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
