#!/usr/bin/env python3
"""Download and extract the complete SONIC G1 and SMPL training sources."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from unilab.assets.sonic import download_sonic_training_data


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/bones_seed_source"),
        help="raw data destination (default: data/bones_seed_source)",
    )
    parser.add_argument(
        "--token",
        help="Hugging Face token; defaults to the token saved by `hf auth login`",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        paths = download_sonic_training_data(args.output, token=args.token)
    except (OSError, PermissionError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(paths.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
