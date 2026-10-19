#!/usr/bin/env python3
"""Pack paired SONIC walking clips into a dedicated mmap dataset."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from unilab.tasks.motion_tracking.g1.sonic_data import resolve_sonic_pairs
from unilab.tools.pack_sonic_data import pack_sonic_dataset


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/sonic"),
        help="converted SONIC root containing robot_filtered/ and smpl_filtered/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sonic/packed_walk"),
    )
    parser.add_argument(
        "--keyword",
        default="walk",
        help="case-insensitive clip-name substring (default: walk)",
    )
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--split-discontinuities", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    robot_input = args.source / "robot_filtered"
    smpl_input = args.source / "smpl_filtered"
    try:
        pairs = resolve_sonic_pairs(
            str(robot_input),
            str(smpl_input),
            robot_suffix=".npz",
            smpl_suffix=".npz",
        )
        keyword = args.keyword.casefold()
        clip_names = [robot.stem for robot, _ in pairs if keyword in robot.stem.casefold()]
        if not clip_names:
            raise FileNotFoundError(
                f"No paired SONIC clip name contains {args.keyword!r} under {args.source}"
            )
        summary = pack_sonic_dataset(
            robot_input,
            smpl_input,
            args.output,
            clip_names=clip_names,
            progress_interval=args.progress_interval,
            split_discontinuities=args.split_discontinuities,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    summary["keyword"] = args.keyword
    summary["source_clips"] = clip_names
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
