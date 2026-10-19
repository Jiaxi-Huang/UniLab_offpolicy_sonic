#!/usr/bin/env python3
"""Pack approximately a fixed total number of frames from converted SONIC data.

This operates only on existing SONIC NPZ files. Clips are taken in deterministic
filename order; the last selected clip is cropped so the packed store contains
exactly ``--frames`` frames (unless the source dataset is shorter).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from unilab.tasks.motion_tracking.g1.sonic_data import (
    _DEFAULT_REFERENCE_FRAMES,
    resolve_sonic_pairs,
)
from unilab.tools.pack_sonic_data import pack_sonic_dataset


def _write_subset(source: Path, destination: Path, end: int) -> None:
    """Copy one converted NPZ, retaining only its first ``end`` frames."""

    with np.load(source, allow_pickle=False) as payload:
        count = int(np.asarray(payload["num_frames"]).item())
        if end <= 0 or end > count:
            raise ValueError(f"Invalid frame limit {end} for {source} with {count} frames")
        values: dict[str, np.ndarray] = {}
        for name in payload.files:
            value = np.asarray(payload[name])
            if name == "num_frames":
                values[name] = np.asarray(end, dtype=np.int32)
            elif value.ndim > 0 and value.shape[0] == count:
                values[name] = value[:end]
            else:
                values[name] = value
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **values)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/sonic"),
        help="existing converted SONIC root with robot_filtered/ and smpl_filtered/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sonic/packed_2000"),
        help="new packed-store directory",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=2000,
        help="target total frames across all selected clips (default: 2000)",
    )
    parser.add_argument("--max-clips", type=int, help="consider at most the first N clips")
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--split-discontinuities", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.frames < _DEFAULT_REFERENCE_FRAMES:
        raise SystemExit(
            f"--frames must be at least {_DEFAULT_REFERENCE_FRAMES} for SONIC references"
        )
    if args.max_clips is not None and args.max_clips <= 0:
        raise SystemExit("--max-clips must be positive")

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    try:
        pairs = resolve_sonic_pairs(
            str(source / "robot_filtered"),
            str(source / "smpl_filtered"),
            robot_suffix=".npz",
            smpl_suffix=".npz",
        )
        if args.max_clips is not None:
            pairs = pairs[: args.max_clips]
        if not pairs:
            raise FileNotFoundError(f"No converted SONIC pairs found under {source}")

        with tempfile.TemporaryDirectory(prefix="sonic_subset_") as temporary:
            temporary_root = Path(temporary)
            robot_subset = temporary_root / "robot_filtered"
            smpl_subset = temporary_root / "smpl_filtered"
            remaining = args.frames
            selected: list[str] = []
            for robot_path, smpl_path in pairs:
                with np.load(robot_path, allow_pickle=False) as robot:
                    count = int(np.asarray(robot["num_frames"]).item())
                if count < _DEFAULT_REFERENCE_FRAMES:
                    continue
                take = min(count, remaining)
                if take < _DEFAULT_REFERENCE_FRAMES:
                    break
                _write_subset(robot_path, robot_subset / robot_path.name, take)
                _write_subset(smpl_path, smpl_subset / smpl_path.name, take)
                selected.append(robot_path.stem)
                remaining -= take
                if remaining == 0:
                    break
            if remaining:
                raise ValueError(
                    f"Converted dataset contains only {args.frames - remaining} usable frames; "
                    f"cannot build the requested {args.frames}"
                )

            packing = pack_sonic_dataset(
                robot_subset,
                smpl_subset,
                output,
                progress_interval=args.progress_interval,
                split_discontinuities=args.split_discontinuities,
            )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(json.dumps({"packed": packing, "selected_clips": selected}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
