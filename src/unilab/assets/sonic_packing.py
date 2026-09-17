#!/usr/bin/env python3
"""Pack paired SONIC NPZ clips into a shared read-only mmap store."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from unilab.tasks.motion_tracking.g1.sonic_data import (
    _DEFAULT_REFERENCE_FRAMES,
    _SONIC_ROBOT_NPZ_FORMAT,
    _SONIC_SMPL_NPZ_FORMAT,
    _TARGET_FPS,
    _WRIST_POLICY_INDICES,
    G1_POLICY_JOINT_NAMES,
    SONIC_PACKED_ARRAY_NAMES,
    SONIC_PACKED_FORMAT,
    resolve_sonic_pairs,
)
from unilab.utils.rotation import (
    np_quat_apply_batched,
    np_quat_conjugate_batched,
    np_quat_mul_batched,
)

_MAX_SMPL_LOCAL_JOINT_STEP_M = 0.5
_MAX_SMPL_RELATIVE_ROOT_STEP_DEG = 20.0
_MAX_WRIST_JOINT_STEP_RAD = 0.5


def _pair_metadata(
    robot_path: Path, smpl_path: Path
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    with np.load(robot_path, allow_pickle=False) as robot:
        robot_format = str(np.asarray(robot["source_format"]).item())
        robot_fps = int(np.asarray(robot["fps"]).item())
        robot_frames = int(np.asarray(robot["num_frames"]).item())
        joint_names = tuple(str(name) for name in robot["joint_names"])
        body_names = tuple(str(name) for name in robot["body_names"])
    with np.load(smpl_path, allow_pickle=False) as smpl:
        smpl_format = str(np.asarray(smpl["source_format"]).item())
        smpl_fps = int(np.asarray(smpl["fps"]).item())
        smpl_frames = int(np.asarray(smpl["num_frames"]).item())
    if robot_format != _SONIC_ROBOT_NPZ_FORMAT:
        raise ValueError(f"Unsupported SONIC robot NPZ format in {robot_path}")
    if smpl_format != _SONIC_SMPL_NPZ_FORMAT:
        raise ValueError(f"Unsupported SONIC SMPL NPZ format in {smpl_path}")
    if robot_fps != _TARGET_FPS or smpl_fps != robot_fps:
        raise ValueError(f"SONIC NPZ pair {robot_path.stem!r} must be {_TARGET_FPS} Hz")
    if robot_frames != smpl_frames or robot_frames < _DEFAULT_REFERENCE_FRAMES:
        raise ValueError(f"SONIC NPZ pair {robot_path.stem!r} has incompatible frame metadata")
    return robot_frames, joint_names, body_names


def _array_shapes(num_frames: int, num_joints: int, num_bodies: int) -> dict[str, tuple[int, ...]]:
    return {
        "joint_pos": (num_frames, num_joints),
        "joint_vel": (num_frames, num_joints),
        "body_pos_w": (num_frames, num_bodies, 3),
        "body_quat_w": (num_frames, num_bodies, 4),
        "body_lin_vel_w": (num_frames, num_bodies, 3),
        "body_ang_vel_w": (num_frames, num_bodies, 3),
        "smpl_joints": (num_frames, 24, 3),
        "smpl_root_quat": (num_frames, 4),
    }


def _continuity_segments(
    robot_path: Path,
    smpl_path: Path,
    *,
    count: int,
    joint_names: tuple[str, ...],
    body_names: tuple[str, ...],
) -> tuple[list[tuple[int, int]], int, int]:
    """Return continuous half-open frame ranges for the SONIC tokenizer input."""

    try:
        pelvis_index = body_names.index("pelvis")
        wrist_indices = np.asarray(
            [joint_names.index(G1_POLICY_JOINT_NAMES[index]) for index in _WRIST_POLICY_INDICES],
            dtype=np.intp,
        )
    except ValueError as error:
        raise ValueError(
            f"SONIC NPZ pair {robot_path.stem!r} lacks continuity-check metadata"
        ) from error

    with np.load(robot_path, allow_pickle=False) as robot:
        robot_root_quat = np.asarray(robot["body_quat_w"][:, pelvis_index], dtype=np.float32)
        wrist_joint_pos = np.asarray(robot["joint_pos"][:, wrist_indices], dtype=np.float32)
    with np.load(smpl_path, allow_pickle=False) as smpl:
        smpl_joints = np.asarray(smpl["smpl_joints"], dtype=np.float32)
        smpl_root_quat = np.asarray(smpl["smpl_root_quat"], dtype=np.float32)

    values = (robot_root_quat, wrist_joint_pos, smpl_joints, smpl_root_quat)
    if any(len(value) != count for value in values) or not all(
        np.isfinite(value).all() for value in values
    ):
        raise ValueError(f"SONIC NPZ pair {robot_path.stem!r} has invalid continuity inputs")

    smpl_local = np_quat_apply_batched(
        np_quat_conjugate_batched(smpl_root_quat[:, None]),
        smpl_joints,
    )
    relative_root = np_quat_mul_batched(
        np_quat_conjugate_batched(robot_root_quat),
        smpl_root_quat,
    )
    local_joint_step = np.max(
        np.linalg.norm(np.diff(smpl_local, axis=0), axis=-1),
        axis=-1,
    )
    relative_root_dot = np.abs(np.sum(relative_root[1:] * relative_root[:-1], axis=-1))
    relative_root_step_deg = np.degrees(2.0 * np.arccos(np.clip(relative_root_dot, 0.0, 1.0)))
    wrist_joint_step = np.max(np.abs(np.diff(wrist_joint_pos, axis=0)), axis=-1)
    discontinuous = (
        (local_joint_step > _MAX_SMPL_LOCAL_JOINT_STEP_M)
        | (relative_root_step_deg > _MAX_SMPL_RELATIVE_ROOT_STEP_DEG)
        | (wrist_joint_step > _MAX_WRIST_JOINT_STEP_RAD)
    )

    boundaries = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(discontinuous).astype(np.int64) + 1,
            np.asarray([count], dtype=np.int64),
        )
    )
    segments: list[tuple[int, int]] = []
    dropped_frames = 0
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        start_i = int(start)
        end_i = int(end)
        if end_i - start_i < _DEFAULT_REFERENCE_FRAMES:
            dropped_frames += end_i - start_i
        else:
            segments.append((start_i, end_i))
    if not segments:
        raise ValueError(
            f"SONIC NPZ pair {robot_path.stem!r} has no continuous segment of {_DEFAULT_REFERENCE_FRAMES} frames"
        )
    return segments, int(np.count_nonzero(discontinuous)), dropped_frames


def pack_sonic_dataset(
    robot_input: str | Path,
    smpl_input: str | Path,
    output: str | Path,
    *,
    progress_interval: int = 1000,
    split_discontinuities: bool = False,
) -> dict[str, Any]:
    """Build an atomic, versioned mmap store from paired SONIC NPZ clips."""

    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"SONIC packed output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pairs = resolve_sonic_pairs(
        str(robot_input),
        str(smpl_input),
        robot_suffix=".npz",
        smpl_suffix=".npz",
    )

    clip_names: list[str] = []
    clip_lengths: list[int] = []
    indexed_pairs: list[tuple[Path, Path, list[tuple[int, int, str]]]] = []
    joint_names: tuple[str, ...] | None = None
    body_names: tuple[str, ...] | None = None
    discontinuity_count = 0
    dropped_frames = 0
    print(f"Indexing {len(pairs):,} SONIC NPZ pairs...", flush=True)
    for index, (robot_path, smpl_path) in enumerate(pairs, start=1):
        count, current_joint_names, current_body_names = _pair_metadata(robot_path, smpl_path)
        if joint_names is None:
            joint_names = current_joint_names
            body_names = current_body_names
        elif current_joint_names != joint_names or current_body_names != body_names:
            raise ValueError(f"SONIC NPZ pair {robot_path.stem!r} has inconsistent metadata")
        if split_discontinuities:
            segments, pair_discontinuities, pair_dropped = _continuity_segments(
                robot_path,
                smpl_path,
                count=count,
                joint_names=current_joint_names,
                body_names=current_body_names,
            )
        else:
            segments = [(0, count)]
            pair_discontinuities = 0
            pair_dropped = 0
        split_pair = len(segments) != 1 or segments[0] != (0, count)
        named_segments: list[tuple[int, int, str]] = []
        for segment_index, (start, end) in enumerate(segments):
            segment_name = (
                f"{robot_path.stem}__segment_{segment_index:04d}" if split_pair else robot_path.stem
            )
            named_segments.append((start, end, segment_name))
            clip_names.append(segment_name)
            clip_lengths.append(end - start)
        indexed_pairs.append((robot_path, smpl_path, named_segments))
        discontinuity_count += pair_discontinuities
        dropped_frames += pair_dropped
        if index % progress_interval == 0 or index == len(pairs):
            print(f"Indexed {index:,}/{len(pairs):,} clips", flush=True)

    assert joint_names is not None and body_names is not None
    total_frames = int(sum(clip_lengths))
    if total_frames > np.iinfo(np.int32).max:
        raise ValueError("SONIC packed store exceeds the int32 global-frame contract")
    shapes = _array_shapes(total_frames, len(joint_names), len(body_names))
    required_bytes = sum(int(np.prod(shape, dtype=np.int64)) * 4 for shape in shapes.values())
    required_bytes += len(clip_lengths) * np.dtype(np.int32).itemsize
    free_bytes = shutil.disk_usage(output.parent).free
    headroom = max(1 << 30, required_bytes // 20)
    if free_bytes < required_bytes + headroom:
        raise OSError(
            f"SONIC packed store needs {required_bytes + headroom:,} free bytes including "
            f"headroom, but only {free_bytes:,} are available"
        )

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=str(output.parent)))
    arrays: dict[str, np.memmap] = {}
    try:
        for name, shape in shapes.items():
            arrays[name] = np.lib.format.open_memmap(
                temporary / f"{name}.npy",
                mode="w+",
                dtype=np.float32,
                shape=shape,
            )

        print(
            f"Packing {total_frames:,} frames into {required_bytes / 2**30:.2f} GiB...",
            flush=True,
        )
        offset = 0
        started = time.monotonic()
        robot_names = SONIC_PACKED_ARRAY_NAMES[:6]
        smpl_names = SONIC_PACKED_ARRAY_NAMES[6:]
        packed_clips = 0
        for source_index, (robot_path, smpl_path, segments) in enumerate(indexed_pairs, start=1):
            with np.load(robot_path, allow_pickle=False) as robot:
                robot_values = {
                    name: np.asarray(robot[name], dtype=np.float32) for name in robot_names
                }
            with np.load(smpl_path, allow_pickle=False) as smpl:
                smpl_values = {
                    name: np.asarray(smpl[name], dtype=np.float32) for name in smpl_names
                }
            source_count = len(robot_values["joint_pos"])
            for name, value in {**robot_values, **smpl_values}.items():
                if value.shape != (source_count, *shapes[name][1:]) or not np.isfinite(value).all():
                    raise ValueError(
                        f"SONIC NPZ array {name!r} in {robot_path.stem!r} violates "
                        f"its shape/finite-value contract: {value.shape}"
                    )
            for start, source_end, _segment_name in segments:
                count = source_end - start
                end = offset + count
                for name, value in robot_values.items():
                    arrays[name][offset:end] = value[start:source_end]
                for name, value in smpl_values.items():
                    arrays[name][offset:end] = value[start:source_end]
                offset = end
                packed_clips += 1
            if source_index % progress_interval == 0 or source_index == len(indexed_pairs):
                elapsed = max(time.monotonic() - started, 1.0e-9)
                rate = source_index / elapsed
                eta = (len(indexed_pairs) - source_index) / max(rate, 1.0e-9)
                print(
                    f"Packed {source_index:,}/{len(indexed_pairs):,} source clips "
                    f"into {packed_clips:,} continuous clips "
                    f"({rate:.1f} clips/s, ETA {eta / 60.0:.1f} min)",
                    flush=True,
                )

        for array in arrays.values():
            array.flush()
        arrays.clear()
        np.save(temporary / "clip_lengths.npy", np.asarray(clip_lengths, dtype=np.int32))
        (temporary / "clip_names.txt").write_text(
            "".join(f"{name}\n" for name in clip_names), encoding="utf-8"
        )
        manifest = {
            "format": SONIC_PACKED_FORMAT,
            "fps": _TARGET_FPS,
            "num_clips": len(clip_lengths),
            "num_frames": total_frames,
            "continuity": {
                "split_discontinuities": split_discontinuities,
                "source_num_clips": len(pairs),
                "discontinuity_count": discontinuity_count,
                "dropped_frames": dropped_frames,
                "max_smpl_local_joint_step_m": _MAX_SMPL_LOCAL_JOINT_STEP_M,
                "max_smpl_relative_root_step_deg": _MAX_SMPL_RELATIVE_ROOT_STEP_DEG,
                "max_wrist_joint_step_rad": _MAX_WRIST_JOINT_STEP_RAD,
            },
            "joint_names": list(joint_names),
            "body_names": list(body_names),
            "clip_lengths_file": "clip_lengths.npy",
            "clip_names_file": "clip_names.txt",
            "arrays": {
                name: {"file": f"{name}.npy", "dtype": "float32", "shape": list(shape)}
                for name, shape in shapes.items()
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(output)
    except BaseException:
        arrays.clear()
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "output": str(output),
        "format": SONIC_PACKED_FORMAT,
        "num_clips": len(clip_lengths),
        "num_frames": total_frames,
        "size_bytes": required_bytes,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-input", type=Path, required=True)
    parser.add_argument("--smpl-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument(
        "--split-discontinuities",
        action="store_true",
        help="split discontinuities as a diagnostic mode; SONIC-compatible packing keeps source clips",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        summary = pack_sonic_dataset(
            args.robot_input,
            args.smpl_input,
            args.output,
            progress_interval=args.progress_interval,
            split_discontinuities=args.split_discontinuities,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
