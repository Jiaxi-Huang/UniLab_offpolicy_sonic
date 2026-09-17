"""Tests for the SONIC packed mmap store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from unilab.assets.sonic_packing import pack_sonic_dataset
from unilab.tasks.motion_tracking.g1.sonic_data import (
    _SONIC_ROBOT_NPZ_FORMAT,
    _SONIC_SMPL_NPZ_FORMAT,
    G1_POLICY_JOINT_NAMES,
    SONIC_PACKED_FORMAT,
    SonicNpzMotionLoader,
    SonicPackedMotionLoader,
)

_BODY_NAMES = ("pelvis", "torso_link", "left_wrist_yaw_link")


class _Backend:
    def get_actuator_names(self) -> tuple[str, ...]:
        return G1_POLICY_JOINT_NAMES


def _values(shape: tuple[int, ...], offset: float) -> np.ndarray:
    return np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + offset


def _write_pair(root: Path, name: str, count: int, offset: float) -> None:
    robot_dir = root / "robot"
    smpl_dir = root / "smpl"
    robot_dir.mkdir(parents=True, exist_ok=True)
    smpl_dir.mkdir(parents=True, exist_ok=True)
    num_joints = len(G1_POLICY_JOINT_NAMES)
    num_bodies = len(_BODY_NAMES)
    np.savez_compressed(
        robot_dir / f"{name}.npz",
        source_format=np.asarray(_SONIC_ROBOT_NPZ_FORMAT),
        fps=np.int32(50),
        num_frames=np.int32(count),
        joint_names=np.asarray(G1_POLICY_JOINT_NAMES),
        body_names=np.asarray(_BODY_NAMES),
        joint_pos=_values((count, num_joints), offset),
        joint_vel=_values((count, num_joints), offset + 1.0),
        body_pos_w=_values((count, num_bodies, 3), offset + 2.0),
        body_quat_w=_values((count, num_bodies, 4), offset + 3.0),
        body_lin_vel_w=_values((count, num_bodies, 3), offset + 4.0),
        body_ang_vel_w=_values((count, num_bodies, 3), offset + 5.0),
    )
    np.savez_compressed(
        smpl_dir / f"{name}.npz",
        source_format=np.asarray(_SONIC_SMPL_NPZ_FORMAT),
        fps=np.int32(50),
        num_frames=np.int32(count),
        smpl_joints=_values((count, 24, 3), offset + 6.0),
        smpl_root_quat=_values((count, 4), offset + 7.0),
    )


def _build_store(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    _write_pair(source, "clip_a", 10, 0.0)
    _write_pair(source, "clip_b", 12, 100_000.0)
    output = tmp_path / "packed"
    summary = pack_sonic_dataset(
        source / "robot",
        source / "smpl",
        output,
        progress_interval=1,
        split_discontinuities=False,
    )
    assert summary == {
        "output": str(output),
        "format": SONIC_PACKED_FORMAT,
        "num_clips": 2,
        "num_frames": 22,
        "size_bytes": 22 * (29 * 2 + 3 * (3 + 4 + 3 + 3) + 24 * 3 + 4) * 4 + 8,
    }
    return source, output


def _write_continuity_pair(root: Path, name: str, count: int, jump_at: int) -> None:
    robot_dir = root / "robot"
    smpl_dir = root / "smpl"
    robot_dir.mkdir(parents=True, exist_ok=True)
    smpl_dir.mkdir(parents=True, exist_ok=True)
    body_quat = np.zeros((count, len(_BODY_NAMES), 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    smpl_root_quat = np.zeros((count, 4), dtype=np.float32)
    smpl_root_quat[:, 0] = 1.0
    smpl_joints = np.zeros((count, 24, 3), dtype=np.float32)
    smpl_joints[jump_at:, 22, 0] = 1.0
    np.savez_compressed(
        robot_dir / f"{name}.npz",
        source_format=np.asarray(_SONIC_ROBOT_NPZ_FORMAT),
        fps=np.int32(50),
        num_frames=np.int32(count),
        joint_names=np.asarray(G1_POLICY_JOINT_NAMES),
        body_names=np.asarray(_BODY_NAMES),
        joint_pos=np.zeros((count, len(G1_POLICY_JOINT_NAMES)), dtype=np.float32),
        joint_vel=np.zeros((count, len(G1_POLICY_JOINT_NAMES)), dtype=np.float32),
        body_pos_w=np.zeros((count, len(_BODY_NAMES), 3), dtype=np.float32),
        body_quat_w=body_quat,
        body_lin_vel_w=np.zeros((count, len(_BODY_NAMES), 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((count, len(_BODY_NAMES), 3), dtype=np.float32),
    )
    np.savez_compressed(
        smpl_dir / f"{name}.npz",
        source_format=np.asarray(_SONIC_SMPL_NPZ_FORMAT),
        fps=np.int32(50),
        num_frames=np.int32(count),
        smpl_joints=smpl_joints,
        smpl_root_quat=smpl_root_quat,
    )


def test_packer_splits_smpl_discontinuities_into_safe_clip_boundaries(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_continuity_pair(source, "jump", count=30, jump_at=15)
    output = tmp_path / "packed"

    summary = pack_sonic_dataset(
        source / "robot", source / "smpl", output, split_discontinuities=True
    )

    assert summary["num_clips"] == 2
    assert summary["num_frames"] == 30
    np.testing.assert_array_equal(np.load(output / "clip_lengths.npy"), [15, 15])
    assert (output / "clip_names.txt").read_text(encoding="utf-8").splitlines() == [
        "jump__segment_0000",
        "jump__segment_0001",
    ]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["continuity"] == {
        "discontinuity_count": 1,
        "dropped_frames": 0,
        "max_smpl_local_joint_step_m": 0.5,
        "max_smpl_relative_root_step_deg": 20.0,
        "max_wrist_joint_step_rad": 0.5,
        "source_num_clips": 1,
        "split_discontinuities": True,
    }


def test_packer_default_preserves_source_clip_semantics(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_continuity_pair(source, "jump", count=30, jump_at=15)
    output = tmp_path / "packed"

    summary = pack_sonic_dataset(source / "robot", source / "smpl", output)

    assert summary["num_clips"] == 1
    assert summary["num_frames"] == 30
    np.testing.assert_array_equal(np.load(output / "clip_lengths.npy"), [30])
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["continuity"]["split_discontinuities"] is False
    assert manifest["continuity"]["source_num_clips"] == 1
    assert manifest["continuity"]["dropped_frames"] == 0


def test_packer_drops_continuity_fragments_shorter_than_history(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_continuity_pair(source, "early_jump", count=30, jump_at=5)
    output = tmp_path / "packed"

    summary = pack_sonic_dataset(
        source / "robot", source / "smpl", output, split_discontinuities=True
    )

    assert summary["num_clips"] == 1
    assert summary["num_frames"] == 25
    np.testing.assert_array_equal(np.load(output / "clip_lengths.npy"), [25])
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["continuity"]["dropped_frames"] == 5
    with np.load(source / "smpl" / "early_jump.npz", allow_pickle=False) as smpl:
        np.testing.assert_array_equal(
            np.load(output / "smpl_joints.npy", mmap_mode="r"),
            smpl["smpl_joints"][5:],
        )


def test_packed_loader_matches_npz_loader_without_materializing_arrays(tmp_path: Path) -> None:
    source, output = _build_store(tmp_path)
    backend = _Backend()
    legacy = SonicNpzMotionLoader(
        str(source / "robot"),
        str(source / "smpl"),
        backend=backend,
        body_names=_BODY_NAMES,
    )
    packed = SonicPackedMotionLoader(output, backend=backend, body_names=_BODY_NAMES)

    np.testing.assert_array_equal(packed.clip_lengths, [10, 12])
    np.testing.assert_array_equal(packed.clip_offsets, [0, 10])
    np.testing.assert_array_equal(packed.clip_end_frames, [9, 21])
    frame_ids = np.asarray([0, 9, 10, 21], dtype=np.int32)
    for name in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "smpl_joints",
        "smpl_root_quat",
    ):
        packed_array = getattr(packed, name)
        assert isinstance(packed_array, np.memmap)
        assert not packed_array.flags.writeable
        np.testing.assert_array_equal(packed_array[frame_ids], getattr(legacy, name)[frame_ids])
    np.testing.assert_array_equal(
        packed.future_indices(np.asarray([8, 20], dtype=np.int32), stride=2),
        [[8, 9, 9, 9, 9, 9, 9, 9, 9, 9], [20, 21, 21, 21, 21, 21, 21, 21, 21, 21]],
    )


def test_packed_store_is_versioned_and_refuses_existing_output(tmp_path: Path) -> None:
    source, output = _build_store(tmp_path)
    with pytest.raises(FileExistsError, match="already exists"):
        pack_sonic_dataset(source / "robot", source / "smpl", output)

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format"] = "future-format"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported SONIC packed format"):
        SonicPackedMotionLoader(output, backend=_Backend(), body_names=_BODY_NAMES)


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        (np.zeros((1, 29), dtype=np.float32), "shape/dtype contract"),
        (np.zeros((22, 29), dtype=np.float64), "shape/dtype contract"),
    ],
)
def test_packed_loader_rejects_array_shape_and_dtype_corruption(
    tmp_path: Path, replacement: np.ndarray, match: str
) -> None:
    _, output = _build_store(tmp_path)
    np.save(output / "joint_pos.npy", replacement)
    with pytest.raises(ValueError, match=match):
        SonicPackedMotionLoader(output, backend=_Backend(), body_names=_BODY_NAMES)


def test_packed_loader_rejects_incomplete_store_and_model_metadata(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(FileNotFoundError, match="missing manifest"):
        SonicPackedMotionLoader(incomplete, backend=_Backend(), body_names=_BODY_NAMES)

    _, output = _build_store(tmp_path)
    with pytest.raises(ValueError, match="body order"):
        SonicPackedMotionLoader(output, backend=_Backend(), body_names=tuple(reversed(_BODY_NAMES)))
