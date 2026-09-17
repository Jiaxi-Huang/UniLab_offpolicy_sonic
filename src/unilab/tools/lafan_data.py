#!/usr/bin/env python3
"""Download, convert, and pack paired LAFAN1 G1/SMPL training data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from unisim.backend.mujoco.reference_motion import materialize_reference_kinematics

from unilab.assets.lafan import download_lafan_training_data
from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.tasks.motion_tracking.g1.sonic_data import (
    _SONIC_ROBOT_NPZ_FORMAT,
    _SONIC_SMPL_NPZ_FORMAT,
    _TARGET_FPS,
    SONIC_PACKED_FORMAT,
    _slerp,
    resolve_sonic_pairs,
)

_HISTORY = 10

from hydra import compose, initialize_config_dir

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.envs import ManagerBasedRlEnvCfg
from unilab.tasks.motion_tracking.g1.sonic_manager import G1_SONIC_BODY_NAMES, SonicMotionCommandCfg
from unilab.tools.pack_sonic_data import pack_sonic_dataset
from unilab.utils.rotation import (
    np_quat_angular_velocity,
    np_quat_mul,
)

G1_CSV_JOINT_ORDER = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Adapted from jaraujo98/lafan_to_smplx revision
# d465aa7202ddc94f2cd557b682854437573e0dca (MIT, Copyright 2025 Joao Pedro Araujo).
_SMPL_TO_LAFAN = {
    1: "LeftUpLeg",
    2: "RightUpLeg",
    3: "Spine",
    4: "LeftLeg",
    5: "RightLeg",
    6: "Spine1",
    7: "LeftFoot",
    8: "RightFoot",
    9: "Spine2",
    10: "LeftToe",
    11: "RightToe",
    12: "Neck",
    13: "LeftShoulder",
    14: "RightShoulder",
    15: "Head",
    16: "LeftArm",
    17: "RightArm",
    18: "LeftForeArm",
    19: "RightForeArm",
    20: "LeftHand",
    21: "RightHand",
}
_SMPL_BETAS = np.asarray(
    [0.9597, 1.0887, -2.1717, -0.8611, 1.3940, 0.1401, -0.2469, 0.3182, -0.2482, 0.3085],
    dtype=np.float32,
)


@dataclass(frozen=True)
class _BvhMotion:
    names: tuple[str, ...]
    parents: np.ndarray
    local_quat_wxyz: np.ndarray
    offsets: np.ndarray
    fps: float


def _rotation_dependencies():
    """Load optional conversion dependencies in one injectable cold-path call."""
    try:
        import smplx
        import torch
        from scipy.spatial.transform import Rotation
    except ImportError as error:
        raise RuntimeError(
            "LAFAN conversion requires `uv sync --extra mujoco --extra lafan`"
        ) from error
    return smplx, torch, Rotation


def _rotation_dependency():
    try:
        from scipy.spatial.transform import Rotation
    except ImportError as error:
        raise RuntimeError(
            "LAFAN conversion requires `uv sync --extra mujoco --extra lafan`"
        ) from error
    return Rotation


def _smpl_dependencies():
    _, torch, _ = _rotation_dependencies()
    return torch


def _load_bvh(path: Path) -> _BvhMotion:
    Rotation = _rotation_dependency()
    lines = path.read_text(encoding="utf-8").splitlines()
    names: list[str] = []
    parents: list[int] = []
    offsets: list[np.ndarray] = []
    offset_seen: list[bool] = []
    active = -1
    end_site = False
    rotation_orders: list[tuple[str, ...]] = []
    frames_line = frame_time_line = -1
    num_frames = 0
    frame_time = 0.0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("ROOT ", "JOINT ")):
            names.append(stripped.split()[1])
            parents.append(active)
            active = len(names) - 1
            offsets.append(np.zeros(3, dtype=np.float32))
            offset_seen.append(False)
        elif stripped == "End Site":
            end_site = True
        elif stripped.startswith("OFFSET ") and active >= 0 and not end_site:
            parts = stripped.split()
            if len(parts) != 4:
                raise ValueError(f"Malformed BVH offset in {path}: {stripped!r}")
            offsets[active] = np.asarray(parts[1:], dtype=np.float32)
            offset_seen[active] = True
        elif stripped == "}":
            if end_site:
                end_site = False
            elif active >= 0:
                active = parents[active]
        elif stripped.startswith("CHANNELS "):
            parts = stripped.split()
            count = int(parts[1])
            channels = tuple(parts[2 : 2 + count])
            rotations = tuple(channel for channel in channels if channel.endswith("rotation"))
            if rotations != ("Zrotation", "Yrotation", "Xrotation"):
                raise ValueError(f"Unsupported BVH rotation channels in {path}: {rotations}")
            rotation_orders.append(rotations)
        elif stripped.startswith("Frames:"):
            frames_line = index
            num_frames = int(stripped.split(":", 1)[1])
        elif stripped.startswith("Frame Time:"):
            frame_time_line = index
            frame_time = float(stripped.split(":", 1)[1])
            break
    if (
        len(rotation_orders) != len(names)
        or not all(offset_seen)
        or frames_line < 0
        or frame_time_line < 0
    ):
        raise ValueError(f"Malformed LAFAN BVH hierarchy or MOTION header in {path}")
    values = np.loadtxt(lines[frame_time_line + 1 :], dtype=np.float64, ndmin=2)
    expected_columns = 6 + 3 * (len(names) - 1)
    if values.shape != (num_frames, expected_columns) or not np.isfinite(values).all():
        raise ValueError(
            f"Invalid BVH motion shape {values.shape} in {path}; "
            f"expected {(num_frames, expected_columns)}"
        )
    euler_zyx = values[:, 3:].reshape(num_frames, len(names), 3)
    local_quat = (
        Rotation.from_euler("ZYX", euler_zyx.reshape(-1, 3), degrees=True)
        .as_quat(scalar_first=True)
        .reshape(num_frames, len(names), 4)
    )
    for frame in range(1, num_frames):
        flip = np.sum(local_quat[frame - 1] * local_quat[frame], axis=-1) < 0.0
        local_quat[frame, flip] *= -1.0
    offsets[0] = np.zeros(3, dtype=np.float32)
    return _BvhMotion(
        names=tuple(names),
        parents=np.asarray(parents, dtype=np.int32),
        offsets=np.asarray(offsets, dtype=np.float32),
        local_quat_wxyz=local_quat.astype(np.float32),
        fps=1.0 / frame_time,
    )


def _resample_quaternions(
    quaternions: np.ndarray, source_times: np.ndarray, target_times: np.ndarray
) -> np.ndarray:
    return np.stack(
        [
            _slerp(quaternions[:, joint], source_times, target_times)
            for joint in range(quaternions.shape[1])
        ],
        axis=1,
    ).astype(np.float32)


def _global_quaternions(local: np.ndarray, parents: np.ndarray) -> np.ndarray:
    result = np.empty_like(local)
    result[:, 0] = local[:, 0]
    for joint in range(1, local.shape[1]):
        result[:, joint] = np_quat_mul(result[:, parents[joint]], local[:, joint])
    return result


def _frame_offsets(Rotation) -> dict[str, Any]:
    euler = Rotation.from_euler
    return {
        "Hips": euler("z", -np.pi / 2) * euler("y", -np.pi / 2),
        **{
            name: euler("z", np.pi / 2) * euler("y", np.pi / 2)
            for name in ("LeftUpLeg", "LeftLeg", "RightUpLeg", "RightLeg")
        },
        **{
            name: euler("z", 0.37117860986509) * euler("y", np.pi / 2)
            for name in ("LeftFoot", "RightFoot")
        },
        **{name: euler("y", np.pi / 2) for name in ("LeftToe", "RightToe")},
        **{
            name: euler("z", -np.pi / 2) * euler("y", -np.pi / 2)
            for name in ("Spine", "Spine1", "Spine2", "Neck", "Head")
        },
        **{
            name: euler("x", np.pi / 2)
            for name in ("LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand")
        },
        **{
            name: euler("z", np.pi) * euler("x", -np.pi / 2)
            for name in ("RightShoulder", "RightArm", "RightForeArm", "RightHand")
        },
    }


def _smpl_pose(motion: _BvhMotion, target_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Rotation = _rotation_dependency()
    source_times = np.arange(motion.local_quat_wxyz.shape[0], dtype=np.float64) / motion.fps
    local = _resample_quaternions(motion.local_quat_wxyz, source_times, target_times)
    global_quat = _global_quaternions(local, motion.parents)
    indices = {name: index for index, name in enumerate(motion.names)}
    missing = sorted((set(_SMPL_TO_LAFAN.values()) | {"Hips"}) - set(indices))
    if missing:
        raise ValueError(f"LAFAN BVH is missing required joints: {missing}")
    offsets = _frame_offsets(Rotation)
    global_rot = {
        name: Rotation.from_quat(global_quat[:, index], scalar_first=True) * offsets[name]
        for name, index in indices.items()
        if name in offsets
    }
    root = global_rot["Hips"]
    body_pose = np.zeros((len(target_times), 21, 3), dtype=np.float32)
    for smpl_index, name in _SMPL_TO_LAFAN.items():
        parent_name = motion.names[motion.parents[indices[name]]]
        body_pose[:, smpl_index - 1] = (
            global_rot[parent_name].inv() * global_rot[name]
        ).as_rotvec()
    return root.as_rotvec().astype(np.float32), body_pose


def _apply_lafan_smpl_frame_correction(
    joints: np.ndarray,
    root_quat: np.ndarray,
    Rotation,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert paired LAFAN SMPL features into the SONIC z-up frame.

    LAFAN BVH/SMPL root poses use a heading frame rotated 90 degrees from the
    paired G1 retarget frame. The correction is a world-frame transform and
    therefore applies to both the materialized joints and root orientation.
    Applying it to only one would corrupt ``root^-1 * joints``, the exact SMPL
    feature consumed by the released SONIC policy.
    """

    correction = Rotation.from_euler("z", np.pi / 2)
    root = Rotation.from_quat(root_quat, scalar_first=True)
    corrected_joints = correction.apply(joints.reshape(-1, 3)).reshape(joints.shape)
    corrected_root = (correction * root).as_quat(scalar_first=True)
    return (
        corrected_joints.astype(np.float32),
        corrected_root.astype(np.float32),
    )


def _materialize_human_joints(
    root_pose: np.ndarray,
    body_pose: np.ndarray,
    info_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-kinematics conversion using gear_sonic human_joints_info.pkl.

    SONIC stores SMPL joints in the local SMPL-root frame.  The paired G1 root
    translation belongs to the robot reference stream and must not be baked
    into ``smpl_joints``; playback rotates these local joints into the robot
    root frame using ``smpl_root_quat``.
    """
    root_pose = np.asarray(root_pose, dtype=np.float32)
    body_pose = np.asarray(body_pose, dtype=np.float32)
    if root_pose.ndim != 2 or root_pose.shape[1] != 3:
        raise ValueError(f"root_pose must have shape (N, 3), got {root_pose.shape}")
    if body_pose.ndim != 3 or body_pose.shape[0] != len(root_pose) or body_pose.shape[2] != 3:
        raise ValueError(
            "body_pose must have shape (N, joints, 3), "
            f"got {body_pose.shape} for N={len(root_pose)}"
        )
    Rotation = _rotation_dependency()
    torch = _smpl_dependencies()
    info = torch.load(info_path, weights_only=False, map_location="cpu")
    rest = np.asarray(info["J"], dtype=np.float32)
    parents = np.asarray(info["parents_list"], dtype=np.intp)
    if rest.shape != (55, 3) or parents.shape != (55,):
        raise ValueError(f"Invalid human joints metadata in {info_path}")
    full_pose = np.concatenate(
        [
            root_pose,
            body_pose.reshape(len(body_pose), -1),
            np.zeros((len(root_pose), 99), dtype=np.float32),
        ],
        axis=-1,
    ).reshape(-1, 55, 3)
    # Older SciPy releases only accept a 2-D ``(N, 3)`` rotvec array.
    local_rot = (
        Rotation.from_rotvec(full_pose.reshape(-1, 3)).as_matrix().reshape(len(root_pose), 55, 3, 3)
    )
    rel = np.broadcast_to(rest, (len(root_pose), 55, 3)).copy()
    rel[:, 1:] -= rest[parents[1:]]
    global_rot = np.empty_like(local_rot)
    joints = np.empty((len(root_pose), 55, 3), dtype=np.float32)
    global_rot[:, 0] = local_rot[:, 0]
    joints[:, 0] = rest[0]
    for i in range(1, 55):
        p = parents[i]
        global_rot[:, i] = global_rot[:, p] @ local_rot[:, i]
        joints[:, i] = joints[:, p] + np.einsum("nij,nj->ni", global_rot[:, p], rel[:, i])
    joints_y_up = joints[:, np.concatenate([np.arange(22), [39, 54]])]
    y_up_to_z_up = Rotation.from_euler("x", np.pi / 2)
    joints_z_up = y_up_to_z_up.apply(joints_y_up.reshape(-1, 3)).reshape(joints_y_up.shape)
    root = Rotation.from_rotvec(root_pose)
    fixed = Rotation.from_quat([0.5, -0.5, -0.5, -0.5], scalar_first=True)
    root_z_up = (y_up_to_z_up * root * fixed).as_quat(scalar_first=True)
    corrected_joints, corrected_root = _apply_lafan_smpl_frame_correction(
        joints_z_up, root_z_up, Rotation
    )
    # Remove the neutral model root offset.  The resulting joints are local to
    # the SMPL root, matching the released SONIC human-reference contract.
    corrected_joints -= corrected_joints[:, :1, :]
    return corrected_joints, corrected_root


_SMPL_DOWNLOAD_HELP = """\
The licensed neutral SMPL model is required for LAFAN conversion.
Download it from the official SMPL website after accepting its license:
  https://smpl.is.tue.mpg.de/
Then either pass the model file/directory with --smpl-model-root or set
SMPL_MODELS_ROOT. Supported layouts are:
  /path/to/body_models/smpl/SMPL_NEUTRAL.pkl
  /path/to/body_models/smpl/SMPL_NEUTRAL.npz
  /path/to/SMPL_NEUTRAL.pkl
  /path/to/SMPL_NEUTRAL.npz
The model is license-restricted and is not downloaded automatically."""


def _resolve_smpl_model_path(value: str | Path | None) -> Path:
    candidate = value or os.environ.get("SMPL_MODELS_ROOT")
    if not candidate:
        raise FileNotFoundError(_SMPL_DOWNLOAD_HELP)
    path = Path(candidate).expanduser().resolve()
    if path.is_file():
        if path.name not in {"SMPL_NEUTRAL.pkl", "SMPL_NEUTRAL.npz"}:
            raise FileNotFoundError(
                f"Unsupported neutral SMPL model filename: {path}\n{_SMPL_DOWNLOAD_HELP}"
            )
        return path
    for relative in (
        Path("smpl/SMPL_NEUTRAL.pkl"),
        Path("smpl/SMPL_NEUTRAL.npz"),
        Path("SMPL_NEUTRAL.pkl"),
        Path("SMPL_NEUTRAL.npz"),
    ):
        model_path = path / relative
        if model_path.is_file():
            return model_path
    raise FileNotFoundError(f"Neutral SMPL model not found under: {path}\n{_SMPL_DOWNLOAD_HELP}")


def _load_smpl_model(smplx, model_path: Path):
    if model_path.suffix == ".npz":
        with np.load(model_path, allow_pickle=False) as archive:
            data_struct = smplx.utils.Struct(**{name: archive[name] for name in archive.files})
        return smplx.create(
            str(model_path),
            model_type="smpl",
            gender="neutral",
            use_pca=False,
            data_struct=data_struct,
        )
    return smplx.create(str(model_path), model_type="smpl", gender="neutral", use_pca=False)


def _load_g1(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if values.shape[1] != 36 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid headerless LAFAN G1 CSV shape/content in {path}: {values.shape}")
    quat = values[:, 3:7][:, (3, 0, 1, 2)]
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-8):
        raise ValueError(f"Zero-length root quaternion in {path}")
    return values[:, :3], quat / norm, values[:, 7:]


def _write_robot_npz(
    robot_path: Path,
    kinematics,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    joint_names: tuple[str, ...],
    body_names: tuple[str, ...],
) -> None:
    robot_tmp = robot_path.with_suffix(".npz.tmp")
    try:
        with robot_tmp.open("wb") as stream:
            np.savez_compressed(
                stream,
                source_format=np.asarray(_SONIC_ROBOT_NPZ_FORMAT),
                fps=np.int32(_TARGET_FPS),
                num_frames=np.int32(len(joint_pos)),
                joint_names=np.asarray(joint_names),
                body_names=np.asarray(body_names),
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                body_pos_w=kinematics.body_pos_w,
                body_quat_w=kinematics.body_quat_w,
                body_lin_vel_w=kinematics.body_lin_vel_w,
                body_ang_vel_w=kinematics.body_ang_vel_w,
            )
        robot_tmp.replace(robot_path)
    finally:
        robot_tmp.unlink(missing_ok=True)


def _write_pair(
    robot_path: Path,
    smpl_path: Path,
    kinematics,
    smpl_joints: np.ndarray,
    smpl_root_quat: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    joint_names: tuple[str, ...],
    body_names: tuple[str, ...],
) -> None:
    smpl_root = np.asarray(smpl_joints[:, 0], dtype=np.float32)
    if smpl_root.shape != (len(joint_pos), 3) or not np.allclose(
        smpl_root, 0.0, atol=2.0e-5, rtol=0.0
    ):
        max_error = float(np.max(np.abs(smpl_root))) if smpl_root.size else float("inf")
        raise ValueError(
            f"LAFAN conversion produced non-local SMPL roots: max position error={max_error:.6g} m"
        )
    smpl_tmp = smpl_path.with_suffix(".npz.tmp")
    try:
        with smpl_tmp.open("wb") as stream:
            np.savez_compressed(
                stream,
                source_format=np.asarray(_SONIC_SMPL_NPZ_FORMAT),
                fps=np.int32(_TARGET_FPS),
                num_frames=np.int32(len(smpl_joints)),
                smpl_joints=smpl_joints,
                smpl_root_quat=smpl_root_quat,
            )
        _write_robot_npz(robot_path, kinematics, joint_pos, joint_vel, joint_names, body_names)
        smpl_tmp.replace(smpl_path)
    finally:
        smpl_tmp.unlink(missing_ok=True)


def convert_lafan_dataset(
    source: str | Path,
    output: str | Path,
    *,
    human_joints_info: str | Path = Path("data/lafan1_source/human_joints_info.pkl"),
    max_clips: int | None = None,
    start_frame: int = 0,
    max_frames: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if max_clips is not None and max_clips <= 0:
        raise ValueError("max_clips must be positive")
    if start_frame < 0 or (max_frames is not None and max_frames < _HISTORY):
        raise ValueError(f"start_frame must be non-negative and max_frames at least {_HISTORY}")
    source = Path(source).expanduser().resolve()
    pairs = resolve_sonic_pairs(
        str(source / "g1"), str(source / "bvh"), robot_suffix=".csv", smpl_suffix=".bvh"
    )
    if max_clips is not None:
        pairs = pairs[:max_clips]
    output = Path(output).expanduser().resolve()
    robot_output = output / "robot_filtered"
    human_output = output / "smpl_filtered"
    robot_output.mkdir(parents=True, exist_ok=True)
    human_output.mkdir(parents=True, exist_ok=True)

    info_path = Path(human_joints_info).expanduser().resolve()
    if not info_path.is_file():
        raise FileNotFoundError(f"human_joints_info.pkl not found: {info_path}")
    config_dir = Path(__file__).parents[1] / "conf" / "flashsac"
    registry.ensure_registries()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        composed = compose(config_name="config_sonic", overrides=["task=g1_sonic/mujoco"])
    cfg = cast(ManagerBasedRlEnvCfg, registry.materialize_env_config("G1SonicManager"))
    root_dir = Path(__file__).parents[3]
    apply_cfg_overrides(
        cfg, BackendAdapter(composed, root_dir=root_dir).build_task_env_cfg_override()
    )
    motion_cfg = cfg.commands["motion"]
    if not isinstance(motion_cfg, SonicMotionCommandCfg) or cfg.scene is None:
        raise TypeError("G1SonicManager configuration is incomplete")
    backend = create_backend(
        "mujoco",
        cfg.scene,
        1,
        cfg.sim_dt,
        base_name=motion_cfg.anchor_body_name,
        add_body_sensors=True,
        **env_backend_kwargs(cfg),
    )
    converted: list[str] = []
    skipped: list[str] = []
    output_frames: dict[str, int] = {}
    for csv_path, bvh_path in pairs:
        robot_path = robot_output / f"{csv_path.stem}.npz"
        smpl_path = human_output / f"{csv_path.stem}.npz"
        exists = robot_path.exists(), smpl_path.exists()
        if all(exists) and not overwrite:
            with (
                np.load(robot_path, allow_pickle=False) as robot,
                np.load(smpl_path, allow_pickle=False) as smpl,
            ):
                if int(robot["num_frames"]) != int(smpl["num_frames"]):
                    raise ValueError(f"Existing converted pair {csv_path.stem!r} is not aligned")
            skipped.append(csv_path.stem)
            continue
        if any(exists) and not all(exists) and not overwrite:
            raise FileExistsError(f"Incomplete converted pair for {csv_path.stem!r}")

        root_pos_30, root_quat_30, joint_pos_30 = _load_g1(csv_path)
        bvh = _load_bvh(bvh_path)
        if abs(bvh.fps - 30.0) > 1.0e-3 or len(bvh.local_quat_wxyz) != len(root_pos_30):
            raise ValueError(f"G1/BVH frame-rate or frame-count mismatch for {csv_path.stem}")
        source_times = np.arange(len(root_pos_30), dtype=np.float64) / 30.0
        target_times = np.arange(0.0, source_times[-1], 1.0 / _TARGET_FPS, dtype=np.float64)
        root_pos = np.stack(
            [np.interp(target_times, source_times, root_pos_30[:, axis]) for axis in range(3)],
            axis=-1,
        ).astype(np.float32)
        root_quat = _slerp(root_quat_30, source_times, target_times).astype(np.float32)
        joint_pos = np.stack(
            [np.interp(target_times, source_times, joint_pos_30[:, axis]) for axis in range(29)],
            axis=-1,
        ).astype(np.float32)
        dt = 1.0 / _TARGET_FPS
        joint_vel = np.gradient(joint_pos, dt, axis=0).astype(np.float32)
        root_lin_vel = np.gradient(root_pos, dt, axis=0).astype(np.float32)
        root_ang_vel = np_quat_angular_velocity(root_quat, dt).astype(np.float32)
        end = (
            len(target_times)
            if max_frames is None
            else min(len(target_times), start_frame + max_frames)
        )
        selection = slice(start_frame, end)
        if end - start_frame < _HISTORY:
            raise ValueError(f"Paired LAFAN clip {csv_path.stem!r} is too short after cropping")

        root_pose, body_pose = _smpl_pose(bvh, target_times[selection])
        kinematics = materialize_reference_kinematics(
            backend,
            root_pos=root_pos[selection],
            root_quat_wxyz=root_quat[selection],
            root_lin_vel=root_lin_vel[selection],
            root_ang_vel=root_ang_vel[selection],
            joint_pos=joint_pos[selection],
            joint_vel=joint_vel[selection],
            body_names=G1_SONIC_BODY_NAMES,
        )
        smpl_joints, smpl_root_quat = _materialize_human_joints(root_pose, body_pose, info_path)
        _write_pair(
            robot_path,
            smpl_path,
            kinematics,
            smpl_joints,
            smpl_root_quat,
            joint_pos[selection],
            joint_vel[selection],
            G1_CSV_JOINT_ORDER,
            G1_SONIC_BODY_NAMES,
        )
        output_frames[csv_path.stem] = len(smpl_joints)
        converted.append(csv_path.stem)

    summary = {
        "format": "unilab_lafan1_conversion_v1",
        "source": str(source),
        "robot_output": str(robot_output),
        "human_output": str(human_output),
        "source_pairs": len(pairs),
        "converted": converted,
        "skipped": skipped,
        "output_frames": output_frames,
        "fps": _TARGET_FPS,
        "start_frame": start_frame,
        "max_frames": max_frames,
    }
    (output / "conversion_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    backend.cleanup_scene_assets()
    return summary


def _packed_summary(path: Path) -> dict[str, Any] | None:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != SONIC_PACKED_FORMAT:
        raise ValueError(f"Existing packed output has unsupported format: {path}")
    return {"output": str(path), "format": SONIC_PACKED_FORMAT, "skipped": True}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pull = subparsers.add_parser("pull")
    pull.add_argument("--output", type=Path, default=Path("data/lafan1_source"))
    pull.add_argument("--lafan-archive", type=Path)
    pull.add_argument("--g1-source", type=Path)

    convert = subparsers.add_parser("convert")
    convert.add_argument("--source", type=Path, default=Path("data/lafan1_source"))
    convert.add_argument("--output", type=Path, default=Path("data/lafan1"))
    convert.add_argument(
        "--human-joints-info", type=Path, default=Path("data/lafan1_source/human_joints_info.pkl")
    )
    convert.add_argument("--max-clips", type=int)
    convert.add_argument("--start-frame", type=int, default=0)
    convert.add_argument("--max-frames", type=int)
    convert.add_argument("--overwrite", action="store_true")

    pack = subparsers.add_parser("pack")
    pack.add_argument("--source", type=Path, default=Path("data/lafan1"))
    pack.add_argument("--output", type=Path, default=Path("data/lafan1/packed"))
    pack.add_argument("--clip", action="append")
    pack.add_argument("--split-discontinuities", action="store_true")

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--raw-output", type=Path, default=Path("data/lafan1_source"))
    prepare.add_argument("--output", type=Path, default=Path("data/lafan1"))
    prepare.add_argument(
        "--human-joints-info", type=Path, default=Path("data/lafan1_source/human_joints_info.pkl")
    )
    prepare.add_argument("--lafan-archive", type=Path)
    prepare.add_argument("--g1-source", type=Path)
    prepare.add_argument("--max-clips", type=int)
    prepare.add_argument("--start-frame", type=int, default=0)
    prepare.add_argument("--max-frames", type=int)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.add_argument("--split-discontinuities", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0].startswith("-"):
        argv = ["prepare", *argv]
    args = _parser().parse_args(argv)
    try:
        if args.command == "pull":
            result: Any = download_lafan_training_data(
                args.output, lafan_archive=args.lafan_archive, g1_source=args.g1_source
            ).as_dict()
        elif args.command == "convert":
            result = convert_lafan_dataset(
                args.source,
                args.output,
                human_joints_info=args.human_joints_info,
                max_clips=args.max_clips,
                start_frame=args.start_frame,
                max_frames=args.max_frames,
                overwrite=args.overwrite,
            )
        elif args.command == "pack":
            result = pack_sonic_dataset(
                args.source / "robot_filtered",
                args.source / "smpl_filtered",
                args.output,
                clip_names=args.clip,
                split_discontinuities=args.split_discontinuities,
            )
        else:
            raw = download_lafan_training_data(
                args.raw_output, lafan_archive=args.lafan_archive, g1_source=args.g1_source
            )
            conversion = convert_lafan_dataset(
                raw.root,
                args.output,
                human_joints_info=args.human_joints_info,
                max_clips=args.max_clips,
                start_frame=args.start_frame,
                max_frames=args.max_frames,
                overwrite=args.overwrite,
            )
            packed_path = args.output / "packed"
            packed = _packed_summary(packed_path) or pack_sonic_dataset(
                args.output / "robot_filtered",
                args.output / "smpl_filtered",
                packed_path,
                split_discontinuities=args.split_discontinuities,
            )
            result = {"source": raw.as_dict(), "conversion": conversion, "packed": packed}
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
