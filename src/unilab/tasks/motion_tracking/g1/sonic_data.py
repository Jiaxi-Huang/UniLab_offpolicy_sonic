"""SONIC reference-data formats and cold-path conversion helpers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from unilab.assets.bones_seed_csv import load_header, parse_joint_names
from unilab.utils.rotation import (
    np_quat_angular_velocity,
    np_quat_apply_batched,
    np_quat_conjugate_batched,
    np_quat_from_euler_xyz,
    np_quat_mul,
)

from ..common.motion_loader import MotionLoader

G1_POLICY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)
_SONIC_ACTUATOR_PARAMETERS = {
    **{
        name: (99.098427777, 6.308801854, 139.0, 0.025101925)
        for name in (
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_knee_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_knee_joint",
        )
    },
    **{
        name: (40.179238471, 2.557889765, 88.0, 0.010177520)
        for name in ("left_hip_yaw_joint", "right_hip_yaw_joint", "waist_yaw_joint")
    },
    **{
        name: (28.501246196, 1.814445687, 50.0, 0.00721945)
        for name in (
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        )
    },
    **{
        name: (14.250623098, 0.907222843, 25.0, 0.003609725)
        for name in (
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
        )
    },
    **{
        name: (16.778327481, 1.068141502, 5.0, 0.00425)
        for name in (
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        )
    },
}
_WRIST_POLICY_INDICES = np.arange(23, 29, dtype=np.int32)
_DEFAULT_REFERENCE_FRAMES = 10
_G1_FUTURE_STRIDE = 5
_SMPL_FUTURE_STRIDE = 1
_TARGET_FPS = 50


def _rotation_6d(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(quat_wxyz, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
        ),
        axis=-1,
    )


def _pack_g1_reference(command: np.ndarray, relative_root: np.ndarray) -> np.ndarray:
    """Pack the G1 tokenizer terms in the released SONIC checkpoint order.

    The upstream observation manager flattens each declared term before it
    concatenates terms.  In particular, root orientation is *not* interleaved
    with the ten command frames.  Keep this term-major layout even though the
    actor later reshapes the resulting vector to ``(10, 64)``.
    """
    return np.concatenate(
        (command.reshape(len(command), -1), _rotation_6d(relative_root).reshape(len(command), -1)),
        axis=-1,
    )


def _pack_smpl_reference(
    human_local: np.ndarray, human_relative: np.ndarray, wrist: np.ndarray
) -> np.ndarray:
    """Pack the SMPL tokenizer terms in the released SONIC checkpoint order."""
    human_feature = np.concatenate((human_local, _rotation_6d(human_relative)), axis=-1)
    return np.concatenate(
        (human_feature.reshape(len(human_feature), -1), wrist.reshape(len(wrist), -1)), axis=-1
    )


def _root_local_proprioception(
    root_quat_w: np.ndarray, root_lin_vel_w: np.ndarray, root_ang_vel_w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match SONIC's pelvis-root local velocity and projected-gravity terms."""
    root_quat_inv = np_quat_conjugate_batched(root_quat_w)
    gravity_w = np.zeros_like(root_lin_vel_w)
    gravity_w[:, 2] = -1.0
    return (
        np_quat_apply_batched(root_quat_inv, root_lin_vel_w),
        np_quat_apply_batched(root_quat_inv, root_ang_vel_w),
        np_quat_apply_batched(root_quat_inv, gravity_w),
    )


def _axis_angle_to_quat_wxyz(axis_angle: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.where(angle > 1.0e-8, np.sin(half) / angle, 0.5 - angle * angle / 48.0)
    return np.concatenate((np.cos(half), axis_angle * scale), axis=-1)


def _slerp(quat: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    upper = np.searchsorted(source_times, target_times, side="right")
    upper = np.clip(upper, 1, len(source_times) - 1)
    lower = upper - 1
    q0 = quat[lower]
    q1 = quat[upper].copy()
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    fraction = ((target_times - source_times[lower]) / (source_times[upper] - source_times[lower]))[
        :, None
    ]
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    linear = (1.0 - fraction) * q0 + fraction * q1
    spherical = (
        np.sin((1.0 - fraction) * theta) / np.where(sin_theta > 1.0e-7, sin_theta, 1.0) * q0
        + np.sin(fraction * theta) / np.where(sin_theta > 1.0e-7, sin_theta, 1.0) * q1
    )
    result = np.where(sin_theta > 1.0e-7, spherical, linear)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def _load_joblib_dict(path: Path) -> dict:
    payload = joblib.load(path)
    if isinstance(payload, dict) and "smpl_joints" in payload:
        return payload
    if isinstance(payload, dict) and len(payload) == 1:
        nested = next(iter(payload.values()))
        if isinstance(nested, dict) and "smpl_joints" in nested:
            return nested
    raise ValueError(f"Unsupported SONIC SMPL payload in {path}")


def resolve_sonic_pairs(
    robot_source: str | list[str],
    smpl_source: str | list[str],
    *,
    robot_suffix: str = ".csv",
    smpl_suffix: str = ".pkl",
) -> list[tuple[Path, Path]]:
    def index(source: str | list[str], suffix: str) -> dict[str, Path]:
        values = [source] if isinstance(source, str) else source
        paths: list[Path] = []
        for value in values:
            path = Path(value).expanduser().resolve()
            if path.is_dir():
                paths.extend(path.rglob(f"*{suffix}"))
            elif path.suffix.lower() == suffix:
                paths.append(path)
            else:
                raise ValueError(f"Expected a {suffix} file or directory, got {path}")
        result: dict[str, Path] = {}
        for path in sorted(paths):
            if path.stem in result:
                raise ValueError(f"Duplicate SONIC clip stem {path.stem!r}")
            result[path.stem] = path
        return result

    robots = index(robot_source, robot_suffix)
    humans = index(smpl_source, smpl_suffix)
    names = sorted(set(robots) & set(humans))
    if not names:
        raise FileNotFoundError("No paired SONIC CSV/SMPL clips were found")
    return [(robots[name], humans[name]) for name in names]


class _SonicMotionLoader(MotionLoader):
    smpl_joints: np.ndarray
    smpl_root_quat: np.ndarray

    def future_indices(
        self, frames: np.ndarray, stride: int, num_frames: int = _DEFAULT_REFERENCE_FRAMES
    ) -> np.ndarray:
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        clip_ids = self.get_clip_indices(frames)
        offsets = np.arange(num_frames, dtype=np.int32) * stride
        return np.minimum(frames[:, None] + offsets, self.clip_end_frames[clip_ids, None])


class SonicRawMotionLoader(_SonicMotionLoader):
    """In-memory loader for paired original CSV and SMPL joblib files."""

    def __init__(
        self,
        robot_source: str | list[str],
        smpl_source: str | list[str],
        *,
        backend,
        body_names: tuple[str, ...],
        start_frame: int = 0,
        max_frames: int | None = None,
    ) -> None:
        if start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if max_frames is not None and max_frames < _DEFAULT_REFERENCE_FRAMES:
            raise ValueError(f"max_frames must be at least {_DEFAULT_REFERENCE_FRAMES}")
        pairs = resolve_sonic_pairs(robot_source, smpl_source)
        model_joint_names = backend.get_actuator_names()
        if set(model_joint_names) != set(G1_POLICY_JOINT_NAMES):
            raise ValueError("G1 actuator set does not match the SONIC 29-DoF contract")

        joint_pos_chunks = []
        joint_vel_chunks = []
        body_pos_chunks = []
        body_quat_chunks = []
        body_lin_chunks = []
        body_ang_chunks = []
        smpl_joint_chunks = []
        smpl_quat_chunks = []
        clip_lengths = []
        self.motion_files = tuple(str(robot) for robot, _ in pairs)

        for robot_path, smpl_path in pairs:
            header = load_header(robot_path)
            csv_joint_names = tuple(parse_joint_names(header, robot_path))
            if set(csv_joint_names) != set(model_joint_names):
                raise ValueError(f"BONES joint set in {robot_path} does not match the G1 model")
            raw = np.loadtxt(robot_path, delimiter=",", skiprows=1, ndmin=2, dtype=np.float64)
            if raw.shape[1] != len(header) or raw.shape[0] < 2:
                raise ValueError(f"Invalid BONES CSV shape {raw.shape} in {robot_path}")
            root_pos_30 = raw[::4, 1:4] / 100.0
            euler = np.deg2rad(raw[::4, 4:7])
            root_quat_wxyz_30 = np_quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
            joint_csv_30 = np.deg2rad(raw[::4, 7:]).astype(np.float32)
            source_times = np.arange(len(root_pos_30), dtype=np.float64) / 30.0
            target_times = np.arange(0.0, source_times[-1], 1.0 / _TARGET_FPS, dtype=np.float64)
            root_pos = np.stack(
                [np.interp(target_times, source_times, root_pos_30[:, axis]) for axis in range(3)],
                axis=-1,
            ).astype(np.float32)
            root_quat_wxyz = _slerp(root_quat_wxyz_30, source_times, target_times).astype(
                np.float32
            )
            joint_csv = np.stack(
                [
                    np.interp(target_times, source_times, joint_csv_30[:, axis])
                    for axis in range(29)
                ],
                axis=-1,
            ).astype(np.float32)
            csv_indices = [csv_joint_names.index(name) for name in model_joint_names]
            joint_pos = np.ascontiguousarray(joint_csv[:, csv_indices])
            dt = 1.0 / _TARGET_FPS
            joint_vel = np.gradient(joint_pos, dt, axis=0).astype(np.float32)
            root_lin_vel = np.gradient(root_pos, dt, axis=0).astype(np.float32)
            root_ang_vel = np_quat_angular_velocity(root_quat_wxyz, dt).astype(np.float32)

            smpl = _load_joblib_dict(smpl_path)
            if int(round(float(smpl["fps"]))) != _TARGET_FPS:
                raise ValueError(f"Expected 50 Hz SMPL data in {smpl_path}")
            pose = np.asarray(smpl["pose_aa"], dtype=np.float32).reshape(-1, 24, 3)
            smpl_joints = np.asarray(smpl["smpl_joints"], dtype=np.float32)
            if smpl_joints.shape != (len(pose), 24, 3):
                raise ValueError(
                    f"Expected SMPL joints {(len(pose), 24, 3)}, got {smpl_joints.shape}"
                )
            root_pose_quat = _axis_angle_to_quat_wxyz(pose[:, 0])
            y_up_to_z_up = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], dtype=np.float32)
            fixed = np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
            smpl_root_quat = np_quat_mul(
                np_quat_mul(np.broadcast_to(y_up_to_z_up, root_pose_quat.shape), root_pose_quat),
                np.broadcast_to(fixed, root_pose_quat.shape),
            ).astype(np.float32)

            available = min(len(joint_pos), len(smpl_joints))
            end = available if max_frames is None else min(available, start_frame + max_frames)
            selection = slice(start_frame, end)
            count = end - start_frame
            if count < _DEFAULT_REFERENCE_FRAMES:
                raise ValueError(
                    f"Paired SONIC clip {robot_path.stem!r} is too short: {count} frames"
                )
            if backend.backend_type == "mujoco":
                from unisim.backend.mujoco.reference_motion import (
                    materialize_reference_kinematics,
                )

                kinematics = materialize_reference_kinematics(
                    backend,
                    root_pos=root_pos[selection],
                    root_quat_wxyz=root_quat_wxyz[selection],
                    root_lin_vel=root_lin_vel[selection],
                    root_ang_vel=root_ang_vel[selection],
                    joint_pos=joint_pos[selection],
                    joint_vel=joint_vel[selection],
                    body_names=body_names,
                )
            else:
                from unisim.backend.motrix.reference_motion import (
                    materialize_reference_kinematics,
                )

                kinematics = materialize_reference_kinematics(
                    backend,
                    root_pos=root_pos[selection],
                    root_quat_xyzw=root_quat_wxyz[selection][:, (1, 2, 3, 0)],
                    root_lin_vel=root_lin_vel[selection],
                    root_ang_vel=root_ang_vel[selection],
                    joint_pos=joint_pos[selection],
                    joint_vel=joint_vel[selection],
                    body_names=body_names,
                )
            clip_lengths.append(count)
            joint_pos_chunks.append(joint_pos[selection])
            joint_vel_chunks.append(joint_vel[selection])
            body_pos_chunks.append(kinematics.body_pos_w)
            body_quat_chunks.append(kinematics.body_quat_w)
            body_lin_chunks.append(kinematics.body_lin_vel_w)
            body_ang_chunks.append(kinematics.body_ang_vel_w)
            smpl_joint_chunks.append(smpl_joints[selection])
            smpl_quat_chunks.append(smpl_root_quat[selection])

        self.fps = _TARGET_FPS
        self.joint_names = tuple(model_joint_names)
        self.body_names = tuple(body_names)
        self.num_joints = len(model_joint_names)
        self.num_bodies = len(body_names)
        self.clip_lengths = np.asarray(clip_lengths, dtype=np.int32)
        self.num_clips = len(clip_lengths)
        self.clip_offsets = np.zeros(self.num_clips, dtype=np.int32)
        if self.num_clips > 1:
            self.clip_offsets[1:] = np.cumsum(self.clip_lengths[:-1], dtype=np.int32)
        self.clip_end_frames = self.clip_offsets + self.clip_lengths - 1
        self.joint_pos = np.concatenate(joint_pos_chunks)
        self.joint_vel = np.concatenate(joint_vel_chunks)
        self.body_pos_w = np.concatenate(body_pos_chunks)
        self.body_quat_w = np.concatenate(body_quat_chunks)
        self.body_lin_vel_w = np.concatenate(body_lin_chunks)
        self.body_ang_vel_w = np.concatenate(body_ang_chunks)
        self.smpl_joints = np.concatenate(smpl_joint_chunks)
        self.smpl_root_quat = np.concatenate(smpl_quat_chunks)
        self.num_frames = len(self.joint_pos)


_SONIC_ROBOT_NPZ_FORMAT = "unilab_sonic_robot_v1"
_SONIC_SMPL_NPZ_FORMAT = "unilab_sonic_smpl_v1"
SONIC_PACKED_FORMAT = "unilab_sonic_packed_v1"
SONIC_PACKED_ARRAY_NAMES = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "smpl_joints",
    "smpl_root_quat",
)


def write_sonic_npz_pair(
    loader: SonicRawMotionLoader,
    robot_path: str | Path,
    smpl_path: str | Path,
) -> tuple[Path, Path]:
    """Write one converted CSV/PKL pair as aligned SONIC NPZ files."""

    if loader.num_clips != 1:
        raise ValueError("write_sonic_npz_pair requires exactly one source clip")
    robot_path = Path(robot_path)
    smpl_path = Path(smpl_path)
    robot_path.parent.mkdir(parents=True, exist_ok=True)
    smpl_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        robot_path,
        source_format=np.asarray(_SONIC_ROBOT_NPZ_FORMAT),
        fps=np.int32(loader.fps),
        num_frames=np.int32(loader.num_frames),
        joint_names=np.asarray(loader.joint_names),
        body_names=np.asarray(loader.body_names),
        joint_pos=loader.joint_pos,
        joint_vel=loader.joint_vel,
        body_pos_w=loader.body_pos_w,
        body_quat_w=loader.body_quat_w,
        body_lin_vel_w=loader.body_lin_vel_w,
        body_ang_vel_w=loader.body_ang_vel_w,
    )
    np.savez_compressed(
        smpl_path,
        source_format=np.asarray(_SONIC_SMPL_NPZ_FORMAT),
        fps=np.int32(loader.fps),
        num_frames=np.int32(loader.num_frames),
        smpl_joints=loader.smpl_joints,
        smpl_root_quat=loader.smpl_root_quat,
    )
    return robot_path, smpl_path


class SonicNpzMotionLoader(_SonicMotionLoader):
    """Load aligned robot/SMPL NPZ pairs produced by UniLab's converter."""

    def __init__(
        self,
        robot_source: str | list[str],
        smpl_source: str | list[str],
        *,
        backend,
        body_names: tuple[str, ...],
    ) -> None:
        pairs = resolve_sonic_pairs(
            robot_source,
            smpl_source,
            robot_suffix=".npz",
            smpl_suffix=".npz",
        )
        model_joint_names = tuple(backend.get_actuator_names())
        joint_pos_chunks = []
        joint_vel_chunks = []
        body_pos_chunks = []
        body_quat_chunks = []
        body_lin_chunks = []
        body_ang_chunks = []
        smpl_joint_chunks = []
        smpl_quat_chunks = []
        clip_lengths = []
        self.motion_files = tuple(str(robot) for robot, _ in pairs)

        for robot_path, smpl_path in pairs:
            with np.load(robot_path, allow_pickle=False) as robot:
                source_format = str(np.asarray(robot["source_format"]).item())
                if source_format != _SONIC_ROBOT_NPZ_FORMAT:
                    raise ValueError(f"Unsupported SONIC robot NPZ format in {robot_path}")
                fps = int(np.asarray(robot["fps"]).item())
                robot_num_frames = int(np.asarray(robot["num_frames"]).item())
                joint_names = tuple(str(name) for name in robot["joint_names"])
                stored_body_names = tuple(str(name) for name in robot["body_names"])
                robot_arrays = {
                    name: np.asarray(robot[name], dtype=np.float32)
                    for name in (
                        "joint_pos",
                        "joint_vel",
                        "body_pos_w",
                        "body_quat_w",
                        "body_lin_vel_w",
                        "body_ang_vel_w",
                    )
                }
            with np.load(smpl_path, allow_pickle=False) as smpl:
                source_format = str(np.asarray(smpl["source_format"]).item())
                if source_format != _SONIC_SMPL_NPZ_FORMAT:
                    raise ValueError(f"Unsupported SONIC SMPL NPZ format in {smpl_path}")
                smpl_fps = int(np.asarray(smpl["fps"]).item())
                smpl_num_frames = int(np.asarray(smpl["num_frames"]).item())
                smpl_joints = np.asarray(smpl["smpl_joints"], dtype=np.float32)
                smpl_root_quat = np.asarray(smpl["smpl_root_quat"], dtype=np.float32)

            if fps != _TARGET_FPS or smpl_fps != fps:
                raise ValueError(f"SONIC NPZ pair {robot_path.stem!r} must be 50 Hz")
            if joint_names != model_joint_names:
                raise ValueError(f"Joint order in {robot_path} does not match the backend model")
            if stored_body_names != tuple(body_names):
                raise ValueError(f"Body order in {robot_path} does not match the SONIC task")
            count = len(robot_arrays["joint_pos"])
            expected_frames = {
                **{name: len(value) for name, value in robot_arrays.items()},
                "smpl_joints": len(smpl_joints),
                "smpl_root_quat": len(smpl_root_quat),
            }
            expected_shapes = {
                "joint_pos": (count, len(model_joint_names)),
                "joint_vel": (count, len(model_joint_names)),
                "body_pos_w": (count, len(body_names), 3),
                "body_quat_w": (count, len(body_names), 4),
                "body_lin_vel_w": (count, len(body_names), 3),
                "body_ang_vel_w": (count, len(body_names), 3),
                "smpl_joints": (count, 24, 3),
                "smpl_root_quat": (count, 4),
            }
            actual_shapes = {
                **{name: value.shape for name, value in robot_arrays.items()},
                "smpl_joints": smpl_joints.shape,
                "smpl_root_quat": smpl_root_quat.shape,
            }
            if (
                count < _DEFAULT_REFERENCE_FRAMES
                or robot_num_frames != count
                or smpl_num_frames != count
                or any(value != count for value in expected_frames.values())
                or actual_shapes != expected_shapes
            ):
                raise ValueError(
                    f"SONIC NPZ pair {robot_path.stem!r} has an incompatible shape contract: "
                    f"{actual_shapes}"
                )
            clip_lengths.append(count)
            joint_pos_chunks.append(robot_arrays["joint_pos"])
            joint_vel_chunks.append(robot_arrays["joint_vel"])
            body_pos_chunks.append(robot_arrays["body_pos_w"])
            body_quat_chunks.append(robot_arrays["body_quat_w"])
            body_lin_chunks.append(robot_arrays["body_lin_vel_w"])
            body_ang_chunks.append(robot_arrays["body_ang_vel_w"])
            smpl_joint_chunks.append(smpl_joints)
            smpl_quat_chunks.append(smpl_root_quat)

        self.fps = _TARGET_FPS
        self.joint_names = model_joint_names
        self.body_names = tuple(body_names)
        self.num_joints = len(model_joint_names)
        self.num_bodies = len(body_names)
        self.clip_lengths = np.asarray(clip_lengths, dtype=np.int32)
        self.num_clips = len(clip_lengths)
        self.clip_offsets = np.zeros(self.num_clips, dtype=np.int32)
        if self.num_clips > 1:
            self.clip_offsets[1:] = np.cumsum(self.clip_lengths[:-1], dtype=np.int32)
        self.clip_end_frames = self.clip_offsets + self.clip_lengths - 1
        self.joint_pos = np.concatenate(joint_pos_chunks)
        self.joint_vel = np.concatenate(joint_vel_chunks)
        self.body_pos_w = np.concatenate(body_pos_chunks)
        self.body_quat_w = np.concatenate(body_quat_chunks)
        self.body_lin_vel_w = np.concatenate(body_lin_chunks)
        self.body_ang_vel_w = np.concatenate(body_ang_chunks)
        self.smpl_joints = np.concatenate(smpl_joint_chunks)
        self.smpl_root_quat = np.concatenate(smpl_quat_chunks)
        self.num_frames = len(self.joint_pos)


class SonicPackedMotionLoader(_SonicMotionLoader):
    """Read a versioned SONIC store through shared, read-only NumPy mappings."""

    def __init__(self, store: str | Path, *, backend, body_names: tuple[str, ...]) -> None:
        root = Path(store).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"SONIC packed store is incomplete or missing manifest.json: {root}. "
                "Build it with `uv run scripts/motion/pack_sonic_data.py`."
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid SONIC packed manifest: {manifest_path}") from error
        if manifest.get("format") != SONIC_PACKED_FORMAT:
            raise ValueError(
                f"Unsupported SONIC packed format {manifest.get('format')!r} in {manifest_path}"
            )

        def store_member(value: object, label: str) -> Path:
            path = (root / str(value)).resolve()
            if root not in path.parents:
                raise ValueError(f"SONIC packed {label} must stay within {root}")
            return path

        model_joint_names = tuple(backend.get_actuator_names())
        stored_joint_names = tuple(str(name) for name in manifest.get("joint_names", ()))
        stored_body_names = tuple(str(name) for name in manifest.get("body_names", ()))
        if stored_joint_names != model_joint_names:
            raise ValueError("SONIC packed joint order does not match the backend model")
        if stored_body_names != tuple(body_names):
            raise ValueError("SONIC packed body order does not match the SONIC task")
        if int(manifest.get("fps", -1)) != _TARGET_FPS:
            raise ValueError(f"SONIC packed store must be {_TARGET_FPS} Hz")

        clip_lengths_path = store_member(manifest.get("clip_lengths_file", ""), "clip lengths")
        try:
            clip_lengths = np.load(clip_lengths_path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"Invalid SONIC packed clip lengths: {clip_lengths_path}") from error
        if clip_lengths.dtype != np.dtype(np.int32) or clip_lengths.ndim != 1:
            raise ValueError("SONIC packed clip_lengths must be a one-dimensional int32 array")
        self.clip_lengths = np.asarray(clip_lengths, dtype=np.int32)
        self.num_clips = int(manifest.get("num_clips", -1))
        self.num_frames = int(manifest.get("num_frames", -1))
        if (
            self.num_clips <= 0
            or self.num_frames <= 0
            or len(self.clip_lengths) != self.num_clips
            or np.any(self.clip_lengths < _DEFAULT_REFERENCE_FRAMES)
            or int(self.clip_lengths.sum(dtype=np.int64)) != self.num_frames
        ):
            raise ValueError("SONIC packed clip metadata is inconsistent")

        self.fps = _TARGET_FPS
        self.joint_names = model_joint_names
        self.body_names = tuple(body_names)
        self.num_joints = len(model_joint_names)
        self.num_bodies = len(body_names)
        expected_shapes = {
            "joint_pos": (self.num_frames, self.num_joints),
            "joint_vel": (self.num_frames, self.num_joints),
            "body_pos_w": (self.num_frames, self.num_bodies, 3),
            "body_quat_w": (self.num_frames, self.num_bodies, 4),
            "body_lin_vel_w": (self.num_frames, self.num_bodies, 3),
            "body_ang_vel_w": (self.num_frames, self.num_bodies, 3),
            "smpl_joints": (self.num_frames, 24, 3),
            "smpl_root_quat": (self.num_frames, 4),
        }
        array_specs = manifest.get("arrays")
        if not isinstance(array_specs, dict) or set(array_specs) != set(SONIC_PACKED_ARRAY_NAMES):
            raise ValueError("SONIC packed manifest has an incompatible array set")
        for name in SONIC_PACKED_ARRAY_NAMES:
            spec = array_specs[name]
            if not isinstance(spec, dict):
                raise ValueError(f"SONIC packed array spec for {name!r} must be a mapping")
            array_path = store_member(spec.get("file", ""), f"array {name!r}")
            try:
                array = np.load(array_path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as error:
                raise ValueError(f"Invalid SONIC packed array: {array_path}") from error
            if (
                not isinstance(array, np.memmap)
                or array.dtype != np.dtype(np.float32)
                or array.shape != expected_shapes[name]
                or spec.get("dtype") != "float32"
                or tuple(spec.get("shape", ())) != expected_shapes[name]
            ):
                raise ValueError(f"SONIC packed array {name!r} violates its shape/dtype contract")
            setattr(self, name, array)

        self.clip_offsets = np.zeros(self.num_clips, dtype=np.int32)
        if self.num_clips > 1:
            self.clip_offsets[1:] = np.cumsum(self.clip_lengths[:-1], dtype=np.int32)
        self.clip_end_frames = self.clip_offsets + self.clip_lengths - 1
        self.motion_files = (str(root),)
