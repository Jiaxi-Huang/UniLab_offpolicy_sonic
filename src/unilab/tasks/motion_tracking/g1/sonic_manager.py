"""Manager-Based G1 SONIC task I/O and reference-motion contract."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, cast

import gymnasium as gym
import numpy as np

from unilab.base import registry
from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.np_env import NpEnvState
from unilab.dtype_config import get_global_dtype
from unilab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from unilab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from unilab.managers import (
    CommandTerm,
    CommandTermCfg,
    EventTermCfg,
    ManagerTermBase,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)


# Vendored from the old locomotion.common.base module (moved downstream with
# the production locomotion tasks); SONIC only needs the flat noise scales.
@dataclass
class NoiseConfig:
    level: float = 0.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 0.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1
    seed: int | None = None


from unilab.tasks.motion_tracking.common import rewards as motion_rewards
from unilab.tasks.motion_tracking.common.motion_loader import MotionData, MotionSampler
from unilab.utils.rotation import (
    np_quat_apply_batched,
    np_quat_apply_inverse_batched,
    np_quat_conjugate_batched,
    np_quat_error_magnitude_squared_batched,
    np_quat_heading,
    np_quat_mul_batched,
)


def _write_body_pos_in_anchor_frame(
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    body_pos_w: np.ndarray,
    out: np.ndarray,
    *,
    body_vec_error: np.ndarray | None = None,
) -> None:
    """Write body positions in the robot anchor frame using public math helpers."""
    relative = body_pos_w - root_pos[:, None, :]
    out[...] = np_quat_apply_inverse_batched(root_quat[:, None, :], relative)
    if body_vec_error is not None:
        np.subtract(body_pos_w, root_pos[:, None, :], out=body_vec_error)


def _write_body_ori6_in_anchor_frame(
    root_quat: np.ndarray, body_quat_w: np.ndarray, out: np.ndarray
) -> None:
    """Write first two columns of body rotation matrices in anchor coordinates."""
    relative = np_quat_mul_batched(np_quat_conjugate_batched(root_quat[:, None, :]), body_quat_w)
    w, x, y, z = (relative[..., i] for i in range(4))
    out[..., 0] = 1.0 - 2.0 * (y * y + z * z)
    out[..., 1] = 2.0 * (x * y - w * z)
    out[..., 2] = 2.0 * (x * y + w * z)
    out[..., 3] = 1.0 - 2.0 * (x * x + z * z)
    out[..., 4] = 2.0 * (x * z - w * y)
    out[..., 5] = 2.0 * (y * z + w * x)


from .sonic_data import (
    _DEFAULT_REFERENCE_FRAMES,
    _G1_FUTURE_STRIDE,
    _SMPL_FUTURE_STRIDE,
    _SONIC_ACTUATOR_PARAMETERS,
    _WRIST_POLICY_INDICES,
    G1_POLICY_JOINT_NAMES,
    SonicNpzMotionLoader,
    SonicPackedMotionLoader,
    SonicRawMotionLoader,
    _pack_g1_reference,
    _pack_smpl_reference,
    _root_local_proprioception,
    _SonicMotionLoader,
    np_quat_from_euler_xyz,
    packed_store_clip_count,
)
from .sonic_kernels import (
    configure_motion_kernel_runtime,
    push_history_and_assemble_kernel,
)

G1_SONIC_JOINTS = G1_POLICY_JOINT_NAMES
G1_SONIC_ACTION_SCALE = 2.0


def _sonic_action_scale(base_scale: float) -> np.ndarray:
    """Expand a base action scale with the released G1 actuator contract."""

    base = float(base_scale)
    if not np.isfinite(base) or base <= 0.0:
        raise ValueError("SONIC base action scale must be a positive finite value")
    return np.asarray(
        [
            base * _SONIC_ACTUATOR_PARAMETERS[name][2] / _SONIC_ACTUATOR_PARAMETERS[name][0]
            for name in G1_SONIC_JOINTS
        ],
        dtype=np.float32,
    )


def _sonic_policy_action_scale() -> np.ndarray:
    """Return gear_sonic's released per-joint action scale in policy order."""

    return _sonic_action_scale(0.25)


G1_SONIC_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
G1_SONIC_EE_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
SONIC_TERMINATION_REASON_NAMES = (
    "anchor_pos_z",
    "anchor_ori",
    "ee_body_pos_z",
    "feet_pos",
    "time_out",
    "clip_end",
)
_SONIC_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
_SONIC_ANTI_SHAKE_BODY_NAMES = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
_SONIC_FOOT_JOINT_NAMES = (
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)
_SONIC_UNDESIRED_CONTACT_HISTORY = 3
_PROPRIO_FRAME_DIM = 93
_SONIC_UNDESIRED_CONTACT_FORCE_THRESHOLD = 1.0
_SONIC_UNDESIRED_CONTACT_GEOM_GROUPS = (
    ("pelvis", ("pelvis_collision",)),
    ("left_hip_roll_link", ("left_hip_collision",)),
    ("left_hip_yaw_link", ("left_thigh_collision",)),
    ("left_knee_link", ("left_shin_collision", "left_linkage_brace_collision")),
    ("right_hip_roll_link", ("right_hip_collision",)),
    ("right_hip_yaw_link", ("right_thigh_collision",)),
    ("right_knee_link", ("right_shin_collision", "right_linkage_brace_collision")),
    ("torso_link", ("torso_collision", "head_collision")),
    ("left_shoulder_yaw_link", ("left_shoulder_yaw_collision",)),
    ("right_shoulder_yaw_link", ("right_shoulder_yaw_collision",)),
)
_SONIC_COMMON_REWARD_FUNCTIONS = motion_rewards.build_reward_functions()
_SONIC_CUSTOM_REWARD_METHODS = {
    "tracking_vr_5point_local": "reward_tracking_vr_5point_local",
    "joint_limit": "reward_joint_limit",
    "undesired_contacts": "reward_undesired_contacts",
    "anti_shake": "reward_anti_shake",
    "feet_acc": "reward_feet_acc",
}
_SONIC_REWARD_STD_ATTRIBUTES = {
    "motion_global_root_pos": "std_root_pos",
    "motion_global_root_ori": "std_root_ori",
    "motion_body_pos": "std_body_pos",
    "motion_body_ori": "std_body_ori",
    "motion_body_lin_vel": "std_body_lin_vel",
    "motion_body_ang_vel": "std_body_ang_vel",
    "tracking_vr_5point_local": "std_vr_5point_local",
    "motion_ee_body_pos_z": "std_body_pos",
    "motion_joint_pos": "std_joint_pos",
    "motion_joint_vel": "std_joint_vel",
}


@dataclass
class SonicNoiseConfig(NoiseConfig):
    """Released SONIC actor-observation noise profile."""

    level: float = 1.0
    scale_joint_angle: float = 0.01
    scale_joint_vel: float = 0.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05


@dataclass(kw_only=True)
class SonicMotionCommandParamsCfg:
    """Nested motion parameters shared with the generic motion command shape."""

    motion_file: str | list[str] = ""
    smpl_motion_file: str | list[str] = ""
    motion_store_file: str = ""
    reference_format: Literal["auto", "raw", "npz", "packed"] = "auto"
    # Subset loading for oversized packed training stores (upstream motion-lib
    # contract): materialize only this many uniformly-sampled clips as the
    # working set and rotate it every ``clip_rotation_interval_steps``
    # control steps.  None keeps the legacy full-store load; rotation
    # requires an active subset (``max_loaded_clips`` smaller than the
    # store's clip count).
    max_loaded_clips: int | None = None
    clip_rotation_interval_steps: int | None = None
    body_names: tuple[str, ...] = G1_SONIC_BODY_NAMES
    anchor_body_name: str = "pelvis"
    ee_body_names: tuple[str, ...] = G1_SONIC_EE_BODY_NAMES
    num_future_frames: int = _DEFAULT_REFERENCE_FRAMES
    sampling_mode: Literal["start", "clip_start", "uniform", "adaptive", "mixed"] = "adaptive"
    sampling_start_ratio: float = 0.0
    adaptive_lambda: float = 0.8
    adaptive_kernel_size: int = 1
    adaptive_uniform_ratio: float = 0.2
    adaptive_alpha: float = 0.001
    # SONIC trains on the cumulative failure-rate statistic; the legacy EMA
    # path (and therefore ``adaptive_alpha``) stays available but unused.
    adaptive_failure_stat: Literal["ema", "cumulative"] = "cumulative"
    adaptive_failure_prior: float = 1.0
    adaptive_pre_failure_window: int = 200
    adaptive_failure_rate_max_over_mean: float = 200.0
    adaptive_sampling_update_interval: int = 200
    adaptive_attribution: Literal["episode_end", "trajectory"] = "trajectory"
    adaptive_max_prob_per_motion: float | None = None
    advance_reference_before_update: bool = False
    truncate_on_clip_end: bool = True
    encoder_sampling: Literal["mixed", "g1", "smpl"] = "mixed"
    anchor_pos_z_threshold: float = 0.15
    anchor_ori_threshold: float = 0.2
    ee_body_pos_z_threshold: float = 0.15
    low_reference_height_threshold: float = 0.5
    low_reference_anchor_pos_z_threshold: float = 0.75
    low_reference_ee_body_pos_z_threshold: float = 0.75
    foot_pos_threshold: float = 0.2
    # Match gear_sonic's released 5-point tracking contract: pelvis, both
    # wrist points and both ankle points.  Wrist offsets target the same
    # forearm/hand points used by the upstream reward rather than the wrist
    # link origins, while ankle points provide an explicit foot-stability
    # signal in addition to the generic body rewards.
    reward_point_body_names: tuple[str, ...] = (
        "pelvis",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    reward_point_body_offsets: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),
        (0.18, -0.025, 0.0),
        (0.18, 0.025, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    pose_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
    )
    velocity_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
    )
    joint_position_range: tuple[float, float] = (0.0, 0.0)
    joint_default_position_range: tuple[float, float] = (0.0, 0.0)
    joint_velocity_range: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_future_frames, bool)
            or not isinstance(self.num_future_frames, int)
            or self.num_future_frames <= 0
        ):
            raise ValueError("num_future_frames must be a positive integer")
        if not np.isfinite(self.adaptive_lambda) or not 0.0 < self.adaptive_lambda <= 1.0:
            raise ValueError("adaptive_lambda must be finite and within (0, 1]")
        if self.adaptive_kernel_size < 1:
            raise ValueError("adaptive_kernel_size must be a positive integer")
        if not 0.0 <= self.adaptive_uniform_ratio <= 1.0:
            raise ValueError("adaptive_uniform_ratio must be within [0, 1]")
        if not 0.0 < self.adaptive_alpha <= 1.0:
            raise ValueError("adaptive_alpha must be within (0, 1]")
        if self.adaptive_failure_stat not in ("ema", "cumulative"):
            raise ValueError("adaptive_failure_stat must be 'ema' or 'cumulative'")
        if (
            isinstance(self.adaptive_failure_prior, bool)
            or not np.isfinite(self.adaptive_failure_prior)
            or self.adaptive_failure_prior <= 0.0
        ):
            raise ValueError("adaptive_failure_prior must be a finite positive number")
        if (
            isinstance(self.adaptive_sampling_update_interval, bool)
            or self.adaptive_sampling_update_interval < 1
        ):
            raise ValueError("adaptive_sampling_update_interval must be a positive integer")
        if self.adaptive_attribution not in ("episode_end", "trajectory"):
            raise ValueError("adaptive_attribution must be 'episode_end' or 'trajectory'")
        if self.adaptive_max_prob_per_motion is not None and (
            isinstance(self.adaptive_max_prob_per_motion, bool)
            or not np.isfinite(self.adaptive_max_prob_per_motion)
            or self.adaptive_max_prob_per_motion < 1.0
        ):
            raise ValueError("adaptive_max_prob_per_motion must be None or a finite value >= 1")
        if self.sampling_mode != "mixed" and self.sampling_start_ratio != 0.0:
            raise ValueError("sampling_start_ratio is only effective when sampling_mode='mixed'")


def _sonic_reward_config() -> motion_rewards.RewardConfig:
    config = motion_rewards.RewardConfig()
    config.scales.update(
        {
            "motion_global_root_pos": 0.5,
            "tracking_vr_5point_local": 2.0,
            "undesired_contacts": -0.1,
            "anti_shake": -0.005,
            "feet_acc": -0.0000025,
        }
    )
    config.std_root_pos = 0.3
    return config


@dataclass(kw_only=True)
class SonicMotionCommandCfg(CommandTermCfg):
    """YAML-owned configuration for the SONIC reference-motion manager."""

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    entity_name: str = "robot"
    params: SonicMotionCommandParamsCfg = field(default_factory=SonicMotionCommandParamsCfg)

    def __getattr__(self, name: str):
        params = self.__dict__.get("params")
        if params is not None and hasattr(params, name):
            return getattr(params, name)
        raise AttributeError(name)

    @property
    def reset_pose_range(self) -> tuple[float, ...]:
        return tuple(
            value
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
            for value in self.params.pose_range[axis]
        )

    @reset_pose_range.setter
    def reset_pose_range(self, value: tuple[float, ...]) -> None:
        self.params.pose_range = {
            axis: tuple(value[index : index + 2])
            for index, axis in zip(
                range(0, 12, 2), ("x", "y", "z", "roll", "pitch", "yaw"), strict=True
            )
        }

    @property
    def reset_velocity_range(self) -> tuple[float, ...]:
        return tuple(
            value
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
            for value in self.params.velocity_range[axis]
        )

    @reset_velocity_range.setter
    def reset_velocity_range(self, value: tuple[float, ...]) -> None:
        self.params.velocity_range = {
            axis: tuple(value[index : index + 2])
            for index, axis in zip(
                range(0, 12, 2), ("x", "y", "z", "roll", "pitch", "yaw"), strict=True
            )
        }

    @property
    def reset_joint_position_range(self) -> tuple[float, float]:
        return self.params.joint_position_range

    @reset_joint_position_range.setter
    def reset_joint_position_range(self, value: tuple[float, float]) -> None:
        self.params.joint_position_range = value

    @property
    def reset_joint_velocity_range(self) -> tuple[float, float]:
        return self.params.joint_velocity_range

    @reset_joint_velocity_range.setter
    def reset_joint_velocity_range(self, value: tuple[float, float]) -> None:
        self.params.joint_velocity_range = value

    def validate(self) -> None:
        axes = ("x", "y", "z", "roll", "pitch", "yaw")
        for name in ("pose_range", "velocity_range"):
            ranges = getattr(self.params, name)
            if set(ranges) != set(axes):
                raise ValueError(f"SONIC {name} must define exactly {axes}")
            for axis in axes:
                values = np.asarray(ranges[axis], dtype=np.float64)
                if values.shape != (2,) or not np.isfinite(values).all() or values[0] > values[1]:
                    raise ValueError(f"SONIC {name}.{axis} must be a finite (lower, upper) range")
        if self.anchor_body_name not in self.body_names:
            raise ValueError("SONIC anchor_body_name must be present in body_names")
        for name in ("max_loaded_clips", "clip_rotation_interval_steps"):
            value = getattr(self.params, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"SONIC {name} must be None or a positive integer")
        if tuple(self.body_names) != G1_SONIC_BODY_NAMES:
            raise ValueError("SONIC body_names must preserve the released body order")
        if any(name not in self.body_names for name in self.ee_body_names):
            raise ValueError("SONIC ee_body_names must be present in body_names")
        if len(self.reward_point_body_names) != len(self.reward_point_body_offsets):
            raise ValueError("SONIC reward-point body names and offsets must have equal length")
        offsets = np.asarray(self.reward_point_body_offsets)
        if (
            offsets.shape != (len(self.reward_point_body_names), 3)
            or not np.isfinite(offsets).all()
        ):
            raise ValueError("SONIC reward-point offsets must be finite XYZ vectors")
        thresholds = (
            self.anchor_pos_z_threshold,
            self.anchor_ori_threshold,
            self.ee_body_pos_z_threshold,
            self.low_reference_height_threshold,
            self.low_reference_anchor_pos_z_threshold,
            self.low_reference_ee_body_pos_z_threshold,
            self.foot_pos_threshold,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in thresholds):
            raise ValueError("SONIC termination thresholds must be finite and non-negative")
        for name, values, expected in (
            ("reset_pose_range", self.reset_pose_range, 12),
            ("reset_velocity_range", self.reset_velocity_range, 12),
        ):
            if len(values) != expected or not np.isfinite(values).all():
                raise ValueError(f"SONIC {name} must contain {expected} finite bounds")
            if any(lower > upper for lower, upper in zip(values[::2], values[1::2], strict=True)):
                raise ValueError(f"SONIC {name} lower bounds must not exceed upper bounds")
        for name, values in (
            ("reset_joint_position_range", self.reset_joint_position_range),
            ("reset_joint_velocity_range", self.reset_joint_velocity_range),
        ):
            if len(values) != 2 or not np.isfinite(values).all() or values[0] > values[1]:
                raise ValueError(f"SONIC {name} must be a finite (lower, upper) range")
        if self.advance_reference_before_update:
            raise ValueError(
                "SONIC reference advancement must occur after reward/termination computation"
            )
        if self.reference_format not in ("auto", "raw", "npz", "packed"):
            raise ValueError("SONIC reference_format must be auto, raw, npz, or packed")

    def build(self, env) -> SonicMotionCommand:
        return SonicMotionCommand(self, env)


def _motion_command(env) -> SonicMotionCommand:
    term = env.command_manager.get_term("motion")
    if not isinstance(term, SonicMotionCommand):
        raise TypeError("G1SonicManager command 'motion' has an incompatible term type")
    return term


class SonicMotionCommand(CommandTerm):
    """Cold-path motion owner and hot-path reference-frame state."""

    cfg: SonicMotionCommandCfg

    def __init__(self, cfg: SonicMotionCommandCfg, env):
        super().__init__(cfg, env)
        task_cfg = cfg
        backend = cast(ManagerBasedRlEnv, env)._backend
        model_names = tuple(backend.get_actuator_names())
        if set(model_names) != set(G1_SONIC_JOINTS):
            raise ValueError("G1 actuator names do not satisfy the SONIC policy contract")
        self._policy_from_model = np.asarray(
            [model_names.index(name) for name in G1_SONIC_JOINTS], dtype=np.intp
        )
        reference_format = task_cfg.reference_format
        if reference_format == "auto":
            if task_cfg.motion_store_file:
                reference_format = "packed"
            else:
                candidates = (
                    [task_cfg.motion_file]
                    if isinstance(task_cfg.motion_file, str)
                    else task_cfg.motion_file
                )
                reference_format = (
                    "raw"
                    if any(str(value).lower().endswith(".csv") for value in candidates)
                    else "npz"
                )
        self._clip_subset_size: int | None = None
        self._clip_rotation_interval: int | None = None
        self._rotation_step_counter = 0
        self._clip_rotations = 0
        self._packed_backend = None
        if reference_format == "packed":
            if not task_cfg.motion_store_file:
                raise ValueError("SONIC packed references require motion_store_file")
            self._packed_backend = backend
            max_loaded = task_cfg.max_loaded_clips
            # The manifest is only consulted when a subset is requested; the
            # default full-load path stays loader-implementation-agnostic.
            if max_loaded is None or max_loaded >= packed_store_clip_count(
                task_cfg.motion_store_file
            ):
                loader = SonicPackedMotionLoader(
                    task_cfg.motion_store_file,
                    backend=backend,
                    body_names=task_cfg.body_names,
                )
            else:
                self._clip_subset_size = int(max_loaded)
                loader = SonicPackedMotionLoader(
                    task_cfg.motion_store_file,
                    backend=backend,
                    body_names=task_cfg.body_names,
                    clip_indices=np.asarray(
                        env.rng.choice(
                            packed_store_clip_count(task_cfg.motion_store_file),
                            size=max_loaded,
                            replace=False,
                        )
                    ),
                )
            if task_cfg.clip_rotation_interval_steps is not None:
                if self._clip_subset_size is None:
                    raise ValueError(
                        "SONIC clip_rotation_interval_steps requires an active "
                        "max_loaded_clips subset smaller than the store"
                    )
                self._clip_rotation_interval = int(task_cfg.clip_rotation_interval_steps)
        else:
            if (
                task_cfg.max_loaded_clips is not None
                or task_cfg.clip_rotation_interval_steps is not None
            ):
                raise ValueError("SONIC clip subsetting requires a packed motion store")
            if not task_cfg.motion_file or not task_cfg.smpl_motion_file:
                raise ValueError(
                    "G1SonicManager requires motion_store_file or paired "
                    "motion_file/smpl_motion_file"
                )
            loader_cls = SonicRawMotionLoader if reference_format == "raw" else SonicNpzMotionLoader
            loader = loader_cls(
                task_cfg.motion_file,
                task_cfg.smpl_motion_file,
                backend=backend,
                body_names=task_cfg.body_names,
            )
        self.loader: _SonicMotionLoader = loader
        self.sampler = self._build_sampler(loader)
        self.anchor_body_idx = task_cfg.body_names.index(task_cfg.anchor_body_name)
        self.ee_body_indices = np.asarray(
            [task_cfg.body_names.index(name) for name in task_cfg.ee_body_names], dtype=np.intp
        )
        self._foot_body_indices = np.asarray(
            [task_cfg.body_names.index(name) for name in _SONIC_FOOT_BODY_NAMES], dtype=np.intp
        )
        self._anti_shake_indices = np.asarray(
            [task_cfg.body_names.index(name) for name in _SONIC_ANTI_SHAKE_BODY_NAMES],
            dtype=np.intp,
        )
        self._foot_joint_model_indices = np.asarray(
            [model_names.index(name) for name in _SONIC_FOOT_JOINT_NAMES], dtype=np.intp
        )
        self._foot_joint_policy_indices = np.asarray(
            [G1_SONIC_JOINTS.index(name) for name in _SONIC_FOOT_JOINT_NAMES], dtype=np.intp
        )
        self.motion_data = loader.make_motion_data_buffer(env.num_envs)
        self._motion_scratch = loader.make_motion_data_buffer(env.num_envs)
        self._g1_command_cache: np.ndarray | None = None
        self._g1_command_cache_step = -1
        self._init_reference_feature_cache(loader)
        self.encoder_index = np.zeros((env.num_envs, 2), dtype=np.float32)
        self.encoder_index[:, 0] = 1.0
        self.clip_end = np.zeros(env.num_envs, dtype=np.bool_)
        self._next_reset_clip_indices = np.full(env.num_envs, -1, dtype=np.int32)
        self._command = np.zeros((env.num_envs, 1), dtype=np.float32)
        self._init_transition_context(task_cfg, model_names)
        self._refresh_motion()

    def _init_transition_context(
        self, task_cfg: SonicMotionCommandCfg, model_names: tuple[str, ...]
    ) -> None:
        dtype = get_global_dtype()
        num_envs = self.num_envs
        num_bodies = len(task_cfg.body_names)
        num_joints = len(model_names)
        self.body_pos_relative_w = np.empty((num_envs, num_bodies, 3), dtype=dtype)
        self.body_quat_relative_w = np.empty((num_envs, num_bodies, 4), dtype=dtype)
        self._robot_body_pos_w = np.empty_like(self.body_pos_relative_w)
        self._robot_body_quat_w = np.empty_like(self.body_quat_relative_w)
        self._robot_body_lin_vel_w = np.empty_like(self.body_pos_relative_w)
        self._robot_body_ang_vel_w = np.empty_like(self.body_pos_relative_w)
        self._dof_pos = np.empty((num_envs, num_joints), dtype=dtype)
        self._dof_vel = np.empty_like(self._dof_pos)
        self._delta_pos_w = np.empty((num_envs, 3), dtype=dtype)
        self._delta_ori_w = np.empty((num_envs, 4), dtype=dtype)
        self._body_vec_error = np.empty_like(self.body_pos_relative_w)
        self._joint_error = np.empty_like(self._dof_pos)
        self._joint_error_upper = np.empty_like(self._dof_pos)
        self._env_error = np.empty(num_envs, dtype=dtype)
        self._env_error2 = np.empty(num_envs, dtype=dtype)
        self._reward_term = np.empty(num_envs, dtype=dtype)
        self._weighted_reward = np.empty(num_envs, dtype=dtype)
        self._quat_error_w = np.empty((num_envs, num_bodies), dtype=dtype)
        self._quat_error_x = np.empty((num_envs, num_bodies), dtype=dtype)
        self._ee_pos_error_z = np.empty((num_envs, len(self.ee_body_indices)), dtype=dtype)

        backend = cast(ManagerBasedRlEnv, self._env)._backend
        joint_range = backend.get_joint_range()
        joint_lower = joint_upper = None
        if joint_range is not None:
            joint_range = np.asarray(joint_range, dtype=dtype)
            midpoint = 0.5 * (joint_range[:, 0] + joint_range[:, 1])
            half_range = 0.45 * (joint_range[:, 1] - joint_range[:, 0])
            joint_lower = midpoint - half_range
            joint_upper = midpoint + half_range
        self.reward_context = motion_rewards.RewardContext(
            info={},
            motion_data=self.motion_data,
            robot_body_pos_w=self._robot_body_pos_w,
            robot_body_quat_w=self._robot_body_quat_w,
            robot_body_lin_vel_w=self._robot_body_lin_vel_w,
            robot_body_ang_vel_w=self._robot_body_ang_vel_w,
            ref_body_pos_w=self.body_pos_relative_w,
            ref_body_quat_w=self.body_quat_relative_w,
            dof_pos=self._dof_pos,
            dof_vel=self._dof_vel,
            reward_config=_sonic_reward_config(),
            anchor_body_idx=self.anchor_body_idx,
            ee_body_indices=self.ee_body_indices,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            num_envs=num_envs,
            body_vec_error=self._body_vec_error,
            joint_error=self._joint_error,
            joint_error_upper=self._joint_error_upper,
            env_error=self._env_error,
            env_error2=self._env_error2,
            reward_term=self._reward_term,
            weighted_reward=self._weighted_reward,
            quat_error_w=self._quat_error_w,
            quat_error_x=self._quat_error_x,
            ee_pos_error_z=self._ee_pos_error_z,
        )

        point_indices = [
            task_cfg.body_names.index(name) for name in task_cfg.reward_point_body_names
        ]
        self._reward_point_indices = np.asarray(point_indices, dtype=np.intp)
        self._reward_point_offsets = np.asarray(task_cfg.reward_point_body_offsets, dtype=dtype)
        point_shape = (num_envs, len(point_indices), 3)
        self._reward_point_reference_w = np.empty(point_shape, dtype=dtype)
        self._reward_point_robot_w = np.empty(point_shape, dtype=dtype)
        self._reward_point_reference_local = np.empty(point_shape, dtype=dtype)
        self._reward_point_robot_local = np.empty(point_shape, dtype=dtype)
        self._reward_point_error = np.empty(point_shape, dtype=dtype)
        self._reward_point_squared_error = np.empty(point_shape[:-1], dtype=dtype)

        self._running_ref_root_height = np.zeros(num_envs, dtype=dtype)
        self._low_reference = np.zeros(num_envs, dtype=np.bool_)
        self._termination_threshold = np.empty(num_envs, dtype=dtype)
        self._termination_error = np.empty(num_envs, dtype=dtype)
        self._termination_reason_mask = np.zeros((num_envs, 4), dtype=np.bool_)
        self._ee_termination_error = np.empty((num_envs, len(self.ee_body_indices)), dtype=dtype)
        self._ee_termination_exceeded = np.empty(
            (num_envs, len(self.ee_body_indices)), dtype=np.bool_
        )
        self._foot_termination_error = np.empty(
            (num_envs, len(self._foot_body_indices), 3), dtype=dtype
        )
        self._foot_termination_norm = np.empty(
            (num_envs, len(self._foot_body_indices)), dtype=dtype
        )
        self._foot_termination_exceeded = np.empty(
            (num_envs, len(self._foot_body_indices)), dtype=np.bool_
        )

        sensor_kind = "force" if backend.backend_type == "mujoco" else "found"
        sensor_names = tuple(
            f"sonic_undesired_{sensor_kind}_{geom_name}"
            for _, geom_names in _SONIC_UNDESIRED_CONTACT_GEOM_GROUPS
            for geom_name in geom_names
        )
        self._undesired_contact_sensor_view = self._env.scene.bind_sensor_data(sensor_names)
        expected_sensor_width = len(sensor_names) * (3 if sensor_kind == "force" else 1)
        if self._undesired_contact_sensor_view.width != expected_sensor_width:
            raise ValueError(
                "SONIC undesired-contact sensor width mismatch: "
                f"expected {expected_sensor_width}, got "
                f"{self._undesired_contact_sensor_view.width}"
            )
        self._undesired_contact_group_slices: list[slice] = []
        sensor_start = 0
        for _, geom_names in _SONIC_UNDESIRED_CONTACT_GEOM_GROUPS:
            sensor_end = sensor_start + len(geom_names)
            self._undesired_contact_group_slices.append(slice(sensor_start, sensor_end))
            sensor_start = sensor_end
        num_groups = len(self._undesired_contact_group_slices)
        self._undesired_contact_history = np.zeros(
            (num_envs, _SONIC_UNDESIRED_CONTACT_HISTORY, num_groups), dtype=np.bool_
        )
        self._undesired_contact_current = np.empty((num_envs, num_groups), dtype=np.bool_)

    def refresh_transition_context(self) -> None:
        robot = self._env.scene["robot"].data
        np.copyto(self._robot_body_pos_w, robot.body_link_pos_w)
        np.copyto(self._robot_body_quat_w, robot.body_link_quat_w)
        np.copyto(self._robot_body_lin_vel_w, robot.body_link_lin_vel_w)
        np.copyto(self._robot_body_ang_vel_w, robot.body_link_ang_vel_w)
        np.copyto(self._dof_pos, robot.joint_pos)
        np.copyto(self._dof_vel, robot.joint_vel)
        self.reward_context.info["current_actions"] = self._env.action_manager.action
        self.reward_context.info["last_actions"] = self._env.action_manager.prev_action
        anchor_pos = self._robot_body_pos_w[:, self.anchor_body_idx]
        anchor_quat = self._robot_body_quat_w[:, self.anchor_body_idx]
        motion_anchor_pos = self.motion_data.body_pos_w[:, self.anchor_body_idx]
        motion_anchor_quat = self.motion_data.body_quat_w[:, self.anchor_body_idx]
        # Match gear_sonic's reference transform: align only the anchor
        # heading, preserve the reference anchor height, and leave roll/pitch
        # out of the world-frame placement.  Using the full anchor quaternion
        # here rotates reference body offsets when the robot is tilted and can
        # spuriously trip EE/feet termination.
        np.copyto(self._delta_pos_w, anchor_pos)
        self._delta_pos_w[:, 2] = motion_anchor_pos[:, 2]
        relative_anchor = np_quat_mul_batched(
            anchor_quat, np_quat_conjugate_batched(motion_anchor_quat)
        )
        np.copyto(self._delta_ori_w, np_quat_heading(relative_anchor))
        delta = self.motion_data.body_pos_w - motion_anchor_pos[:, None, :]
        self.body_pos_relative_w[...] = self._delta_pos_w[:, None, :] + np_quat_apply_batched(
            self._delta_ori_w[:, None, :], delta
        )
        self.body_quat_relative_w[...] = np_quat_mul_batched(
            self._delta_ori_w[:, None, :], self.motion_data.body_quat_w
        )
        self._running_ref_root_height *= 0.9
        self._running_ref_root_height += (
            0.1 * self.motion_data.body_pos_w[:, self.anchor_body_idx, 2]
        )
        np.less(
            self._running_ref_root_height,
            self.cfg.low_reference_height_threshold,
            out=self._low_reference,
        )

    @property
    def command(self) -> np.ndarray:
        return self._command

    def reset(self, env_ids: np.ndarray | slice | None) -> dict[str, float]:
        ids = np.arange(self.num_envs)[env_ids] if isinstance(env_ids, slice) else env_ids
        if ids is None:
            ids = np.arange(self.num_envs)
        self.command_counter[ids] = 0
        self.time_left[ids] = self.cfg.resampling_time_range[0]
        self.clip_end[ids] = False
        self._update_command(ids)
        return {}

    def stage_reset_clip_indices(self, env_ids: np.ndarray, clip_indices: np.ndarray) -> None:
        """Select exact clips for the next reset of the specified environments."""
        ids = np.asarray(env_ids)
        clips = np.asarray(clip_indices)
        if ids.ndim != 1 or clips.ndim != 1 or ids.shape != clips.shape:
            raise ValueError("env_ids and clip_indices must be equal-length vectors")
        if not np.issubdtype(ids.dtype, np.integer) or not np.issubdtype(clips.dtype, np.integer):
            raise TypeError("env_ids and clip_indices must contain integers")
        if np.any(ids < 0) or np.any(ids >= self.num_envs):
            raise ValueError("env_ids are outside the command environment range")
        if np.any(clips < 0) or np.any(clips >= self.loader.num_clips):
            raise ValueError("clip_indices are outside the motion dataset range")
        self._next_reset_clip_indices[ids.astype(np.intp, copy=False)] = clips.astype(
            np.int32, copy=False
        )

    def tracking_body_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the robot/reference positions used by the latest transition.

        Rows are copied because command advancement refreshes the next reference
        frame immediately after reward and termination computation.
        """
        return self._robot_body_pos_w.copy(), self.body_pos_relative_w.copy()

    def reset_reference(self, env_ids: np.ndarray) -> None:
        staged_clips = self._next_reset_clip_indices[env_ids]
        staged = staged_clips >= 0
        frames = np.empty(len(env_ids), dtype=np.int32)
        unstaged = ~staged
        if np.any(unstaged):
            frames[unstaged] = self.sampler.sample_frames(env_ids[unstaged])
        if np.any(staged):
            staged_ids = env_ids[staged]
            frames[staged] = self.sampler.set_clip_starts(staged_ids, staged_clips[staged])
            self._next_reset_clip_indices[staged_ids] = -1
        # One gather serves both the motion_data scatter and the local reset
        # views; the previous form gathered the same frames twice.  The views
        # stay valid until the next _gather_motion call, which nothing below
        # triggers.
        motion = self._gather_motion(frames)
        self._scatter_motion(env_ids, motion)
        # Reset rows carry freshly sampled frames: drop the step-keyed
        # g1_command cache so the reset-path observation rebuild sees them.
        self._g1_command_cache = None
        self._g1_command_cache_step = -1
        self._sample_encoder(env_ids)
        self._running_ref_root_height[env_ids] = motion.body_pos_w[:, self.anchor_body_idx, 2]
        self._undesired_contact_history[env_ids] = False
        robot = self._env.scene[self.cfg.entity_name]
        self._env.scene.reset_to_default(env_ids, term_name="sonic_reference_reset")
        root_pose = np.concatenate(
            (
                motion.body_pos_w[:, self.anchor_body_idx],
                motion.body_quat_w[:, self.anchor_body_idx],
            ),
            axis=-1,
        )
        root_velocity = np.concatenate(
            (
                motion.body_lin_vel_w[:, self.anchor_body_idx],
                motion.body_ang_vel_w[:, self.anchor_body_idx],
            ),
            axis=-1,
        )
        pose_noise = self._env.rng.uniform(
            np.asarray(self.cfg.reset_pose_range[::2], dtype=root_pose.dtype),
            np.asarray(self.cfg.reset_pose_range[1::2], dtype=root_pose.dtype),
            size=(len(env_ids), 6),
        )
        root_pose[:, :3] += pose_noise[:, :3]
        root_pose[:, 3:] = np_quat_mul_batched(
            np_quat_from_euler_xyz(pose_noise[:, 3], pose_noise[:, 4], pose_noise[:, 5]),
            root_pose[:, 3:],
        )
        velocity_noise = self._env.rng.uniform(
            np.asarray(self.cfg.reset_velocity_range[::2], dtype=root_velocity.dtype),
            np.asarray(self.cfg.reset_velocity_range[1::2], dtype=root_velocity.dtype),
            size=(len(env_ids), 6),
        )
        root_velocity += velocity_noise
        joint_pos = np.array(motion.joint_pos, copy=True)
        joint_vel = np.array(motion.joint_vel, copy=True)
        joint_pos += self._env.rng.uniform(
            self.cfg.reset_joint_position_range[0],
            self.cfg.reset_joint_position_range[1],
            size=joint_pos.shape,
        ).astype(joint_pos.dtype, copy=False)
        joint_vel += self._env.rng.uniform(
            self.cfg.reset_joint_velocity_range[0],
            self.cfg.reset_joint_velocity_range[1],
            size=joint_vel.shape,
        ).astype(joint_vel.dtype, copy=False)
        # Keep reset randomization inside the same soft limits used by the
        # generic motion-tracking command.  The motion store is validated in
        # model joint order, which is also the entity state order here.
        limits = np.asarray(robot.data.soft_joint_pos_limits)
        if limits.ndim == 3:
            limits = limits[env_ids]
        if limits.shape[-2:] != (joint_pos.shape[-1], 2):
            raise ValueError(
                "SONIC soft joint position limits must have shape "
                f"(num_joints, 2) or (num_envs, num_joints, 2), got {limits.shape}"
            )
        np.clip(joint_pos, limits[..., 0], limits[..., 1], out=joint_pos)
        robot.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
        robot.write_root_link_velocity_to_sim(root_velocity, env_ids=env_ids)
        robot.write_joint_state_to_sim(
            # Motion stores are validated against the backend/model joint order,
            # which is also the order declared by the robot entity.  Policy order
            # is only an observation/action boundary and must not be applied to a
            # generalized-state reset.
            joint_pos,
            joint_vel,
            env_ids=env_ids,
        )
        self.clip_end[env_ids] = False

    def reward_tracking_vr_5point_local(self) -> np.ndarray:
        ctx = self.reward_context
        indices = self._reward_point_indices
        offsets = self._reward_point_offsets[None]
        reference_quat = self.motion_data.body_quat_w[:, indices]
        robot_quat = self._robot_body_quat_w[:, indices]
        self._reward_point_reference_w[:] = self.motion_data.body_pos_w[:, indices]
        self._reward_point_reference_w += np_quat_apply_batched(reference_quat, offsets)
        self._reward_point_robot_w[:] = self._robot_body_pos_w[:, indices]
        self._reward_point_robot_w += np_quat_apply_batched(robot_quat, offsets)
        reference_anchor_pos = self.motion_data.body_pos_w[:, self.anchor_body_idx, None]
        robot_anchor_pos = self._robot_body_pos_w[:, self.anchor_body_idx, None]
        reference_anchor_quat = self.motion_data.body_quat_w[:, self.anchor_body_idx, None]
        robot_anchor_quat = self._robot_body_quat_w[:, self.anchor_body_idx, None]
        self._reward_point_reference_local[:] = np_quat_apply_batched(
            np_quat_conjugate_batched(reference_anchor_quat),
            self._reward_point_reference_w - reference_anchor_pos,
        )
        self._reward_point_robot_local[:] = np_quat_apply_batched(
            np_quat_conjugate_batched(robot_anchor_quat),
            self._reward_point_robot_w - robot_anchor_pos,
        )
        np.subtract(
            self._reward_point_reference_local,
            self._reward_point_robot_local,
            out=self._reward_point_error,
        )
        np.square(self._reward_point_error, out=self._reward_point_error)
        np.sum(self._reward_point_error, axis=-1, out=self._reward_point_squared_error)
        np.mean(self._reward_point_squared_error, axis=-1, out=self._reward_term)
        self._reward_term /= -(ctx.reward_config.std_vr_5point_local**2)
        np.exp(self._reward_term, out=self._reward_term)
        return self._reward_term

    def reward_joint_limit(self) -> np.ndarray:
        ctx = self.reward_context
        if ctx.joint_lower is None or ctx.joint_upper is None:
            self._reward_term.fill(0.0)
            return self._reward_term
        np.subtract(ctx.joint_lower, self._dof_pos, out=self._joint_error)
        np.maximum(self._joint_error, 0.0, out=self._joint_error)
        np.subtract(self._dof_pos, ctx.joint_upper, out=self._joint_error_upper)
        np.maximum(self._joint_error_upper, 0.0, out=self._joint_error_upper)
        self._joint_error += self._joint_error_upper
        np.sum(self._joint_error, axis=-1, out=self._reward_term)
        return self._reward_term

    def reward_anti_shake(self) -> np.ndarray:
        angular = self._robot_body_ang_vel_w[:, self._anti_shake_indices]
        excess = np.maximum(np.linalg.norm(angular, axis=-1) - 1.5, 0.0)
        np.mean(np.square(excess), axis=-1, out=self._reward_term)
        return self._reward_term

    def reward_feet_acc(self) -> np.ndarray:
        action = self._env.action_manager.get_term("joint_pos")
        if not isinstance(action, SonicJointPositionAction):
            raise TypeError("G1SonicManager action 'joint_pos' has an incompatible term type")
        current = self._dof_vel[:, self._foot_joint_model_indices]
        previous = action.joint_velocity_before_action[:, self._foot_joint_policy_indices]
        acceleration = (current - previous) / self._env.step_dt
        np.sum(np.square(acceleration), axis=-1, out=self._reward_term)
        return self._reward_term

    def reward_undesired_contacts(self) -> np.ndarray:
        sensor_values = self._undesired_contact_sensor_view.read()
        num_sensors = len(self._undesired_contact_sensor_view.names)
        if self._undesired_contact_sensor_view.backend_type == "mujoco":
            force = sensor_values.reshape(self.num_envs, num_sensors, 3)
            geom_contacts = (
                np.linalg.norm(force, axis=-1) > _SONIC_UNDESIRED_CONTACT_FORCE_THRESHOLD
            )
        else:
            geom_contacts = sensor_values.reshape(self.num_envs, num_sensors) > 0.0
        for group_index, sensor_slice in enumerate(self._undesired_contact_group_slices):
            self._undesired_contact_current[:, group_index] = np.any(
                geom_contacts[:, sensor_slice], axis=1
            )
        self._undesired_contact_history[:, :-1] = self._undesired_contact_history[:, 1:]
        self._undesired_contact_history[:, -1] = self._undesired_contact_current
        np.any(self._undesired_contact_history, axis=1, out=self._undesired_contact_current)
        np.sum(self._undesired_contact_current, axis=1, out=self._reward_term)
        return self._reward_term

    def advance_reference(self, env_ids: np.ndarray | None = None) -> None:
        ids = np.arange(self.num_envs, dtype=np.int32) if env_ids is None else env_ids
        self.clip_end[ids] = False
        done_ids = self.sampler.step(ids)
        if self.cfg.params.truncate_on_clip_end:
            self.clip_end[done_ids] = True
            # ``MotionSampler.step`` detects clip completion after incrementing
            # ``current_frames``. Keep the terminal rows on the last valid frame
            # while the transition is finalized; otherwise the subsequent
            # ``_refresh_motion`` gather indexes one past the packed motion store.
            if len(done_ids):
                self.sampler.current_frames[done_ids] = self.sampler.current_clip_end_frames[
                    done_ids
                ]
            self._refresh_motion()
        else:
            if len(done_ids):
                # Wrap mode: rows that finished their clip teleport onto a
                # freshly sampled reference frame mid-episode. ``reset_reference``
                # reruns the full reference reset (sample + encoder + state
                # writes); episode bookkeeping stays untouched, matching the
                # MotrixLab sim-reset-on-wrap contract.
                self.reset_reference(done_ids)
                self._refresh_motion()
                # Stamp after the teleport so the wrap stays observable for this
                # step; ``reset_reference`` clears the flag for the reset rows.
                self.clip_end[done_ids] = True
            else:
                self._refresh_motion()
        self._command[ids, 0] = self.sampler.current_frames[ids]

    def g1_command(self, rows: np.ndarray) -> np.ndarray:
        # The policy reference term and the critic observation both evaluate
        # the full batch within one update; the cached result is invalidated
        # by reset_reference (frames are resampled there) and by the step
        # counter, so subset callers always slice a fresh full-batch result.
        step = self._env.common_step_counter
        cached = self._g1_command_cache
        if cached is not None and self._g1_command_cache_step == step:
            return cached if len(rows) == len(cached) else cached[rows]
        frames = self.sampler.current_frames[rows]
        future = self.loader.future_indices(frames, _G1_FUTURE_STRIDE, self.cfg.num_future_frames)
        pos = self._ref_joint_pos_policy[future]
        vel = self._ref_joint_vel_policy[future]
        result = np.concatenate(
            (pos.reshape(len(rows), -1), vel.reshape(len(rows), -1)), axis=-1
        ).reshape(len(rows), self.cfg.num_future_frames, 58)
        if len(rows) == self.num_envs:
            result.setflags(write=False)
            self._g1_command_cache = result
            self._g1_command_cache_step = step
        return result

    def g1_reference(self, rows: np.ndarray | None = None) -> np.ndarray:
        rows = np.arange(self.num_envs, dtype=np.intp) if rows is None else rows
        future = self.loader.future_indices(
            self.sampler.current_frames[rows], _G1_FUTURE_STRIDE, self.cfg.num_future_frames
        )
        root_quat = self._env.scene["robot"].data.root_link_quat_w[rows]
        future_root_quat = self.loader.body_quat_w[future, self.anchor_body_idx]
        relative_root = np_quat_mul_batched(
            np_quat_conjugate_batched(root_quat[:, None]), future_root_quat
        )
        return _pack_g1_reference(self.g1_command(rows), relative_root).astype(
            get_global_dtype(), copy=False
        )

    def smpl_reference(self, rows: np.ndarray | None = None) -> np.ndarray:
        rows = np.arange(self.num_envs, dtype=np.intp) if rows is None else rows
        future = self.loader.future_indices(
            self.sampler.current_frames[rows], _SMPL_FUTURE_STRIDE, self.cfg.num_future_frames
        )
        human_quat = self.loader.smpl_root_quat[future]
        human_local = self._ref_smpl_human_local[future]
        wrist = self._ref_wrist_policy[future]
        root_quat = self._env.scene["robot"].data.root_link_quat_w[rows]
        human_relative = np_quat_mul_batched(
            np_quat_conjugate_batched(root_quat[:, None]), human_quat
        )
        return _pack_smpl_reference(
            human_local.reshape(len(rows), self.cfg.num_future_frames, 72),
            human_relative,
            wrist,
        ).astype(get_global_dtype(), copy=False)

    def reference_qpos(self) -> np.ndarray:
        """Return the world-frame reference pose for playback ghost rendering.

        Rows are ``[anchor_pos(3), anchor_quat wxyz(4), joint_pos]`` for the
        most recently refreshed reference frame, with joint columns in
        ``loader.joint_names`` order (= backend actuator order).
        """
        return np.concatenate(
            (
                self.motion_data.body_pos_w[:, self.anchor_body_idx],
                self.motion_data.body_quat_w[:, self.anchor_body_idx],
                self.motion_data.joint_pos,
            ),
            axis=-1,
        ).astype(np.float32, copy=False)

    def _build_sampler(self, loader: _SonicMotionLoader) -> MotionSampler:
        task_cfg = self.cfg
        return MotionSampler(
            loader,
            task_cfg.sampling_mode,
            self.num_envs,
            # Runtime test loaders and legacy packed stores may omit fps;
            # SONIC references are standardized at 50 Hz.
            bin_count=int(loader.num_frames // getattr(loader, "fps", 50)) + 1,
            adaptive_lambda=task_cfg.adaptive_lambda,
            adaptive_kernel_size=task_cfg.adaptive_kernel_size,
            adaptive_uniform_ratio=task_cfg.adaptive_uniform_ratio,
            adaptive_alpha=task_cfg.adaptive_alpha,
            adaptive_failure_stat=task_cfg.adaptive_failure_stat,
            adaptive_failure_prior=task_cfg.adaptive_failure_prior,
            adaptive_pre_failure_window=task_cfg.adaptive_pre_failure_window,
            adaptive_failure_rate_max_over_mean=task_cfg.adaptive_failure_rate_max_over_mean,
            adaptive_sampling_update_interval=task_cfg.adaptive_sampling_update_interval,
            adaptive_attribution=task_cfg.adaptive_attribution,
            adaptive_max_prob_per_motion=task_cfg.adaptive_max_prob_per_motion,
            start_ratio=task_cfg.sampling_start_ratio,
            rng=self._env.rng,
        )

    def rotate_clip_subset(self) -> None:
        """Swap in a fresh uniformly-sampled clip working set (wrap-style).

        Rebuilds the loader, sampler, and derived reference features for the
        new subset, then teleports every environment onto it through
        ``reset_reference`` — the same mid-episode teleport the wrap mode
        applies on clip exhaustion.  Adaptive-sampling statistics restart
        from their prior on each rotation.  No-op without an active subset.
        """
        if self._clip_subset_size is None or self._packed_backend is None:
            return
        task_cfg = self.cfg
        total_clips = packed_store_clip_count(task_cfg.motion_store_file)
        loader = SonicPackedMotionLoader(
            task_cfg.motion_store_file,
            backend=self._packed_backend,
            body_names=task_cfg.body_names,
            clip_indices=np.asarray(
                self._env.rng.choice(total_clips, size=self._clip_subset_size, replace=False)
            ),
        )
        self.loader = loader
        self.sampler = self._build_sampler(loader)
        self._next_reset_clip_indices.fill(-1)
        self._g1_command_cache = None
        self._g1_command_cache_step = -1
        self._init_reference_feature_cache(loader)
        self._clip_rotations += 1
        self.reset_reference(np.arange(self.num_envs, dtype=np.int32))

    def _init_reference_feature_cache(self, loader: _SonicMotionLoader) -> None:
        """Materialize per-frame reference features on the cold path.

        Following the upstream motion-lib contract, every state-independent
        quantity (policy-ordered joint columns, wrist columns, SMPL root-local
        joints) is precomputed here once; the hot path keeps only the window
        gathers and the two live-state quaternion products that depend on the
        robot's current root orientation.
        """
        self._ref_joint_pos_policy = loader.joint_pos[:, self._policy_from_model]
        self._ref_joint_vel_policy = loader.joint_vel[:, self._policy_from_model]
        self._ref_wrist_policy = loader.joint_pos[:, self._policy_from_model][
            :, _WRIST_POLICY_INDICES
        ]
        human_local = np.empty((loader.num_frames, 24, 3), dtype=loader.smpl_joints.dtype)
        chunk_frames = 1 << 16
        for start in range(0, loader.num_frames, chunk_frames):
            stop = min(start + chunk_frames, loader.num_frames)
            human_local[start:stop] = np_quat_apply_batched(
                np_quat_conjugate_batched(loader.smpl_root_quat[start:stop, None]),
                loader.smpl_joints[start:stop],
            )
        self._ref_smpl_human_local = human_local

    def _gather_motion(self, frames: np.ndarray) -> MotionData:
        """Gather motion rows into the reusable scratch buffer.

        The returned fields are views into ``self._motion_scratch`` and stay
        valid until the next gather; callers must copy anything they keep.
        """
        count = len(frames)
        out = MotionData(
            joint_pos=self._motion_scratch.joint_pos[:count],
            joint_vel=self._motion_scratch.joint_vel[:count],
            body_pos_w=self._motion_scratch.body_pos_w[:count],
            body_quat_w=self._motion_scratch.body_quat_w[:count],
            body_lin_vel_w=self._motion_scratch.body_lin_vel_w[:count],
            body_ang_vel_w=self._motion_scratch.body_ang_vel_w[:count],
        )
        self.loader.get_motion_at_frame(frames, out=out)
        return out

    def _scatter_motion(self, rows: np.ndarray, motion: MotionData) -> None:
        for name in (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ):
            getattr(self.motion_data, name)[rows] = getattr(motion, name)

    def _refresh_motion(self, env_ids: np.ndarray | None = None) -> None:
        rows = np.arange(self.num_envs, dtype=np.intp) if env_ids is None else env_ids
        self._scatter_motion(rows, self._gather_motion(self.sampler.current_frames[rows]))

    def _sample_encoder(self, env_ids: np.ndarray) -> None:
        mode = self.cfg.encoder_sampling
        if mode == "g1":
            use_smpl = np.zeros(len(env_ids), dtype=bool)
        elif mode == "smpl":
            use_smpl = np.ones(len(env_ids), dtype=bool)
        else:
            use_smpl = self._env.rng.random(len(env_ids)) < 0.5
        # Keep the selector strictly one-hot.  The previous implementation
        # initialized every row to G1 ``[1, 0]`` and only enabled the SMPL bit,
        # producing ``[1, 1]`` for SMPL samples and failing the SONIC backbone
        # contract during playback/training.
        self.encoder_index[env_ids] = (1.0, 0.0)
        smpl_ids = env_ids[use_smpl]
        self.encoder_index[smpl_ids] = (0.0, 1.0)

    def _update_metrics(self, env_ids: np.ndarray | None = None) -> None:
        del env_ids
        return

    def get_diagnostics(
        self, *, include_histograms: bool = False
    ) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        sampler = self.sampler
        histograms = (
            {
                "sampling_failure_rate": sampler.bin_failure_rate.copy(),
                "sampling_probability": sampler._sampling_probs.copy(),
                "sampling_visit_count": sampler.bin_visit_count.copy(),
                "sampling_failure_count": sampler.bin_failed_count.copy(),
            }
            if include_histograms
            else {}
        )
        return (
            {
                "sampling_entropy": sampler.sampling_entropy,
                "sampling_top1_prob": sampler.sampling_top1_prob,
                "sampling_top1_clip_prob": sampler.sampling_top1_clip_prob,
                "sampling_top1_bin": sampler.sampling_top1_bin,
                "sampling_effective_bin_count": sampler.sampling_effective_bin_count,
                "sampling_visited_bin_fraction": sampler.sampling_visited_bin_fraction,
                "sampling_failure_rate_mean": sampler.sampling_failure_rate_mean,
                "sampling_failure_rate_max": sampler.sampling_failure_rate_max,
                "sampling_failure_count_total": sampler.sampling_failure_count_total,
                "sampling_visit_count_total": sampler.sampling_visit_count_total,
                "sampling_uniform_mass_actual": sampler.sampling_uniform_mass_actual,
                "motion_subset_clip_count": float(self.loader.num_clips),
                "clip_rotations": float(self._clip_rotations),
            },
            histograms,
        )

    def _resample_command(self, env_ids: np.ndarray) -> None:
        del env_ids

    def _update_command(self, env_ids: np.ndarray | None) -> None:
        rows = slice(None) if env_ids is None else env_ids
        if env_ids is None:
            # Attribute failures to the frames that produced this transition,
            # then advance the reference for the next observation.
            termination_manager = self._env.termination_manager
            self.sampler.update_failure_stats(
                termination_manager.terminated,
                episode_done=self._env.reset_buf,
            )
            active_ids = np.flatnonzero(~self._env.reset_buf).astype(np.int32, copy=False)
            self.advance_reference(active_ids)
            if self._clip_rotation_interval is not None:
                self._rotation_step_counter += 1
                if self._rotation_step_counter >= self._clip_rotation_interval:
                    self._rotation_step_counter = 0
                    # Rotate after attribution/advancement so every frame
                    # index stays legal on the loader it was computed
                    # against; the teleport mirrors the wrap-mode reset and
                    # leaves the post-rotation observations on the fresh
                    # subset's start frames.
                    self.rotate_clip_subset()
        self._command[rows, 0] = self.sampler.current_frames[rows]


def reset_sonic_reference(env, env_ids: np.ndarray | None) -> None:
    _motion_command(env).reset_reference(mdp.resolve_env_ids(env, env_ids))


class SonicActuatorDynamics(ManagerTermBase):
    """Cold-bind and stage the released policy-order actuator dynamics."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self._entity = env.scene["robot"]
        actuator_ids, actuator_names = self._entity.find_actuators(
            G1_SONIC_JOINTS, preserve_order=True
        )
        joint_ids, joint_names = self._entity.find_joints(G1_SONIC_JOINTS, preserve_order=True)
        if tuple(actuator_names) != G1_SONIC_JOINTS or tuple(joint_names) != G1_SONIC_JOINTS:
            raise ValueError("SONIC dynamics targets do not match the released policy order")
        self._actuator_ids, _, _ = self._entity.bind_actuator_gain_write(
            actuator_ids,
            term_name="sonic_actuator_gains",
        )
        self._joint_ids, _ = self._entity.bind_joint_armature_write(
            joint_ids,
            term_name="sonic_joint_armature",
        )
        self._kp = np.asarray(
            [_SONIC_ACTUATOR_PARAMETERS[name][0] for name in G1_SONIC_JOINTS],
            dtype=np.float32,
        )
        self._kd = np.asarray(
            [_SONIC_ACTUATOR_PARAMETERS[name][1] for name in G1_SONIC_JOINTS],
            dtype=np.float32,
        )
        self._armature = np.asarray(
            [_SONIC_ACTUATOR_PARAMETERS[name][3] for name in G1_SONIC_JOINTS],
            dtype=np.float32,
        )

    def __call__(self, env: ManagerBasedRlEnv, env_ids: np.ndarray | None) -> None:
        ids = mdp.resolve_env_ids(env, env_ids)
        self._entity.write_actuator_gains_to_sim(
            np.broadcast_to(self._kp, (len(ids), len(self._kp))),
            np.broadcast_to(self._kd, (len(ids), len(self._kd))),
            actuator_ids=self._actuator_ids,
            env_ids=ids,
            term_name="sonic_actuator_gains",
        )
        self._entity.write_joint_armature_to_sim(
            np.broadcast_to(self._armature, (len(ids), len(self._armature))),
            joint_ids=self._joint_ids,
            env_ids=ids,
            term_name="sonic_joint_armature",
        )


@dataclass(kw_only=True)
class SonicJointPositionActionCfg(JointPositionActionCfg):
    """SONIC policy-order position action with scalar or per-joint scaling."""

    scale: float | list[float] | tuple[float, ...] = G1_SONIC_ACTION_SCALE
    simulate_action_latency: bool = False

    def build(self, env) -> SonicJointPositionAction:
        return SonicJointPositionAction(self, env)


class SonicJointPositionAction(JointPositionAction):
    """Released clip/scale/default/latency action transform."""

    def __init__(self, cfg: SonicJointPositionActionCfg, env):
        configured_scale = cfg.scale
        # BaseAction resolves only scalar/dict affine scales. Initialize it
        # with a neutral scalar, then resolve SONIC's optional vector scale
        # below in this owner layer.
        super().__init__(replace(cfg, scale=1.0), env)
        self.cfg = cfg
        target_ids, target_names = self._entity.find_joints(G1_SONIC_JOINTS, preserve_order=True)
        if tuple(target_names) != G1_SONIC_JOINTS:
            raise ValueError("SONIC action targets do not match the released policy order")
        self._target_ids = np.asarray(target_ids, dtype=np.intp)
        self._target_ids.setflags(write=False)
        self._target_names = list(target_names)
        if cfg.use_default_offset:
            self._offset = self._entity.data.default_joint_pos[:, self._target_ids].copy()
        if isinstance(configured_scale, (int, float)) and not isinstance(configured_scale, bool):
            self._scale = _sonic_action_scale(float(configured_scale)).reshape(1, self.action_dim)
        else:
            scale = np.asarray(configured_scale, dtype=np.float32)
            if scale.shape != (self.action_dim,):
                raise ValueError(
                    f"SONIC action scale vector must have shape ({self.action_dim},), "
                    f"got {scale.shape}"
                )
            if not np.isfinite(scale).all() or np.any(scale <= 0.0):
                raise ValueError("SONIC action scale vector must contain positive finite values")
            self._scale = scale.reshape(1, self.action_dim)
        self.joint_velocity_before_action = np.zeros_like(self._raw_actions)

    def process_actions(self, actions: np.ndarray) -> None:
        self.joint_velocity_before_action[:] = self._entity.data.joint_vel[:, self._target_ids]
        self._raw_actions[:] = actions
        cfg = cast(SonicJointPositionActionCfg, self.cfg)
        executed = self._env.action_manager.prev_action if cfg.simulate_action_latency else actions
        np.multiply(executed, self._scale, out=self._processed_actions)
        np.add(self._processed_actions, self._offset, out=self._processed_actions)
        limits = np.asarray(self._entity.data.soft_joint_pos_limits)
        if limits.ndim == 3:
            limits = limits[:, self._target_ids]
        elif limits.ndim == 2:
            limits = limits[self._target_ids]
        else:
            raise ValueError(
                "SONIC soft_joint_pos_limits must have shape (joints, 2) or (envs, joints, 2)"
            )
        action_dim = self._raw_actions.shape[-1]
        if limits.shape[-2:] != (action_dim, 2):
            raise ValueError(
                "SONIC soft_joint_pos_limits do not match the policy joint dimension: "
                f"got {limits.shape}"
            )
        np.clip(
            self._processed_actions, limits[..., 0], limits[..., 1], out=self._processed_actions
        )

    def apply_actions(self) -> None:
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        np.subtract(self._processed_actions, encoder_bias, out=self._target)
        limits = np.asarray(self._entity.data.soft_joint_pos_limits)
        if limits.ndim == 3:
            limits = limits[:, self._target_ids]
        elif limits.ndim == 2:
            limits = limits[self._target_ids]
        else:
            raise ValueError(
                "SONIC soft_joint_pos_limits must have shape (joints, 2) or (envs, joints, 2)"
            )
        np.clip(self._target, limits[..., 0], limits[..., 1], out=self._target)
        self._entity.set_joint_position_target(self._target, joint_ids=self._target_ids)

    def resolved_action_contract(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the (default offset, per-joint scale) pair in policy order.

        Reference-mode actor BC inverts this contract to translate reference
        joint positions into normalized policy actions.
        """
        offset = np.asarray(self._offset, dtype=np.float32)
        scale = np.asarray(self._scale, dtype=np.float32)
        return offset.reshape(-1).copy(), scale.reshape(-1).copy()


class _HistoryObservation:
    # Released checkpoint field slices of the proprioception frame; the output
    # layout concatenates these blocks in order, each block frame-major.
    _history_slices: tuple[tuple[int, int], ...] = ()

    def __init__(self, cfg: ObservationTermCfg, env, width: int):
        self.cfg = cfg
        self._env = env
        policy_joint_ids, policy_joint_names = env.scene["robot"].find_joints(
            G1_SONIC_JOINTS, preserve_order=True
        )
        if tuple(policy_joint_names) != G1_SONIC_JOINTS:
            raise ValueError("SONIC observations do not match the released policy joint order")
        self._policy_joint_ids = np.asarray(policy_joint_ids, dtype=np.intp)
        self._policy_joint_ids.setflags(write=False)
        history_length = int(getattr(cfg, "sonic_history_length", _DEFAULT_REFERENCE_FRAMES))
        if history_length <= 0:
            raise ValueError("SonicObservationTermCfg sonic_history_length must be positive")
        self._history = np.zeros(
            (env.num_envs, history_length, _PROPRIO_FRAME_DIM), dtype=np.float32
        )
        self._output = np.zeros((env.num_envs, width), dtype=get_global_dtype())
        self._reset_pending = np.zeros(env.num_envs, dtype=bool)
        # Physical ring head over the frame axis; the per-step push writes the
        # newest frame here and advances, replacing the legacy full-buffer roll.
        self._head = 0
        widths = [end - start for start, end in self._history_slices]
        block_starts = [0]
        for slice_width in widths[:-1]:
            block_starts.append(block_starts[-1] + slice_width * history_length)
        self._slice_starts = np.asarray(block_starts, dtype=np.intp)
        self._slice_widths = np.asarray(widths, dtype=np.intp)
        self._slice_columns = np.asarray(
            [start for start, _ in self._history_slices], dtype=np.intp
        )
        configure_motion_kernel_runtime()

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        self._reset_pending[slice(None) if env_ids is None else env_ids] = True

    def _rows(self) -> tuple[np.ndarray, bool]:
        pending = np.flatnonzero(self._reset_pending)
        if len(pending):
            return pending, True
        return np.arange(self._env.num_envs, dtype=np.intp), False

    def _push(self, rows: np.ndarray, values: np.ndarray, reset: bool, out_offset: int) -> None:
        """Push the newest proprioception frame and assemble ``self._output``.

        The fused kernel ring-writes ``values`` (all environments on the
        per-step call, only ``rows`` on the reset call) and writes the
        history block of the output for ``rows`` in logical oldest-to-newest
        order, replacing the legacy roll + slice-reshape + concatenate chain.
        """
        self._head = push_history_and_assemble_kernel(
            self._history,
            values,
            rows.astype(np.intp, copy=False),
            reset,
            self._head,
            self._output,
            out_offset,
            self._slice_starts,
            self._slice_widths,
            self._slice_columns,
        )
        if reset:
            self._reset_pending[rows] = False


@dataclass(kw_only=True)
class SonicObservationTermCfg(ObservationTermCfg):
    sonic_noise: SonicNoiseConfig = field(default_factory=SonicNoiseConfig)
    sonic_history_length: int = _DEFAULT_REFERENCE_FRAMES

    def __post_init__(self) -> None:
        if (
            isinstance(self.sonic_history_length, bool)
            or not isinstance(self.sonic_history_length, int)
            or self.sonic_history_length <= 0
        ):
            raise ValueError("SonicObservationTermCfg sonic_history_length must be positive")


class SonicActorObservation(_HistoryObservation):
    # Motrix PPO ``local_dir_hist`` order: angular velocity, joint position,
    # joint velocity, action, gravity (gravity last).
    _history_slices = ((0, 3), (3, 32), (32, 61), (61, 90), (90, 93))

    def __init__(self, cfg: ObservationTermCfg, env):
        if not isinstance(cfg, SonicObservationTermCfg):
            raise TypeError("SonicActorObservation requires SonicObservationTermCfg")
        if (
            isinstance(cfg.sonic_history_length, bool)
            or not isinstance(cfg.sonic_history_length, int)
            or cfg.sonic_history_length <= 0
        ):
            raise ValueError("SonicObservationTermCfg sonic_history_length must be positive")
        super().__init__(cfg, env, _PROPRIO_FRAME_DIM * cfg.sonic_history_length)
        self._sonic_cfg = cfg
        seed = cfg.sonic_noise.seed
        self._noise_rng = None if seed is None else np.random.default_rng(seed)

    def __call__(self, env) -> np.ndarray:
        rows, reset = self._rows()
        robot = env.scene["robot"].data
        _, gyro, gravity = _root_local_proprioception(
            robot.root_link_quat_w[rows],
            robot.root_link_lin_vel_w[rows],
            robot.root_link_ang_vel_w[rows],
        )
        joint_pos = (
            robot.joint_pos[rows][:, self._policy_joint_ids]
            - robot.default_joint_pos[rows][:, self._policy_joint_ids]
        )
        joint_vel = robot.joint_vel[rows][:, self._policy_joint_ids]
        action = env.action_manager.action[rows]
        noise = self._sonic_cfg.sonic_noise
        values = np.concatenate(
            (
                self._add_noise(gyro, noise.scale_gyro),
                self._add_noise(joint_pos, noise.scale_joint_angle),
                self._add_noise(joint_vel, noise.scale_joint_vel),
                action,
                # Gravity is last to match ``local_dir_hist`` order.
                self._add_noise(gravity, noise.scale_gravity),
            ),
            axis=-1,
            dtype=np.float32,
        )
        # The fused kernel ring-writes the newest frame and assembles the
        # whole (history-major) actor output in one pass; the per-frame value
        # order stays base_ang_vel, joint_pos, joint_vel, actions, gravity.
        self._push(rows, values, reset, out_offset=0)
        return self._output

    def _add_noise(self, value: np.ndarray, scale: float) -> np.ndarray:
        level = float(self._sonic_cfg.sonic_noise.level)
        if level <= 0.0:
            return value
        if self._noise_rng is None:
            sample = np.random.uniform(-1.0, 1.0, value.shape).astype(value.dtype)
        else:
            sample = self._noise_rng.uniform(-1.0, 1.0, value.shape).astype(value.dtype)
        return value + sample * level * scale


class SonicCriticObservation(_HistoryObservation):
    _history_slices = ((0, 3), (3, 6), (6, 35), (35, 64), (64, 93))

    def __init__(self, cfg: ObservationTermCfg, env):
        history_length = int(getattr(cfg, "sonic_history_length", _DEFAULT_REFERENCE_FRAMES))
        if history_length <= 0:
            raise ValueError("SonicObservationTermCfg sonic_history_length must be positive")
        # command (58) and proprioception (93) are both history-major; the
        # remaining anchor/body terms contribute a fixed 135 dimensions.
        super().__init__(cfg, env, 135 + history_length * (58 + _PROPRIO_FRAME_DIM))
        self._body_count = len(_motion_command(env).cfg.body_names)

    def __call__(self, env) -> np.ndarray:
        rows, reset = self._rows()
        robot = env.scene["robot"].data
        motion = _motion_command(env)
        root_quat = robot.root_link_quat_w[rows]
        linvel, gyro, _ = _root_local_proprioception(
            root_quat,
            robot.root_link_lin_vel_w[rows],
            robot.root_link_ang_vel_w[rows],
        )
        joint_pos = (
            robot.joint_pos[rows][:, self._policy_joint_ids]
            - robot.default_joint_pos[rows][:, self._policy_joint_ids]
        )
        joint_vel = robot.joint_vel[rows][:, self._policy_joint_ids]
        action = env.action_manager.action[rows]
        command = motion.g1_command(rows)
        reference = motion.motion_data
        anchor_idx = motion.anchor_body_idx
        anchor_pos = np.empty((len(rows), 3), dtype=np.float32)
        anchor_ori = np.empty((len(rows), 6), dtype=np.float32)
        root_pos = robot.root_link_pos_w[rows]
        anchor_delta = reference.body_pos_w[rows, anchor_idx] - root_pos
        anchor_pos[...] = np_quat_apply_inverse_batched(root_quat, anchor_delta)
        anchor_rel = np_quat_mul_batched(
            np_quat_conjugate_batched(root_quat), reference.body_quat_w[rows, anchor_idx]
        )
        w, x, y, z = (anchor_rel[:, i] for i in range(4))
        anchor_ori[...] = np.stack(
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
        body_pos_b = np.empty((len(rows), self._body_count, 3), dtype=np.float32)
        body_ori_b = np.empty((len(rows), self._body_count, 6), dtype=np.float32)
        body_vec_error = np.empty_like(body_pos_b)
        _write_body_pos_in_anchor_frame(
            root_pos,
            root_quat,
            robot.body_link_pos_w[rows],
            body_pos_b,
            body_vec_error=body_vec_error,
        )
        _write_body_ori6_in_anchor_frame(root_quat, robot.body_link_quat_w[rows], body_ori_b)
        values = np.concatenate(
            (linvel, gyro, joint_pos, joint_vel, action), axis=-1, dtype=np.float32
        )
        # The command/anchor/body block leads; the fused kernel fills the
        # trailing history block in one pass.
        history_offset = self._output.shape[1] - self._history.shape[1] * _PROPRIO_FRAME_DIM
        self._output[rows, :history_offset] = np.concatenate(
            (
                command.reshape(len(rows), -1),
                anchor_pos,
                anchor_ori,
                body_pos_b.reshape(len(rows), -1),
                body_ori_b.reshape(len(rows), -1),
            ),
            axis=-1,
        )
        self._push(rows, values, reset, out_offset=history_offset)
        return self._output


class _ResetScopedReferenceObservation:
    """Row-scoped observation base for the SONIC reference terms.

    Mirrors the ``_HistoryObservation`` reset contract: the observation
    manager stamps ``reset(env_ids)`` before the reset-path rebuild, so the
    term recomputes only the reset rows into a persistent full-batch buffer
    instead of re-evaluating the (expensive) reference terms for every
    environment on every reset.
    """

    def __init__(self, cfg: ObservationTermCfg, env) -> None:
        self.cfg = cfg
        self._env = env
        self._output: np.ndarray | None = None
        self._reset_pending = np.zeros(env.num_envs, dtype=bool)

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        self._reset_pending[slice(None) if env_ids is None else env_ids] = True

    def _rows(self) -> tuple[np.ndarray, bool]:
        pending = np.flatnonzero(self._reset_pending)
        if len(pending):
            return pending, True
        return np.arange(self._env.num_envs, dtype=np.intp), False

    def _compute(self, rows: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, env) -> np.ndarray:
        rows, reset = self._rows()
        if not reset:
            # The per-step path returns the freshly computed array directly;
            # the persistent buffer only exists to serve reset-path calls,
            # which must return a full-batch array while recomputing a subset.
            return self._compute(rows)
        result = self._compute(rows)
        if self._output is None or self._output.shape[1:] != result.shape[1:]:
            self._output = np.zeros(
                (self._env.num_envs, *result.shape[1:]), dtype=get_global_dtype()
            )
        self._output[rows] = result
        self._reset_pending[rows] = False
        return self._output


class SonicG1ReferenceObservation(_ResetScopedReferenceObservation):
    def _compute(self, rows: np.ndarray) -> np.ndarray:
        return _motion_command(self._env).g1_reference(rows)


class SonicSmplReferenceObservation(_ResetScopedReferenceObservation):
    def _compute(self, rows: np.ndarray) -> np.ndarray:
        return _motion_command(self._env).smpl_reference(rows)


def sonic_g1_reference(env) -> np.ndarray:
    return _motion_command(env).g1_reference()


def sonic_reference_qpos(env) -> np.ndarray:
    """World-frame reference qpos rows for the playback ghost overlay."""
    return _motion_command(env).reference_qpos()


def sonic_reference_joint_names(env) -> tuple[str, ...]:
    """Joint column order of :func:`sonic_reference_qpos` rows."""
    return tuple(_motion_command(env).loader.joint_names)


def sonic_smpl_reference(env) -> np.ndarray:
    return _motion_command(env).smpl_reference()


def sonic_encoder_index(env) -> np.ndarray:
    return _motion_command(env).encoder_index


def sonic_clip_end(env) -> np.ndarray:
    return _motion_command(env).clip_end.copy()


def sonic_reward(env, term_name: str, std: float | None = None) -> np.ndarray:
    motion = _motion_command(env)
    std_attribute = _SONIC_REWARD_STD_ATTRIBUTES.get(term_name)
    if std_attribute is not None:
        if std is None or not np.isfinite(std) or std <= 0.0:
            raise ValueError(f"SONIC reward term {term_name!r} requires a positive finite std")
        setattr(motion.reward_context.reward_config, std_attribute, float(std))
    method_name = _SONIC_CUSTOM_REWARD_METHODS.get(term_name)
    if method_name is not None:
        return getattr(motion, method_name)()
    try:
        reward_fn = _SONIC_COMMON_REWARD_FUNCTIONS[term_name]
    except KeyError as exc:
        raise ValueError(f"Unknown SONIC reward term {term_name!r}") from exc
    return reward_fn(motion.reward_context)


def sonic_anchor_pos_z_termination(env) -> np.ndarray:
    motion = _motion_command(env)
    cfg = motion.cfg
    motion._termination_threshold.fill(cfg.anchor_pos_z_threshold)
    motion._termination_threshold[motion._low_reference] = cfg.low_reference_anchor_pos_z_threshold
    np.subtract(
        motion.motion_data.body_pos_w[:, motion.anchor_body_idx, 2],
        motion._robot_body_pos_w[:, motion.anchor_body_idx, 2],
        out=motion._termination_error,
    )
    np.abs(motion._termination_error, out=motion._termination_error)
    np.greater(
        motion._termination_error,
        motion._termination_threshold,
        out=motion._termination_reason_mask[:, 0],
    )
    return motion._termination_reason_mask[:, 0]


def sonic_anchor_ori_termination(env) -> np.ndarray:
    motion = _motion_command(env)
    orientation_error = np_quat_error_magnitude_squared_batched(
        motion._robot_body_quat_w[:, motion.anchor_body_idx],
        motion.motion_data.body_quat_w[:, motion.anchor_body_idx],
    )
    np.greater(
        orientation_error,
        motion.cfg.anchor_ori_threshold,
        out=motion._termination_reason_mask[:, 1],
    )
    return motion._termination_reason_mask[:, 1]


def sonic_ee_body_pos_z_termination(env) -> np.ndarray:
    motion = _motion_command(env)
    cfg = motion.cfg
    motion._termination_threshold.fill(cfg.ee_body_pos_z_threshold)
    motion._termination_threshold[motion._low_reference] = cfg.low_reference_ee_body_pos_z_threshold
    np.subtract(
        motion.body_pos_relative_w[:, motion.ee_body_indices, 2],
        motion._robot_body_pos_w[:, motion.ee_body_indices, 2],
        out=motion._ee_termination_error,
    )
    np.abs(motion._ee_termination_error, out=motion._ee_termination_error)
    np.greater(
        motion._ee_termination_error,
        motion._termination_threshold[:, None],
        out=motion._ee_termination_exceeded,
    )
    np.any(
        motion._ee_termination_exceeded,
        axis=-1,
        out=motion._termination_reason_mask[:, 2],
    )
    return motion._termination_reason_mask[:, 2]


def sonic_feet_pos_termination(env) -> np.ndarray:
    motion = _motion_command(env)
    np.subtract(
        motion.body_pos_relative_w[:, motion._foot_body_indices],
        motion._robot_body_pos_w[:, motion._foot_body_indices],
        out=motion._foot_termination_error,
    )
    np.square(motion._foot_termination_error, out=motion._foot_termination_error)
    np.sum(motion._foot_termination_error, axis=-1, out=motion._foot_termination_norm)
    np.sqrt(motion._foot_termination_norm, out=motion._foot_termination_norm)
    np.greater(
        motion._foot_termination_norm,
        motion.cfg.foot_pos_threshold,
        out=motion._foot_termination_exceeded,
    )
    np.any(
        motion._foot_termination_exceeded,
        axis=-1,
        out=motion._termination_reason_mask[:, 3],
    )
    return motion._termination_reason_mask[:, 3]


def make_g1_sonic_manager_cfg() -> ManagerBasedRlEnvCfg:
    """Return the empty manager config; Hydra owns all SONIC term declarations."""
    policy: dict[str, ObservationTermCfg | None] = {
        "obs": ObservationTermCfg(func=SonicActorObservation),
        "g1_reference": ObservationTermCfg(func=SonicG1ReferenceObservation),
        "smpl_reference": ObservationTermCfg(func=SonicSmplReferenceObservation),
        "encoder_index": ObservationTermCfg(func=sonic_encoder_index),
    }
    return ManagerBasedRlEnvCfg(
        observations={
            "policy": ObservationGroupCfg(terms=policy),
            "critic": ObservationGroupCfg(
                terms={"obs": ObservationTermCfg(func=SonicCriticObservation)}
            ),
        },
        events={
            "reset_reference": EventTermCfg(func=reset_sonic_reference, mode="reset"),
            "actuator_gains": EventTermCfg(func=SonicActuatorDynamics, mode="reset"),
        },
        terminations={
            "anchor_pos_z": TerminationTermCfg(func=sonic_anchor_pos_z_termination),
            "anchor_ori": TerminationTermCfg(func=sonic_anchor_ori_termination),
            "ee_body_pos_z": TerminationTermCfg(func=sonic_ee_body_pos_z_termination),
            "feet_pos": TerminationTermCfg(func=sonic_feet_pos_termination),
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "clip_end": TerminationTermCfg(func=sonic_clip_end, time_out=True),
        },
        critic_observation_group="critic",
        scale_rewards_by_dt=True,
    )


# Compatibility import for downstream code; production ownership is the base config
# and YAML manager terms, not a task-specific Python subclass.
G1SonicManagerCfg = ManagerBasedRlEnvCfg


class G1SonicManagerEnv(ManagerBasedRlEnv):
    """Thin scheduler/bounds adapter; task computations remain manager-owned."""

    _cfg: ManagerBasedRlEnvCfg

    def __init__(self, cfg, backend, num_envs):
        super().__init__(cfg, backend, num_envs)
        self._clipped_actions = np.empty((num_envs, 29), dtype=np.float32)
        self._termination_reason_mask = np.zeros(
            (num_envs, len(SONIC_TERMINATION_REASON_NAMES)), dtype=np.bool_
        )

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(-20.0, 20.0, shape=(29,), dtype=np.float32)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        np.clip(actions, -20.0, 20.0, out=self._clipped_actions)
        return super().apply_action(self._clipped_actions, state)

    def update_state(self, state: NpEnvState) -> NpEnvState:
        motion = _motion_command(self)
        motion.refresh_transition_context()
        state = super().update_state(state)
        # Keep a stable diagnostic schema even when an owner disables an
        # optional termination (for example playback removes the generic
        # 10-second ``time_out`` term).  Disabled terms have an all-false
        # column instead of being queried from TerminationManager.
        active_terms = self.termination_manager.active_terms
        self._termination_reason_mask.fill(False)
        for column, name in enumerate(SONIC_TERMINATION_REASON_NAMES):
            if name in active_terms:
                self._termination_reason_mask[:, column] = self.termination_manager.get_term(name)
        state.info["termination_reason_mask"] = self._termination_reason_mask.copy()
        state.info["termination_reason_names"] = SONIC_TERMINATION_REASON_NAMES
        return state


def _make_sonic_env(cfg, *, num_envs=1, backend_type="mujoco"):
    if not isinstance(cfg, ManagerBasedRlEnvCfg):
        raise TypeError("G1SonicManager requires ManagerBasedRlEnvCfg")
    motion_cfg = cfg.commands.get("motion")
    if not isinstance(motion_cfg, SonicMotionCommandCfg):
        raise TypeError("G1SonicManager requires commands['motion'] to be SonicMotionCommandCfg")
    if cfg.scene is None:
        raise ValueError("G1SonicManager requires a scene configuration")
    backend = create_backend(
        backend_type,
        cfg.scene,
        num_envs,
        cfg.sim_dt,
        base_name=motion_cfg.anchor_body_name,
        add_body_sensors=True,
        **env_backend_kwargs(cfg),
    )
    return G1SonicManagerEnv(cfg, backend, num_envs)


registry.register_env_config("G1SonicManager", make_g1_sonic_manager_cfg)
registry.register_env("G1SonicManager", _make_sonic_env, sim_backend="mujoco")
registry.register_env("G1SonicManager", _make_sonic_env, sim_backend="motrix")

__all__ = [
    "G1_SONIC_ACTION_SCALE",
    "_sonic_action_scale",
    "_sonic_policy_action_scale",
    "G1_SONIC_BODY_NAMES",
    "G1_SONIC_JOINTS",
    "SONIC_TERMINATION_REASON_NAMES",
    "G1SonicManagerCfg",
    "G1SonicManagerEnv",
    "SonicActorObservation",
    "SonicCriticObservation",
    "SonicJointPositionAction",
    "SonicJointPositionActionCfg",
    "SonicMotionCommand",
    "SonicMotionCommandCfg",
    "SonicMotionCommandParamsCfg",
    "SonicObservationTermCfg",
    "make_g1_sonic_manager_cfg",
]
