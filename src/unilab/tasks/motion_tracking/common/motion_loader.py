"""Shared motion loading and sampling for motion-tracking tasks."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from unilab.assets.hub import resolve_motion_files
from unilab.utils.rotation import np_quat_angular_velocity, np_quat_ensure_continuity


@dataclass
class MotionData:
    """Container for motion data at specific frame(s)."""

    joint_pos: np.ndarray  # (N, num_joints)
    joint_vel: np.ndarray  # (N, num_joints)
    body_pos_w: np.ndarray  # (N, num_bodies, 3)
    body_quat_w: np.ndarray  # (N, num_bodies, 4)
    body_lin_vel_w: np.ndarray  # (N, num_bodies, 3)
    body_ang_vel_w: np.ndarray  # (N, num_bodies, 3)


def quat_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions (wxyz format).

    The computation runs in the input dtype; pass float64 arrays for a
    float64 interpolation path.
    """
    # Ensure shortest path
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2
        dot = -dot

    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    # Compute angle
    theta = np.arccos(np.clip(dot, -1, 1))
    sin_theta = np.sin(theta)

    # Compute interpolation weights
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta

    return w1 * q1 + w2 * q2


@dataclass
class InterpolatedMotion:
    """Root/joint trajectory resampled to the output frame rate."""

    output_frames: int
    base_poss: np.ndarray  # (N, 3)
    base_rots: np.ndarray  # (N, 4), wxyz
    dof_poss: np.ndarray  # (N, num_joints)
    base_lin_vels: np.ndarray  # (N, 3)
    base_ang_vels: np.ndarray  # (N, 3)
    dof_vels: np.ndarray  # (N, num_joints)


def compute_motion_velocities(
    base_poss: np.ndarray,
    base_rots: np.ndarray,
    dof_poss: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numerically differentiate a trajectory into base/dof velocities."""
    base_lin_vels = np.gradient(base_poss, dt, axis=0)
    dof_vels = np.gradient(dof_poss, dt, axis=0)
    base_ang_vels = np_quat_angular_velocity(base_rots, dt)
    return base_lin_vels, base_ang_vels, dof_vels


def compute_frame_blend(
    times: np.ndarray, duration: float, input_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute frame indices and blend weights for interpolation."""
    phase = times / duration
    index_0 = np.floor(phase * (input_frames - 1)).astype(np.int32)
    index_1 = np.minimum(index_0 + 1, input_frames - 1)
    blend = phase * (input_frames - 1) - index_0
    return index_0, index_1, blend


def interpolate_motion(
    base_poss_input: np.ndarray,
    base_rots_input: np.ndarray,
    dof_poss_input: np.ndarray,
    *,
    input_fps: int,
    output_fps: int,
) -> InterpolatedMotion:
    """Resample a root+joint trajectory from ``input_fps`` to ``output_fps``.

    Positions and joint angles are linearly interpolated, root quaternions
    (wxyz) use slerp, and velocities are computed by numerical
    differentiation at the output rate.
    """
    input_dt = 1.0 / input_fps
    output_dt = 1.0 / output_fps
    input_frames = base_poss_input.shape[0]
    duration = (input_frames - 1) * input_dt

    times = np.arange(0, duration, output_dt, dtype=np.float32)
    output_frames = times.shape[0]
    index_0, index_1, blend = compute_frame_blend(times, duration, input_frames)

    # Linear interpolation for positions
    base_poss = (
        base_poss_input[index_0] * (1 - blend[:, None]) + base_poss_input[index_1] * blend[:, None]
    )

    # Spherical linear interpolation for quaternions
    base_rots = np.zeros((output_frames, 4), dtype=np.float32)
    for i in range(output_frames):
        base_rots[i] = quat_slerp(
            base_rots_input[index_0[i]], base_rots_input[index_1[i]], blend[i]
        )
    base_rots = np_quat_ensure_continuity(base_rots)

    # Linear interpolation for joint positions
    dof_poss = (
        dof_poss_input[index_0] * (1 - blend[:, None]) + dof_poss_input[index_1] * blend[:, None]
    )

    base_lin_vels, base_ang_vels, dof_vels = compute_motion_velocities(
        base_poss, base_rots, dof_poss, output_dt
    )
    return InterpolatedMotion(
        output_frames=output_frames,
        base_poss=base_poss,
        base_rots=base_rots,
        dof_poss=dof_poss,
        base_lin_vels=base_lin_vels,
        base_ang_vels=base_ang_vels,
        dof_vels=dof_vels,
    )


class MotionLoader:
    """Loads and provides access to motion data from NPZ files."""

    def __init__(
        self,
        motion_file: str | Sequence[str],
        body_indices: np.ndarray | None = None,
        target_joint_names: Sequence[str] | None = None,
    ):
        """Initialize motion loader.

        Args:
            motion_file: Path to one NPZ file, or a sequence of NPZ files
            body_indices: Optional indices into the NPZ body axis. The exported
                motion files currently keep MuJoCo body-id layout, so these
                indices are expected to follow that convention.
        """
        motion_file = resolve_motion_files(motion_file)
        if isinstance(motion_file, str) and Path(motion_file).is_dir():
            # A directory containing manifest.json is the versioned packed
            # contract.  Plain NPZ directories (for example robot_filtered)
            # remain a valid cold-path input for manager tasks that do not use
            # packed storage.
            root = Path(motion_file)
            if (root / "manifest.json").is_file():
                self._load_packed_store(motion_file, body_indices, target_joint_names)
                return
            motion_file = sorted(str(path) for path in root.glob("*.npz"))
            if not motion_file:
                raise FileNotFoundError(f"Motion directory contains no NPZ clips: {root}")
        self.motion_files = self._normalize_motion_files(motion_file)

        joint_pos_list: list[np.ndarray] = []
        joint_vel_list: list[np.ndarray] = []
        body_pos_list: list[np.ndarray] = []
        body_quat_list: list[np.ndarray] = []
        body_lin_vel_list: list[np.ndarray] = []
        body_ang_vel_list: list[np.ndarray] = []
        clip_lengths: list[int] = []

        self.fps = 0
        self.num_joints = 0
        self.num_bodies = 0

        for clip_idx, motion_path in enumerate(self.motion_files):
            with np.load(motion_path) as data:
                fps = int(np.asarray(data["fps"]).reshape(-1)[0])
                joint_pos = data["joint_pos"].astype(np.float32)
                joint_vel = data["joint_vel"].astype(np.float32)
                body_pos_w = data["body_pos_w"].astype(np.float32)
                body_quat_w = data["body_quat_w"].astype(np.float32)
                body_lin_vel_w = data["body_lin_vel_w"].astype(np.float32)
                body_ang_vel_w = data["body_ang_vel_w"].astype(np.float32)

            if body_indices is not None:
                body_pos_w = body_pos_w[:, body_indices]
                body_quat_w = body_quat_w[:, body_indices]
                body_lin_vel_w = body_lin_vel_w[:, body_indices]
                body_ang_vel_w = body_ang_vel_w[:, body_indices]

            num_frames = joint_pos.shape[0]
            if num_frames == 0:
                raise ValueError(f"Motion file '{motion_path}' contains no frames")
            if joint_vel.shape[0] != num_frames:
                raise ValueError(
                    f"Motion file '{motion_path}' has inconsistent frame counts between "
                    "'joint_pos' and 'joint_vel'"
                )
            for name, array in (
                ("body_pos_w", body_pos_w),
                ("body_quat_w", body_quat_w),
                ("body_lin_vel_w", body_lin_vel_w),
                ("body_ang_vel_w", body_ang_vel_w),
            ):
                if array.shape[0] != num_frames:
                    raise ValueError(
                        f"Motion file '{motion_path}' has inconsistent frame counts for '{name}'"
                    )

            if clip_idx == 0:
                self.fps = fps
                self.num_joints = joint_pos.shape[1]
                self.num_bodies = body_pos_w.shape[1]
            else:
                if fps != self.fps:
                    raise ValueError(
                        f"Motion file '{motion_path}' has fps={fps}, expected {self.fps}"
                    )
                if joint_pos.shape[1] != self.num_joints or joint_vel.shape[1] != self.num_joints:
                    raise ValueError(
                        f"Motion file '{motion_path}' has incompatible joint dimensions"
                    )
                if (
                    body_pos_w.shape[1] != self.num_bodies
                    or body_quat_w.shape[1] != self.num_bodies
                    or body_lin_vel_w.shape[1] != self.num_bodies
                    or body_ang_vel_w.shape[1] != self.num_bodies
                ):
                    raise ValueError(
                        f"Motion file '{motion_path}' has incompatible body dimensions"
                    )

            clip_lengths.append(num_frames)
            joint_pos_list.append(joint_pos)
            joint_vel_list.append(joint_vel)
            body_pos_list.append(body_pos_w)
            body_quat_list.append(body_quat_w)
            body_lin_vel_list.append(body_lin_vel_w)
            body_ang_vel_list.append(body_ang_vel_w)

        self.clip_lengths = np.asarray(clip_lengths, dtype=np.int32)
        self.num_clips = int(self.clip_lengths.shape[0])
        self.clip_offsets = np.zeros(self.num_clips, dtype=np.int32)
        if self.num_clips > 1:
            self.clip_offsets[1:] = np.cumsum(self.clip_lengths[:-1], dtype=np.int32)
        self.clip_end_frames = self.clip_offsets + self.clip_lengths - 1

        self.joint_pos = np.concatenate(joint_pos_list, axis=0)
        self.joint_vel = np.concatenate(joint_vel_list, axis=0)
        self.body_pos_w = np.concatenate(body_pos_list, axis=0)
        self.body_quat_w = np.concatenate(body_quat_list, axis=0)
        self.body_lin_vel_w = np.concatenate(body_lin_vel_list, axis=0)
        self.body_ang_vel_w = np.concatenate(body_ang_vel_list, axis=0)

        if target_joint_names is not None and self.joint_pos.shape[1] != len(target_joint_names):
            raise ValueError("NPZ motion joint width does not match target model")
        self.num_frames = int(self.joint_pos.shape[0])

    def _load_packed_store(
        self,
        root_path: str,
        body_indices: np.ndarray | None,
        target_joint_names: Sequence[str] | None,
    ) -> None:
        """Load a manifest-backed packed store on the cold path."""
        root = Path(root_path).resolve()
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid packed motion manifest: {root / 'manifest.json'}") from exc
        arrays = manifest.get("arrays")
        required = (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        )
        if not isinstance(arrays, dict) or any(name not in arrays for name in required):
            raise ValueError(f"Packed motion store is missing required arrays: {root}")

        def load_array(name: str) -> np.ndarray:
            spec = arrays[name]
            if not isinstance(spec, dict):
                raise ValueError(f"Invalid packed motion array specification for {name!r}")
            path = (root / str(spec.get("file", ""))).resolve()
            if root not in path.parents:
                raise ValueError(f"Packed motion array escapes store: {path}")
            try:
                value = np.load(path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise ValueError(f"Invalid packed motion array: {path}") from exc
            if value.dtype != np.float32:
                raise ValueError(f"Packed motion array {name!r} must be float32")
            return value

        loaded = {name: load_array(name) for name in required}
        if target_joint_names is not None:
            stored_names = tuple(str(name) for name in manifest.get("joint_names", ()))
            try:
                indices = np.asarray(
                    [stored_names.index(name) for name in target_joint_names], dtype=np.intp
                )
            except ValueError as exc:
                raise ValueError("Packed motion joint names do not match target model") from exc
            for name in ("joint_pos", "joint_vel"):
                loaded[name] = loaded[name][:, indices]
        frames = loaded["joint_pos"].shape[0]
        if any(value.shape[0] != frames for value in loaded.values()):
            raise ValueError("Packed motion arrays have inconsistent frame counts")
        if body_indices is not None:
            for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                loaded[name] = loaded[name][:, body_indices]
        self.fps = int(manifest.get("fps", 0))
        if self.fps <= 0:
            raise ValueError("Packed motion store must declare a positive fps")
        self.num_joints = int(loaded["joint_pos"].shape[1])
        self.num_bodies = int(loaded["body_pos_w"].shape[1])
        for name in required:
            setattr(self, name, loaded[name])
        lengths_path = root / str(manifest.get("clip_lengths_file", "clip_lengths.npy"))
        self.clip_lengths = np.asarray(np.load(lengths_path, allow_pickle=False), dtype=np.int32)
        self.num_clips = int(self.clip_lengths.size)
        self.clip_offsets = np.zeros(self.num_clips, dtype=np.int32)
        if self.num_clips > 1:
            self.clip_offsets[1:] = np.cumsum(self.clip_lengths[:-1], dtype=np.int32)
        self.clip_end_frames = self.clip_offsets + self.clip_lengths - 1
        self.num_frames = int(frames)
        self.motion_files = (str(root),)

    @staticmethod
    def _normalize_motion_files(motion_file: str | Sequence[str]) -> tuple[str, ...]:
        motion_files: tuple[str, ...]
        if isinstance(motion_file, str):
            motion_files = (motion_file,)
        elif isinstance(motion_file, Sequence):
            motion_files = tuple(motion_file)
        else:
            raise TypeError("motion_file must be a string path or a sequence of string paths")

        if not motion_files:
            raise ValueError("motion_file must contain at least one NPZ path")
        if any((not isinstance(path, str)) or (not path) for path in motion_files):
            raise ValueError("motion_file entries must be non-empty strings")
        return motion_files

    def get_clip_indices(self, frame_idx: np.ndarray) -> np.ndarray:
        """Map global frame indices to clip indices."""
        clip_indices = np.searchsorted(self.clip_offsets, frame_idx, side="right") - 1
        return np.asarray(clip_indices, dtype=np.int32)

    def make_motion_data_buffer(self, num_frames: int) -> MotionData:
        """Allocate a reusable ``MotionData`` buffer for frame-index gathers."""
        return MotionData(
            joint_pos=np.empty((num_frames, self.num_joints), dtype=self.joint_pos.dtype),
            joint_vel=np.empty((num_frames, self.num_joints), dtype=self.joint_vel.dtype),
            body_pos_w=np.empty((num_frames, self.num_bodies, 3), dtype=self.body_pos_w.dtype),
            body_quat_w=np.empty((num_frames, self.num_bodies, 4), dtype=self.body_quat_w.dtype),
            body_lin_vel_w=np.empty(
                (num_frames, self.num_bodies, 3), dtype=self.body_lin_vel_w.dtype
            ),
            body_ang_vel_w=np.empty(
                (num_frames, self.num_bodies, 3), dtype=self.body_ang_vel_w.dtype
            ),
        )

    def get_motion_at_frame(
        self, frame_idx: np.ndarray, out: MotionData | None = None
    ) -> MotionData:
        """Get motion data at specified frame indices.

        Args:
            frame_idx: Frame indices (N,)
            out: Optional reusable output buffer.

        Returns:
            MotionData at specified frames
        """
        if out is not None:
            np.take(self.joint_pos, frame_idx, axis=0, out=out.joint_pos)
            np.take(self.joint_vel, frame_idx, axis=0, out=out.joint_vel)
            np.take(self.body_pos_w, frame_idx, axis=0, out=out.body_pos_w)
            np.take(self.body_quat_w, frame_idx, axis=0, out=out.body_quat_w)
            np.take(self.body_lin_vel_w, frame_idx, axis=0, out=out.body_lin_vel_w)
            np.take(self.body_ang_vel_w, frame_idx, axis=0, out=out.body_ang_vel_w)
            return out

        return MotionData(
            joint_pos=self.joint_pos[frame_idx],
            joint_vel=self.joint_vel[frame_idx],
            body_pos_w=self.body_pos_w[frame_idx],
            body_quat_w=self.body_quat_w[frame_idx],
            body_lin_vel_w=self.body_lin_vel_w[frame_idx],
            body_ang_vel_w=self.body_ang_vel_w[frame_idx],
        )


class MotionSampler:
    """Handles motion frame sampling with different strategies."""

    def __init__(
        self,
        motion_loader: MotionLoader,
        mode: Literal["start", "clip_start", "uniform", "adaptive", "mixed"],
        num_envs: int,
        bin_count: int | None = None,
        adaptive_lambda: float = 0.8,
        adaptive_kernel_size: int = 1,
        adaptive_uniform_ratio: float = 0.1,
        adaptive_alpha: float = 0.001,
        adaptive_failure_stat: Literal["ema", "cumulative"] = "ema",
        adaptive_failure_prior: float = 1.0,
        adaptive_pre_failure_window: int = 200,
        adaptive_failure_rate_max_over_mean: float = 200.0,
        adaptive_sampling_update_interval: int = 200,
        adaptive_attribution: Literal["episode_end", "trajectory"] = "trajectory",
        adaptive_max_prob_per_motion: float | None = None,
        start_ratio: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        """Initialize motion sampler.

        Args:
            motion_loader: Motion loader instance
            mode: Sampling mode ("start", "clip_start", "uniform", "adaptive", "mixed")
            num_envs: Number of parallel environments
            bin_count: Number of bins for adaptive sampling (auto if None)
            adaptive_lambda: Decay factor for adaptive kernel
            adaptive_kernel_size: Kernel size for adaptive sampling
            adaptive_uniform_ratio: Uniform sampling ratio for adaptive mode
            adaptive_alpha: EMA alpha for failure count updates (only used
                when ``adaptive_failure_stat`` is "ema")
            adaptive_failure_stat: Failure statistic driving sampling
                weights and metrics: "ema" keeps the legacy per-bin failure
                rate EMA; "cumulative" uses the count ratio
                ``(failed + prior) / (visited + prior)``, whose prior dilutes
                at 1/(N+1) per visit and keeps unvisited bins at rate 1.0.
            adaptive_failure_prior: Laplace pseudo-count for the
                "cumulative" statistic (unused by "ema").
            start_ratio: Fraction of envs forced to frame 0 in "mixed" mode
                (remaining envs are uniformly sampled). Lets buffer concentrate
                launch-transition samples while keeping motion-clip coverage.
        """
        if not 0.0 <= start_ratio <= 1.0:
            raise ValueError(f"start_ratio must be in [0, 1], got {start_ratio}")
        if mode != "mixed" and start_ratio != 0.0:
            raise ValueError("start_ratio is only effective when mode='mixed'")
        if not np.isfinite(adaptive_lambda) or not 0.0 < adaptive_lambda <= 1.0:
            raise ValueError(
                "adaptive_lambda must be finite and within (0, 1], "
                f"got {adaptive_lambda}"
            )
        if adaptive_kernel_size < 1:
            raise ValueError(
                f"adaptive_kernel_size must be a positive integer, got {adaptive_kernel_size}"
            )
        if not 0.0 <= adaptive_uniform_ratio <= 1.0:
            raise ValueError(
                "adaptive_uniform_ratio must be in [0, 1], "
                f"got {adaptive_uniform_ratio}"
            )
        if not 0.0 < adaptive_alpha <= 1.0:
            raise ValueError(f"adaptive_alpha must be in (0, 1], got {adaptive_alpha}")
        if adaptive_failure_stat not in ("ema", "cumulative"):
            raise ValueError("adaptive_failure_stat must be 'ema' or 'cumulative'")
        if (
            isinstance(adaptive_failure_prior, bool)
            or not np.isfinite(adaptive_failure_prior)
            or adaptive_failure_prior <= 0.0
        ):
            raise ValueError(
                "adaptive_failure_prior must be a finite positive number, "
                f"got {adaptive_failure_prior}"
            )
        if isinstance(adaptive_pre_failure_window, bool) or adaptive_pre_failure_window < 0:
            raise ValueError("adaptive_pre_failure_window must be a non-negative integer")
        if not np.isfinite(adaptive_failure_rate_max_over_mean) or adaptive_failure_rate_max_over_mean <= 0.0:
            raise ValueError("adaptive_failure_rate_max_over_mean must be positive and finite")
        if (
            isinstance(adaptive_sampling_update_interval, bool)
            or adaptive_sampling_update_interval < 1
        ):
            raise ValueError("adaptive_sampling_update_interval must be a positive integer")
        if adaptive_attribution not in ("episode_end", "trajectory"):
            raise ValueError("adaptive_attribution must be 'episode_end' or 'trajectory'")
        if adaptive_max_prob_per_motion is not None and (
            isinstance(adaptive_max_prob_per_motion, bool)
            or not np.isfinite(adaptive_max_prob_per_motion)
            or adaptive_max_prob_per_motion < 1.0
        ):
            raise ValueError("adaptive_max_prob_per_motion must be None or a finite value >= 1")
        self.motion_loader = motion_loader
        self.mode = mode
        self.num_envs = num_envs
        self.start_ratio = start_ratio
        self.rng = rng

        # Current frame indices for each environment
        self.current_frames = np.zeros(num_envs, dtype=np.int32)
        # Frame at which each environment's current episode was sampled; the
        # trajectory attribution mode stamps every bin between this frame and
        # the episode's terminal frame as a visit.
        self._episode_start_frames = np.zeros(num_envs, dtype=np.int32)
        self.current_clip_indices = np.zeros(num_envs, dtype=np.int32)
        self.current_clip_end_frames = np.full(
            num_envs, motion_loader.clip_end_frames[0], dtype=np.int32
        )

        # Adaptive sampling parameters
        if bin_count is None:
            # Keep approximately one-second bins, as in the Sonic sampler.
            configured_bin_count = int(motion_loader.num_frames // motion_loader.fps) + 1
        else:
            configured_bin_count = int(bin_count)
        if configured_bin_count < 1:
            raise ValueError("bin_count must be a positive integer")

        # Adaptive bins are clip-local.  ``bin_count`` is retained as the
        # single-clip equivalent for backwards compatibility; for multiple
        # clips it defines the target temporal bin width.
        target_bin_width = max(1, int(math.ceil(motion_loader.num_frames / configured_bin_count)))
        clip_bin_counts = np.maximum(
            1, np.ceil(motion_loader.clip_lengths / target_bin_width).astype(np.int32)
        )
        self._clip_bin_counts = clip_bin_counts
        self._bin_clip_indices = np.repeat(
            np.arange(motion_loader.num_clips, dtype=np.int32), clip_bin_counts
        )
        bin_starts: list[int] = []
        bin_ends: list[int] = []
        bin_lengths: list[int] = []
        self._clip_bin_ranges: list[tuple[int, int]] = []
        cursor = 0
        for clip_idx, num_bins in enumerate(clip_bin_counts):
            length = int(motion_loader.clip_lengths[clip_idx])
            starts = np.linspace(0, length, int(num_bins), endpoint=False, dtype=np.int32)
            ends = np.r_[starts[1:], length].astype(np.int32)
            bin_starts.extend((starts + motion_loader.clip_offsets[clip_idx]).tolist())
            bin_ends.extend((ends + motion_loader.clip_offsets[clip_idx]).tolist())
            bin_lengths.extend((ends - starts).tolist())
            self._clip_bin_ranges.append((cursor, cursor + int(num_bins)))
            cursor += int(num_bins)
        self._bin_start_frames = np.asarray(bin_starts, dtype=np.int32)
        self._bin_end_frames = np.asarray(bin_ends, dtype=np.int32)
        self._bin_lengths = np.asarray(bin_lengths, dtype=np.float32)
        adaptive_bin_count = int(self._bin_start_frames.size)
        # Keep legacy bin-count metrics for non-adaptive modes.
        self.bin_count = adaptive_bin_count if mode == "adaptive" else configured_bin_count
        self._frame_to_bin = np.empty(motion_loader.num_frames, dtype=np.int32)
        for bin_idx, (start, end) in enumerate(zip(self._bin_start_frames, self._bin_end_frames)):
            self._frame_to_bin[start:end] = bin_idx

        self.adaptive_lambda = adaptive_lambda
        self.adaptive_kernel_size = adaptive_kernel_size
        self.adaptive_uniform_ratio = adaptive_uniform_ratio
        self.adaptive_alpha = adaptive_alpha
        self.adaptive_failure_stat = adaptive_failure_stat
        self.adaptive_failure_prior = float(adaptive_failure_prior)
        self.adaptive_pre_failure_window = int(adaptive_pre_failure_window)
        self.adaptive_failure_rate_max_over_mean = float(adaptive_failure_rate_max_over_mean)
        self.adaptive_sampling_update_interval = int(adaptive_sampling_update_interval)
        self.adaptive_attribution = adaptive_attribution
        self.adaptive_max_prob_per_motion = adaptive_max_prob_per_motion

        # Failure tracking for adaptive sampling
        # Raw counters are retained for diagnostics.  Sampling uses the EMA of
        # per-bin failure *rates* below, rather than raw failure counts; this
        # prevents frequently visited bins from creating a self-reinforcing
        # sampling bias.
        self.bin_failed_count = np.zeros(self.bin_count, dtype=np.float32)
        self.bin_visit_count = np.zeros(self.bin_count, dtype=np.float32)
        # Under "cumulative" the rate starts at the prior value 1.0 so that
        # recompute before the first update keeps cold-start exploration.
        self.bin_failure_rate = np.full(
            self.bin_count,
            1.0 if adaptive_failure_stat == "cumulative" else 0.0,
            dtype=np.float32,
        )
        # Non-zero prior prevents one early failure from monopolizing sampling.
        self._bin_failure_rate_ema = np.ones(self.bin_count, dtype=np.float32)
        # Visit EMA has no prior: unvisited bins must remain distinguishable in
        # diagnostics.  Failure-rate EMA intentionally keeps a non-zero prior.
        self._bin_visit_ema = np.zeros(self.bin_count, dtype=np.float32)
        self._current_bin_failed = np.zeros(self.bin_count, dtype=np.float32)
        self._current_bin_visited = np.zeros(self.bin_count, dtype=np.float32)
        self._sampling_probs = np.full(self.bin_count, 1.0 / self.bin_count, dtype=np.float32)
        self._peer_counts = self._clip_bin_counts[self._bin_clip_indices].astype(np.float32)
        self._uniform_probs = np.full(self.bin_count, 1.0 / self.bin_count, dtype=np.float32)
        self._adaptive_steps_since_update = 0
        self._adaptive_probs_initialized = False

        # Precompute adaptive kernel
        self.kernel = np.array(
            [adaptive_lambda**i for i in range(adaptive_kernel_size)], dtype=np.float32
        )
        self.kernel = self.kernel / self.kernel.sum()

        # Metrics
        self.sampling_entropy = 0.0
        self.sampling_top1_prob = 0.0
        self.sampling_top1_bin = 0.0
        self.sampling_top1_clip_prob = float(1.0 / self.motion_loader.num_clips)
        self.sampling_effective_bin_count = float(self.bin_count)
        self.sampling_visited_bin_fraction = 0.0
        self.sampling_failure_rate_mean = 0.0
        self.sampling_failure_rate_max = 0.0
        self.sampling_failure_count_total = 0.0
        self.sampling_visit_count_total = 0.0
        self.sampling_uniform_mass_actual = (
            float(adaptive_uniform_ratio) if mode == "adaptive" else 0.0
        )
        self._done_mask = np.zeros(num_envs, dtype=bool)

    def sample_frames(self, env_ids: np.ndarray) -> np.ndarray:
        """Sample motion frames for specified environments.

        Args:
            env_ids: Environment indices to sample for

        Returns:
            Sampled frame indices
        """
        if self.mode == "start":
            return self._sample_start(env_ids)
        elif self.mode == "clip_start":
            return self._sample_clip_start(env_ids)
        elif self.mode == "uniform":
            return self._sample_uniform(env_ids)
        elif self.mode == "adaptive":
            return self._sample_adaptive(env_ids)
        elif self.mode == "mixed":
            return self._sample_mixed(env_ids)
        else:
            raise ValueError(f"Unknown sampling mode: {self.mode}")

    def set_clip_starts(self, env_ids: np.ndarray, clip_indices: np.ndarray) -> np.ndarray:
        """Assign exact clip starts for deterministic evaluation resets.

        This is the public, cold-path counterpart to stochastic training
        sampling.  It lets evaluation pair policies on the same motion clips
        without reaching into sampler-private arrays.
        """
        ids = np.asarray(env_ids)
        clips = np.asarray(clip_indices)
        if ids.ndim != 1 or clips.ndim != 1 or ids.shape != clips.shape:
            raise ValueError("env_ids and clip_indices must be equal-length vectors")
        if not np.issubdtype(ids.dtype, np.integer) or not np.issubdtype(clips.dtype, np.integer):
            raise TypeError("env_ids and clip_indices must contain integers")
        if np.any(ids < 0) or np.any(ids >= self.num_envs):
            raise ValueError("env_ids are outside the sampler environment range")
        if np.any(clips < 0) or np.any(clips >= self.motion_loader.num_clips):
            raise ValueError("clip_indices are outside the motion dataset range")
        ids = ids.astype(np.intp, copy=False)
        clips = clips.astype(np.intp, copy=False)
        frames = np.asarray(self.motion_loader.clip_offsets[clips], dtype=np.int32)
        self._set_sampled_frames(ids, frames)
        return frames

    def _sample_start(self, env_ids: np.ndarray) -> np.ndarray:
        """Always start from the global first frame (historical behavior)."""
        frames = np.zeros(len(env_ids), dtype=np.int32)
        self._set_sampled_frames(env_ids, frames)
        return frames

    def _sample_clip_start(self, env_ids: np.ndarray) -> np.ndarray:
        """Start from the first frame of a randomly chosen clip."""
        frames: np.ndarray
        if self.motion_loader.num_clips == 1:
            frames = np.zeros(len(env_ids), dtype=np.int32)
        else:
            if self.rng is None:
                clip_indices = np.random.randint(
                    0, self.motion_loader.num_clips, len(env_ids), dtype=np.int32
                )
            else:
                clip_indices = self.rng.integers(
                    0, self.motion_loader.num_clips, len(env_ids), dtype=np.int32
                )
            frames = np.asarray(self.motion_loader.clip_offsets[clip_indices], dtype=np.int32)
        self._set_sampled_frames(env_ids, frames)
        return frames

    def _sample_uniform(self, env_ids: np.ndarray) -> np.ndarray:
        """Sample uniformly across motion."""
        if self.rng is None:
            frames = np.random.randint(
                0, self.motion_loader.num_frames, len(env_ids), dtype=np.int32
            )
        else:
            frames = self.rng.integers(
                0, self.motion_loader.num_frames, len(env_ids), dtype=np.int32
            )
        self._set_sampled_frames(env_ids, frames)

        # Update metrics
        self.sampling_entropy = 1.0  # Maximum entropy for uniform
        self.sampling_top1_prob = 1.0 / self.bin_count
        self.sampling_top1_bin = 0.5  # No specific bin preference
        self.sampling_effective_bin_count = float(self.bin_count)
        self.sampling_uniform_mass_actual = 1.0

        return frames

    def _sample_mixed(self, env_ids: np.ndarray) -> np.ndarray:
        """Per-env Bernoulli mix of ``start`` (frame 0) and ``uniform``.

        Each env independently lands on frame 0 with probability ``start_ratio``
        and on a uniformly sampled frame otherwise. This concentrates buffer
        coverage on the launch transition (frame 0 -> apex) while preserving
        uniform RSI's whole-clip coverage everywhere else.
        """
        n = len(env_ids)
        random_values = np.random.random(n) if self.rng is None else self.rng.random(n)
        if self.rng is None:
            uniform_frames = np.random.randint(0, self.motion_loader.num_frames, n)
        else:
            uniform_frames = self.rng.integers(0, self.motion_loader.num_frames, n)
        use_start = random_values < self.start_ratio
        frames = np.where(
            use_start,
            0,
            uniform_frames,
        ).astype(np.int32)
        self._set_sampled_frames(env_ids, frames)

        # Metrics: mixture of a degenerate start mass and a uniform spread.
        # Reflect the start over-representation in top1.
        start_mass = self.start_ratio + (1.0 - self.start_ratio) / self.bin_count
        self.sampling_top1_prob = float(start_mass)
        self.sampling_top1_bin = 0.0
        self.sampling_uniform_mass_actual = float(1.0 - self.start_ratio)
        if start_mass >= 1.0 - 1e-9:
            self.sampling_entropy = 0.0
        else:
            uniform_mass = (1.0 - self.start_ratio) / self.bin_count
            H = -start_mass * math.log(start_mass + 1e-12)
            H -= (self.bin_count - 1) * uniform_mass * math.log(uniform_mass + 1e-12)
            self.sampling_entropy = (
                float(H / math.log(self.bin_count)) if self.bin_count > 1 else 1.0
            )

        return frames

    def _sample_adaptive(self, env_ids: np.ndarray) -> np.ndarray:
        """Sample adaptively based on failure statistics."""
        if (
            not self._adaptive_probs_initialized
            or self._adaptive_steps_since_update >= self.adaptive_sampling_update_interval
        ):
            self._recompute_adaptive_probs()

        sampling_probs = self._sampling_probs
        # Sample bins
        sampled_bins = (
            np.random.choice(self.bin_count, size=len(env_ids), p=sampling_probs)
            if self.rng is None
            else self.rng.choice(self.bin_count, size=len(env_ids), p=sampling_probs)
        )

        # Add random offset within bin
        bin_offsets = (
            np.random.uniform(0.0, 1.0, len(env_ids))
            if self.rng is None
            else self.rng.uniform(0.0, 1.0, len(env_ids))
        )
        frames = (
            self._bin_start_frames[sampled_bins]
            + bin_offsets * self._bin_lengths[sampled_bins]
        ).astype(np.int32)
        if self.adaptive_pre_failure_window > 0:
            clip_indices = self.motion_loader.get_clip_indices(frames)
            clip_starts = self.motion_loader.clip_offsets[clip_indices]
            if self.rng is None:
                offsets = np.random.randint(
                    0, self.adaptive_pre_failure_window, len(env_ids), dtype=np.int32
                )
            else:
                offsets = self.rng.integers(
                    0, self.adaptive_pre_failure_window, len(env_ids), dtype=np.int32
                )
            frames = np.maximum(frames - offsets, clip_starts).astype(np.int32, copy=False)

        self._set_sampled_frames(env_ids, frames)
        return np.asarray(frames, dtype=np.int32)

    def _recompute_adaptive_probs(self) -> None:
        """Rebuild the adaptive distribution at the configured low frequency."""
        # Compute probabilities from the smoothed failure-rate estimate.  The
        # configured uniform ratio is an actual mixture weight, not an additive
        # epsilon that vanishes as counters grow.
        # Weight each clip equally, then distribute its mass over local bins.
        # This matches gear_sonic's sequence-length-agnostic bin weighting.
        if self.adaptive_failure_stat == "cumulative":
            failure_stat = self.bin_failure_rate
        else:
            failure_stat = self._bin_failure_rate_ema
        adaptive_weights = failure_stat / self._peer_counts
        if not np.any(adaptive_weights > 0.0):
            adaptive_probs = self._uniform_probs.copy()
        else:
            adaptive_weights += np.finfo(np.float32).eps
            upper = float(adaptive_weights.mean()) * self.adaptive_failure_rate_max_over_mean
            np.minimum(adaptive_weights, upper, out=adaptive_weights)
            adaptive_probs = adaptive_weights / adaptive_weights.sum()

        # Apply smoothing kernel (non-causal convolution)
        if self.adaptive_kernel_size > 1:
            smoothed = np.empty_like(adaptive_probs)
            for start, end in self._clip_bin_ranges:
                local = adaptive_probs[start:end]
                if len(local) == 1:
                    smoothed[start:end] = local
                    continue
                width = min(self.adaptive_kernel_size, len(local))
                kernel = self.kernel[:width]
                kernel = kernel / kernel.sum()
                padded = np.pad(local, (0, width - 1), mode="edge")
                smoothed[start:end] = np.convolve(padded, kernel, mode="valid")
            adaptive_probs = smoothed
            adaptive_probs /= adaptive_probs.sum()

        sampling_probs = (
            (1.0 - self.adaptive_uniform_ratio) * adaptive_probs
            + self.adaptive_uniform_ratio * self._uniform_probs
        )
        sampling_probs /= sampling_probs.sum()
        clip_mass = np.bincount(
            self._bin_clip_indices,
            weights=sampling_probs,
            minlength=self.motion_loader.num_clips,
        )
        if self.adaptive_max_prob_per_motion is not None:
            cap = self.adaptive_max_prob_per_motion / self.motion_loader.num_clips
            over = clip_mass > cap
            if np.any(over):
                scale = np.ones_like(clip_mass)
                scale[over] = cap / clip_mass[over]
                sampling_probs = sampling_probs * scale[self._bin_clip_indices]
                # Plain renormalization would push the trimmed excess straight
                # back into the capped clips whenever the remaining clips carry
                # near-zero mass; redistribute it uniformly over the uncapped
                # clips' bins instead.
                excess = float((clip_mass[over] - cap).sum())
                rest_bins = (~over)[self._bin_clip_indices]
                if np.any(rest_bins) and excess > 0.0:
                    sampling_probs[rest_bins] += excess / np.count_nonzero(rest_bins)
                sampling_probs /= sampling_probs.sum()
                clip_mass = np.bincount(
                    self._bin_clip_indices,
                    weights=sampling_probs,
                    minlength=self.motion_loader.num_clips,
                )
        self._sampling_probs[...] = sampling_probs
        self.sampling_top1_clip_prob = float(clip_mass.max())
        H = -(sampling_probs * np.log(sampling_probs + 1e-12)).sum()
        H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else 1.0
        pmax_idx = np.argmax(sampling_probs)
        pmax = sampling_probs[pmax_idx]

        self.sampling_entropy = H_norm
        self.sampling_top1_prob = float(pmax)
        self.sampling_top1_bin = float(pmax_idx) / self.bin_count
        self.sampling_effective_bin_count = float(np.exp(H))
        self.sampling_uniform_mass_actual = float(self.adaptive_uniform_ratio)
        if self.adaptive_failure_stat == "cumulative":
            visited_bins = self.bin_visit_count > 0.0
        else:
            visited_bins = self._bin_visit_ema > 0.0
        self.sampling_visited_bin_fraction = float(
            np.count_nonzero(visited_bins) / max(self.bin_count, 1)
        )
        self.sampling_failure_rate_mean = float(np.mean(failure_stat))
        self.sampling_failure_rate_max = float(np.max(failure_stat))
        self.sampling_failure_count_total = float(self.bin_failed_count.sum())
        self.sampling_visit_count_total = float(self.bin_visit_count.sum())
        self._adaptive_probs_initialized = True
        self._adaptive_steps_since_update = 0

    def _trajectory_bin_indices(self, frames: np.ndarray, completed: np.ndarray) -> np.ndarray:
        """Flat bin indices covered by each completed episode's traversed frames.

        Ranges stay inside the clip the episode started in (defensively
        clamped to that clip's end frame), matching the contract that
        episodes never cross motion clips.
        """
        ends = frames.astype(np.int32, copy=False)[completed]
        starts = np.minimum(self._episode_start_frames[completed], ends)
        loader = self.motion_loader
        start_clips = loader.get_clip_indices(starts)
        ends = np.minimum(ends, loader.clip_end_frames[start_clips]).astype(np.int32, copy=False)
        lengths = (ends - starts + 1).astype(np.int64)
        total = int(lengths.sum())
        if total <= 0:
            return np.empty(0, dtype=np.int64)
        episode_offsets = np.zeros(lengths.size + 1, dtype=np.int64)
        np.cumsum(lengths, out=episode_offsets[1:])
        within = np.arange(total, dtype=np.int64) - np.repeat(episode_offsets[:-1], lengths)
        frame_indices = np.repeat(starts.astype(np.int64), lengths) + within
        episode_ids = np.repeat(np.arange(lengths.size, dtype=np.int64), lengths)
        # Deduplicate to one visit per (episode, bin): an episode spending 21
        # frames inside a bin must count as a single visit, not 21.
        keys = episode_ids * self.bin_count + self._frame_to_bin[frame_indices]
        return (np.unique(keys) % self.bin_count).astype(np.int64)

    def update_failure_stats(
        self,
        terminated: np.ndarray,
        current_frames: np.ndarray | None = None,
        episode_done: np.ndarray | None = None,
    ):
        """Update failure statistics for adaptive sampling.

        Args:
            terminated: Boolean array indicating which environments terminated due to failure
            current_frames: Optional current frame indices (uses internal if None)
            episode_done: Optional boolean array indicating all completed episodes
                (including timeout/truncation). If omitted, all supplied rows are
                treated as completed for backwards compatibility with the direct
                sampler API.

        Attribution modes:

        - ``trajectory`` (default): every completed episode stamps a visit on
          each bin it traversed from its sampled start frame to its terminal
          frame; failures stamp only their terminal bin. This restores
          within-clip contrast, which end-frame attribution destroys (successes
          pile onto the clip-final bin while mid-clip bins only ever receive
          failure stamps and saturate at rate 1.0). Start frames come from the
          internal ``_episode_start_frames`` recorded at sampling time.
        - ``episode_end``: legacy behaviour; visits and failures are both
          stamped on the episode's terminal-frame bin.
        """
        if self.mode != "adaptive":
            return

        self._adaptive_steps_since_update += 1

        if current_frames is None:
            current_frames = self.current_frames

        # Find which bins failed
        frames = np.asarray(current_frames)
        done = np.asarray(terminated, dtype=bool)
        completed = (
            np.ones_like(done, dtype=bool)
            if episode_done is None
            else np.asarray(episode_done, dtype=bool)
        )
        if (
            frames.shape != (self.num_envs,)
            or done.shape != (self.num_envs,)
            or completed.shape != (self.num_envs,)
        ):
            raise ValueError(
                "MotionSampler failure statistics expect current_frames and terminated "
                f"with shape ({self.num_envs},), got {frames.shape}, {done.shape}, "
                f"and {completed.shape}"
            )
        # Statistics are episode-level. Updating on every live environment step
        # incorrectly treats ordinary transitions as successful episodes and
        # rapidly drives the failure EMA toward zero.
        if not np.any(completed):
            return
        if np.any(frames < 0) or np.any(frames >= self.motion_loader.num_frames):
            raise ValueError("MotionSampler failure frames are outside the motion buffer")
        bin_indices = self._frame_to_bin[frames.astype(np.int32, copy=False)]
        if self.adaptive_attribution == "trajectory":
            visited_indices = self._trajectory_bin_indices(frames, completed)
        else:
            visited_indices = bin_indices[completed]
        self._current_bin_visited.fill(0.0)
        np.add.at(self._current_bin_visited, visited_indices, 1.0)
        self._current_bin_failed.fill(0.0)
        if np.any(done):
            np.add.at(self._current_bin_failed, bin_indices[done & completed], 1.0)

        self.bin_visit_count += self._current_bin_visited
        self.bin_failed_count += self._current_bin_failed
        if self.adaptive_failure_stat == "cumulative":
            # Pure function of the cumulative counters; the prior keeps the
            # rate well-defined for bins nobody visited yet.
            prior = np.float32(self.adaptive_failure_prior)
            np.divide(
                self.bin_failed_count + prior,
                self.bin_visit_count + prior,
                out=self.bin_failure_rate,
            )
        else:
            visited = self._current_bin_visited > 0.0
            current_rate = np.zeros_like(self._current_bin_failed)
            np.divide(
                self._current_bin_failed,
                self._current_bin_visited,
                out=current_rate,
                where=visited,
            )
            # Update only bins observed in this collector cycle; otherwise an
            # unvisited bin would decay toward zero merely because it was absent.
            self._bin_failure_rate_ema[visited] = (
                (1.0 - self.adaptive_alpha) * self._bin_failure_rate_ema[visited]
                + self.adaptive_alpha * current_rate[visited]
            )
            self._bin_visit_ema[visited] = (
                (1.0 - self.adaptive_alpha) * self._bin_visit_ema[visited]
                + self.adaptive_alpha * self._current_bin_visited[visited]
            )
            self.bin_failure_rate[...] = self._bin_failure_rate_ema
        self.sampling_failure_count_total = float(self.bin_failed_count.sum())
        self.sampling_visit_count_total = float(self.bin_visit_count.sum())

    def _set_sampled_frames(self, env_ids: np.ndarray, frames: np.ndarray) -> None:
        self.current_frames[env_ids] = frames
        self._episode_start_frames[env_ids] = frames
        clip_indices = self.motion_loader.get_clip_indices(frames)
        self.current_clip_indices[env_ids] = clip_indices
        self.current_clip_end_frames[env_ids] = self.motion_loader.clip_end_frames[clip_indices]

    def step(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        """Advance selected frames by one step and return their clip-end rows."""
        ids = np.arange(self.num_envs, dtype=np.int32) if env_ids is None else env_ids
        self.current_frames[ids] += 1

        # Find environments that reached the end of their current clip.
        self._done_mask.fill(False)
        self._done_mask[ids] = self.current_frames[ids] > self.current_clip_end_frames[ids]
        return np.flatnonzero(self._done_mask)

    def get_current_motion(self, out: MotionData | None = None) -> MotionData:
        """Get motion data at current frames for all environments."""
        return self.motion_loader.get_motion_at_frame(self.current_frames, out=out)
