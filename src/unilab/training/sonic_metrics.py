"""Deterministic, clip-paired evaluation for SONIC checkpoints."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium.spaces import Box
from omegaconf import DictConfig, OmegaConf
from uni_rl.algos.sonic import load_sonic_checkpoint, load_sonic_checkpoint_file
from uni_rl.ipc.dp_launcher import (
    current_dp_rank,
    resolve_dp_rank_device,
    resolve_dp_topology,
)

from unilab.base.config_adapter import create_env
from unilab.base.np_env import NpEnv
from unilab.tasks.motion_tracking.g1.sonic_manager import (
    G1_SONIC_BODY_NAMES,
    SONIC_TERMINATION_REASON_NAMES,
    SonicMotionCommand,
)
from unilab.training.offpolicy_sonic import (
    ROOT_DIR,
    _align_play_env_to_checkpoint,
    _apply_sonic_play_action_scale,
    _resolve_checkpoint_format,
    build_play_actor,
    build_play_env_cfg_override,
)
from unilab.utils.checkpoint import resolve_offpolicy_checkpoint_path
from unilab.utils.sim2sim import policy_load_dim_guard, resolve_sim2sim_config
from unilab.visualization.interactive_playback import default_device, resolve_play_obs_dims


@dataclass
class _MetricTotals:
    position: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0
    frames: int = 0

    def add(self, other: _MetricTotals) -> None:
        self.position += other.position
        self.velocity += other.velocity
        self.acceleration += other.acceleration
        self.frames += other.frames

    def means(self) -> dict[str, float | None]:
        if self.frames == 0:
            return {
                "mpjpe_l_mm": None,
                "velocity_distance_mm_per_frame": None,
                "acceleration_distance_mm_per_frame2": None,
            }
        # Match gear_sonic's im_eval_callback: derivative sums use the full
        # trajectory frame count as their denominator.
        return {
            "mpjpe_l_mm": self.position / self.frames,
            "velocity_distance_mm_per_frame": self.velocity / self.frames,
            "acceleration_distance_mm_per_frame2": self.acceleration / self.frames,
        }


class SonicBodyMetricAccumulator:
    """Streaming equivalent of SONIC's ``compute_metrics_lite`` core metrics."""

    def __init__(self, num_envs: int, num_bodies: int) -> None:
        shape = (num_envs, num_bodies, 3)
        self._previous = np.zeros(shape, dtype=np.float32)
        self._previous_previous = np.zeros(shape, dtype=np.float32)
        self._reference_previous = np.zeros(shape, dtype=np.float32)
        self._reference_previous_previous = np.zeros(shape, dtype=np.float32)
        self._totals = [_MetricTotals() for _ in range(num_envs)]

    def observe(
        self,
        predicted: np.ndarray,
        reference: np.ndarray,
        active: np.ndarray,
    ) -> None:
        predicted = np.asarray(predicted, dtype=np.float32)
        reference = np.asarray(reference, dtype=np.float32)
        active = np.asarray(active, dtype=bool)
        if predicted.shape != self._previous.shape or reference.shape != self._previous.shape:
            raise ValueError("SONIC body position batches do not match the accumulator shape")
        if active.shape != (len(self._totals),):
            raise ValueError("SONIC active mask does not match the accumulator rows")

        for row in np.flatnonzero(active):
            total = self._totals[int(row)]
            pred_local = predicted[row] - predicted[row, :1]
            ref_local = reference[row] - reference[row, :1]
            total.position += float(np.linalg.norm(pred_local - ref_local, axis=-1).mean() * 1000.0)
            if total.frames >= 1:
                pred_velocity = predicted[row] - self._previous[row]
                ref_velocity = reference[row] - self._reference_previous[row]
                total.velocity += float(
                    np.linalg.norm(pred_velocity - ref_velocity, axis=-1).mean() * 1000.0
                )
            if total.frames >= 2:
                pred_acceleration = (
                    predicted[row] - 2.0 * self._previous[row] + self._previous_previous[row]
                )
                ref_acceleration = (
                    reference[row]
                    - 2.0 * self._reference_previous[row]
                    + self._reference_previous_previous[row]
                )
                total.acceleration += float(
                    np.linalg.norm(pred_acceleration - ref_acceleration, axis=-1).mean() * 1000.0
                )
            self._previous_previous[row] = self._previous[row]
            self._previous[row] = predicted[row]
            self._reference_previous_previous[row] = self._reference_previous[row]
            self._reference_previous[row] = reference[row]
            total.frames += 1

    def totals(self, row: int) -> _MetricTotals:
        total = self._totals[row]
        return _MetricTotals(
            position=total.position,
            velocity=total.velocity,
            acceleration=total.acceleration,
            frames=total.frames,
        )


@dataclass
class _PlaybackSession:
    env: NpEnv
    motion: SonicMotionCommand
    actor: Any
    checkpoint_format: str
    obs_normalizer: Any
    device: str
    checkpoint_path: Path


def _resolve_checkpoint(cfg: DictConfig, configured: str | os.PathLike[str]) -> tuple[Path, Path]:
    path, run_dir = resolve_offpolicy_checkpoint_path(
        ROOT_DIR,
        str(cfg.algo.algo_log_name),
        str(cfg.training.task_name),
        configured,
    )
    if path is None or run_dir is None:
        raise FileNotFoundError(f"Could not resolve SONIC checkpoint: {configured}")
    return Path(path).resolve(), Path(run_dir).resolve()


def _force_clip_end_protocol(env_override: dict[str, Any]) -> None:
    """Force the per-clip truncation protocol on the benchmark play env.

    SONIC training defaults to wrap mode (a clip's end resamples the reference
    and teleports the robot mid-episode), but the per-clip metric protocol
    scores exactly one clip per reset and needs episodes truncated at the
    clip's final frame.
    """
    motion_params = (
        env_override.setdefault("commands", {}).setdefault("motion", {}).setdefault("params", {})
    )
    motion_params["truncate_on_clip_end"] = True
    env_override.setdefault("terminations", {})["clip_end"] = {
        "_target_": "unilab.managers.TerminationTermCfg",
        "func": "unilab.tasks.motion_tracking.g1.sonic_manager.sonic_clip_end",
        "time_out": True,
    }


def _build_session(cfg: DictConfig, configured_checkpoint: str) -> _PlaybackSession:
    checkpoint_path, run_dir = _resolve_checkpoint(cfg, configured_checkpoint)
    session_cfg = deepcopy(cfg)
    session_cfg = (
        resolve_sim2sim_config(
            run_dir,
            session_cfg,
            algo_name="flashsac",
            strict=bool(getattr(session_cfg.training, "sim2sim_strict", True)),
        )
        or session_cfg
    )
    checkpoint = load_sonic_checkpoint_file(checkpoint_path)
    checkpoint_format = _resolve_checkpoint_format(session_cfg, checkpoint)
    env_override = build_play_env_cfg_override("flashsac", session_cfg)
    _align_play_env_to_checkpoint(env_override, checkpoint)
    _apply_sonic_play_action_scale(env_override, checkpoint_format=checkpoint_format)
    _force_clip_end_protocol(env_override)
    env = create_env(
        session_cfg,
        num_envs=int(session_cfg.training.play_env_num),
        env_cfg_override=env_override,
    )
    if not isinstance(env, NpEnv) and not hasattr(env, "command_manager"):
        raise TypeError("SONIC metric benchmark requires a Manager runtime with commands")
    motion = env.command_manager.get_term("motion")
    if not isinstance(motion, SonicMotionCommand):
        raise TypeError("SONIC metric benchmark requires SonicMotionCommand")
    # ``NpEnv`` exposes the manager attributes only on manager-based runtimes;
    # reach through ``getattr`` to keep the static contract unchanged.
    termination_terms = getattr(getattr(env, "termination_manager", None), "active_terms", ())
    if not motion.cfg.params.truncate_on_clip_end or "clip_end" not in termination_terms:
        raise ValueError(
            "SONIC metric benchmark requires the clip-end truncation protocol "
            "(truncate_on_clip_end=true with the clip_end termination term)"
        )

    obs_dim, _ = resolve_play_obs_dims(env.obs_groups_spec)
    action_space = env.action_space
    if not isinstance(action_space, Box) or action_space.shape is None:
        raise TypeError("SONIC metric benchmark requires a continuous Box action space")
    action_dim = int(action_space.shape[0])
    devices = resolve_dp_topology(getattr(session_cfg.training, "devices", None))
    device = default_device(torch, resolve_dp_rank_device(current_dp_rank()))
    actor = build_play_actor(
        session_cfg,
        checkpoint,
        obs_dim=obs_dim,
        action_low=np.full(action_dim, -1.0, dtype=np.float32),
        action_high=np.full(action_dim, 1.0, dtype=np.float32),
        device=device,
    )
    obs_normalizer = None
    if checkpoint_format != "sonic_release" and bool(session_cfg.algo.obs_normalization):
        from uni_rl.algos.common.normalization import EmpiricalNormalization

        obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
        normalizer_state = checkpoint.get("obs_normalizer")
        if normalizer_state:
            obs_normalizer.load_state_dict(normalizer_state)
        obs_normalizer.eval()
    with policy_load_dim_guard(
        env_obs_dim=obs_dim,
        env_action_dim=action_dim,
        algo_name="flashsac",
    ):
        if checkpoint_format == "sonic_release":
            load_sonic_checkpoint(actor.backbone, checkpoint_path)
        else:
            actor.load_state_dict(checkpoint["actor"])
    return _PlaybackSession(
        env=env,
        motion=motion,
        actor=actor,
        checkpoint_format=checkpoint_format,
        obs_normalizer=obs_normalizer,
        device=device,
        checkpoint_path=checkpoint_path,
    )


def _actions(session: _PlaybackSession, obs: np.ndarray) -> np.ndarray:
    obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(session.device)
    if session.obs_normalizer is not None:
        obs_tensor = session.obs_normalizer(obs_tensor, update=False)
    action = (
        session.actor.explore_native(obs_tensor)
        if session.checkpoint_format == "sonic_release"
        else session.actor.explore(obs_tensor, deterministic=True)
    )
    return action.cpu().numpy()


def _clip_batch_indices(start: int, stop: int, num_envs: int) -> np.ndarray:
    clips = np.arange(start, stop, dtype=np.int32)
    if len(clips) < num_envs:
        clips = np.pad(clips, (0, num_envs - len(clips)), mode="edge")
    return clips


def evaluate_sonic_checkpoint(
    cfg: DictConfig,
    checkpoint: str,
    *,
    max_clips: int | None = None,
    include_clip_metrics: bool = True,
) -> dict[str, Any]:
    """Evaluate one checkpoint once from the start of every selected clip."""
    session = _build_session(cfg, checkpoint)
    env, motion = session.env, session.motion
    num_envs = env.num_envs
    total_clips = motion.loader.num_clips
    selected_clips = total_clips if max_clips is None else min(total_clips, max_clips)
    if selected_clips <= 0:
        env.close()
        raise ValueError("benchmark.max_clips must be positive or null")

    all_totals = _MetricTotals()
    success_totals = _MetricTotals()
    records: list[dict[str, Any]] = []
    all_ids = np.arange(num_envs, dtype=np.int32)
    reason_names = tuple(SONIC_TERMINATION_REASON_NAMES)
    clip_end_column = reason_names.index("clip_end")
    episode_lengths: list[int] = []
    success_count = 0

    try:
        with torch.inference_mode():
            for batch_start in range(0, selected_clips, num_envs):
                batch_stop = min(batch_start + num_envs, selected_clips)
                real_rows = batch_stop - batch_start
                clip_indices = _clip_batch_indices(batch_start, batch_stop, num_envs)
                motion.stage_reset_clip_indices(all_ids, clip_indices)
                if env.state is None:
                    state = env.init_state()
                    obs = np.asarray(state.obs["obs"], dtype=np.float32)
                else:
                    reset_obs, _ = env.reset(all_ids)
                    obs = np.asarray(reset_obs["obs"], dtype=np.float32)

                completed = np.arange(num_envs) >= real_rows
                accumulator = SonicBodyMetricAccumulator(num_envs, len(G1_SONIC_BODY_NAMES))
                clip_lengths = np.asarray(motion.loader.clip_lengths[clip_indices], dtype=np.int32)
                metric_limits = np.maximum(clip_lengths - 1, 0)
                metric_counts = np.zeros(num_envs, dtype=np.int32)
                episode_steps = np.zeros(num_envs, dtype=np.int32)
                max_steps = int(np.max(clip_lengths[:real_rows])) + 2
                for _ in range(max_steps):
                    live = ~completed
                    if not np.any(live):
                        break
                    state = env.step(torch.as_tensor(_actions(session, obs)))
                    predicted, reference = motion.tracking_body_positions()
                    # The official callback discards the first/reset frame and
                    # retains exactly ``clip_frames - 1`` post-step samples.
                    metric_active = live & (metric_counts < metric_limits)
                    accumulator.observe(predicted, reference, metric_active)
                    metric_counts[metric_active] += 1
                    episode_steps[live] += 1
                    obs = np.asarray(state.obs["obs"], dtype=np.float32)
                    done = live & (np.asarray(state.terminated) | np.asarray(state.truncated))
                    if not np.any(done):
                        continue
                    masks = np.asarray(state.info.get("termination_reason_mask"), dtype=bool)
                    if masks.shape != (num_envs, len(reason_names)):
                        raise ValueError("SONIC termination reason diagnostics are unavailable")
                    for row in np.flatnonzero(done):
                        clip_index = int(clip_indices[row])
                        totals = accumulator.totals(int(row))
                        all_totals.add(totals)
                        survived = bool(masks[row, clip_end_column] and not state.terminated[row])
                        if survived:
                            success_count += 1
                            success_totals.add(totals)
                        episode_length = int(episode_steps[row])
                        episode_lengths.append(episode_length)
                        if include_clip_metrics:
                            failure_reasons = [
                                name
                                for name, flagged in zip(reason_names, masks[row], strict=True)
                                if flagged and name not in ("clip_end", "time_out")
                            ]
                            records.append(
                                {
                                    "clip_index": clip_index,
                                    "clip_frames": int(motion.loader.clip_lengths[clip_index]),
                                    "episode_length": episode_length,
                                    "survived": survived,
                                    "progress": min(
                                        episode_length
                                        / int(motion.loader.clip_lengths[clip_index]),
                                        1.0,
                                    ),
                                    "failure_reasons": failure_reasons,
                                    **totals.means(),
                                }
                            )
                    completed[done] = True
                if not np.all(completed):
                    missing = np.flatnonzero(~completed).tolist()
                    raise RuntimeError(
                        f"SONIC clips did not terminate within their frame bounds; rows={missing}"
                    )
    finally:
        env.close()

    result: dict[str, Any] = {
        "checkpoint": str(session.checkpoint_path),
        "checkpoint_format": session.checkpoint_format,
        "num_trials": selected_clips,
        "num_successes": success_count,
        "num_failures": selected_clips - success_count,
        "mean_ep_len": float(np.mean(episode_lengths)),
        "survival_rate": success_count / selected_clips,
        **success_totals.means(),
        "successful_tracking_frames": success_totals.frames,
        "all_trials": {
            **all_totals.means(),
            "tracking_frames": all_totals.frames,
        },
    }
    if include_clip_metrics:
        result["clips"] = sorted(records, key=lambda item: item["clip_index"])
    return result


def benchmark_sonic_models(cfg: DictConfig) -> dict[str, Any]:
    """Compare the release and locally trained checkpoints on one dataset."""
    release = OmegaConf.select(cfg, "benchmark.release_checkpoint")
    trained = OmegaConf.select(cfg, "benchmark.trained_checkpoint")
    if not release or not trained:
        raise ValueError("Set benchmark.release_checkpoint and benchmark.trained_checkpoint")
    max_clips_value = OmegaConf.select(cfg, "benchmark.max_clips")
    max_clips = None if max_clips_value is None else int(max_clips_value)
    include_clips = bool(OmegaConf.select(cfg, "benchmark.include_clip_metrics", default=True))
    models = {
        "sonic_release": evaluate_sonic_checkpoint(
            cfg, str(release), max_clips=max_clips, include_clip_metrics=include_clips
        ),
        "trained": evaluate_sonic_checkpoint(
            cfg, str(trained), max_clips=max_clips, include_clip_metrics=include_clips
        ),
    }
    comparable = (
        "mean_ep_len",
        "survival_rate",
        "mpjpe_l_mm",
        "velocity_distance_mm_per_frame",
        "acceleration_distance_mm_per_frame2",
    )
    delta = {
        name: (
            None
            if models["trained"][name] is None or models["sonic_release"][name] is None
            else models["trained"][name] - models["sonic_release"][name]
        )
        for name in comparable
    }
    return {
        "schema_version": 1,
        "dataset": str(OmegaConf.select(cfg, "benchmark.dataset")),
        "body_names": list(G1_SONIC_BODY_NAMES),
        "metric_protocol": {
            "tracking_subset": "successful clips only",
            "aggregation": "frame-weighted micro average",
            "mpjpe_l": "pelvis-translation-aligned 14-body position error",
            "velocity": "first difference of global body positions",
            "acceleration": "second difference of global body positions",
        },
        "models": models,
        "trained_minus_release": delta,
    }


def write_sonic_benchmark(result: dict[str, Any], output: str | os.PathLike[str]) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


__all__ = [
    "SonicBodyMetricAccumulator",
    "benchmark_sonic_models",
    "evaluate_sonic_checkpoint",
    "write_sonic_benchmark",
]
