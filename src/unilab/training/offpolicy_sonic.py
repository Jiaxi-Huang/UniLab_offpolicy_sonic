"""SONIC-only assembly for UniLab's unchanged off-policy runtime."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium.spaces import Box
from omegaconf import DictConfig, OmegaConf

from uni_rl.offpolicy.thread_budget import (
    apply_torch_thread_runtime,
    resolve_torch_thread_runtime,
)
from uni_rl.algos.sonic import (
    SonicAuxLossConfig,
    SonicModelConfig,
    classify_sonic_checkpoint,
    load_sonic_checkpoint,
    load_sonic_checkpoint_file,
)
from uni_rl.algos.sonic.flashsac import (
    SONIC_ACTOR_GROUP_NAMES,
    SONIC_FLASHSAC_CHECKPOINT_KIND,
    SonicFlashSACActor,
    SonicReleasePPOActor,
)
from unisim.backend.base import log_playback_plan
from unilab.base.config_adapter import create_env
from unilab.base.np_env import NpEnv
from uni_rl.ipc.dp_launcher import (
    apply_dp_rank_config,
    current_dp_rank,
    resolve_collector_cpu_ids,
    resolve_dp_rank_device,
    resolve_dp_rendezvous_path,
    resolve_dp_topology,
)
from unilab.utils.checkpoint import resolve_offpolicy_checkpoint_path
from unilab.utils.nan_guard import NanGuardCfg
from unilab.utils.sim2sim import policy_load_dim_guard, resolve_sim2sim_config
from unilab.visualization.interactive_playback import (
    build_offpolicy_env_cfg_override,
    default_device,
    resolve_play_obs_dims,
)

ROOT_DIR = Path(__file__).parents[3]


def _model_config(values: Any) -> SonicModelConfig:
    if OmegaConf.is_config(values):
        values = OmegaConf.to_container(values, resolve=True)
    if not isinstance(values, dict):
        raise ValueError("SONIC model configuration must be a mapping")
    values = dict(values)
    # Algorithm-level actor width is not part of the SONIC backbone schema.
    values.pop("actor_hidden_dim", None)
    for key in (
        "g1_encoder_hidden_dims",
        "smpl_encoder_hidden_dims",
        "g1_motion_decoder_hidden_dims",
        "g1_control_decoder_hidden_dims",
    ):
        if key in values:
            values[key] = tuple(values[key])
    return SonicModelConfig(**values)


def _auxiliary_config(values: Any) -> SonicAuxLossConfig:
    if OmegaConf.is_config(values):
        values = OmegaConf.to_container(values, resolve=True)
    if not isinstance(values, dict):
        raise ValueError("SONIC auxiliary configuration must be a mapping")
    return SonicAuxLossConfig(**dict(values))


def build_runner(algo_name: str, cfg: DictConfig, log_dir: str | None = None):
    """Build the SONIC FlashSAC runner while preserving generic dispatch."""

    if algo_name != "flashsac":
        raise ValueError("train_offpolicy_sonic.py only supports algo=flashsac")
    model_cfg = _model_config(cfg.algo.sonic.model)
    model_cfg.validate_observation_contract()
    print(
        "SONIC training contract: "
        f"profile_frames={model_cfg.num_future_frames}, "
        f"obs_dim={model_cfg.actor_obs_dim + model_cfg.g1_input_dim + model_cfg.smpl_input_dim + 2}, "
        f"backbone_params={model_cfg.parameter_count():,}, "
        "training_action_scale=2.0 x effort_limit/stiffness; "
        "release_play_base_scale=0.25 x effort_limit/stiffness"
    )
    env_cfg_override = build_offpolicy_env_cfg_override(algo_name, cfg, root_dir=ROOT_DIR)
    dp_devices = resolve_dp_topology(cfg.training.devices)
    dp_world_size = len(dp_devices) if dp_devices is not None else 1
    dp_rank = current_dp_rank()
    from unilab.utils.device import get_default_device

    rank_device = resolve_dp_rank_device(dp_devices, dp_rank) or get_default_device()
    host_cpu_count = os.cpu_count() or 1
    explicit_cpu_ids = getattr(cfg.training, "dp_collector_cpu_ids", None)
    if explicit_cpu_ids is not None:
        explicit_cpu_ids = OmegaConf.to_container(explicit_cpu_ids, resolve=True)
        if not isinstance(explicit_cpu_ids, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in explicit_cpu_ids
        ):
            raise ValueError("training.dp_collector_cpu_ids must be a list of integer CPU ids")
    collector_cpu_ids = resolve_collector_cpu_ids(
        dp_world_size,
        dp_rank,
        host_cpu_count,
        explicit=explicit_cpu_ids,
    )

    dp_sync = None
    if dp_world_size > 1:
        if log_dir is None:
            raise ValueError("SONIC multi-GPU runner requires log_dir on every rank")
        from uni_rl.ipc.dp_sync import DpParameterSync

        dp_sync = DpParameterSync(
            world_size=dp_world_size,
            rank=dp_rank,
            rendezvous_path=resolve_dp_rendezvous_path(log_dir, rank=dp_rank),
            device=rank_device,
        )

    torch_thread_runtime = resolve_torch_thread_runtime(
        getattr(cfg.training, "torch_threads", None),
        cpu_count=host_cpu_count // dp_world_size if dp_world_size > 1 else None,
    )
    apply_torch_thread_runtime(torch_thread_runtime, role="learner")
    nan_guard_cfg = getattr(cfg.training, "nan_guard", None)
    resolved_nan_guard: NanGuardCfg | None = None
    if nan_guard_cfg is not None and getattr(nan_guard_cfg, "enabled", False):
        resolved_nan_guard = NanGuardCfg(
            enabled=True,
            buffer_size=int(getattr(nan_guard_cfg, "buffer_size", 100)),
            max_envs_to_dump=int(getattr(nan_guard_cfg, "max_envs_to_dump", 5)),
            output_dir=getattr(nan_guard_cfg, "output_dir", None),
        )
    replay_prefetch_mode = getattr(cfg.training, "replay_prefetch_mode", "one_tick")
    from uni_rl.ipc.replay_pipelines.gpu_resident import require_offpolicy_replay_device

    replay_device = require_offpolicy_replay_device(rank_device)
    from unilab.training.sonic_double_buffer import build_sonic_flashsac_runner

    return build_sonic_flashsac_runner(
        cfg,
        env_cfg_override=env_cfg_override,
        replay_prefetch_mode=replay_prefetch_mode,
        device=replay_device,
        nan_guard_cfg=resolved_nan_guard,
        torch_thread_runtime=torch_thread_runtime,
        collector_cpu_ids=collector_cpu_ids,
        dp_sync=dp_sync,
    )


def _resolve_checkpoint_format(cfg: DictConfig, checkpoint: dict[str, Any]) -> str:
    detected = classify_sonic_checkpoint(checkpoint)
    configured = str(OmegaConf.select(cfg, "algo.sonic.checkpoint_format", default="auto"))
    if configured not in ("auto", "sonic_release", "unilab"):
        raise ValueError("algo.sonic.checkpoint_format must be auto, sonic_release, or unilab")
    if configured != "auto" and configured != detected:
        raise ValueError(
            f"Configured SONIC checkpoint format {configured!r} does not match "
            f"detected format {detected!r}"
        )
    return detected


def build_play_actor(
    cfg: DictConfig,
    checkpoint: dict[str, Any],
    *,
    obs_dim: int,
    action_low: Any,
    action_high: Any,
    device: str,
) -> SonicFlashSACActor | SonicReleasePPOActor:
    checkpoint_format = _resolve_checkpoint_format(cfg, checkpoint)
    model_config = _model_config(checkpoint.get("sonic_model_config", cfg.algo.sonic.model))
    model_config.validate_observation_contract()
    auxiliary_config = _auxiliary_config(
        checkpoint.get("sonic_auxiliary_config", cfg.algo.sonic.auxiliary)
    )
    expected_group_dims = (
        model_config.actor_obs_dim,
        model_config.g1_input_dim,
        model_config.smpl_input_dim,
        2,
    )
    if "actor_group_names" in checkpoint:
        if (
            tuple(checkpoint["actor_group_names"]) != SONIC_ACTOR_GROUP_NAMES
            or tuple(int(dim) for dim in checkpoint["actor_group_dims"]) != expected_group_dims
        ):
            raise ValueError("UniLab SONIC checkpoint actor-group metadata is incompatible")
    if checkpoint_format == "sonic_release":
        actor = SonicReleasePPOActor(model_config, auxiliary_config, device=device)
        if actor.input_dim != obs_dim:
            raise ValueError(f"SONIC play actor expects obs dim {actor.input_dim}, got {obs_dim}")
        actor.eval()
        return actor

    cfg_log_std_min = OmegaConf.select(cfg, "algo.algo_params.log_std_min", default=-5.0)
    cfg_log_std_max = OmegaConf.select(cfg, "algo.algo_params.log_std_max", default=-2.0)
    is_internal = checkpoint.get("checkpoint_kind") == SONIC_FLASHSAC_CHECKPOINT_KIND
    if is_internal and checkpoint.get("sonic_std_conditioning") != "flashsac_actor_hidden":
        raise ValueError(
            "UniLab SONIC checkpoint uses unsupported std conditioning; "
            "retrain with the current SONIC FlashSAC policy head"
        )
    actor = SonicFlashSACActor(
        model_config,
        auxiliary_config,
        action_low=torch.as_tensor(action_low, dtype=torch.float32, device=device),
        action_high=torch.as_tensor(action_high, dtype=torch.float32, device=device),
        log_std_min=float(checkpoint.get("sonic_log_std_min", cfg_log_std_min)),
        log_std_max=float(checkpoint.get("sonic_log_std_max", cfg_log_std_max)),
        noise_zeta_mu=float(
            checkpoint.get("sonic_noise_zeta_mu", cfg.algo.algo_params.actor_noise_zeta_mu)
        ),
        noise_zeta_max=int(
            checkpoint.get("sonic_noise_zeta_max", cfg.algo.algo_params.actor_noise_zeta_max)
        ),
        actor_hidden_dim=(
            int(checkpoint["sonic_actor_hidden_dim"])
            if "sonic_actor_hidden_dim" in checkpoint
            else int(OmegaConf.select(cfg, "algo.actor_hidden_dim", default=128))
        ),
        actor_num_blocks=int(
            checkpoint.get(
                "sonic_actor_num_blocks",
                OmegaConf.select(cfg, "algo.algo_params.actor_num_blocks", default=2),
            )
        ),
        compute_action_decoder=True,
        device=device,
    )
    if actor.input_dim != obs_dim:
        raise ValueError(f"SONIC play actor expects obs dim {actor.input_dim}, got {obs_dim}")
    actor.eval()
    return actor


def build_play_env_cfg_override(algo_name: str, cfg: DictConfig) -> dict[str, Any]:
    """Build SONIC play overrides without changing the training env limit."""

    env_cfg_override = deepcopy(
        build_offpolicy_env_cfg_override(algo_name, cfg, root_dir=ROOT_DIR) or {}
    )
    try:
        actor_obs = env_cfg_override["observations"]["policy"]["terms"]["obs"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "SONIC play requires observations.policy.terms.obs in the task owner config"
        ) from exc
    if not isinstance(actor_obs, dict):
        raise TypeError("SONIC policy observation override must be a mapping")
    actor_obs["sonic_noise"] = {"level": 0.0}
    if "play_max_episode_seconds" not in cfg.training:
        return env_cfg_override

    play_limit = cfg.training.play_max_episode_seconds
    if play_limit is None:
        return env_cfg_override
    play_limit = float(play_limit)
    if play_limit <= 0.0:
        raise ValueError("training.play_max_episode_seconds must be positive or null")
    env_cfg_override["max_episode_seconds"] = play_limit
    return env_cfg_override


def _align_play_env_to_checkpoint(
    env_cfg_override: dict[str, Any], checkpoint: dict[str, Any]
) -> None:
    """Align temporal observation widths with a UniLab SONIC checkpoint.

    Task owners normally default to the released 10-frame contract, while
    locally trained profiles (for example LAFAN) may use a different
    ``num_future_frames``.  The checkpoint is authoritative for playback;
    otherwise the actor and packed environment dimensions disagree before the
    first inference step.
    """
    model_config = checkpoint.get("sonic_model_config")
    if not isinstance(model_config, dict) or "num_future_frames" not in model_config:
        return
    frames = int(model_config["num_future_frames"])
    if frames <= 0:
        raise ValueError("SONIC checkpoint num_future_frames must be positive")
    motion_params = env_cfg_override.get("commands", {}).get("motion", {}).get("params", {})
    if isinstance(motion_params, dict):
        motion_params["num_future_frames"] = frames
    observations = env_cfg_override.get("observations", {})
    for group_name in ("policy", "critic"):
        group = observations.get(group_name, {}) if isinstance(observations, dict) else {}
        terms = group.get("terms", {}) if isinstance(group, dict) else {}
        obs_term = terms.get("obs", {}) if isinstance(terms, dict) else {}
        if isinstance(obs_term, dict):
            obs_term["sonic_history_length"] = frames


def _apply_sonic_play_action_scale(
    env_cfg_override: dict[str, Any], *, checkpoint_format: str
) -> None:
    """Apply the action scale required by the loaded SONIC checkpoint format.

    Official SONIC/PPO checkpoints use gear_sonic's base action scale ``0.25``;
    the environment expands it with the per-joint effort/stiffness ratio.
    UniLab FlashSAC checkpoints use the task owner's base scale (``2.0``).
    """

    if checkpoint_format != "sonic_release":
        return

    actions = env_cfg_override.setdefault("actions", {})
    joint_pos = actions.setdefault("joint_pos", {})
    joint_pos["scale"] = 0.25


def play_offpolicy(
    algo_name: str, cfg: DictConfig, *, stats_output: str | os.PathLike[str] | None = None
) -> str | None:
    """Play SONIC release or UniLab checkpoints through the packed adapter env."""

    if algo_name != "flashsac":
        raise ValueError("SONIC play only supports algo=flashsac")
    load_path, load_path_dir = resolve_offpolicy_checkpoint_path(
        ROOT_DIR,
        cfg.algo.algo_log_name,
        cfg.training.task_name,
        cfg.algo.load_run,
    )
    if not load_path or not os.path.exists(load_path):
        print(f"Could not find checkpoint. load_path={load_path}")
        return None
    cfg = (
        resolve_sim2sim_config(
            load_path_dir,
            cfg,
            algo_name=algo_name,
            strict=bool(getattr(cfg.training, "sim2sim_strict", True)),
        )
        or cfg
    )
    checkpoint = load_sonic_checkpoint_file(load_path)
    checkpoint_format = _resolve_checkpoint_format(cfg, checkpoint)
    env_cfg_override = build_play_env_cfg_override(algo_name, cfg)
    _align_play_env_to_checkpoint(env_cfg_override, checkpoint)
    # The action term expands each checkpoint's base scale with effort/kp.
    _apply_sonic_play_action_scale(
        env_cfg_override,
        checkpoint_format=checkpoint_format,
    )
    devices = resolve_dp_topology(cfg.training.devices)
    device = default_device(torch, resolve_dp_rank_device(devices, current_dp_rank()))
    print(f"Using device for play: {device}")
    env = create_env(
        cfg,
        num_envs=cfg.training.play_env_num,
        env_cfg_override=env_cfg_override,
    )
    if not isinstance(env, NpEnv):
        raise TypeError("SONIC playback requires the NumPy environment contract")
    obs_dim, _critic_obs_dim = resolve_play_obs_dims(env.obs_groups_spec)
    action_space = env.action_space
    if not isinstance(action_space, Box):
        raise TypeError("SONIC playback requires a continuous Box action space")
    action_shape = action_space.shape
    if action_shape is None:
        raise ValueError("env.action_space.shape must be defined")
    action_dim = int(action_shape[0])
    actor = build_play_actor(
        cfg,
        checkpoint,
        obs_dim=obs_dim,
        action_low=np.full(action_dim, -1.0, dtype=np.float32),
        action_high=np.full(action_dim, 1.0, dtype=np.float32),
        device=device,
    )
    obs_normalizer = None
    if checkpoint_format != "sonic_release" and bool(getattr(cfg.algo, "obs_normalization", False)):
        from uni_rl.algos.common.normalization import EmpiricalNormalization

        obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
        normalizer_state = checkpoint.get("obs_normalizer")
        if normalizer_state:
            obs_normalizer.load_state_dict(normalizer_state)
    print(f"Loading model: {load_path}")
    with policy_load_dim_guard(
        env_obs_dim=obs_dim,
        env_action_dim=action_dim,
        algo_name=algo_name,
    ):
        if checkpoint_format == "sonic_release":
            load_sonic_checkpoint(actor.backbone, load_path)
        else:
            actor.load_state_dict(checkpoint["actor"])
    if bool(getattr(cfg.training, "export_onnx", False)):
        raise ValueError("SONIC ONNX export is not supported; set training.export_onnx=false")
    if env.state is None:
        env.init_state()

    ghost_joint_names = None
    if bool(getattr(cfg.training, "play_motion_ghost", False)):
        from unilab.tasks.motion_tracking.g1.sonic_manager import (
            sonic_reference_joint_names,
            sonic_reference_qpos,
        )

        def ghost_state_getter() -> np.ndarray:
            return sonic_reference_qpos(env)

        ghost_joint_names = sonic_reference_joint_names(env)
        print("Reference-motion ghost overlay enabled for playback video.")
    else:
        ghost_state_getter = None

    stats = None
    if stats_output is not None:
        from unilab.training.playback_stats import PlaybackStats

        stats = PlaybackStats(env, output_path=stats_output)

    def initialize() -> np.ndarray:
        observations, _info = env.reset(np.arange(cfg.training.play_env_num, dtype=np.int32))
        return np.asarray(observations["obs"], dtype=np.float32)

    def step(obs_np: np.ndarray) -> np.ndarray:
        obs_torch = torch.from_numpy(obs_np).to(device)
        if obs_normalizer is not None:
            obs_torch = obs_normalizer(obs_torch, update=False)
        actions = (
            (
                actor.explore_native(obs_torch)
                if checkpoint_format == "sonic_release"
                else actor.explore(obs_torch, deterministic=True)
            )
            .cpu()
            .numpy()
        )
        state = env.step(actions)
        if stats is not None:
            stats.observe(state)
        return np.asarray(state.obs["obs"], dtype=np.float32)

    try:
        with torch.inference_mode():
            video_path = env.run_playback_mode(
                play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
                play_steps=getattr(cfg.training, "play_steps", None),
                output_video=(
                    os.path.join(load_path_dir, "play_video.mp4") if load_path_dir else None
                ),
                initialize=initialize,
                step=step,
                camera_kwargs={
                    "cam_distance": cfg.training.cam_distance,
                    "cam_elevation": cfg.training.cam_elevation,
                    "cam_azimuth": cfg.training.cam_azimuth,
                },
                ghost_state_getter=ghost_state_getter,
                ghost_joint_names=ghost_joint_names,
                on_plan=log_playback_plan,
            )
    finally:
        if stats is not None:
            stats.write(
                metadata={
                    "algo": algo_name,
                    "checkpoint": str(load_path),
                    "play_render_mode": getattr(cfg.training, "play_render_mode", "auto"),
                }
            )
    if video_path is not None:
        print(f"Saving video to {video_path} ...")
    print("Done.")
    return video_path


def apply_rank_config(cfg: DictConfig) -> str | None:
    """Expose the same DP rank configuration helper for focused tests."""

    devices = resolve_dp_topology(cfg.training.devices)
    return apply_dp_rank_config(cfg, devices, current_dp_rank())


__all__ = [
    "apply_rank_config",
    "build_play_actor",
    "build_play_env_cfg_override",
    "_apply_sonic_play_action_scale",
    "build_runner",
    "play_offpolicy",
]
