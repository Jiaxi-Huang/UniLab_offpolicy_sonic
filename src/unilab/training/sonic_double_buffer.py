"""SONIC-owned FlashSAC builder using the unchanged shared runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from gymnasium.spaces import Box
from omegaconf import DictConfig, OmegaConf
from uni_rl.algos.sonic.config import SonicAuxLossConfig, SonicModelConfig
from uni_rl.algos.sonic.flashsac import (
    SONIC_ACTOR_GROUP_NAMES,
    SonicFlashSACLearner,
)
from uni_rl.ipc.replay_pipelines.gpu_resident import require_offpolicy_replay_device
from uni_rl.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner

from unilab.base.env_factory import registry_env_factory
from unilab.envs import ManagerBasedRlEnv
from unilab.tasks.motion_tracking.g1.sonic_manager import SonicJointPositionAction
from unilab.training import create_env, ensure_registries
from unilab.utils.device import get_default_device
from unilab.utils.nan_guard import NanGuardCfg
from unilab.utils.seed import apply_training_seed

if TYPE_CHECKING:
    from uni_rl.ipc.dp_sync import DpParameterSync


def _string_keyed_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} keys must be strings, got {key!r}")
        result[key] = item
    return result


def _model_config(cfg: DictConfig) -> SonicModelConfig:
    values = _string_keyed_mapping(
        OmegaConf.to_container(cfg.algo.sonic.model, resolve=True),
        "algo.sonic.model",
    )
    for key in (
        "g1_encoder_hidden_dims",
        "smpl_encoder_hidden_dims",
        "g1_motion_decoder_hidden_dims",
        "g1_control_decoder_hidden_dims",
    ):
        if key in values:
            values[key] = tuple(values[key])
    return SonicModelConfig(**values)


def _auxiliary_config(cfg: DictConfig) -> SonicAuxLossConfig:
    values = _string_keyed_mapping(
        OmegaConf.to_container(cfg.algo.sonic.auxiliary, resolve=True),
        "algo.sonic.auxiliary",
    )
    return SonicAuxLossConfig(**values)


def build_sonic_flashsac_runner(
    cfg: DictConfig,
    *,
    env_cfg_override: dict[str, Any] | None,
    replay_prefetch_mode: str,
    device: str | None = None,
    nan_guard_cfg: NanGuardCfg | None = None,
    torch_thread_runtime: dict[str, Any] | None = None,
    collector_cpu_ids: list[int] | None = None,
    dp_sync: DpParameterSync | None = None,
) -> DoubleBufferOffPolicyRunner:
    """Build SONIC on UniLab FlashSAC without modifying the generic builder."""

    if not bool(cfg.algo.sonic.enabled):
        raise ValueError("SONIC runner requires algo.sonic.enabled=true")

    from uni_rl.utils.observations import get_obs_dims

    if replay_prefetch_mode != "one_tick":
        raise ValueError("SONIC FlashSAC requires replay_prefetch_mode='one_tick'")
    if cfg.algo.algo_params.n_step != 1:
        raise ValueError("SONIC FlashSAC supports n_step=1 only")
    device = require_offpolicy_replay_device(device or get_default_device())
    ensure_registries()
    apply_training_seed(cfg.algo.seed, torch_runtime=True, cuda=True)
    env = create_env(cfg, num_envs=1, env_cfg_override=env_cfg_override)
    try:
        obs_dim, critic_obs_dim = get_obs_dims(env.obs_groups_spec)
        action_space = env.action_space
        if not isinstance(action_space, Box):
            raise TypeError("SONIC requires a continuous Box action space")
        action_shape = action_space.shape
        if action_shape is None:
            raise ValueError("SONIC action_space.shape must be defined")
        action_dim = int(action_shape[0])
        # Sonic policy outputs are normalized; the task action term owns the
        # physical (scalar or per-joint) position scale.
        action_low = torch.full((action_dim,), -1.0, dtype=torch.float32)
        action_high = torch.full((action_dim,), 1.0, dtype=torch.float32)
        # Reference-mode BC inverts the env action contract (default offset +
        # per-joint scale, policy order) to build expert actions from the
        # packed reference terms.
        bc_kwargs: dict[str, Any] = {}
        if str(cfg.algo.algo_params.actor_bc_target) == "reference":
            action_term = cast(ManagerBasedRlEnv, env).action_manager.get_term("joint_pos")
            if not isinstance(action_term, SonicJointPositionAction):
                raise TypeError("reference BC requires the SONIC joint position action term")
            default_offset, action_scale = action_term.resolved_action_contract()
            bc_kwargs["bc_joint_default"] = torch.as_tensor(default_offset, dtype=torch.float32)
            bc_kwargs["bc_action_scale"] = torch.as_tensor(action_scale, dtype=torch.float32)
    finally:
        env.close()

    model_config = _model_config(cfg)
    model_config.validate_observation_contract()
    expected_obs_dim = (
        model_config.actor_obs_dim + model_config.g1_input_dim + model_config.smpl_input_dim + 2
    )
    if obs_dim != expected_obs_dim:
        raise ValueError(f"SONIC packed env obs dim must be {expected_obs_dim}, got {obs_dim}")
    learner = SonicFlashSACLearner(
        model_config=model_config,
        auxiliary_config=_auxiliary_config(cfg),
        action_low=action_low,
        action_high=action_high,
        actor_group_names=SONIC_ACTOR_GROUP_NAMES,
        actor_group_dims=(
            model_config.actor_obs_dim,
            model_config.g1_input_dim,
            model_config.smpl_input_dim,
            2,
        ),
        log_std_min=float(cfg.algo.algo_params.log_std_min),
        log_std_max=float(cfg.algo.algo_params.log_std_max),
        obs_dim=obs_dim,
        action_dim=action_dim,
        critic_obs_dim=critic_obs_dim,
        device=device,
        gamma=cfg.algo.gamma,
        tau=cfg.algo.tau,
        actor_lr=cfg.algo.actor_lr,
        critic_lr=cfg.algo.critic_lr,
        actor_hidden_dim=model_config.actor_hidden_dim or int(cfg.algo.actor_hidden_dim),
        critic_hidden_dim=model_config.critic_hidden_dim,
        actor_num_blocks=cfg.algo.algo_params.actor_num_blocks,
        critic_num_blocks=cfg.algo.algo_params.critic_num_blocks,
        num_atoms=cfg.algo.num_atoms,
        critic_min_v=cfg.algo.algo_params.critic_min_v,
        critic_max_v=cfg.algo.algo_params.critic_max_v,
        temp_initial_value=cfg.algo.algo_params.temp_initial_value,
        temp_target_sigma=cfg.algo.algo_params.temp_target_sigma,
        temp_target_entropy=cfg.algo.algo_params.temp_target_entropy,
        actor_bc_alpha=cfg.algo.algo_params.actor_bc_alpha,
        actor_bc_alpha_end=cfg.algo.algo_params.actor_bc_alpha_end,
        actor_bc_target=str(cfg.algo.algo_params.actor_bc_target),
        **bc_kwargs,
        actor_noise_zeta_mu=cfg.algo.algo_params.actor_noise_zeta_mu,
        actor_noise_zeta_max=cfg.algo.algo_params.actor_noise_zeta_max,
        learning_rate_init=cfg.algo.algo_params.learning_rate_init,
        learning_rate_peak=cfg.algo.algo_params.learning_rate_peak,
        learning_rate_end=cfg.algo.algo_params.learning_rate_end,
        learning_rate_warmup_steps=cfg.algo.algo_params.learning_rate_warmup_steps,
        learning_rate_decay_steps=cfg.algo.algo_params.learning_rate_decay_steps,
        normalize_reward=cfg.algo.algo_params.normalize_reward,
        normalized_g_max=cfg.algo.algo_params.normalized_g_max,
        n_step=cfg.algo.algo_params.n_step,
        obs_normalization=bool(cfg.algo.obs_normalization),
        use_amp=cfg.training.use_amp,
        amp_dtype=cfg.algo.algo_params.amp_dtype,
        use_compile=bool(cfg.algo.algo_params.use_compile),
        use_cuda_graph_critic=cfg.algo.algo_params.use_cuda_graph_critic,
        use_cuda_graph_actor=cfg.algo.algo_params.use_cuda_graph_actor,
        use_cuda_graph_critic_packed_staging=(
            cfg.algo.algo_params.use_cuda_graph_critic_packed_staging
        ),
        use_cuda_graph_actor_packed_staging=(
            cfg.algo.algo_params.use_cuda_graph_actor_packed_staging
        ),
        pretrained_checkpoint=OmegaConf.select(cfg, "algo.sonic.finetune_checkpoint", default=None),
        freeze_sonic_backbone=bool(
            OmegaConf.select(cfg, "algo.sonic.freeze_backbone", default=False)
        ),
    )
    return DoubleBufferOffPolicyRunner(
        learner=learner,
        env_name=cfg.training.task_name,
        algo_type="flashsac",
        env_factory=registry_env_factory(
            str(cfg.training.task_name), str(cfg.training.sim_backend)
        ),
        num_envs=cfg.algo.num_envs,
        replay_buffer_n=cfg.algo.replay_buffer_n,
        batch_size=cfg.algo.batch_size,
        learning_starts=cfg.algo.learning_starts,
        updates_per_step=cfg.algo.updates_per_step,
        policy_frequency=cfg.algo.policy_frequency,
        env_steps_per_sync=cfg.training.env_steps_per_sync,
        device=device,
        obs_normalization=bool(cfg.algo.obs_normalization),
        sim_backend=cfg.training.sim_backend,
        env_cfg_override=env_cfg_override,
        seed=cfg.algo.seed,
        trace_enabled=cfg.training.trace_enabled,
        trace_output_dir=cfg.training.trace_output_dir,
        trace_thread_time=cfg.training.trace_thread_time,
        trace_cuda_events=cfg.training.trace_cuda_events,
        replay_prefetch_mode=replay_prefetch_mode,
        nan_guard_cfg=nan_guard_cfg,
        torch_thread_runtime=torch_thread_runtime,
        collector_cpu_ids=collector_cpu_ids,
        dp_sync=dp_sync,
    )


__all__ = ["build_sonic_flashsac_runner"]
