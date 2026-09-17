from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from uni_rl.algos.flash_sac.layers import safe_tanh_log_det_jacobian
from uni_rl.algos.sonic import SonicAuxLossConfig, SonicModelConfig
from uni_rl.algos.sonic.checkpoint import (
    classify_sonic_checkpoint,
    load_sonic_checkpoint_file,
)
from uni_rl.algos.sonic.flashsac import (
    SONIC_ACTION_SCALE,
    SONIC_ACTOR_GROUP_NAMES,
    SONIC_FLASHSAC_CHECKPOINT_KIND,
    SonicFlashSACActor,
    SonicFlashSACLearner,
    SonicReleasePPOActor,
)
from uni_rl.ipc.dp_sync import DpParameterSync

_CONF_DIR = Path(__file__).parents[2] / "src" / "unilab" / "conf" / "flashsac"


def _training_config(*overrides: str):
    with initialize_config_dir(config_dir=str(_CONF_DIR), version_base="1.3"):
        return compose(
            config_name="config_sonic",
            overrides=["task=g1_sonic/mujoco", *overrides],
        )


def _config() -> SonicModelConfig:
    return SonicModelConfig(
        num_future_frames=2,
        num_tokens=1,
        token_dim=4,
        fsq_levels=4,
        actor_obs_dim=6,
        g1_frame_dim=7,
        smpl_frame_dim=8,
        action_dim=2,
        g1_encoder_hidden_dims=(8,),
        smpl_encoder_hidden_dims=(8,),
        g1_motion_decoder_hidden_dims=(8,),
        g1_control_decoder_hidden_dims=(8,),
    )


def _packed(batch_size: int = 4) -> torch.Tensor:
    cfg = _config()
    values = torch.randn(batch_size, cfg.actor_obs_dim + cfg.g1_input_dim + cfg.smpl_input_dim)
    encoder_index = torch.tensor([[1.0, 1.0], [1.0, 0.0]]).repeat((batch_size + 1) // 2, 1)[
        :batch_size
    ]
    return torch.cat((values, encoder_index), dim=-1)


def test_sonic_flashsac_actor_is_bounded_and_backpropagates_auxiliary_loss() -> None:
    cfg = _config()
    actor = SonicFlashSACActor(
        cfg,
        SonicAuxLossConfig(),
        action_low=torch.full((cfg.action_dim,), -1.0),
        action_high=torch.full((cfg.action_dim,), 1.0),
    )

    actions, info = actor(_packed(), training=True)

    assert actions.shape == (4, 2)
    assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)
    assert torch.isfinite(info["log_prob"]).all()
    assert torch.isfinite(info["auxiliary_loss"])
    (actions.square().mean() + info["auxiliary_loss"]).backward()
    assert actor.backbone.decoders["g1_dyn"].module[0].weight.grad is not None


def test_flashsac_actor_can_skip_legacy_g1_dyn_decoder(monkeypatch) -> None:
    cfg = _config()
    actor = SonicFlashSACActor(
        cfg,
        SonicAuxLossConfig(),
        action_low=torch.full((cfg.action_dim,), -1.0),
        action_high=torch.full((cfg.action_dim,), 1.0),
        actor_hidden_dim=8,
        actor_num_blocks=1,
        compute_action_decoder=False,
    )

    def fail_decode(*_args, **_kwargs):
        raise AssertionError("legacy g1_dyn decoder must be skipped")

    monkeypatch.setattr(actor.backbone, "decode_action_features", fail_decode)
    actions, _info = actor(_packed(), training=True)
    assert actions.shape == (4, cfg.action_dim)


def test_sonic_flashsac_deterministic_action_matches_decoder_mean() -> None:
    cfg = _config()
    actor = SonicFlashSACActor(
        cfg,
        SonicAuxLossConfig(),
        action_low=torch.full((cfg.action_dim,), -1.0),
        action_high=torch.full((cfg.action_dim,), 1.0),
    )
    packed = _packed(2)
    actor_obs, g1, smpl, encoder_index = actor._unpack(packed)
    backbone_out = actor.backbone(actor_obs, g1, smpl, encoder_index)
    hidden = actor.backbone.decode_action_features(backbone_out.selected_tokens, actor_obs)
    expected, _ = actor.action_head.get_mean_and_std(hidden)

    actual = actor.explore(packed, deterministic=True)

    torch.testing.assert_close(actual, torch.tanh(expected), atol=1e-5, rtol=1e-5)


def test_sonic_native_action_uses_flashsac_native_squash() -> None:
    cfg = _config()
    actor = SonicFlashSACActor(
        cfg,
        SonicAuxLossConfig(),
        action_low=torch.full((cfg.action_dim,), -1.0),
        action_high=torch.full((cfg.action_dim,), 1.0),
    )
    packed = _packed(2)
    actor_obs, g1, smpl, encoder_index = actor._unpack(packed)
    backbone_out = actor.backbone(actor_obs, g1, smpl, encoder_index)
    hidden = actor.backbone.decode_action_features(backbone_out.selected_tokens, actor_obs)
    expected, _ = actor.action_head.get_mean_and_std(hidden)
    actual = actor.explore_native(packed)
    torch.testing.assert_close(actual, torch.tanh(expected), atol=1e-5, rtol=1e-5)


def test_sonic_release_actor_uses_raw_g1_dyn_mean() -> None:
    cfg = _config()
    actor = SonicReleasePPOActor(cfg, SonicAuxLossConfig())
    with torch.no_grad():
        decoder = actor.backbone.decoders["g1_dyn"].module
        decoder[-1].weight.zero_()
        decoder[-1].bias.fill_(2.0)
    action = actor.explore_native(_packed(2))
    torch.testing.assert_close(action, torch.full_like(action, 2.0))
    assert not hasattr(actor, "action_head")


def test_sonic_log_prob_uses_normalized_tanh_entropy_convention() -> None:
    """Affine action scaling must not shift SONIC temperature/entropy stats."""
    cfg = _config()
    actor = SonicFlashSACActor(
        cfg,
        SonicAuxLossConfig(),
        action_low=torch.full((cfg.action_dim,), -1.0),
        action_high=torch.full((cfg.action_dim,), 1.0),
    )
    packed = _packed(3)
    mean, std, _ = actor._policy_parameters(packed, compute_auxiliary=False)
    torch.manual_seed(1234)
    actions, info = actor(packed, training=False)
    normalized = ((actions - actor.action_bias) / actor.action_scale).clamp(
        -1.0 + 1.0e-6, 1.0 - 1.0e-6
    )
    raw = torch.atanh(normalized)
    distribution = torch.distributions.Normal(mean, std, validate_args=False)
    expected = (distribution.log_prob(raw) - safe_tanh_log_det_jacobian(raw)).sum(dim=-1)
    torch.testing.assert_close(info["log_prob"], expected, atol=1.0e-5, rtol=1.0e-5)


def test_sonic_std_head_is_conditioned_on_flashsac_actor_hidden() -> None:
    cfg = _config()
    actor = SonicFlashSACActor(
        cfg,
        SonicAuxLossConfig(),
        action_low=torch.full((cfg.action_dim,), -1.0),
        action_high=torch.full((cfg.action_dim,), 1.0),
    )
    first = _packed(1)
    second = first.clone()
    # Reference terms influence the selected token and therefore the g1_dyn
    # hidden feature used by the shared SAC action head.
    second[:, cfg.actor_obs_dim : -2] += 100.0
    _, std_first, _ = actor._policy_parameters(first, compute_auxiliary=False)
    _, std_second, _ = actor._policy_parameters(second, compute_auxiliary=False)
    assert not torch.allclose(std_first, std_second)


def test_sonic_decoder_overflow_keeps_finite_nonzero_gradient() -> None:
    cfg = _config()
    actor = SonicFlashSACActor(
        cfg,
        SonicAuxLossConfig(),
        action_low=torch.full((cfg.action_dim,), -1.0),
        action_high=torch.full((cfg.action_dim,), 1.0),
    )
    with torch.no_grad():
        actor.action_head.mean_bias.fill_(5.0)
    actions, info = actor(_packed(2), training=True)
    assert torch.isfinite(actions).all()
    assert info["sonic_decoder_action_overflow"] == 1.0
    actions.mean().backward()
    grad = actor.action_head.mean_bias.grad
    assert grad is not None and torch.isfinite(grad).all() and torch.any(grad != 0)


def test_release_tokenizer_terms_are_reassembled_per_frame_for_the_backbone() -> None:
    """The release flattens observation-manager terms before concatenating them."""
    cfg = SonicModelConfig.smoke()
    actor = SonicFlashSACActor(
        cfg,
        SonicAuxLossConfig(),
        action_low=torch.full((cfg.action_dim,), -20.0),
        action_high=torch.full((cfg.action_dim,), 20.0),
    )
    command = torch.arange(10 * 58, dtype=torch.float32).reshape(1, 10, 58)
    root_ori = torch.arange(10 * 6, dtype=torch.float32).reshape(1, 10, 6) + 1_000
    human = torch.arange(10 * 78, dtype=torch.float32).reshape(1, 10, 78) + 2_000
    wrist = torch.arange(10 * 6, dtype=torch.float32).reshape(1, 10, 6) + 3_000
    packed = torch.cat(
        (
            torch.zeros(1, cfg.actor_obs_dim),
            command.flatten(1),
            root_ori.flatten(1),
            human.flatten(1),
            wrist.flatten(1),
            torch.tensor([[1.0, 0.0]]),
        ),
        dim=-1,
    )

    _, g1_reference, smpl_reference, _ = actor._unpack(packed)

    torch.testing.assert_close(g1_reference, torch.cat((command, root_ori), dim=-1))
    torch.testing.assert_close(smpl_reference, torch.cat((human, wrist), dim=-1))


def test_sonic_flashsac_learner_updates_and_marks_checkpoint(tmp_path: Path) -> None:
    cfg = _config()
    input_dim = cfg.actor_obs_dim + cfg.g1_input_dim + cfg.smpl_input_dim + 2
    learner = SonicFlashSACLearner(
        model_config=cfg,
        auxiliary_config=SonicAuxLossConfig(),
        action_low=torch.full((cfg.action_dim,), -20.0),
        action_high=torch.full((cfg.action_dim,), 20.0),
        actor_group_names=SONIC_ACTOR_GROUP_NAMES,
        actor_group_dims=(cfg.actor_obs_dim, cfg.g1_input_dim, cfg.smpl_input_dim, 2),
        log_std_min=-5.0,
        log_std_max=0.0,
        obs_dim=input_dim,
        action_dim=cfg.action_dim,
        critic_obs_dim=5,
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        actor_num_blocks=1,
        critic_num_blocks=1,
        num_atoms=5,
        device="cpu",
        normalize_reward=False,
        pretrained_checkpoint=None,
        freeze_sonic_backbone=False,
        use_cuda_graph_actor=True,
        use_cuda_graph_actor_packed_staging=True,
    )
    batch_size = 4
    batch = {
        "obs": _packed(batch_size),
        "next_obs": _packed(batch_size),
        "critic": torch.randn(batch_size, 5),
        "next_critic": torch.randn(batch_size, 5),
        "actions": torch.zeros(batch_size, cfg.action_dim),
        "rewards": torch.randn(batch_size),
        "dones": torch.zeros(batch_size),
        "truncated": torch.zeros(batch_size),
    }

    critic_metrics = learner.update_critic(batch)
    actor_metrics = learner.update_actor(batch)
    state = learner.get_state_dict()

    assert torch.isfinite(torch.tensor(critic_metrics["critic_loss"]))
    assert torch.isfinite(torch.tensor(actor_metrics["actor_loss"]))
    assert learner.use_cuda_graph_actor is True
    assert learner.use_cuda_graph_actor_packed_staging is True
    assert "sonic_total" in actor_metrics
    assert "sonic_cycle_consistency" in actor_metrics
    assert "sonic_latent_alignment" in actor_metrics
    assert "sonic_reconstruction" in actor_metrics
    assert "action_mean" in actor_metrics
    assert "q_mean" in actor_metrics
    assert state["checkpoint_kind"] == SONIC_FLASHSAC_CHECKPOINT_KIND
    assert state["sonic_action_scale"] == SONIC_ACTION_SCALE
    assert state["sonic_std_conditioning"] == "flashsac_actor_hidden"
    assert state["actor_group_names"] == SONIC_ACTOR_GROUP_NAMES
    assert classify_sonic_checkpoint(state) == "unilab"
    checkpoint_path = tmp_path / "model_1.pt"
    torch.save(state, checkpoint_path)
    state = load_sonic_checkpoint_file(checkpoint_path)
    assert classify_sonic_checkpoint(state) == "unilab"

    from unilab.training.offpolicy_sonic import build_play_actor

    play_cfg = OmegaConf.create(
        {
            "algo": {
                "sonic": {
                    "enabled": True,
                    "model": state["sonic_model_config"],
                    "auxiliary": state["sonic_auxiliary_config"],
                    "log_std_min": -9.0,
                    "log_std_max": 1.0,
                },
                "algo_params": {"actor_noise_zeta_mu": 2.0, "actor_noise_zeta_max": 4},
            }
        }
    )
    restored = build_play_actor(
        play_cfg,
        state,
        obs_dim=input_dim,
        action_low=torch.full((cfg.action_dim,), -20.0),
        action_high=torch.full((cfg.action_dim,), 20.0),
        device="cpu",
    )
    restored.load_state_dict(state["actor"])
    assert restored.log_std_min == -5.0
    assert restored.log_std_max == 0.0
    play_obs = _packed(2)
    torch.testing.assert_close(
        restored.explore(play_obs, deterministic=True),
        learner.actor.explore(play_obs, deterministic=True),
    )


def test_sonic_mujoco_builder_uses_shared_multi_gpu_contract(monkeypatch, tmp_path: Path) -> None:
    from unilab.training import sonic_double_buffer as owner_module
    from unilab.training import offpolicy_sonic

    cfg = _training_config("training.devices=[0,1]", "training.nan_guard.enabled=false")
    monkeypatch.setattr(offpolicy_sonic.os, "cpu_count", lambda: 128)
    monkeypatch.setattr(offpolicy_sonic, "build_offpolicy_env_cfg_override", lambda *a, **k: {})
    monkeypatch.setattr(offpolicy_sonic, "apply_torch_thread_runtime", lambda *a, **k: None)
    monkeypatch.setattr(
        owner_module,
        "build_sonic_flashsac_runner",
        lambda _cfg, **kwargs: kwargs,
    )

    kwargs = offpolicy_sonic.build_runner("flashsac", cfg, log_dir=str(tmp_path))

    assert cfg.training.sim_backend == "mujoco"
    assert kwargs["device"] == "cuda:0"
    assert kwargs["collector_cpu_ids"] == list(range(64))
    assert isinstance(kwargs["dp_sync"], DpParameterSync)
    assert kwargs["dp_sync"].world_size == 2
    assert kwargs["dp_sync"].rank == 0
    assert kwargs["dp_sync"].backend == "nccl"
    assert "max_episode_seconds" not in kwargs["env_cfg_override"]


def test_sonic_play_time_limit_override_does_not_mutate_training_override(monkeypatch) -> None:
    from unilab.training import offpolicy_sonic

    cfg = _training_config()
    training_override = {
        "motion_file": "robot",
        "smpl_motion_file": "smpl",
        "observations": {"policy": {"terms": {"obs": {"func": "actor"}}}},
    }
    monkeypatch.setattr(
        offpolicy_sonic,
        "build_offpolicy_env_cfg_override",
        lambda *args, **kwargs: training_override,
    )

    play_override = offpolicy_sonic.build_play_env_cfg_override("flashsac", cfg)

    assert "max_episode_seconds" not in play_override
    assert play_override["observations"]["policy"]["terms"]["obs"]["sonic_noise"] == {"level": 0.0}
    assert "max_episode_seconds" not in training_override
    assert "sonic_noise" not in training_override["observations"]["policy"]["terms"]["obs"]

    cfg.training.play_max_episode_seconds = 3.5
    finite_play_override = offpolicy_sonic.build_play_env_cfg_override("flashsac", cfg)
    assert finite_play_override["max_episode_seconds"] == 3.5


def test_sonic_release_play_uses_gear_sonic_base_action_scale() -> None:
    from unilab.training.offpolicy_sonic import _apply_sonic_play_action_scale

    env_override = {"actions": {"joint_pos": {"scale": 2.0}}}
    _apply_sonic_play_action_scale(env_override, checkpoint_format="sonic_release")
    assert env_override["actions"]["joint_pos"]["scale"] == pytest.approx(0.25)


def test_unilab_flashsac_play_preserves_training_action_scale() -> None:
    from unilab.training.offpolicy_sonic import _apply_sonic_play_action_scale

    env_override = {"actions": {"joint_pos": {"scale": 2.0}}}
    _apply_sonic_play_action_scale(env_override, checkpoint_format="unilab")
    assert env_override["actions"]["joint_pos"]["scale"] == 2.0


def test_sonic_checkpoint_format_detection_is_explicit() -> None:
    assert classify_sonic_checkpoint({"policy_state_dict": {}}) == "sonic_release"
