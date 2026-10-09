"""Hydra and Manager-Based materialization tests for SONIC owners."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from uni_rl.algos.sonic.config import SonicModelConfig

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.registry import apply_cfg_overrides
from unilab.tasks.motion_tracking.g1.sonic_manager import (
    G1_SONIC_JOINTS,
    SonicJointPositionActionCfg,
    SonicMotionCommandCfg,
    _sonic_action_scale,
    _sonic_policy_action_scale,
)

CONF_DIR = Path(__file__).parents[2] / "src" / "unilab" / "conf" / "flashsac"
ROOT_DIR = Path(__file__).parents[2]


def _compose(*overrides: str):
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose(config_name="config_sonic", overrides=list(overrides))


def test_sonic_model_profiles_compose() -> None:
    release = _compose()
    smoke = _compose("sonic_model=smoke")
    assert release.algo.algo == "flashsac"
    assert release.algo.sonic.model.token_dim == 32
    assert smoke.algo.sonic.model.token_dim == 8
    assert smoke.algo.sonic.model.num_future_frames == 1
    assert "actor_obs_dim" not in smoke.algo.sonic.model
    assert smoke.env.commands.motion.params.num_future_frames == 1
    assert smoke.env.observations.policy.terms.obs.sonic_history_length == 1

    configs = []
    for profile in ("smoke", "lafan", "release"):
        cfg = _compose(f"sonic_model={profile}")
        model = cfg.algo.sonic.model
        runtime = SonicModelConfig(**dict(model))
        configs.append(
            (
                profile,
                int(model.num_future_frames),
                int(runtime.actor_obs_dim),
                int(runtime.actor_hidden_dim),
            )
        )
        assert int(runtime.actor_obs_dim) == 93 * int(model.num_future_frames)
        assert "critic_hidden_dim" in model
    assert [item[1] for item in configs] == [1, 4, 10]
    assert [item[2] for item in configs] == [93, 372, 930]
    assert [item[3] for item in configs] == [128, 256, 512]


def test_sonic_lafan_training_defaults_are_update_aligned() -> None:
    cfg = _compose("task=g1_sonic/mujoco")
    assert cfg.algo.learning_starts == 24
    assert cfg.algo.updates_per_step == 2
    assert cfg.algo.max_iterations == 200000
    assert cfg.algo.tau == pytest.approx(0.01)
    assert cfg.algo.algo_params.learning_rate_decay_steps == 150000
    assert cfg.algo.algo_params.use_compile is False


def test_sonic_training_action_scale_matches_canonical_g1_contract() -> None:
    cfg = _compose("task=g1_sonic/mujoco")
    assert float(cfg.env.actions.joint_pos.scale) == pytest.approx(2.0)


def test_sonic_training_compose_enables_full_objective_compile() -> None:
    cfg = _compose("task=g1_sonic/mujoco")
    assert cfg.algo.algo_params.use_compile is True
    assert cfg.algo.algo_params.compile_full_objectives is True


def test_sonic_motion_subset_params_default_off() -> None:
    cfg = _compose("task=g1_sonic/mujoco")
    params = cfg.env.commands.motion.params
    # The composed task YAML leaves both subset knobs unset; the dataclass
    # defaults (None) keep the legacy full-store load.
    assert "max_loaded_clips" not in params
    assert "clip_rotation_interval_steps" not in params


def test_sonic_base_randomization_matches_gear_sonic_semantics() -> None:
    cfg = _compose("task=g1_sonic/base")
    motion = cfg.env.commands.motion.params
    action = cfg.env.actions.joint_pos
    mass = cfg.env.events.base_mass
    com = cfg.env.events.base_com
    encoder_bias = cfg.env.events.encoder_bias

    assert action.simulate_action_latency is False
    assert tuple(motion.pose_range["x"]) == (-0.05, 0.05)
    assert tuple(motion.pose_range["z"]) == (-0.01, 0.01)
    assert tuple(motion.pose_range["yaw"]) == (-0.2, 0.2)
    assert tuple(motion.velocity_range["x"]) == (-0.5, 0.5)
    assert tuple(motion.velocity_range["z"]) == (-0.2, 0.2)
    assert tuple(motion.velocity_range["yaw"]) == (-0.78, 0.78)
    assert tuple(motion.joint_position_range) == (-0.1, 0.1)
    assert tuple(motion.joint_velocity_range) == (0.0, 0.0)

    assert mass.mode == "reset"
    assert list(mass.params.asset_cfg.body_names) == [
        "torso_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]
    assert tuple(mass.params.mass_distribution_params) == (0.8, 2.5)
    assert mass.params.operation == "scale"
    assert mass.params.recompute_inertia is False

    assert com.mode == "reset"
    assert com.params.asset_cfg.body_names == "torso_link"
    assert tuple(com.params.com_range.x) == (-0.025, 0.025)
    assert tuple(com.params.com_range.y) == (-0.05, 0.05)
    assert tuple(com.params.com_range.z) == (-0.05, 0.05)

    assert encoder_bias.mode == "reset"
    assert encoder_bias.params.asset_cfg.joint_names == ".*"
    assert tuple(encoder_bias.params.bias_range) == (-0.01, 0.01)


def test_sonic_tracking_points_match_gear_sonic_release_contract() -> None:
    cfg = _compose("task=g1_sonic/base")
    motion = cfg.env.commands.motion.params
    assert list(motion.reward_point_body_names) == [
        "torso_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]
    assert [list(offset) for offset in motion.reward_point_body_offsets] == [
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]


def test_released_sonic_policy_action_scale_matches_policy_order() -> None:
    scale = _sonic_policy_action_scale()
    assert scale.shape == (len(G1_SONIC_JOINTS),)
    assert scale.dtype == "float32"
    assert scale[0] == pytest.approx(0.3506614663776252)
    assert scale[2] == pytest.approx(0.5475464652193159)
    assert scale[-1] == pytest.approx(0.07450087032903109)


def test_sonic_base_scale_expands_with_effort_over_stiffness() -> None:
    base = _sonic_action_scale(2.0)
    release = _sonic_policy_action_scale()
    assert base.shape == release.shape == (len(G1_SONIC_JOINTS),)
    assert base[0] == pytest.approx(2.0 * 139.0 / 99.098427777)
    assert base[2] == pytest.approx(2.0 * 88.0 / 40.179238471)
    for actual, expected in zip(base, release):
        assert actual == pytest.approx(8.0 * expected)


def test_sonic_mujoco_local_uses_canonical_training_action_scale() -> None:
    cfg = _compose("task=g1_sonic/mujoco_local", "sonic_model=smoke")
    action = cfg.env.actions.joint_pos
    assert float(action.scale) == pytest.approx(2.0)
    assert len(action.actuator_names) == len(G1_SONIC_JOINTS)


@pytest.mark.parametrize(
    ("owner", "backend"),
    (
        ("mujoco", "mujoco"),
        ("motrix", "motrix"),
    ),
)
def test_sonic_owner_materializes_manager_contract(owner: str, backend: str) -> None:
    cfg = _compose(f"task=g1_sonic/{owner}")
    registry.ensure_registries()
    override = BackendAdapter(cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env_cfg = registry.materialize_env_config("G1SonicManager")
    apply_cfg_overrides(env_cfg, override)
    env_cfg.validate()

    command = env_cfg.commands["motion"]
    action = env_cfg.actions["joint_pos"]
    assert isinstance(command, SonicMotionCommandCfg)
    assert command.entity_name == "robot"
    assert command.resampling_time_range == [1.0e9, 1.0e9]
    if owner == "mujoco":
        assert tuple(command.params.pose_range["x"]) == (-0.0, 0.0)
        assert tuple(command.params.velocity_range["yaw"]) == (-0.0, 0.0)
        assert tuple(command.params.joint_position_range) == (-0.0, 0.0)
    else:
        assert tuple(command.params.pose_range["x"]) == (-0.05, 0.05)
        assert tuple(command.params.velocity_range["yaw"]) == (-0.78, 0.78)
        assert tuple(command.params.joint_position_range) == (-0.1, 0.1)
    assert tuple(command.params.joint_default_position_range) == (0.0, 0.0)
    assert isinstance(action, SonicJointPositionActionCfg)
    assert cfg.training.sim_backend == backend
    # Wrap-mode training contract: a clip's end resamples the reference
    # mid-episode instead of truncating it.
    assert command.truncate_on_clip_end is False
    assert env_cfg.terminations["clip_end"] is None
    assert command.anchor_pos_z_threshold == pytest.approx(0.15)
    assert command.low_reference_anchor_pos_z_threshold == pytest.approx(0.75)
    assert float(action.scale) == pytest.approx(2.0)
    assert action.simulate_action_latency is False
    assert command.params.adaptive_lambda == pytest.approx(0.8)
    assert command.params.adaptive_kernel_size == 3
    assert command.params.adaptive_attribution == "trajectory"
    assert command.params.adaptive_max_prob_per_motion == pytest.approx(30.0)
    assert set(action.actuator_names) == set(G1_SONIC_JOINTS)
    assert env_cfg.scene is not None
    assert env_cfg.scene.default_keyframe_name == "stand"
    assert env_cfg.critic_observation_group == "critic"
    assert env_cfg.scale_rewards_by_dt is True
    assert env_cfg.rewards["undesired_contacts"].weight == pytest.approx(-0.1)


@pytest.mark.parametrize("owner", ("motrix", "motrix_local"))
def test_sonic_motrix_training_disables_reset_randomization(owner: str) -> None:
    cfg = _compose(f"task=g1_sonic/{owner}", "sonic_model=smoke")
    assert cfg.training.sim_backend == "motrix"
    assert cfg.env.events.base_mass is None
    assert cfg.env.events.base_com is None
    assert cfg.env.events.encoder_bias is None


def test_sonic_motrix_play_disables_episode_timeout() -> None:
    cfg = _compose("task=g1_sonic/motrix_play", "sonic_model=smoke")
    assert cfg.env.terminations.time_out is None
    assert cfg.env.commands.motion.params.sampling_mode == "clip_start"
    # Clip boundaries remain the explicit motion playback loop boundary.
    assert cfg.env.terminations.clip_end.time_out is True
    assert cfg.env.commands.motion.params.truncate_on_clip_end is True


def test_sonic_mujoco_play_uses_random_clip_starts_without_episode_timeout() -> None:
    cfg = _compose("task=g1_sonic/mujoco_play", "sonic_model=smoke")
    assert cfg.env.terminations.time_out is None
    assert cfg.env.commands.motion.params.sampling_mode == "clip_start"
    assert cfg.env.terminations.clip_end.time_out is True
    assert cfg.env.commands.motion.params.truncate_on_clip_end is True


def test_sonic_training_uses_plain_motion_tracking_action_contract() -> None:
    cfg = _compose("task=g1_sonic/mujoco")
    assert "actuator_gains" not in cfg.env.events
    assert float(cfg.env.actions.joint_pos.scale) == pytest.approx(2.0)
    registry.ensure_registries(packages=["unilab.tasks.motion_tracking"])
    override = BackendAdapter(cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env_cfg = registry.materialize_env_config("G1SonicManager")
    apply_cfg_overrides(env_cfg, override)
    env_cfg.validate()
    assert "actuator_gains" not in env_cfg.events
    action = env_cfg.actions["joint_pos"]
    assert float(action.scale) == pytest.approx(2.0)
    assert action.use_release_scale_rule is False


def test_sonic_release_play_injects_release_action_contract() -> None:
    from unilab.training.offpolicy_sonic import _apply_sonic_play_action_scale

    override: dict = {}
    _apply_sonic_play_action_scale(override, checkpoint_format="unilab")
    assert override == {}

    _apply_sonic_play_action_scale(override, checkpoint_format="sonic_release")
    joint_pos = override["actions"]["joint_pos"]
    assert float(joint_pos["scale"]) == pytest.approx(0.25)
    assert joint_pos["use_release_scale_rule"] is True
    assert joint_pos["clip_to_joint_limits"] is True
    event = override["events"]["actuator_gains"]
    assert event["func"].endswith("SonicActuatorDynamics")
    assert event["mode"] == "reset"


def test_sonic_backends_keep_policy_contract_equal() -> None:
    mujoco = _compose("task=g1_sonic/mujoco")
    motrix = _compose("task=g1_sonic/motrix")
    assert mujoco.algo.sonic == motrix.algo.sonic
    assert mujoco.reward == motrix.reward
    assert (
        mujoco.env.actions.joint_pos.actuator_names == motrix.env.actions.joint_pos.actuator_names
    )
    assert mujoco.env.actions.joint_pos.scale == motrix.env.actions.joint_pos.scale
    assert mujoco.env.actions.joint_pos.simulate_action_latency is False
    assert motrix.env.actions.joint_pos.simulate_action_latency is False
    assert mujoco.env.observations == motrix.env.observations
    # Motion paths and sampling/randomization ranges are owner-specific; the
    # policy-facing temporal and termination contract must remain identical.
    for cfg in (mujoco, motrix):
        motion = cfg.env.commands.motion.params
        assert motion.num_future_frames == 10
        assert motion.sampling_mode == "adaptive"
        assert motion.adaptive_attribution == "trajectory"
        assert motion.truncate_on_clip_end is False
        assert motion.anchor_pos_z_threshold == pytest.approx(0.15)
        assert motion.low_reference_anchor_pos_z_threshold == pytest.approx(0.75)
