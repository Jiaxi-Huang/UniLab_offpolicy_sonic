"""Manager-Based SONIC task I/O and reference lifecycle tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.managers import ObservationTermCfg
from unilab.tasks.motion_tracking.g1 import sonic_manager
from unilab.tasks.motion_tracking.g1.sonic_data import (
    _SONIC_ACTUATOR_PARAMETERS,
    _SonicMotionLoader,
    np_quat_from_euler_xyz,
)
from unilab.tasks.motion_tracking.g1.sonic_manager import (
    G1_SONIC_ACTION_SCALE,
    G1_SONIC_BODY_NAMES,
    G1_SONIC_JOINTS,
    SONIC_TERMINATION_REASON_NAMES,
    G1SonicManagerCfg,
    G1SonicManagerEnv,
    SonicActorObservation,
    SonicActuatorDynamics,
    SonicCriticObservation,
    SonicJointPositionAction,
    SonicJointPositionActionCfg,
    SonicMotionCommand,
    SonicMotionCommandCfg,
    SonicNoiseConfig,
    SonicObservationTermCfg,
    make_g1_sonic_manager_cfg,
    sonic_anchor_ori_termination,
    sonic_anchor_pos_z_termination,
    sonic_ee_body_pos_z_termination,
    sonic_feet_pos_termination,
    sonic_reference_joint_names,
    sonic_reference_qpos,
    sonic_reward,
)
from unilab.utils.rotation import (
    np_quat_apply_batched,
    np_quat_conjugate_batched,
    np_quat_heading,
    np_quat_mul_batched,
)


def test_sonic_manager_config_declares_released_policy_contract() -> None:
    cfg = make_g1_sonic_manager_cfg()
    assert isinstance(cfg, G1SonicManagerCfg)
    assert cfg.policy_observation_group == "policy"
    assert cfg.critic_observation_group == "critic"
    assert cfg.actions == {}
    assert cfg.commands == {}
    assert set(cfg.observations) == {"policy", "critic"}
    assert set(cfg.observations["policy"].terms) == {
        "obs",
        "g1_reference",
        "smpl_reference",
        "encoder_index",
    }
    assert set(cfg.observations["critic"].terms) == {"obs", "encoder_index"}
    assert tuple(cfg.terminations) == (
        "anchor_pos_z",
        "anchor_ori",
        "ee_body_pos_z",
        "feet_pos",
        "time_out",
        "clip_end",
    )
    assert cfg.rewards == {}
    assert cfg.scale_rewards_by_dt is True
    # Release-only actuator gains are injected by playback overrides, never
    # declared by the default task contract.
    assert set(cfg.events) == {"reset_reference"}


def test_sonic_action_scale_is_scalar_by_default_and_release_rule_opt_in() -> None:
    joints = list(G1_SONIC_JOINTS)
    ids = np.arange(len(joints), dtype=np.intp)
    robot = SimpleNamespace(
        find_joints_by_actuator_names=lambda patterns: (ids, tuple(joints)),
        find_joints=lambda names, preserve_order=False: (ids, tuple(joints)),
        data=SimpleNamespace(default_joint_pos=np.zeros((1, len(joints)), dtype=np.float32)),
    )
    env = SimpleNamespace(num_envs=1, scene={"robot": robot})

    default_term = SonicJointPositionAction(
        SonicJointPositionActionCfg(entity_name="robot", actuator_names=joints), env
    )
    assert default_term._scale.shape == (1, len(joints))
    np.testing.assert_allclose(default_term._scale, G1_SONIC_ACTION_SCALE)

    release_term = SonicJointPositionAction(
        SonicJointPositionActionCfg(
            entity_name="robot",
            actuator_names=joints,
            scale=0.25,
            use_release_scale_rule=True,
        ),
        env,
    )
    np.testing.assert_allclose(
        release_term._scale[0], sonic_manager._sonic_policy_action_scale(), rtol=1e-6
    )


def test_sonic_manager_is_registered_for_both_backends() -> None:
    registry.ensure_registries(packages=["unilab.tasks.motion_tracking"])
    metadata = registry.list_registered_envs()["G1SonicManager"]
    assert metadata["available_backends"] == ["mujoco", "motrix"]
    assert isinstance(registry.materialize_env_config("G1SonicManager"), G1SonicManagerCfg)


class _RobotData:
    def __init__(self, num_envs: int):
        self.root_link_pos_w = np.zeros((num_envs, 3), dtype=np.float32)
        self.root_link_quat_w = np.zeros((num_envs, 4), dtype=np.float32)
        self.root_link_quat_w[:, 0] = 1.0
        self.root_link_lin_vel_w = np.arange(num_envs * 3, dtype=np.float32).reshape(num_envs, 3)
        self.root_link_ang_vel_w = self.root_link_lin_vel_w + 10.0
        self.joint_pos = np.arange(num_envs * 29, dtype=np.float32).reshape(num_envs, 29)
        self.joint_vel = self.joint_pos + 100.0
        self.default_joint_pos = np.ones((num_envs, 29), dtype=np.float32)
        self.body_link_pos_w = np.zeros((num_envs, len(G1_SONIC_BODY_NAMES), 3), dtype=np.float32)
        self.body_link_quat_w = np.zeros((num_envs, len(G1_SONIC_BODY_NAMES), 4), dtype=np.float32)
        self.body_link_quat_w[:, :, 0] = 1.0
        self.body_link_lin_vel_w = np.zeros_like(self.body_link_pos_w)
        self.body_link_ang_vel_w = np.zeros_like(self.body_link_pos_w)


def _observation_env(num_envs: int = 2):
    cfg = make_g1_sonic_manager_cfg()
    robot = SimpleNamespace(
        data=_RobotData(num_envs),
        find_joints=lambda names, preserve_order: (np.arange(len(names)), tuple(names)),
    )
    action = np.arange(num_envs * 29, dtype=np.float32).reshape(num_envs, 29) + 200.0
    return SimpleNamespace(
        num_envs=num_envs,
        _cfg=cfg,
        scene={"robot": robot},
        action_manager=SimpleNamespace(action=action),
    )


def _pack_actor_frame(env, rows: np.ndarray) -> np.ndarray:
    robot = env.scene["robot"].data
    gravity = np.zeros((len(rows), 3), dtype=np.float32)
    gravity[:, 2] = -1.0
    return np.concatenate(
        (
            robot.root_link_ang_vel_w[rows],
            robot.joint_pos[rows] - robot.default_joint_pos[rows],
            robot.joint_vel[rows],
            env.action_manager.action[rows],
            gravity,
        ),
        axis=-1,
    )


def _pack_history(frame: np.ndarray) -> np.ndarray:
    history = np.broadcast_to(frame[:, None, :], (len(frame), 10, 93))
    return np.concatenate(
        tuple(
            history[:, :, start:end].reshape(len(frame), -1)
            for start, end in (
                (0, 3),
                (3, 32),
                (32, 61),
                (61, 90),
                (90, 93),
            )
        ),
        axis=-1,
    )


def test_actor_observation_backfills_history_and_scopes_partial_reset() -> None:
    env = _observation_env()
    term = SonicActorObservation(
        SonicObservationTermCfg(
            func=SonicActorObservation,
            sonic_noise=SonicNoiseConfig(level=0.0),
        ),
        env,
    )
    all_rows = np.arange(env.num_envs)
    term.reset(all_rows)
    actual = term(env)
    np.testing.assert_allclose(actual, _pack_history(_pack_actor_frame(env, all_rows)))
    previous_env_one = actual[1].copy()

    env.scene["robot"].data.joint_pos[0] += 5.0
    term.reset(np.asarray([0], dtype=np.int32))
    partial = term(env)
    np.testing.assert_allclose(partial[0], _pack_history(_pack_actor_frame(env, all_rows[:1]))[0])
    np.testing.assert_array_equal(partial[1], previous_env_one)
    assert partial.shape == (2, 930)


def test_actor_observation_preserves_official_field_order(monkeypatch) -> None:
    """The 930-D release actor input is gyro, joints, action, then gravity."""
    env = _observation_env(num_envs=1)
    robot = env.scene["robot"].data
    robot.joint_pos[:] = 3.0
    robot.default_joint_pos[:] = 1.0
    robot.joint_vel[:] = 4.0
    env.action_manager.action[:] = 5.0
    monkeypatch.setattr(
        sonic_manager,
        "_root_local_proprioception",
        lambda *_args: (
            np.zeros((1, 3), dtype=np.float32),
            np.full((1, 3), 1.0, dtype=np.float32),
            np.full((1, 3), 6.0, dtype=np.float32),
        ),
    )
    term = SonicActorObservation(
        SonicObservationTermCfg(
            func=SonicActorObservation,
            sonic_noise=SonicNoiseConfig(level=0.0),
        ),
        env,
    )
    term.reset(np.asarray([0], dtype=np.int32))
    output = term(env)
    # Each field is history-major (10 frames), then fields are concatenated.
    expected = ([(1.0, 3), (2.0, 29), (4.0, 29), (5.0, 29), (6.0, 3)],)
    offset = 0
    for value, width in expected[0]:
        size = width * 10
        np.testing.assert_allclose(output[0, offset : offset + size], value)
        offset += size
    assert offset == 930


class _FakeMotion(SonicMotionCommand):
    def __init__(self, num_envs: int):
        self.cfg = SonicMotionCommandCfg()
        self.anchor_body_idx = 0
        quat = np.zeros((num_envs, len(G1_SONIC_BODY_NAMES), 4), dtype=np.float32)
        quat[:, :, 0] = 1.0
        self.motion_data = SimpleNamespace(
            body_pos_w=np.zeros((num_envs, len(G1_SONIC_BODY_NAMES), 3), dtype=np.float32),
            body_quat_w=quat,
        )

    def g1_command(self, rows: np.ndarray) -> np.ndarray:
        return np.zeros((len(rows), 10, 58), dtype=np.float32)

    def g1_reference(self, rows: np.ndarray) -> np.ndarray:
        return np.zeros((len(rows), 640), dtype=np.float32)


def _reference_motion(num_envs: int = 2, anchor_body_idx: int = 2) -> SonicMotionCommand:
    motion = object.__new__(SonicMotionCommand)
    motion.anchor_body_idx = anchor_body_idx
    rng = np.random.default_rng(7)
    num_bodies, num_joints = 5, 3
    quat = np.zeros((num_envs, num_bodies, 4), dtype=np.float32)
    quat[..., 0] = 1.0
    motion.motion_data = SimpleNamespace(
        body_pos_w=rng.normal(size=(num_envs, num_bodies, 3)).astype(np.float32),
        body_quat_w=quat,
        joint_pos=rng.normal(size=(num_envs, num_joints)).astype(np.float32),
    )
    return motion


def test_reference_qpos_packs_anchor_pose_then_loader_order_joints() -> None:
    motion = _reference_motion(anchor_body_idx=2)

    rows = motion.reference_qpos()

    assert rows.shape == (2, 10)
    assert rows.dtype == np.float32
    np.testing.assert_allclose(rows[:, 0:3], motion.motion_data.body_pos_w[:, 2])
    np.testing.assert_allclose(rows[:, 3:7], motion.motion_data.body_quat_w[:, 2])
    np.testing.assert_allclose(rows[:, 7:], motion.motion_data.joint_pos)


def test_sonic_reference_accessors_route_through_motion_command() -> None:
    motion = _reference_motion()
    motion.loader = SimpleNamespace(joint_names=("joint_a", "joint_b"))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: motion))

    np.testing.assert_allclose(sonic_reference_qpos(env), motion.reference_qpos())
    assert sonic_reference_joint_names(env) == ("joint_a", "joint_b")


def test_critic_observation_has_released_width_and_reset_history() -> None:
    env = _observation_env()
    motion = _FakeMotion(env.num_envs)
    env.command_manager = SimpleNamespace(get_term=lambda name: motion)
    term = SonicCriticObservation(ObservationTermCfg(func=SonicCriticObservation), env)
    term.reset(np.arange(env.num_envs))
    actual = term(env)
    assert actual.shape == (2, 1705)
    # The first 640 entries are the ten-frame future G1 reference (future
    # joint command plus relative-root 6D), matching the actor term layout.
    np.testing.assert_array_equal(actual[:, :640], 0.0)


def test_sonic_action_term_uses_current_or_previous_clipped_action() -> None:
    num_envs = 2
    term = object.__new__(SonicJointPositionAction)
    current = np.full((num_envs, 29), 2.0, dtype=np.float32)
    previous = np.full((num_envs, 29), 3.0, dtype=np.float32)
    term._raw_actions = np.zeros_like(current)
    term._processed_actions = np.zeros_like(current)
    term._scale = np.full((1, 29), G1_SONIC_ACTION_SCALE, dtype=np.float32)
    term._offset = np.ones_like(current)
    term._entity = SimpleNamespace(
        data=SimpleNamespace(
            joint_vel=np.full_like(current, 7.0),
            soft_joint_pos_limits=np.stack(
                (np.full((29,), -10.0, dtype=np.float32), np.full((29,), 10.0, dtype=np.float32)),
                axis=-1,
            ),
        )
    )
    term._target_ids = np.arange(29, dtype=np.intp)
    term.joint_velocity_before_action = np.zeros_like(current)
    term.cfg = SimpleNamespace(simulate_action_latency=False, clip_to_joint_limits=False)
    term._env = SimpleNamespace(
        action_manager=SimpleNamespace(prev_action=previous),
    )
    term.process_actions(current)
    np.testing.assert_allclose(term._processed_actions, current * G1_SONIC_ACTION_SCALE + 1.0)
    np.testing.assert_array_equal(term.joint_velocity_before_action, 7.0)

    term.cfg.simulate_action_latency = True
    term.process_actions(current)
    np.testing.assert_allclose(term._processed_actions, previous * G1_SONIC_ACTION_SCALE + 1.0)

    # Default contract does not clip to soft joint limits (motion-tracking
    # parity); the release contract does.
    term.cfg.simulate_action_latency = False
    term._entity.data.soft_joint_pos_limits[term._target_ids[0]] = (-0.1, 0.1)
    term.process_actions(current)
    np.testing.assert_allclose(term._processed_actions[:, 0], 2.0 * G1_SONIC_ACTION_SCALE + 1.0)

    term.cfg.clip_to_joint_limits = True
    term.process_actions(current)
    np.testing.assert_allclose(term._processed_actions[:, 0], 0.1)


def test_sonic_reset_stages_policy_order_actuator_gains() -> None:
    captured = {}

    def write(kp, kd, **kwargs):
        captured.update(kp=kp, kd=kd, kwargs=kwargs)

    def write_armature(values, **kwargs):
        captured.update(armature=values, armature_kwargs=kwargs)

    target_ids = np.arange(29, dtype=np.intp)[::-1]
    env = SimpleNamespace(
        num_envs=2,
        scene={
            "robot": SimpleNamespace(
                write_actuator_gains_to_sim=write,
                write_joint_armature_to_sim=write_armature,
            )
        },
    )
    ids = np.asarray([0, 1], dtype=np.int32)
    term = object.__new__(SonicActuatorDynamics)
    term._entity = env.scene["robot"]
    term._actuator_ids = target_ids
    term._joint_ids = target_ids
    term._kp = np.asarray(
        [_SONIC_ACTUATOR_PARAMETERS[name][0] for name in G1_SONIC_JOINTS], dtype=np.float32
    )
    term._kd = np.asarray(
        [_SONIC_ACTUATOR_PARAMETERS[name][1] for name in G1_SONIC_JOINTS], dtype=np.float32
    )
    term._armature = np.asarray(
        [_SONIC_ACTUATOR_PARAMETERS[name][3] for name in G1_SONIC_JOINTS], dtype=np.float32
    )
    term(env, ids)
    expected_kp = [_SONIC_ACTUATOR_PARAMETERS[name][0] for name in G1_SONIC_JOINTS]
    expected_kd = [_SONIC_ACTUATOR_PARAMETERS[name][1] for name in G1_SONIC_JOINTS]
    expected_armature = [_SONIC_ACTUATOR_PARAMETERS[name][3] for name in G1_SONIC_JOINTS]
    np.testing.assert_allclose(captured["kp"], np.broadcast_to(expected_kp, (2, 29)))
    np.testing.assert_allclose(captured["kd"], np.broadcast_to(expected_kd, (2, 29)))
    np.testing.assert_array_equal(captured["kwargs"]["actuator_ids"], target_ids)
    np.testing.assert_allclose(captured["armature"], np.broadcast_to(expected_armature, (2, 29)))
    np.testing.assert_array_equal(captured["armature_kwargs"]["joint_ids"], target_ids)
    np.testing.assert_array_equal(captured["kwargs"]["env_ids"], ids)


def test_sonic_env_inherits_unbounded_action_space() -> None:
    # g1_motion_tracking parity: the env declares no action-space override, so
    # the ManagerBasedRlEnv unbounded Box over the action-term dim applies.
    assert "action_space" not in G1SonicManagerEnv.__dict__
    assert "apply_action" not in G1SonicManagerEnv.__dict__


class _TransitionBackend:
    def __init__(self, num_envs: int, backend_type: str):
        self.backend_type = backend_type
        self.dof_pos = np.zeros((num_envs, 29), dtype=np.float32)
        self.dof_vel = np.zeros_like(self.dof_pos)
        self.robot_data = None

    def get_body_ids(self, names):
        return np.arange(len(names), dtype=np.intp)

    def copy_body_state_w(self, body_ids, out_pos, out_quat, out_lin_vel, out_ang_vel):
        del body_ids
        np.copyto(out_pos, self.robot_data.body_link_pos_w)
        np.copyto(out_quat, self.robot_data.body_link_quat_w)
        np.copyto(out_lin_vel, self.robot_data.body_link_lin_vel_w)
        np.copyto(out_ang_vel, self.robot_data.body_link_ang_vel_w)
        return out_pos, out_quat, out_lin_vel, out_ang_vel

    def get_joint_range(self):
        return np.broadcast_to(np.asarray([-1.0, 1.0]), (29, 2)).copy()

    def get_dof_pos(self):
        return self.dof_pos

    def get_dof_vel(self):
        return self.dof_vel


class _TransitionScene(dict):
    def __init__(self, robot, backend_type: str, num_envs: int):
        super().__init__(robot=robot)
        self.backend_type = backend_type
        width = 3 if backend_type == "mujoco" else 1
        self.sensor_values = np.zeros((num_envs, 13 * width), dtype=np.float32)

    def bind_sensor_data(self, names):
        return SimpleNamespace(
            names=tuple(names),
            width=self.sensor_values.shape[1],
            backend_type=self.backend_type,
            read=lambda: self.sensor_values,
        )


def _transition_motion(backend_type: str = "mujoco", num_envs: int = 1):
    cfg = SonicMotionCommandCfg()
    backend = _TransitionBackend(num_envs, backend_type)
    robot = SimpleNamespace(data=_RobotData(num_envs))
    robot.data.joint_pos = backend.dof_pos
    robot.data.joint_vel = backend.dof_vel
    backend.robot_data = robot.data
    scene = _TransitionScene(robot, backend_type, num_envs)
    action_term = object.__new__(SonicJointPositionAction)
    action_term.joint_velocity_before_action = np.zeros((num_envs, 29), dtype=np.float32)
    action_manager = SimpleNamespace(
        action=np.zeros((num_envs, 29), dtype=np.float32),
        prev_action=np.zeros((num_envs, 29), dtype=np.float32),
        get_term=lambda name: action_term,
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        _cfg=SimpleNamespace(ctrl_dt=0.02),
        step_dt=0.02,
        _backend=backend,
        scene=scene,
        action_manager=action_manager,
    )
    motion = object.__new__(SonicMotionCommand)
    motion._env = env
    motion.cfg = cfg
    motion._body_ids = backend.get_body_ids(G1_SONIC_BODY_NAMES)
    motion._copy_body_state_w = backend.copy_body_state_w
    motion.anchor_body_idx = 0
    motion.ee_body_indices = np.asarray([3, 6, 10, 13], dtype=np.intp)
    motion._foot_body_indices = np.asarray([3, 6], dtype=np.intp)
    motion._anti_shake_indices = np.asarray([10, 13, 7], dtype=np.intp)
    model_names = tuple(G1_SONIC_JOINTS)
    motion._foot_joint_model_indices = np.asarray(
        [model_names.index(name) for name in sonic_manager._SONIC_FOOT_JOINT_NAMES]
    )
    motion._foot_joint_policy_indices = motion._foot_joint_model_indices.copy()
    quat = np.zeros((num_envs, len(G1_SONIC_BODY_NAMES), 4), dtype=np.float32)
    quat[..., 0] = 1.0
    motion.motion_data = SimpleNamespace(
        body_pos_w=np.zeros((num_envs, len(G1_SONIC_BODY_NAMES), 3), dtype=np.float32),
        body_quat_w=quat,
        body_lin_vel_w=np.zeros((num_envs, len(G1_SONIC_BODY_NAMES), 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((num_envs, len(G1_SONIC_BODY_NAMES), 3), dtype=np.float32),
        joint_pos=np.zeros((num_envs, 29), dtype=np.float32),
        joint_vel=np.zeros((num_envs, 29), dtype=np.float32),
    )
    motion._init_transition_context(cfg, model_names)
    env.command_manager = SimpleNamespace(get_term=lambda name: motion)
    motion.refresh_transition_context()
    return env, motion, action_term


def test_sonic_common_rewards_dispatch_to_exact_shared_formulas() -> None:
    env, motion, _ = _transition_motion()
    for name in (
        "motion_global_root_pos",
        "motion_global_root_ori",
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
        "motion_ee_body_pos_z",
        "motion_joint_pos",
        "motion_joint_vel",
        "action_rate_l2",
    ):
        expected = sonic_manager._SONIC_COMMON_REWARD_FUNCTIONS[name](motion.reward_context).copy()
        std = {
            "motion_global_root_pos": 0.3,
            "motion_global_root_ori": 0.4,
            "motion_body_pos": 0.3,
            "motion_body_ori": 0.4,
            "motion_body_lin_vel": 1.0,
            "motion_body_ang_vel": 3.14,
            "motion_ee_body_pos_z": 0.3,
            "motion_joint_pos": 0.2,
            "motion_joint_vel": 1.0,
        }.get(name)
        np.testing.assert_allclose(sonic_reward(env, name, std=std), expected)


def test_sonic_reward_std_is_owned_by_manager_term_params() -> None:
    env, motion, _ = _transition_motion()
    motion.motion_data.body_pos_w[0, motion.anchor_body_idx, 0] = 0.5
    np.testing.assert_allclose(sonic_reward(env, "motion_global_root_pos", std=0.25), np.exp(-4.0))
    with pytest.raises(ValueError, match="positive finite std"):
        sonic_reward(env, "motion_global_root_pos", std=0.0)


def test_sonic_custom_rewards_match_released_numeric_semantics() -> None:
    _, motion, action_term = _transition_motion()
    motion._dof_pos[0, :2] = (1.0, -1.1)
    np.testing.assert_allclose(motion.reward_joint_limit(), 0.3, atol=1.0e-6)

    motion._robot_body_ang_vel_w[0, motion._anti_shake_indices, 0] = (2.5, 1.5, 3.5)
    np.testing.assert_allclose(motion.reward_anti_shake(), 5.0 / 3.0)

    motion._dof_vel[0, motion._foot_joint_model_indices] = (0.02, 0.04, 0.06, 0.08)
    action_term.joint_velocity_before_action[0, motion._foot_joint_policy_indices] = 0.0
    np.testing.assert_allclose(motion.reward_feet_acc(), 30.0)

    point = motion._reward_point_indices[0]
    motion._robot_body_pos_w[0, point, 0] = 0.1
    # This fixture drives the dataclass default point set, whose first point
    # (pelvis) is the anchor body: perturbing the anchor shifts the other
    # four points' local coordinates by 0.1, so the mean squared error is
    # 4 * 0.01 / 5 and the reward is exp(-0.8).
    np.testing.assert_allclose(motion.reward_tracking_vr_5point_local(), np.exp(-0.8), rtol=1.0e-6)


@pytest.mark.parametrize("backend_type", ["mujoco", "motrix"])
def test_sonic_undesired_contact_threshold_grouping_and_history(backend_type: str) -> None:
    _, motion, _ = _transition_motion(backend_type)
    values = motion._env.scene.sensor_values
    if backend_type == "mujoco":
        values[0, 0] = 1.0
        np.testing.assert_array_equal(motion.reward_undesired_contacts(), 0.0)
        values.fill(0.0)
        values[0, 3 * 3] = 1.01
        values[0, 4 * 3] = 2.0
    else:
        values[0, 0] = 0.0
        np.testing.assert_array_equal(motion.reward_undesired_contacts(), 0.0)
        values[0, 3:5] = 1.0
    # Two geoms in the same upstream body group count once.
    np.testing.assert_array_equal(motion.reward_undesired_contacts(), 1.0)
    values.fill(0.0)
    np.testing.assert_array_equal(motion.reward_undesired_contacts(), 1.0)
    np.testing.assert_array_equal(motion.reward_undesired_contacts(), 1.0)
    np.testing.assert_array_equal(motion.reward_undesired_contacts(), 0.0)


def test_sonic_termination_thresholds_full_xyz_and_reason_order() -> None:
    env, motion, _ = _transition_motion()
    motion._low_reference[0] = False
    motion.motion_data.body_pos_w[0, 0, 2] = motion.cfg.anchor_pos_z_threshold
    assert not sonic_anchor_pos_z_termination(env)[0]
    motion.motion_data.body_pos_w[0, 0, 2] += 0.001
    assert sonic_anchor_pos_z_termination(env)[0]
    motion._low_reference[0] = True
    motion.motion_data.body_pos_w[0, 0, 2] = motion.cfg.low_reference_anchor_pos_z_threshold
    assert not sonic_anchor_pos_z_termination(env)[0]
    motion.motion_data.body_pos_w[0, 0, 2] += 0.001
    assert sonic_anchor_pos_z_termination(env)[0]

    angle = 0.5
    motion.motion_data.body_quat_w[0, 0] = (
        np.cos(angle / 2),
        np.sin(angle / 2),
        0.0,
        0.0,
    )
    assert sonic_anchor_ori_termination(env)[0]
    motion._low_reference[0] = False
    motion.body_pos_relative_w[0, motion.ee_body_indices[0], 2] = motion.cfg.ee_body_pos_z_threshold
    assert not sonic_ee_body_pos_z_termination(env)[0]
    motion.body_pos_relative_w[0, motion.ee_body_indices[0], 2] += 0.001
    assert sonic_ee_body_pos_z_termination(env)[0]
    motion._low_reference[0] = True
    motion.body_pos_relative_w[0, motion.ee_body_indices[0], 2] = (
        motion.cfg.low_reference_ee_body_pos_z_threshold
    )
    assert not sonic_ee_body_pos_z_termination(env)[0]
    motion.body_pos_relative_w[0, motion.ee_body_indices[0], 2] += 0.001
    assert sonic_ee_body_pos_z_termination(env)[0]
    # XY-only displacement must trigger the full-XYZ foot constraint.
    motion.body_pos_relative_w[0, motion._foot_body_indices[0], 0] = 0.201
    assert sonic_feet_pos_termination(env)[0]
    assert motion._termination_reason_mask.all()


def test_sonic_reference_height_ema_updates_once_per_transition() -> None:
    env, motion, _ = _transition_motion()
    motion._running_ref_root_height[:] = 0.4
    motion.motion_data.body_pos_w[:, motion.anchor_body_idx, 2] = 0.6
    motion.refresh_transition_context()
    np.testing.assert_allclose(motion._running_ref_root_height, 0.42)
    for term in (
        sonic_anchor_pos_z_termination,
        sonic_anchor_ori_termination,
        sonic_ee_body_pos_z_termination,
        sonic_feet_pos_termination,
    ):
        term(env)
    np.testing.assert_allclose(motion._running_ref_root_height, 0.42)


def test_sonic_reference_relative_transform_matches_heading_only_contract() -> None:
    """Reference body transforms must match gear_sonic's heading-only formula."""
    env, motion, _ = _transition_motion()
    robot = env.scene["robot"].data
    robot.body_link_pos_w[0, motion.anchor_body_idx] = (1.0, 2.0, 3.0)
    robot.body_link_quat_w[0, motion.anchor_body_idx] = np_quat_from_euler_xyz(0.5, -0.4, 0.7)
    motion.motion_data.body_pos_w[0, motion.anchor_body_idx] = (0.4, -0.3, 1.5)
    motion.motion_data.body_quat_w[0, motion.anchor_body_idx] = np_quat_from_euler_xyz(
        -0.2, 0.3, -0.6
    )
    body_idx = 3
    motion.motion_data.body_pos_w[0, body_idx] = (0.8, 0.1, 2.2)
    motion.motion_data.body_quat_w[0, body_idx] = np_quat_from_euler_xyz(0.1, -0.2, 0.4)

    motion.refresh_transition_context()

    robot_anchor_pos = robot.body_link_pos_w[:, motion.anchor_body_idx]
    robot_anchor_quat = robot.body_link_quat_w[:, motion.anchor_body_idx]
    ref_anchor_pos = motion.motion_data.body_pos_w[:, motion.anchor_body_idx]
    ref_anchor_quat = motion.motion_data.body_quat_w[:, motion.anchor_body_idx]
    delta_pos = robot_anchor_pos.copy()
    delta_pos[:, 2] = ref_anchor_pos[:, 2]
    delta_ori = np_quat_heading(
        np_quat_mul_batched(robot_anchor_quat, np_quat_conjugate_batched(ref_anchor_quat))
    )
    expected_pos = delta_pos[:, None] + np_quat_apply_batched(
        delta_ori[:, None], motion.motion_data.body_pos_w - ref_anchor_pos[:, None]
    )
    expected_quat = np_quat_mul_batched(delta_ori[:, None], motion.motion_data.body_quat_w)
    np.testing.assert_allclose(motion.body_pos_relative_w, expected_pos, atol=1.0e-6)
    np.testing.assert_allclose(motion.body_quat_relative_w, expected_quat, atol=1.0e-6)
    # In particular, heading alignment must not tilt the reference offset in Z.
    assert motion.body_pos_relative_w[0, body_idx, 2] == pytest.approx(
        ref_anchor_pos[0, 2]
        + (motion.motion_data.body_pos_w[0, body_idx, 2] - ref_anchor_pos[0, 2]),
        abs=1.0e-6,
    )


class _RuntimeMotionLoader(_SonicMotionLoader):
    def __init__(self, store, *, backend, body_names):
        del store
        frames = 12
        joints = len(backend.get_actuator_names())
        bodies = len(body_names)
        self.num_frames = frames
        self.num_clips = 1
        self.num_joints = joints
        self.num_bodies = bodies
        self.clip_lengths = np.asarray([frames], dtype=np.int32)
        self.clip_offsets = np.asarray([0], dtype=np.int32)
        self.clip_end_frames = np.asarray([frames - 1], dtype=np.int32)
        self.joint_pos = (
            np.arange(frames, dtype=np.float32)[:, None]
            + np.arange(joints, dtype=np.float32)[None, :] / 100.0
        )
        self.joint_vel = np.zeros_like(self.joint_pos)
        self.body_pos_w = np.zeros((frames, bodies, 3), dtype=np.float32)
        self.body_pos_w[:, 0, 2] = 1.0
        self.body_quat_w = np.zeros((frames, bodies, 4), dtype=np.float32)
        self.body_quat_w[:, :, 0] = 1.0
        self.body_lin_vel_w = np.zeros_like(self.body_pos_w)
        self.body_ang_vel_w = np.zeros_like(self.body_pos_w)
        self.smpl_joints = np.zeros((frames, 24, 3), dtype=np.float32)
        self.smpl_root_quat = np.zeros((frames, 4), dtype=np.float32)
        self.smpl_root_quat[:, 0] = 1.0


class _SubsetRuntimeLoader(_RuntimeMotionLoader):
    """Synthetic multi-clip packed loader honoring ``clip_indices`` selection.

    Each selected global clip contributes the base 12-frame synthesis with
    its joint positions offset by the global clip id, so different subsets
    (and rotated working sets) are distinguishable through the observations.
    """

    def __init__(self, store, *, backend, body_names, clip_indices=None):
        super().__init__(store, backend=backend, body_names=body_names)
        if clip_indices is None:
            self.subset_clip_indices = np.zeros(1, dtype=np.int64)
            return
        selected = np.asarray(clip_indices, dtype=np.int64)
        count = len(selected)
        self.joint_pos = np.concatenate(
            [self.joint_pos + selected[i].astype(np.float32) * 10.0 for i in range(count)],
            axis=0,
        )
        self.joint_vel = np.concatenate([self.joint_vel] * count, axis=0)
        for name in (
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "smpl_joints",
            "smpl_root_quat",
        ):
            setattr(self, name, np.concatenate([getattr(self, name)] * count, axis=0))
        self.clip_lengths = np.full(count, 12, dtype=np.int32)
        self.clip_offsets = np.zeros(count, dtype=np.int32)
        if count > 1:
            self.clip_offsets[1:] = np.cumsum(self.clip_lengths[:-1], dtype=np.int32)
        self.clip_end_frames = self.clip_offsets + self.clip_lengths - 1
        self.num_clips = count
        self.num_frames = int(self.clip_lengths.sum())
        self.subset_clip_indices = selected


@pytest.mark.parametrize("backend_type", ["mujoco", "motrix"])
def test_sonic_manager_runtime_materializes_and_steps(monkeypatch, backend_type: str) -> None:
    monkeypatch.setattr(sonic_manager, "SonicPackedMotionLoader", _RuntimeMotionLoader)
    config_dir = Path(__file__).parents[2] / "src" / "unilab" / "conf" / "flashsac"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        hydra_cfg = compose(
            config_name="config_sonic",
            overrides=[f"task=g1_sonic/{backend_type}"],
        )
    env_cfg_override = BackendAdapter(
        hydra_cfg, root_dir=Path(__file__).parents[2]
    ).build_task_env_cfg_override()
    command = env_cfg_override["commands"]["motion"]
    params = command["params"]
    params["motion_store_file"] = "synthetic"
    params["sampling_mode"] = "start"
    params["anchor_pos_z_threshold"] = 100.0
    params["anchor_ori_threshold"] = 100.0
    params["ee_body_pos_z_threshold"] = 100.0
    params["foot_pos_threshold"] = 100.0
    params["pose_range"] = {axis: [0.0, 0.0] for axis in ("x", "y", "z", "roll", "pitch", "yaw")}
    params["velocity_range"] = {
        axis: [0.0, 0.0] for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    params["joint_position_range"] = [0.0, 0.0]
    params["joint_velocity_range"] = [0.0, 0.0]
    # The training owner defaults to wrap mode (clip end resamples mid-episode);
    # pin the per-clip truncation protocol here to keep validating the clip_end
    # termination term and its termination-reason mask column.
    params["truncate_on_clip_end"] = True
    env_cfg_override["terminations"]["clip_end"] = {
        "_target_": "unilab.managers.TerminationTermCfg",
        "func": "unilab.tasks.motion_tracking.g1.sonic_manager.sonic_clip_end",
        "time_out": True,
    }
    env_cfg_override["observations"]["policy"]["terms"]["obs"]["sonic_noise"] = {"level": 0.0}
    env = registry.make(
        "G1SonicManager",
        sim_backend=backend_type,
        env_cfg_override=env_cfg_override,
        num_envs=2,
    )
    try:
        state = env.init_state()
        assert state.obs["obs"].shape == (2, 2412)
        # critic = g1 reference window (640) + anchor/body error (135)
        #         + proprio history (930) + encoder index (2)
        assert state.obs["critic"].shape == (2, 1707)
        motion = env.command_manager.get_term("motion")
        assert motion.sampler.rng is env.rng
        np.testing.assert_allclose(
            env.scene["robot"].data.joint_pos,
            motion.motion_data.joint_pos,
        )
        np.testing.assert_allclose(
            env.scene["robot"].data.joint_vel,
            motion.motion_data.joint_vel,
        )
        anchor = motion.anchor_body_idx
        np.testing.assert_allclose(
            env.scene["robot"].data.body_link_pos_w[:, anchor],
            motion.motion_data.body_pos_w[:, anchor],
        )
        np.testing.assert_allclose(
            env.scene["robot"].data.body_link_quat_w[:, anchor],
            motion.motion_data.body_quat_w[:, anchor],
        )
        np.testing.assert_allclose(
            env.scene["robot"].data.body_link_lin_vel_w[:, anchor],
            motion.motion_data.body_lin_vel_w[:, anchor],
        )
        np.testing.assert_allclose(
            env.scene["robot"].data.body_link_ang_vel_w[:, anchor],
            motion.motion_data.body_ang_vel_w[:, anchor],
        )
        policy_ids, _ = env.scene["robot"].find_joints(G1_SONIC_JOINTS, preserve_order=True)
        expected_joint_obs = (
            env.scene["robot"].data.joint_pos[:, policy_ids]
            - env.scene["robot"].data.default_joint_pos[:, policy_ids]
        )
        # Actor history follows the Motrix PPO SONIC order: angular velocity,
        # joint position, joint velocity, action, then gravity.
        np.testing.assert_allclose(state.obs["obs"][:, 30:59], expected_joint_obs)
        motion._undesired_contact_history[:] = True
        env.reset(env_ids=np.asarray([0], dtype=np.int32))
        assert not motion._undesired_contact_history[0].any()
        assert motion._undesired_contact_history[1].all()
        state = env.step(np.zeros((2, 29), dtype=np.float32))
        assert np.isfinite(state.obs["obs"]).all()
        assert state.info["termination_reason_names"] == SONIC_TERMINATION_REASON_NAMES
        assert SONIC_TERMINATION_REASON_NAMES[-2:] == ("time_out", "clip_end")
        assert state.info["termination_reason_mask"].shape == (
            2,
            len(SONIC_TERMINATION_REASON_NAMES),
        )
        np.testing.assert_array_equal(
            state.info["termination_reason_mask"][:, -2],
            env.termination_manager.get_term("time_out"),
        )
        np.testing.assert_array_equal(
            state.info["termination_reason_mask"][:, -1],
            env.termination_manager.get_term("clip_end"),
        )
        # The first G1-reference value follows the frame-1 position, proving
        # reference advancement happens before transition observations.
        np.testing.assert_array_equal(state.obs["obs"][:, 930], 1.0)
        np.testing.assert_array_equal(
            env.command_manager.get_term("motion").sampler.current_frames, 1
        )

        # A done row must not be advanced by the per-step command update;
        # reset_reference() owns its next frame selection on the reset path.
        motion.sampler.current_frames[:] = 0
        env.reset_buf[:] = False
        env.reset_buf[0] = True
        motion._update_command(None)
        np.testing.assert_array_equal(motion.sampler.current_frames, [0, 1])

        # Reset randomization is clipped to the entity's soft joint limits,
        # matching the generic motion-tracking command contract.
        motion.cfg.reset_joint_position_range = (-100.0, 100.0)
        env.reset(env_ids=np.asarray([0], dtype=np.int32))
        limits = np.asarray(env.scene["robot"].data.soft_joint_pos_limits)
        joint_pos = env.scene["robot"].data.joint_pos[0]
        np.testing.assert_array_less(limits[:, 0] - 1.0e-6, joint_pos)
        np.testing.assert_array_less(joint_pos, limits[:, 1] + 1.0e-6)
    finally:
        env.close()


@pytest.mark.parametrize("backend_type", ["mujoco", "motrix"])
def test_sonic_reference_terms_are_row_scoped_on_reset(monkeypatch, backend_type: str) -> None:
    """A subset reset must recompute only the reset rows of the reference terms."""

    monkeypatch.setattr(sonic_manager, "SonicPackedMotionLoader", _RuntimeMotionLoader)
    config_dir = Path(__file__).parents[2] / "src" / "unilab" / "conf" / "flashsac"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        hydra_cfg = compose(
            config_name="config_sonic",
            overrides=[f"task=g1_sonic/{backend_type}"],
        )
    env_cfg_override = BackendAdapter(
        hydra_cfg, root_dir=Path(__file__).parents[2]
    ).build_task_env_cfg_override()
    params = env_cfg_override["commands"]["motion"]["params"]
    params["motion_store_file"] = "synthetic"
    params["sampling_mode"] = "start"
    params["pose_range"] = {axis: [0.0, 0.0] for axis in ("x", "y", "z", "roll", "pitch", "yaw")}
    params["velocity_range"] = {
        axis: [0.0, 0.0] for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    params["joint_position_range"] = [0.0, 0.0]
    params["joint_velocity_range"] = [0.0, 0.0]
    env_cfg_override["observations"]["policy"]["terms"]["obs"]["sonic_noise"] = {"level": 0.0}
    env = registry.make(
        "G1SonicManager",
        sim_backend=backend_type,
        env_cfg_override=env_cfg_override,
        num_envs=3,
    )
    try:
        env.init_state()
        env.step(np.zeros((env.num_envs,) + env.action_space.shape, dtype=np.float32))
        motion = env.command_manager.get_term("motion")
        term = env.observation_manager.get_term_cfg("policy", "g1_reference").func
        assert hasattr(term, "reset"), "g1_reference term must be a row-scoped class term"

        full = term(env).copy()
        # Simulate the observation manager's reset stamping for row 1 only,
        # then advance row 1's reference so a fresh row must differ.
        term.reset(np.asarray([1], dtype=np.intp))
        motion.sampler.current_frames[1] = np.minimum(
            motion.sampler.current_frames[1] + 3, motion.loader.clip_end_frames[0]
        )
        motion._g1_command_cache = None
        out = term(env)
        np.testing.assert_array_equal(out[[0, 2]], full[[0, 2]])
        np.testing.assert_array_equal(
            out[1], motion.g1_reference(np.asarray([1], dtype=np.intp))[0]
        )
        assert not term._reset_pending.any()
    finally:
        env.close()


@pytest.mark.parametrize("backend_type", ["mujoco", "motrix"])
def test_sonic_clip_subset_rotation_swaps_working_set(monkeypatch, backend_type: str) -> None:
    """Subset loading bounds the working set and rotation teleports onto it."""

    monkeypatch.setattr(sonic_manager, "SonicPackedMotionLoader", _SubsetRuntimeLoader)
    monkeypatch.setattr(sonic_manager, "packed_store_clip_count", lambda store: 6)
    config_dir = Path(__file__).parents[2] / "src" / "unilab" / "conf" / "flashsac"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        hydra_cfg = compose(
            config_name="config_sonic",
            overrides=[f"task=g1_sonic/{backend_type}"],
        )
    env_cfg_override = BackendAdapter(
        hydra_cfg, root_dir=Path(__file__).parents[2]
    ).build_task_env_cfg_override()
    params = env_cfg_override["commands"]["motion"]["params"]
    params["motion_store_file"] = "synthetic"
    params["max_loaded_clips"] = 2
    params["clip_rotation_interval_steps"] = 3
    params["sampling_mode"] = "start"
    params["pose_range"] = {axis: [0.0, 0.0] for axis in ("x", "y", "z", "roll", "pitch", "yaw")}
    params["velocity_range"] = {
        axis: [0.0, 0.0] for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    params["joint_position_range"] = [0.0, 0.0]
    params["joint_velocity_range"] = [0.0, 0.0]
    env_cfg_override["observations"]["policy"]["terms"]["obs"]["sonic_noise"] = {"level": 0.0}
    env = registry.make(
        "G1SonicManager",
        sim_backend=backend_type,
        env_cfg_override=env_cfg_override,
        num_envs=3,
    )
    try:
        state = env.init_state()
        motion = env.command_manager.get_term("motion")
        first_loader = motion.loader
        assert motion.loader.num_clips == 2
        assert len(motion.loader.subset_clip_indices) == 2
        assert motion._ref_smpl_human_local.shape[0] == motion.loader.num_frames

        state = None
        for _ in range(3):
            state = env.step(np.zeros((env.num_envs,) + env.action_space.shape, dtype=np.float32))

        assert motion._clip_rotations == 1
        assert motion.loader is not first_loader
        assert motion.loader.num_clips == 2
        # Every in-flight frame index is legal on the rotated loader.
        clips = motion.loader.get_clip_indices(motion.sampler.current_frames)
        assert int(clips.min()) >= 0 and int(clips.max()) < motion.loader.num_clips
        # Staged clip starts from the old working set are dropped and the
        # derived reference cache is rebuilt against the new one.
        assert (motion._next_reset_clip_indices == -1).all()
        assert motion._ref_smpl_human_local.shape[0] == motion.loader.num_frames
        assert motion._ref_joint_pos_policy.shape[0] == motion.loader.num_frames
        reference = motion.g1_reference()
        assert reference.shape == (env.num_envs, 640)
        assert np.isfinite(reference).all()
        assert state is not None and np.isfinite(state.obs["obs"]).all()

        # Without an active subset the rotation is a no-op.
        motion._clip_subset_size = None
        motion.rotate_clip_subset()
        assert motion.loader.num_clips == 2
        assert motion._clip_rotations == 1
    finally:
        env.close()


@pytest.mark.parametrize("backend_type", ["mujoco", "motrix"])
def test_sonic_reference_cache_matches_legacy_computation(monkeypatch, backend_type: str) -> None:
    """The cold-path reference feature cache must reproduce the legacy math."""

    monkeypatch.setattr(sonic_manager, "SonicPackedMotionLoader", _RuntimeMotionLoader)
    config_dir = Path(__file__).parents[2] / "src" / "unilab" / "conf" / "flashsac"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        hydra_cfg = compose(
            config_name="config_sonic",
            overrides=[f"task=g1_sonic/{backend_type}"],
        )
    env_cfg_override = BackendAdapter(
        hydra_cfg, root_dir=Path(__file__).parents[2]
    ).build_task_env_cfg_override()
    params = env_cfg_override["commands"]["motion"]["params"]
    params["motion_store_file"] = "synthetic"
    params["sampling_mode"] = "start"
    params["pose_range"] = {axis: [0.0, 0.0] for axis in ("x", "y", "z", "roll", "pitch", "yaw")}
    params["velocity_range"] = {
        axis: [0.0, 0.0] for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    params["joint_position_range"] = [0.0, 0.0]
    params["joint_velocity_range"] = [0.0, 0.0]
    env_cfg_override["observations"]["policy"]["terms"]["obs"]["sonic_noise"] = {"level": 0.0}
    env = registry.make(
        "G1SonicManager",
        sim_backend=backend_type,
        env_cfg_override=env_cfg_override,
        num_envs=3,
    )
    try:
        state = env.init_state()
        motion = env.command_manager.get_term("motion")
        rows = np.arange(env.num_envs, dtype=np.intp)

        def _compare() -> None:
            """Match the cold-path cache against the legacy on-the-fly math."""
            frames = motion.sampler.current_frames
            g1_future = motion.loader.future_indices(
                frames, sonic_manager._G1_FUTURE_STRIDE, motion.cfg.num_future_frames
            )
            legacy_pos = motion.loader.joint_pos[g1_future][..., motion._policy_from_model]
            legacy_vel = motion.loader.joint_vel[g1_future][..., motion._policy_from_model]
            legacy_cmd = (
                np.concatenate(
                    (
                        legacy_pos.reshape(len(frames), -1),
                        legacy_vel.reshape(len(frames), -1),
                    ),
                    axis=-1,
                )
                .reshape(len(frames), motion.cfg.num_future_frames, 58)
                .copy()
            )
            np.testing.assert_array_equal(motion.g1_command(rows), legacy_cmd)

            smpl_future = motion.loader.future_indices(
                frames, sonic_manager._SMPL_FUTURE_STRIDE, motion.cfg.num_future_frames
            )
            legacy_hum_quat = motion.loader.smpl_root_quat[smpl_future]
            legacy_hum_joints = motion.loader.smpl_joints[smpl_future]
            legacy_human_local = sonic_manager.np_quat_apply_batched(
                sonic_manager.np_quat_conjugate_batched(legacy_hum_quat[:, :, None]),
                legacy_hum_joints,
            )
            legacy_wrist = motion.loader.joint_pos[smpl_future][..., motion._policy_from_model][
                ..., sonic_manager._WRIST_POLICY_INDICES
            ]
            smpl = motion.smpl_reference()
            expected_human_local = legacy_human_local.reshape(
                len(frames), motion.cfg.num_future_frames, 72
            )
            # The packed layout is [human_local | rot6d(human_relative) | wrist]
            # per future frame; compare the cached gather against the fresh
            # computation by rebuilding the pack with the legacy features.
            root_quat = env.scene["robot"].data.root_link_quat_w
            legacy_human_relative = sonic_manager.np_quat_mul_batched(
                sonic_manager.np_quat_conjugate_batched(root_quat[:, None]), legacy_hum_quat
            )
            legacy_packed = sonic_manager._pack_smpl_reference(
                expected_human_local, legacy_human_relative, legacy_wrist
            ).astype(smpl.dtype, copy=False)
            np.testing.assert_array_equal(smpl, legacy_packed)

        _compare()
        env.step(np.zeros((env.num_envs,) + env.action_space.shape, dtype=np.float32))
        _compare()
        assert state.obs["obs"].shape == (3, 2412)
    finally:
        env.close()


@pytest.mark.parametrize("backend_type", ["mujoco", "motrix"])
def test_sonic_transition_uses_current_frame_for_reward_and_next_frame_for_observation(
    monkeypatch, backend_type: str
) -> None:
    """A transition must score frame t and expose frame t+1 as next observation."""
    monkeypatch.setattr(sonic_manager, "SonicPackedMotionLoader", _RuntimeMotionLoader)
    config_dir = Path(__file__).parents[2] / "src" / "unilab" / "conf" / "flashsac"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        hydra_cfg = compose(
            config_name="config_sonic",
            overrides=[f"task=g1_sonic/{backend_type}"],
        )
    env_cfg_override = BackendAdapter(
        hydra_cfg, root_dir=Path(__file__).parents[2]
    ).build_task_env_cfg_override()
    command = env_cfg_override["commands"]["motion"]
    params = command["params"]
    params.update(
        {
            "motion_store_file": "synthetic",
            "sampling_mode": "start",
            "anchor_pos_z_threshold": 100.0,
            "anchor_ori_threshold": 100.0,
            "ee_body_pos_z_threshold": 100.0,
            "foot_pos_threshold": 100.0,
            "pose_range": {axis: [0.0, 0.0] for axis in ("x", "y", "z", "roll", "pitch", "yaw")},
            "velocity_range": {
                axis: [0.0, 0.0] for axis in ("x", "y", "z", "roll", "pitch", "yaw")
            },
            "joint_position_range": [0.0, 0.0],
            "joint_velocity_range": [0.0, 0.0],
        }
    )
    env_cfg_override["observations"]["policy"]["terms"]["obs"]["sonic_noise"] = {"level": 0.0}
    env = registry.make(
        "G1SonicManager",
        sim_backend=backend_type,
        env_cfg_override=env_cfg_override,
        num_envs=1,
    )
    try:
        env.init_state()
        motion = env.command_manager.get_term("motion")
        # Distinguish reference frames in the reward-side motion buffer and
        # in the packed G1 reference (the synthetic loader's joint values are
        # equal to the global frame index).
        motion.motion_data.body_pos_w[:, 0, 0] = 7.0
        sampled_frames: list[int] = []
        original_compute = env.reward_manager.compute

        def capture_reward(*args, **kwargs):
            sampled_frames.append(int(motion.sampler.current_frames[0]))
            return original_compute(*args, **kwargs)

        monkeypatch.setattr(env.reward_manager, "compute", capture_reward)
        state = env.step(np.zeros((1, 29), dtype=np.float32))
        assert sampled_frames == [0]
        np.testing.assert_array_equal(motion.sampler.current_frames, [1])
        # Synthetic joint_pos[frame, 0] == frame; g1 reference begins with
        # the frame-1 command after the command manager advances the sampler.
        assert state.obs["obs"][0, 930] == pytest.approx(1.0)
    finally:
        env.close()


@pytest.mark.parametrize("backend_type", ["mujoco", "motrix"])
def test_sonic_manager_wrap_mode_continues_episode_past_clip_end(
    monkeypatch, backend_type: str
) -> None:
    """Wrap mode resamples the reference and teleports at a clip's end.

    The episode itself survives the wrap: no termination fires, episode
    bookkeeping keeps counting, and the robot moves back onto the freshly
    sampled reference frame (the MotrixLab sim-reset-on-wrap contract).
    """
    monkeypatch.setattr(sonic_manager, "SonicPackedMotionLoader", _RuntimeMotionLoader)
    config_dir = Path(__file__).parents[2] / "src" / "unilab" / "conf" / "flashsac"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        hydra_cfg = compose(
            config_name="config_sonic",
            overrides=[f"task=g1_sonic/{backend_type}"],
        )
    env_cfg_override = BackendAdapter(
        hydra_cfg, root_dir=Path(__file__).parents[2]
    ).build_task_env_cfg_override()
    params = env_cfg_override["commands"]["motion"]["params"]
    params.update(
        {
            "motion_store_file": "synthetic",
            "sampling_mode": "start",
            "anchor_pos_z_threshold": 100.0,
            "anchor_ori_threshold": 100.0,
            "ee_body_pos_z_threshold": 100.0,
            "foot_pos_threshold": 100.0,
            "pose_range": {axis: [0.0, 0.0] for axis in ("x", "y", "z", "roll", "pitch", "yaw")},
            "velocity_range": {
                axis: [0.0, 0.0] for axis in ("x", "y", "z", "roll", "pitch", "yaw")
            },
            "joint_position_range": [0.0, 0.0],
            "joint_velocity_range": [0.0, 0.0],
        }
    )
    env_cfg_override["observations"]["policy"]["terms"]["obs"]["sonic_noise"] = {"level": 0.0}
    env = registry.make(
        "G1SonicManager",
        sim_backend=backend_type,
        env_cfg_override=env_cfg_override,
        num_envs=1,
    )
    try:
        env.init_state()
        motion = env.command_manager.get_term("motion")
        assert motion.cfg.params.truncate_on_clip_end is False
        assert "clip_end" not in env.termination_manager.active_terms
        num_frames = int(motion.loader.num_frames)
        for step in range(1, num_frames):
            state = env.step(np.zeros((1, 29), dtype=np.float32))
            assert not state.terminated.any()
            assert not state.truncated.any()
            assert not motion.clip_end[0]
            np.testing.assert_array_equal(motion.sampler.current_frames, [step])
        # The clip's final frame has now been current for one step; the next
        # advance wraps instead of truncating the episode.
        length_before_wrap = int(env.episode_length_buf[0])
        state = env.step(np.zeros((1, 29), dtype=np.float32))
        assert not state.terminated.any()
        assert not state.truncated.any()
        assert not env.reset_buf[0]
        assert motion.clip_end[0]
        # "start" sampling resamples frame 0; the reference buffer and the
        # robot both move onto the new frame.
        np.testing.assert_array_equal(motion.sampler.current_frames, [0])
        assert motion.motion_data.joint_pos[0, 0] == pytest.approx(0.0)
        np.testing.assert_allclose(
            env.scene["robot"].data.joint_pos[0],
            motion.motion_data.joint_pos[0],
            atol=1.0e-6,
        )
        # Episode bookkeeping is untouched by the wrap.
        assert env.episode_length_buf[0] == length_before_wrap + 1
        # The wrap flag clears on the following advance and the reference
        # keeps playing from the resampled frame.
        state = env.step(np.zeros((1, 29), dtype=np.float32))
        assert not motion.clip_end[0]
        np.testing.assert_array_equal(motion.sampler.current_frames, [1])
        assert env.episode_length_buf[0] == length_before_wrap + 2
    finally:
        env.close()
