"""Cold-path reference-motion forward kinematics for the SONIC data pipeline.

Moved here from the retired unisim sonic branch so the upstream unisim
dependency stays pristine. These adapters run only during dataset
conversion/tooling (never in ``step``/``reset``) and reproduce the released
SONIC kinematics contract: MuJoCo consumes wxyz quaternions, Motrix xyzw, and
both return world-frame body velocities in UniLab's wxyz layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from unilab.base.rotation import np_quat_apply_inverse_batched

if TYPE_CHECKING:
    import motrixsim as mtx
    from unisim.backend.base import SimBackend


@dataclass(frozen=True)
class ReferenceKinematics:
    """Reference link states in UniLab's wxyz quaternion contract."""

    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray


def materialize_reference_kinematics_mujoco(
    backend: "SimBackend",
    *,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    root_lin_vel: np.ndarray,
    root_ang_vel: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    body_names: tuple[str, ...],
) -> ReferenceKinematics:
    """Run ``mj_forward`` once per source frame without stepping dynamics.

    ``root_ang_vel`` is the world-frame root angular velocity. MuJoCo's
    free-joint ``qvel[3:6]`` slot is body-local, so it is rotated into the
    root frame here; the returned body velocities are world-frame throughout.
    """
    if backend.backend_type != "mujoco":
        raise ValueError("MuJoCo reference materialization requires the MuJoCo backend")

    import mujoco

    model = backend._model  # pyright: ignore[reportPrivateUsage]
    count = int(root_pos.shape[0])
    expected = {
        "root_pos": (count, 3),
        "root_quat_wxyz": (count, 4),
        "root_lin_vel": (count, 3),
        "root_ang_vel": (count, 3),
        "joint_pos": (count, model.nu),
        "joint_vel": (count, model.nu),
    }
    values = {
        "root_pos": root_pos,
        "root_quat_wxyz": root_quat_wxyz,
        "root_lin_vel": root_lin_vel,
        "root_ang_vel": root_ang_vel,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
    }
    for name, shape in expected.items():
        value = np.asarray(values[name])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite with shape {shape}, got {value.shape}")

    body_ids = np.asarray(backend.get_body_ids(body_names), dtype=np.intp)
    actuator_joint_ids = model.actuator_trnid[:, 0].astype(np.intp)
    qpos_ids = model.jnt_qposadr[actuator_joint_ids]
    qvel_ids = model.jnt_dofadr[actuator_joint_ids]
    quat = np.asarray(root_quat_wxyz, dtype=np.float64)
    quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12)
    root_ang_vel_local = np_quat_apply_inverse_batched(
        quat, np.asarray(root_ang_vel, dtype=np.float64)
    ).astype(np.float32)
    pos = np.empty((count, len(body_ids), 3), dtype=np.float32)
    quat_out = np.empty((count, len(body_ids), 4), dtype=np.float32)
    lin_vel = np.empty_like(pos)
    ang_vel = np.empty_like(pos)
    data = mujoco.MjData(model)
    for frame in range(count):
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_quat_wxyz[frame]
        data.qvel[:3] = root_lin_vel[frame]
        data.qvel[3:6] = root_ang_vel_local[frame]
        data.qpos[qpos_ids] = joint_pos[frame]
        data.qvel[qvel_ids] = joint_vel[frame]
        mujoco.mj_forward(model, data)
        pos[frame] = data.xpos[body_ids]
        quat_out[frame] = data.xquat[body_ids]
        for column, body_id in enumerate(body_ids):
            velocity = np.empty(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, int(body_id), velocity, 0
            )
            ang_vel[frame, column] = velocity[:3]
            lin_vel[frame, column] = velocity[3:]
    return ReferenceKinematics(pos, quat_out, lin_vel, ang_vel)


def materialize_reference_kinematics_motrix(
    backend: "SimBackend",
    *,
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    root_lin_vel: np.ndarray,
    root_ang_vel: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    body_names: tuple[str, ...],
) -> ReferenceKinematics:
    """Run batched FK once using the already materialized Motrix scene model."""
    if backend.backend_type != "motrix":
        raise ValueError("Motrix reference materialization requires the Motrix backend")

    import motrixsim as mtx

    model: mtx.SceneModel = backend._model  # pyright: ignore[reportPrivateUsage]
    count = int(root_pos.shape[0])
    expected = {
        "root_quat_xyzw": (count, 4),
        "root_lin_vel": (count, 3),
        "root_ang_vel": (count, 3),
        "joint_pos": (count, len(model.joint_names)),
        "joint_vel": (count, len(model.joint_names)),
    }
    values = {
        "root_quat_xyzw": root_quat_xyzw,
        "root_lin_vel": root_lin_vel,
        "root_ang_vel": root_ang_vel,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
    }
    if root_pos.shape != (count, 3):
        raise ValueError(f"root_pos must have shape {(count, 3)}, got {root_pos.shape}")
    for name, shape in expected.items():
        if values[name].shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {values[name].shape}")

    qpos = np.concatenate((root_pos, root_quat_xyzw, joint_pos), axis=-1).astype(
        np.float32, copy=False
    )
    qvel = np.concatenate((root_lin_vel, np.zeros_like(root_ang_vel), joint_vel), axis=-1).astype(
        np.float32, copy=False
    )
    data = mtx.SceneData(model, batch=[count])
    data.set_dof_pos(np.ascontiguousarray(qpos), model)
    data.set_dof_vel(np.ascontiguousarray(qvel))
    model.floating_bases[0].set_global_angular_velocity(
        data, np.ascontiguousarray(root_ang_vel, dtype=np.float32)
    )
    model.forward_kinematic(data)

    body_ids = []
    for name in body_names:
        body_id = model.get_link_index(name)
        if body_id is None or body_id < 0:
            raise ValueError(f"Body {name!r} not found in Motrix model")
        body_ids.append(int(body_id))
    poses = np.asarray(model.get_link_poses(data), dtype=np.float32)[:, body_ids]
    quat_wxyz = poses[..., (6, 3, 4, 5)]
    return ReferenceKinematics(
        body_pos_w=np.ascontiguousarray(poses[..., :3]),
        body_quat_w=np.ascontiguousarray(quat_wxyz),
        body_lin_vel_w=np.ascontiguousarray(
            np.asarray(model.get_link_linear_velocities(data), dtype=np.float32)[:, body_ids]
        ),
        body_ang_vel_w=np.ascontiguousarray(
            np.asarray(model.get_link_angular_velocities(data), dtype=np.float32)[:, body_ids]
        ),
    )


__all__ = [
    "ReferenceKinematics",
    "materialize_reference_kinematics_mujoco",
    "materialize_reference_kinematics_motrix",
]
