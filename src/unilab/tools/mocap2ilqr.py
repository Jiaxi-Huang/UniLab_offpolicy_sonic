#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Re-solve converted SONIC mocap clips with receding-horizon iLQR dynamics.

Dataset-agnostic: reads any paired SONIC robot/SMPL NPZ dataset (the
``unilab_sonic_robot_v1``/``unilab_sonic_smpl_v1`` contract, e.g.
``data/lafan1`` produced by ``unilab.tools.lafan_data`` or the release
collection in ``data/sonic``) and writes a derived dataset (for example
``data/lafan1_ilqr``) whose robot stream follows a dynamically feasible iLQR
plan that tracks each source clip's body trajectories.  The source dataset is
never modified; the SMPL stream is copied unchanged because the optimizer only
touches the robot kinematics and frame counts stay aligned.

The solver core is ported from the dbm_uni receding-horizon iLQR tracker
(``examples/g1_flip.py``, Apache-2.0).  The physics rate must match the
training environment: ``dt * sub`` equals one 50 Hz control step, and the
scene is assembled from the same model/fragment XMLs as the ``g1_sonic``
owner config, so the plan's contact modes agree with training dynamics.
The output is solved-only: clips whose solve diverges (non-finite states, or
deviation from the source beyond the acceptance bounds) land nothing on disk,
are flagged in ``ilqr_manifest.json``, and are retried by a rerun (no output
pair exists to skip).
"""

# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import mujoco
import numpy as np
from hydra import compose, initialize_config_dir
from mjbatch import Batch
from unisim.backend.mujoco.reference_motion import materialize_reference_kinematics
from unisim.backend.mujoco.xml import materialize_scene_fragments

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.envs import ManagerBasedRlEnvCfg
from unilab.tasks.motion_tracking.g1.sonic_data import (
    _TARGET_FPS,
    resolve_sonic_pairs,
)
from unilab.tasks.motion_tracking.g1.sonic_manager import (
    G1_SONIC_BODY_NAMES,
    SonicMotionCommandCfg,
)
from unilab.tools.lafan_data import _write_robot_npz
from unilab.tools.pack_sonic_data import pack_sonic_dataset

# Bodies whose world-frame features the tracking cost follows.  This is the
# dbm_uni backflip set: the SONIC limbs without the wrist yaw links.
DEFAULT_TRACKED_BODIES = (
    "pelvis",
    "torso_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
)
ILQR_MANIFEST_FORMAT = "unilab_mocap_ilqr_v1"
# Small vertical offset that keeps the initial reference out of the floor.
LIFT = 0.001


@dataclass(frozen=True)
class IlqrCostWeights:
    """Cost weights per m^2, rad^2, (m/s)^2, (rad/s)^2, and rad^2 of command."""

    w_pos: float = 100.0
    w_rot: float = 100.0
    w_vel: float = 5.0
    w_ang: float = 1.0
    w_ctrl: float = 1.0
    w_root: tuple[float, float, float] = (1.0, 1.0, 1000.0)
    huber: float = 0.05


@dataclass(frozen=True)
class IlqrSolverConfig:
    horizon: int = 50
    step: int = 10
    iterations: int = 20
    tolerance: float = 1.0e-4
    weights: IlqrCostWeights = field(default_factory=IlqrCostWeights)
    # Early abort when the committed plan stays fallen; None disables.
    abort_pelvis_z: float | None = 0.3
    abort_patience: int = 25
    # A low plan only counts as fallen while the reference is standing this
    # high: intentional floor motion (fall-and-get-up families) must not trip
    # the fall detector.  None keeps the legacy plan-only check.
    abort_ref_pelvis_z: float | None = 0.45
    # Impatience applied while the window is fallen: no recovery has been
    # observed from a sustained fall, so the patience shrinks.
    abort_patience_fallen: int = 8
    # Committed windows sampled before the cost-abort baseline activates.
    abort_warmup: int = 50
    # Early abort when the committed plan's window cost stays far above the
    # clip's healthy baseline (a fallen-but-hovering tracker grinds through
    # every remaining window at hopeless cost); non-positive disables.
    cost_abort_multiple: float = 50.0
    # Abort as soon as the committed prefix makes the acceptance gates
    # mathematically impossible to pass.
    early_gate_reject: bool = True
    # Render the solve-time view (plan-state robot, reference ghost, window
    # plan trails) into ``<video_output_dir>/<stem>_opt.mp4``.
    record_video: bool = False
    video_output_dir: str | None = None
    # Acceptance gates mirrored for the early-reject monitor; ``convert``
    # injects the run's gate values so workers judge with the real ones.
    max_joint_deviation: float = 0.15
    max_joint_dev_p95: float = 0.4
    max_root_deviation: float = 0.05
    max_root_dev_p95: float = 0.15
    root_path_kappa: float = 0.0


# ── Quaternion helpers on (..., 4) arrays in MuJoCo (w, x, y, z) order. ──────


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w = a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdims=True)
    v = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])
    return np.concatenate([w, v], axis=-1)


def _quat_exp(v: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    axis = np.divide(v, angle, out=np.zeros_like(v), where=angle > 0)
    return np.concatenate([np.cos(angle / 2), np.sin(angle / 2) * axis], axis=-1)


def _quat_log(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    axis = np.divide(q[..., 1:], norm, out=np.zeros_like(q[..., 1:]), where=norm > 0)
    angle = 2.0 * np.arctan2(norm, q[..., :1])
    return np.where(angle > np.pi, angle - 2.0 * np.pi, angle) * axis


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    padded = np.concatenate([np.zeros_like(v[..., :1]), v], axis=-1)
    return _quat_mul(_quat_mul(q, padded), q * [1.0, -1.0, -1.0, -1.0])[..., 1:]


def _ensure_quaternion_continuity(quats: np.ndarray) -> np.ndarray:
    """Flip equivalent q/-q representations onto one shortest-path branch."""
    result = quats.copy()
    for frame in range(1, len(result)):
        flip = np.sum(result[frame - 1] * result[frame], axis=-1) < 0.0
        result[frame, flip] *= -1.0
    return result


# ── Solver core (ported from dbm_uni examples/g1_flip.py, Apache-2.0). ───────


class IlqrPlanner:
    """iLQR over a batched MuJoCo model; subclasses supply cost and expansion."""

    def __init__(self, model: mujoco.MjModel, horizon: int, sub: int):
        self.T, self.sub = horizon, sub
        self.nq, self.nv, self.nu = model.nq, model.nv, model.nu
        self.nx = 2 * self.nv
        self.lo, self.hi = model.actuator_ctrlrange.T
        joint_types = model.jnt_type
        if np.any(joint_types == 1):
            raise ValueError("ball joints are not supported")
        free = list(
            zip(
                model.jnt_qposadr[joint_types == 0],
                model.jnt_dofadr[joint_types == 0],
                strict=True,
            )
        )
        quat = [address + i for address, _ in free for i in range(3, 7)]
        spin = [dof + i for _, dof in free for i in range(3, 6)]
        self._lin_q = np.setdiff1d(range(self.nq), quat)
        self._lin_v = np.setdiff1d(range(self.nv), spin)
        self._free = free
        # mjbatch defaults to one thread per logical CPU; under taskset that
        # oversubscribes each shard's core slice and burns it on scheduling.
        # Size the pools by the CPUs we may actually run on.
        threads = len(os.sched_getaffinity(0))
        self.batch = Batch(model, self.T * (1 + self.nx + self.nu), num_threads=threads)
        self.qpos, self.qvel = self.batch.bind("qpos"), self.batch.bind("qvel")
        self.ctrl, self.warm = self.batch.bind("ctrl"), self.batch.bind("qacc_warmstart")
        alphas = 0.5 ** np.arange(9)
        self.alphas = alphas
        self.line = Batch(model, len(alphas), forward=True, num_threads=threads)
        self._line_fields = [
            self.line.bind(name) for name in ("qpos", "qvel", "ctrl", "qacc_warmstart")
        ]
        self.warning = self.line.bind("warning")

    def integrate(self, qpos: np.ndarray, dq: np.ndarray) -> np.ndarray:
        out = qpos.copy()
        out[:, self._lin_q] += dq[:, self._lin_v]
        for address, dof in self._free:
            quat = _quat_mul(
                qpos[:, address + 3 : address + 7], _quat_exp(dq[:, dof + 3 : dof + 6])
            )
            out[:, address + 3 : address + 7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
        return out

    def difference(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dq = np.empty((len(x), self.nv))
        dq[:, self._lin_v] = y[:, self._lin_q] - x[:, self._lin_q]
        for address, dof in self._free:
            conj = x[:, address + 3 : address + 7] * [1.0, -1.0, -1.0, -1.0]
            dq[:, dof + 3 : dof + 6] = _quat_log(_quat_mul(conj, y[:, address + 3 : address + 7]))
        return np.concatenate([dq, y[:, self.nq :] - x[:, self.nq :]], axis=1)

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        count = len(x)
        self.qpos[:count], self.qvel[:count] = x[:, : self.nq], x[:, self.nq :]
        self.ctrl[:count], self.warm[:count] = u, 0.0
        ids = np.arange(count) if count < self.batch.num_sims else None
        self.batch.step(ids, nstep=1)
        self.probe(count)
        if self.sub > 1:
            self.batch.step(ids, nstep=self.sub - 1)
        return np.concatenate([self.qpos[:count], self.qvel[:count]], axis=1)

    def probe(self, count: int) -> None:  # noqa: ARG002 (subclasses override)
        return None

    def cost(self, t: int, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def advance(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        qpos, qvel, ctrl, warm = self._line_fields
        qpos[:], qvel[:], ctrl[:], warm[:] = x[:, : self.nq], x[:, self.nq :], u, 0.0
        self.line.step(nstep=self.sub)
        return np.concatenate([qpos[:], qvel[:]], axis=1)

    def linearize(self, xs: np.ndarray, us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Jacobians A_t = df/dx and B_t = df/du per knot, by forward differences."""
        horizon, nx, nu, nv = self.T, self.nx, self.nu, self.nv
        columns = 1 + nx + nu
        eps = 1.0e-6
        x, u = np.repeat(xs[:-1], columns, axis=0), np.repeat(us, columns, axis=0)
        delta = eps * np.tile(np.eye(columns)[:, 1:], (horizon, 1))
        # MuJoCo clamps ctrl: perturb downwards at the upper bound.
        delta[:, nx:] *= np.where(u + eps > self.hi, -1.0, 1.0)
        x[:, : self.nq] = self.integrate(x[:, : self.nq], delta[:, :nv])
        x[:, self.nq :] += delta[:, nv:nx]
        out = self.step(x, u + delta[:, nx:]).reshape(horizon, columns, -1)
        base = np.repeat(out[:, 0], columns - 1, axis=0)
        signed = delta.reshape(horizon, columns, -1)[:, 1:].sum(axis=2).ravel()
        jac = self.difference(base, out[:, 1:].reshape(-1, out.shape[2])) / signed[:, None]
        jac = np.swapaxes(jac.reshape(horizon, nx + nu, nx), 1, 2)
        return jac[:, :, :nx], jac[:, :, nx:]


class MotionClip:
    """A converted SONIC robot NPZ as MuJoCo states plus tracked-body features."""

    def __init__(self, npz_path: Path, model: mujoco.MjModel, tracked_bodies):
        label = npz_path.name
        with np.load(npz_path, allow_pickle=False) as loaded:
            data = {
                name: (
                    loaded[name].astype(np.float64)
                    if loaded[name].dtype.kind in "fiu"
                    else loaded[name]
                )
                for name in loaded.files
            }
        joint_names = tuple(str(name) for name in data["joint_names"])
        body_names = tuple(str(name) for name in data["body_names"])
        self.frames = int(data["num_frames"])
        if int(data["fps"]) != _TARGET_FPS:
            raise ValueError(f"{npz_path.name}: expected {_TARGET_FPS} Hz clip")
        pelvis = body_names.index("pelvis")
        pos = data["body_pos_w"][:, pelvis] + [0.0, 0.0, LIFT]
        quat = data["body_quat_w"][:, pelvis]
        quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
        model_joint_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint))
            for joint in np.arange(1, model.njnt)
        ]
        missing = [name for name in model_joint_names if name not in joint_names]
        if missing:
            raise ValueError(f"{label}: clip lacks model joints {missing}")
        keep = np.asarray([joint_names.index(name) for name in model_joint_names], dtype=np.intp)
        # MuJoCo free-joint angular velocity is body-local; the NPZ stores
        # world-frame values.
        spin = _rotate(quat * [1.0, -1.0, -1.0, -1.0], data["body_ang_vel_w"][:, pelvis])
        self.qpos = np.concatenate([pos, quat, data["joint_pos"][:, keep]], axis=1)
        self.qvel = np.concatenate(
            [data["body_lin_vel_w"][:, pelvis], spin, data["joint_vel"][:, keep]], axis=1
        )
        tracked = list(tracked_bodies)
        ids = np.asarray([body_names.index(name) for name in tracked], dtype=np.intp)
        self.features = np.concatenate(
            (
                data["body_pos_w"][:, ids] + [0.0, 0.0, LIFT],
                data["body_quat_w"][:, ids],
                data["body_lin_vel_w"][:, ids],
                data["body_ang_vel_w"][:, ids],
            ),
            axis=2,
        )

    def warm_start(self, model: mujoco.MjModel) -> np.ndarray:
        """The clip's joint angles as initial position commands, per actuator."""
        joint = model.actuator_trnid[:, 0]
        lo, hi = model.actuator_ctrlrange.T
        return np.clip(self.qpos[:, model.jnt_qposadr[joint]], lo, hi)


class IlqrTracker(IlqrPlanner):
    """Cost: tracking error of the clip's tracked bodies via FRAME sensors."""

    def __init__(
        self,
        model: mujoco.MjModel,
        clip: MotionClip,
        config: IlqrSolverConfig,
        sub: int,
        tracked_bodies,
    ):
        super().__init__(model, config.horizon, sub)
        self.clip = clip
        self.config = config
        self.start = 0
        self.tracked = list(tracked_bodies)
        self.sensor, self.line_sensor = self.batch.bind("sensordata"), self.line.bind("sensordata")
        # The FRAME sensors were appended after the shipped ones; they are
        # body-major and contiguous.
        self.frames = slice(int(model.sensor(f"{self.tracked[0]}_pos").adr.item()), None)

    def window(self, start: int, length: int) -> None:
        self.start, self.T = start, length

    def features(self, x: np.ndarray) -> np.ndarray:
        """Sensor readings at each state via a batch forward: (n, bodies, 13)."""
        count = len(x)
        self.qpos[:count], self.qvel[:count] = x[:, : self.nq], x[:, self.nq :]
        self.batch.forward(np.arange(count))
        return self.sensor[:count][:, self.frames].reshape(count, len(self.tracked), 13)

    def probe(self, count: int) -> None:
        columns = 1 + self.nx + self.nu
        self.feat = (
            self.sensor[:count][:, self.frames]
            .reshape(-1, columns, len(self.tracked), 13)[:, : 1 + self.nx]
            .copy()
        )

    def residual(self, t: np.ndarray | int, feat: np.ndarray):
        """Weighted errors vs clip knot t plus the pseudo-Huber position slope."""
        weights = self.config.weights
        end = self.frames_stop
        ref = self.clip.features[np.minimum(self.start + t, end)]
        pos, quat, vel = feat[..., :3], feat[..., 3:7], feat[..., 7:]
        rot = _quat_log(_quat_mul(quat, ref[..., 3:7] * [1.0, -1.0, -1.0, -1.0]))
        rel = (pos - pos[..., :1, :]) - (ref[..., :3] - ref[..., :1, :3])
        root = pos[..., :1, :] - ref[..., :1, :3]
        lin, ang = vel[..., :3] - ref[..., 7:10], vel[..., 3:] - ref[..., 10:]
        weighted = (
            (rel, weights.w_pos),
            (root, np.asarray(weights.w_root)),
            (rot, weights.w_rot),
            (lin, weights.w_vel),
            (ang, weights.w_ang),
        )
        errors = [error * np.sqrt(weight) for error, weight in weighted]
        robust = np.concatenate(errors[:2], axis=-2)
        slope = 1.0 / np.sqrt(1.0 + (robust**2).sum(-1) / (weights.w_pos * weights.huber**2))
        flattened = np.concatenate(
            [error.reshape(*error.shape[:-2], -1) for error in errors], axis=-1
        )
        return flattened, slope

    def cost(self, t: int, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # At t == 0 the line batch has not advanced yet, so the state's sensor
        # readings come from a forward on the main batch; afterwards the line
        # sensors already hold x.
        feat = (
            self.features(x)
            if t == 0
            else self.line_sensor.reshape(len(x), -1)[:, self.frames].reshape(
                len(x), len(self.tracked), 13
            )
        )
        residual, slope = self.residual(t, feat)
        robust_count = 3 * slope.shape[-1]
        weights = self.config.weights
        robust = 2.0 * weights.w_pos * weights.huber**2 * (1.0 / slope - 1.0)
        return (
            robust.sum(-1)
            + (residual[..., robust_count:] ** 2).sum(-1)
            + weights.w_ctrl * (u**2).sum(-1)
        )

    def expand(
        self, xs: np.ndarray, us: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Gauss-Newton gradient/Hessian with IRLS reweighting of robust rows."""
        horizon, nx, nv, nq = self.T, self.nx, self.nv, self.nq
        eps = 1.0e-6
        last = np.repeat(xs[-1:], 1 + nx, axis=0)
        delta = eps * np.eye(1 + nx)[:, 1:]
        last[:, :nq] = self.integrate(last[:, :nq], delta[:, :nv])
        last[:, nq:] += delta[:, nv:]
        feat = np.concatenate([self.feat, self.features(last)[None]])
        residual, slope = self.residual(np.arange(horizon + 1)[:, None], feat)
        jac = (residual[:, 1:] - residual[:, :1]) / eps
        weight = np.ones(residual.shape[::2])
        weight[:, : 3 * slope.shape[-1]] = np.repeat(slope[:, 0], 3, axis=-1)
        lx = 2.0 * np.einsum("tkr,tr->tk", jac, weight * residual[:, 0])
        lxx = 2.0 * np.einsum("tkr,tr,tlr->tkl", jac, weight, jac, optimize=True)
        ctrl_weight = 2.0 * self.config.weights.w_ctrl
        return (
            lx,
            lxx,
            ctrl_weight * us,
            ctrl_weight * np.tile(np.eye(self.nu), (horizon, 1, 1)),
        )

    @property
    def frames_stop(self) -> int:
        return self.clip.frames - 1


def _boxqp(
    q_matrix: np.ndarray, q_vector: np.ndarray, lo: np.ndarray, hi: np.ndarray, k: np.ndarray
):
    """Minimize 1/2 k'Qk + q'k subject to lo <= k <= hi (Tassa 2012)."""
    k = np.clip(k, lo, hi)
    count = len(q_vector)
    index = np.empty(count, np.int32)
    free = np.zeros(count, bool)
    free[
        index[
            : mujoco.mju_boxQP(k, np.empty((count, count + 7)), index, q_matrix, q_vector, lo, hi)
        ]
    ] = True
    return k, free


def _backward(
    a_jac: np.ndarray,
    b_jac: np.ndarray,
    lx: np.ndarray,
    lxx: np.ndarray,
    lu: np.ndarray,
    luu: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    mu: float,
):
    horizon, nu, nx = len(lu), lu.shape[1], lx.shape[1]
    vx, vxx = lx[-1], lxx[-1]
    k = np.zeros((horizon + 1, nu))
    kalman = np.empty((horizon, nu, nx))
    for t in range(horizon - 1, -1, -1):
        qx = lx[t] + a_jac[t].T @ vx
        qu = lu[t] + b_jac[t].T @ vx
        qxx = lxx[t] + a_jac[t].T @ vxx @ a_jac[t]
        quu = luu[t] + b_jac[t].T @ vxx @ b_jac[t]
        qux = b_jac[t].T @ vxx @ a_jac[t]
        reg = vxx + mu * np.eye(nx)
        quu_reg = luu[t] + b_jac[t].T @ reg @ b_jac[t]
        qux_reg = b_jac[t].T @ reg @ a_jac[t]
        if not (np.isfinite(quu_reg).all() and np.linalg.eigvalsh(quu_reg).min() > 0.0):
            return None
        k[t], free = _boxqp(quu_reg, qu, lo[t], hi[t], k[t + 1])
        kalman[t] = 0.0  # no feedback on clamped controls
        kalman[t, free] = -np.linalg.solve(quu_reg[np.ix_(free, free)], qux_reg[free])
        vx = qx + kalman[t].T @ quu @ k[t] + kalman[t].T @ qu + qux.T @ k[t]
        vxx = qxx + kalman[t].T @ quu @ kalman[t] + kalman[t].T @ qux + qux.T @ kalman[t]
        vxx = 0.5 * (vxx + vxx.T)
    return k[:-1], kalman


def _ilqr(
    planner: IlqrTracker,
    x0: np.ndarray,
    us: np.ndarray,
    xs: np.ndarray,
    config: IlqrSolverConfig,
    watch=None,
):
    total = np.inf
    if watch is not None:
        watch(xs)

    def derivatives():
        return (
            *planner.linearize(xs, us),
            *planner.expand(xs, us),
            planner.lo - us,
            planner.hi - us,
        )

    mu, kalman, d = 1.0, np.zeros((planner.T, planner.nu, planner.nx)), derivatives()
    stop = False
    for _ in range(config.iterations):
        accepted = False
        a_jac, b_jac, lx, lxx, lu, luu, lo, hi = d
        sweep = _backward(a_jac, b_jac, lx, lxx, lu, luu, lo, hi, mu)
        if sweep is not None:
            k, kalman = sweep
            new_xs, new_us, totals = _rollout(planner, x0, us, (xs, k, kalman))
            best = int(np.argmin(totals))
            if accepted := bool(totals[best] < total):
                xs, us = new_xs[:, best], new_us[:, best]
                drop = total - totals[best]
                total, mu = totals[best], max(mu / 10.0, 1.0e-6)
                stop = drop < config.tolerance * total
        if not accepted:
            mu *= 10.0
            stop = mu > 1.0e6
        if watch is not None:
            watch(xs)
        if stop:
            break
        if accepted:
            d = derivatives()
    return xs, us, kalman, total


def _rollout(planner: IlqrPlanner, x0: np.ndarray, us: np.ndarray, gains=None):
    """Roll out on every line-search sim; with gains (xs, k, K), apply feedback."""
    count = len(planner.alphas)
    horizon = planner.T
    x = np.tile(x0, (count, 1))
    warned = planner.warning.copy()
    new_xs = np.empty((horizon + 1, count, planner.nq + planner.nv))
    new_us = np.empty((horizon, count, planner.nu))
    total = np.zeros(count)
    new_xs[0] = x
    for t in range(horizon):
        u = np.tile(us[t], (count, 1))
        if gains is not None:
            xs, k, kalman = gains
            u += (
                planner.alphas[:, None] * k[t]
                + planner.difference(np.tile(xs[t], (count, 1)), x) @ kalman[t].T
            )
        u = np.clip(u, planner.lo, planner.hi)
        total += planner.cost(t, x, u)
        x = planner.advance(x, u)
        new_xs[t + 1], new_us[t] = x, u
    total += planner.cost(horizon, x, np.zeros((count, planner.nu)))
    total[(planner.warning != warned).any((1, 2))] = np.inf
    return new_xs, new_us, total


def _window_fallen(
    plan_z: float,
    ref_z: float,
    abort_pelvis_z: float | None,
    abort_ref_pelvis_z: float | None,
) -> bool:
    """A committed window counts as fallen only when the plan is low AND the
    reference is standing: intentional floor motion (fall-and-get-up families)
    tracks a low reference by design and must not trip the fall detector."""
    if abort_pelvis_z is None or plan_z >= abort_pelvis_z:
        return False
    return abort_ref_pelvis_z is None or ref_z > abort_ref_pelvis_z


class _PrefixGateMonitor:
    """Provable early rejection against the acceptance gates.

    The final deviation statistics are bounded below by the committed prefix:
    the final mean is at least ``prefix_sum / total_entries`` even when every
    remaining frame tracks perfectly, and the final p95 exceeds a gate once
    more than 5% of ALL entries already exceed it.  Once either bound crosses
    a gate the clip cannot pass acceptance, so the solve aborts instead of
    grinding through the rest of the motion.  Entries are counted against the
    clip length (the diagnostics comparison truncates to it as well), so the
    sliding-window overshoot past the end is not accumulated.
    """

    def __init__(
        self,
        clip: MotionClip,
        *,
        max_joint_deviation: float,
        max_joint_dev_p95: float,
        max_root_deviation: float,
        max_root_dev_p95: float,
        root_path_kappa: float = 0.0,
    ) -> None:
        frames, nq = clip.qpos.shape
        self._reference = clip.qpos
        self._frames = int(frames)
        self._joint_slice = slice(7, nq)
        self._joint_total = self._frames * (nq - 7)
        self._joint_gate_mean = max_joint_deviation
        self._joint_gate_p95 = max_joint_dev_p95
        self._joint_sum = 0.0
        self._joint_breaches = 0
        self._root3 = root_path_kappa <= 0.0
        self._root3_gate_mean = max_root_deviation
        self._root3_gate_p95 = max_root_dev_p95
        self._root3_total = self._frames * 3
        self._root3_sum = 0.0
        self._root3_breaches = 0
        path = _root_path_length(clip.qpos[:, :2])
        self._horiz_gate_mean = max(max_root_deviation, root_path_kappa * path)
        self._horiz_gate_p95 = max(max_root_dev_p95, root_path_kappa * path)
        self._z_gate_mean = max_root_deviation
        self._z_gate_p95 = max_root_dev_p95
        self._frame_totals = self._frames
        self._horiz_sum = 0.0
        self._horiz_breaches = 0
        self._z_sum = 0.0
        self._z_breaches = 0

    def update(self, block: np.ndarray, start: int) -> str | None:
        """Fold committed knots ``[start, start + len(block))`` into the
        running totals and return a rejection reason once doom is proven."""
        end = min(start + len(block), self._frames)
        if end <= start:
            return None
        plan = block[: end - start]
        ref = self._reference[start:end]
        joint = np.abs(plan[:, self._joint_slice] - ref[:, self._joint_slice])
        self._joint_sum += float(joint.sum())
        self._joint_breaches += int((joint > self._joint_gate_p95).sum())
        if self._joint_sum > self._joint_gate_mean * self._joint_total:
            return (
                f"early gate: mean joint deviation {self._joint_sum / self._joint_total:.3f} rad "
                "cannot recover below the gate"
            )
        if self._joint_breaches > 0.05 * self._joint_total:
            return (
                f"early gate: {self._joint_breaches} joint entries over "
                f"{self._joint_gate_p95} rad exceed the 5% p95 budget"
            )
        root = np.abs(plan[:, :3] - ref[:, :3])
        if self._root3:
            self._root3_sum += float(root.sum())
            self._root3_breaches += int((root > self._root3_gate_p95).sum())
            if self._root3_sum > self._root3_gate_mean * self._root3_total:
                return (
                    f"early gate: mean root deviation {self._root3_sum / self._root3_total:.3f} m "
                    "cannot recover below the gate"
                )
            if self._root3_breaches > 0.05 * self._root3_total:
                return f"early gate: root p95 budget exceeded ({self._root3_breaches} entries)"
        else:
            horiz = np.linalg.norm(plan[:, :2] - ref[:, :2], axis=1)
            self._horiz_sum += float(horiz.sum())
            self._horiz_breaches += int((horiz > self._horiz_gate_p95).sum())
            self._z_sum += float(root[:, 2].sum())
            self._z_breaches += int((root[:, 2] > self._z_gate_p95).sum())
            if self._horiz_sum > self._horiz_gate_mean * self._frame_totals:
                return "early gate: mean horizontal root deviation cannot recover"
            if self._horiz_breaches > 0.05 * self._frame_totals:
                return (
                    f"early gate: {self._horiz_breaches} frames over the horizontal "
                    f"p95 allowance {self._horiz_gate_p95:.3f} m"
                )
            if self._z_sum > self._z_gate_mean * self._frame_totals:
                return "early gate: mean vertical root deviation cannot recover"
            if self._z_breaches > 0.05 * self._frame_totals:
                return f"early gate: vertical root p95 budget exceeded ({self._z_breaches} frames)"
        return None


def solve_motion(
    tracker: IlqrTracker,
    x0: np.ndarray,
    feedforward: np.ndarray,
    config: IlqrSolverConfig,
    watch=None,
    *,
    abort_pelvis_z: float | None = 0.3,
    abort_patience: int = 25,
    abort_warmup: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Plan ``horizon`` knots at a time and commit ``step`` of them.

    The first window grows from ``step`` to ``horizon`` because a full-length
    plan from a cold start can fall over, and iLQR cannot recover from a fall.
    Three early exits keep hopeless tracks from burning compute: a sustained
    fall (pelvis below ``abort_pelvis_z`` while the reference stands above
    ``abort_ref_pelvis_z``), a window cost stuck far above the clip's healthy
    baseline, and the prefix gate monitor aborting once the committed prefix
    makes the acceptance gates mathematically impossible to pass.  The caller
    falls aborted clips back to the source instead of burning hours on an
    unrecoverable track.
    """
    total_steps = tracker.frames_stop
    nu = tracker.nu
    stages = [(0, end) for end in range(config.step, config.horizon, config.step)]
    stages += [(at, config.horizon) for at in range(0, total_steps, config.step)]
    held = np.concatenate([feedforward, np.tile(feedforward[-1:], (config.horizon, 1))])
    monitor = (
        _PrefixGateMonitor(
            tracker.clip,
            max_joint_deviation=config.max_joint_deviation,
            max_joint_dev_p95=config.max_joint_dev_p95,
            max_root_deviation=config.max_root_deviation,
            max_root_dev_p95=config.max_root_dev_p95,
            root_path_kappa=config.root_path_kappa,
        )
        if config.early_gate_reject
        else None
    )
    xs_all: list[np.ndarray] = [x0]
    us_all: list[np.ndarray] = []
    k_all: list[np.ndarray] = []
    us = np.zeros((0, nu))
    committed = False
    low_windows = 0
    costly_windows = 0
    healthy_costs: list[float] = []
    info: dict[str, Any] = {"aborted": False, "abort_reason": None}
    started = time.perf_counter()
    for index, (at, length) in enumerate(stages):
        tracker.window(at, length)
        shift = config.step if committed else 0
        us = np.concatenate([us[shift:], held[at + len(us) - shift : at + length]])
        xs = _rollout(tracker, x0, us)[0][:, 0]
        xs, us, kalman, cost = _ilqr(tracker, x0, us, xs, config, watch)
        committed = length == config.horizon or at > 0
        count = min(config.step, length) if committed else 0
        if watch is not None:
            watch(xs, commit=count)
        committed_start = len(us_all)
        xs_all += list(xs[1 : count + 1])
        us_all += list(us[:count])
        k_all += list(kalman[:count])
        x0 = xs[count]
        done = (index + 1) / len(stages)
        abort_reason = None
        if monitor is not None and count > 0:
            abort_reason = monitor.update(xs[1 : count + 1], committed_start)
        # Reference-relative fall detection: intentional floor motion must not
        # trip the detector, and a sustained fall earns less patience.
        knot = min(len(us_all) - 1, total_steps - 1)
        ref_z = float(tracker.clip.qpos[knot, 2])
        fallen = _window_fallen(float(x0[2]), ref_z, abort_pelvis_z, config.abort_ref_pelvis_z)
        patience = config.abort_patience_fallen if fallen else abort_patience
        if abort_reason is None and abort_pelvis_z is not None and committed:
            low_windows = low_windows + 1 if fallen else 0
            if low_windows >= patience:
                abort_reason = (
                    f"pelvis z below {abort_pelvis_z} m (ref {ref_z:.2f} m) for "
                    f"{low_windows} committed windows (knot {len(us_all)}/{total_steps})"
                )
        if abort_reason is None and committed and config.cost_abort_multiple > 0:
            # Track every committed window cost and use the cheapest one as the
            # healthy baseline.  Positional statistics (first windows, lower
            # half) get poisoned when a clip falls early and fallen windows
            # dominate the history; the minimum stays at the healthy level as
            # long as the tracker was ever on track.
            healthy_costs.append(float(cost))
            if len(healthy_costs) >= max(abort_warmup, 2):
                baseline = float(min(healthy_costs))
                if float(cost) > config.cost_abort_multiple * max(baseline, 1.0):
                    costly_windows += 1
                else:
                    costly_windows = 0
                if costly_windows >= patience:
                    abort_reason = (
                        f"window cost > {config.cost_abort_multiple}x baseline "
                        f"{baseline:.1f} for {costly_windows} committed windows "
                        f"(knot {len(us_all)}/{total_steps})"
                    )
        if abort_reason is not None:
            info["aborted"] = True
            info["abort_reason"] = abort_reason
            print(f"  abort: {abort_reason}", file=sys.stderr, flush=True)
            break
        if done >= 0.999 or (index + 1) % 50 == 0:
            print(
                f"  knot {len(us_all):6d}/{total_steps} cost {cost:10.1f} "
                f"height {x0[2]:.2f} elapsed {time.perf_counter() - started:7.1f}s",
                file=sys.stderr,
                flush=True,
            )
    return np.array(xs_all), np.array(us_all), np.array(k_all), info


# ── Scene / model plumbing. ──────────────────────────────────────────────────


def _sonic_scene_cfg() -> tuple[ManagerBasedRlEnvCfg, SonicMotionCommandCfg]:
    """Materialize the g1_sonic/mujoco owner config exactly like convert does."""
    config_dir = Path(__file__).parents[1] / "conf" / "flashsac"
    registry.ensure_registries()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        composed = compose(config_name="config_sonic", overrides=["task=g1_sonic/mujoco"])
    cfg = cast(ManagerBasedRlEnvCfg, registry.materialize_env_config("G1SonicManager"))
    root_dir = Path(__file__).parents[3]
    apply_cfg_overrides(
        cfg, BackendAdapter(composed, root_dir=root_dir).build_task_env_cfg_override()
    )
    motion_cfg = cfg.commands["motion"]
    if not isinstance(motion_cfg, SonicMotionCommandCfg) or cfg.scene is None:
        raise TypeError("G1SonicManager configuration is incomplete")
    return cfg, motion_cfg


def _scene_sources(cfg: ManagerBasedRlEnvCfg) -> tuple[str, tuple[str, ...]]:
    scene = cfg.scene
    assert scene is not None
    model_file = str(scene.model_file)
    fragments = tuple(str(path) for path in (scene.fragment_files or ()))
    return model_file, fragments


def _build_solver_model(
    model_file: str,
    fragment_files: tuple[str, ...],
    dt: float,
    tracked_bodies,
) -> tuple[mujoco.MjModel, str | None]:
    """Compile the SONIC scene with FRAME sensors for the tracked bodies.

    Sensors are appended body-major after the shipped ones, matching the
    tracker's contiguous (bodies, 13) feature slice.  Position-actuator
    ctrlranges are bounded by the driven joints' ranges.
    """
    merged: str | None = None
    if fragment_files:
        merged = materialize_scene_fragments(model_file, fragment_files=fragment_files)
        source = merged
    else:
        source = model_file
    try:
        spec = mujoco.MjSpec.from_file(source)
        spec.option.timestep = dt
        objtype = mujoco.mjtObj.mjOBJ_XBODY
        for body in tracked_bodies:
            for name in ("pos", "quat", "linvel", "angvel"):
                spec.add_sensor(
                    name=f"{body}_{name}",
                    type=getattr(mujoco.mjtSensor, f"mjSENS_FRAME{name.upper()}"),
                    objtype=objtype,
                    objname=body,
                )
        model = spec.compile()
    finally:
        if merged is not None:
            Path(merged).unlink(missing_ok=True)
    joint = model.actuator_trnid[:, 0]
    ranges = model.jnt_range[joint]
    bounded = ranges[:, 1] > ranges[:, 0]
    model.actuator_ctrlrange[bounded] = ranges[bounded]
    model.actuator_ctrllimited[:] = True
    return model, merged


_RECORD_GHOST_PREFIX = "ghost_"
_RECORD_GHOST_RGBA = (0.62, 0.30, 0.20, 0.35)
_RECORD_OLD_RGBA = (0.62, 0.30, 0.20, 0.30)
_RECORD_PLAN_RGBA = (0.95, 0.45, 0.10, 0.90)
_RECORD_HEAD_OFFSET = (0.0, 0.0, 0.43)
_RECORD_HISTORY = 5
_RECORD_PATH_STRIDE = 3  # subsample window plans for trail segments


def _add_scene_trail(
    scene: mujoco.MjvScene,
    points: np.ndarray,
    *,
    start: int,
    rgba: tuple[float, float, float, float],
) -> None:
    """Append a subsampled polyline to the render scene as capsules.

    ``points`` is one ``(..., knot, 3)`` polyline; only its final knot axis
    is time.  Earlier axes select paths (head/feet) or batch elements.
    """
    stride = max(_RECORD_PATH_STRIDE, 1)
    window = np.asarray(points[..., start::stride, :], dtype=np.float64)
    if window.ndim == 2:
        window = window[None]
    point_as = window[..., :-1, :].reshape(-1, 3)
    point_bs = window[..., 1:, :].reshape(-1, 3)
    count = min(len(point_as), max(scene.maxgeom - scene.ngeom, 0))
    for point_a, point_b in zip(
        point_as[:count],
        point_bs[:count],
    ):
        geom = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            [0.004, 0.0, 0.0],
            np.zeros(3),
            np.eye(3).flatten(),
            rgba,
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.004, point_a, point_b)


class _SolveRecorder:
    """Render the solve-time view into ``<stem>_opt.mp4`` (dbm g1_flip style).

    One frame per committed knot: the solid robot shows the accepted window
    plan's states, the translucent ghost shows the reference clip, and capsule
    trails draw the recent window plans (thin, faded) plus the committed plan
    from the current knot onward (thick).  Rendering never touches the solve
    -- the scene is a separate model and the paths come from the tracker's
    batch forward, exactly like the solver's own cost evaluation.  Any render
    fault disables the recording for the clip instead of failing the solve.
    """

    def __init__(
        self,
        stem: str,
        out_path: Path,
        model_sources: tuple[str, tuple[str, ...]],
        robot_file: str,
        tracker: IlqrTracker,
        clip: MotionClip,
        tracked_bodies: tuple[str, ...],
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        import imageio.v2 as imageio

        self.tracker = tracker
        self.clip = clip
        self.nq = clip.qpos.shape[1]
        self.frames = len(clip.qpos)
        self.knot_base = 0
        self.broken = False
        self.trail_history: deque[np.ndarray] = deque(maxlen=_RECORD_HISTORY)

        self.torso_idx = tracked_bodies.index("torso_link")
        self.foot_idx = (
            tracked_bodies.index("left_ankle_roll_link"),
            tracked_bodies.index("right_ankle_roll_link"),
        )

        model_file, fragment_files = model_sources
        merged = (
            materialize_scene_fragments(model_file, fragment_files=fragment_files)
            if fragment_files
            else None
        )
        try:
            spec = mujoco.MjSpec.from_file(merged or model_file)
            ghost_spec = mujoco.MjSpec.from_file(robot_file)
            frame = spec.worldbody.add_frame()
            spec.attach(ghost_spec, prefix=_RECORD_GHOST_PREFIX, frame=frame)
            self.model = spec.compile()
        finally:
            if merged is not None:
                Path(merged).unlink(missing_ok=True)
        # Ghost styling only; the render model is never stepped, so its
        # collision geoms are irrelevant.
        visual = (self.model.geom_contype == 0) & (self.model.geom_conaffinity == 0)
        for geom_id, body_id in enumerate(self.model.geom_bodyid):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(body_id))
            if body_name is not None and body_name.startswith(_RECORD_GHOST_PREFIX):
                self.model.geom_rgba[geom_id] = (
                    _RECORD_GHOST_RGBA if visual[geom_id] else [0, 0, 0, 0]
                )
        self.data = mujoco.MjData(self.model)
        # The scene ships a small default offscreen framebuffer; raise it to
        # the render resolution so the Renderer's frames fit.
        self.model.vis.global_.offwidth = width
        self.model.vis.global_.offheight = height

        ghost_free_adr, ghost_joint_adrs = self._ghost_layout()
        expected = self.nq - 7
        if len(ghost_joint_adrs) != expected:
            raise ValueError(
                f"ghost layout mismatch: {len(ghost_joint_adrs)} joints, expected {expected}"
            )
        self.ghost_free_adr = ghost_free_adr
        self.ghost_joint_adrs = ghost_joint_adrs
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        self.camera.distance = 3.0
        self.camera.elevation = -10.0
        self.camera.azimuth = -90.0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(
            str(out_path), fps=fps, codec="libx264", pixelformat="yuv420p", quality=8
        )
        self.out_path = out_path

    def _ghost_layout(self) -> tuple[int, list[int]]:
        ghost_free_adr, ghost_joints = None, []
        for joint in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if name is None or not name.startswith(_RECORD_GHOST_PREFIX):
                continue
            if self.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                ghost_free_adr = int(self.model.jnt_qposadr[joint])
            else:
                ghost_joints.append(joint)
        if ghost_free_adr is None:
            raise ValueError("ghost robot has no free joint")
        ghost_joints.sort(key=lambda joint: int(self.model.jnt_qposadr[joint]))
        return ghost_free_adr, [int(self.model.jnt_qposadr[joint]) for joint in ghost_joints]

    def _paths(self, xs: np.ndarray) -> np.ndarray:
        """Head and feet polylines of a window plan, from the tracker batch."""
        feat = self.tracker.features(xs)
        torso_pos = feat[:, self.torso_idx, :3]
        torso_quat = feat[:, self.torso_idx, 3:7]
        head = torso_pos + _rotate(torso_quat, np.asarray(_RECORD_HEAD_OFFSET, dtype=np.float64))
        return np.stack((head, feat[:, self.foot_idx[0], :3], feat[:, self.foot_idx[1], :3]))

    def __call__(self, xs: np.ndarray, commit: int = 0) -> None:
        if self.broken or commit <= 0:
            return
        try:
            self.trail_history.append(self._paths(xs))
            for step in range(1, commit + 1):
                self._render_frame(xs[step], self.knot_base + step, step)
        except Exception as error:  # noqa: BLE001 (recording must not kill solves)
            print(f"[ilqr] video recording stopped: {error!r}", file=sys.stderr, flush=True)
            self.broken = True
        self.knot_base += commit

    def _render_frame(self, state: np.ndarray, knot: int, step: int) -> None:
        # ``state`` rows are full solver states (nq + nv); qpos takes the front.
        self.data.qpos[: self.nq] = state[: self.nq]
        reference = self.clip.qpos[min(knot, self.frames - 1)]
        self.data.qpos[self.ghost_free_adr : self.ghost_free_adr + 7] = reference[:7]
        self.data.qpos[self.ghost_joint_adrs] = reference[7:]
        mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = state[:3]
        renderer = self.renderer
        renderer.update_scene(self.data, camera=self.camera)
        scene = renderer.scene

        for old_path in list(self.trail_history)[:-1]:
            _add_scene_trail(scene, old_path, start=0, rgba=_RECORD_OLD_RGBA)
        if self.trail_history:
            _add_scene_trail(
                scene,
                self.trail_history[-1],
                start=step,
                rgba=_RECORD_PLAN_RGBA,
            )
        self.writer.append_data(renderer.render())

    def close(self) -> Path | None:
        self.renderer.close()
        self.writer.close()
        return self.out_path


def _solve_task(
    args: tuple[
        str, str, IlqrSolverConfig, float, int, tuple[str, tuple[str, ...]], tuple[str, ...]
    ],
):
    """Worker entry: solve one clip and return states plus diagnostics."""
    stem, npz_path, config, dt, sub, model_sources, tracked_bodies = args
    mujoco.set_mju_user_warning(lambda _: None)
    model, _ = _build_solver_model(model_sources[0], model_sources[1], dt, tracked_bodies)
    clip = MotionClip(Path(npz_path), model, tracked_bodies)
    tracker = IlqrTracker(model, clip, config, sub=sub, tracked_bodies=tracked_bodies)
    x0 = np.concatenate([clip.qpos[0], clip.qvel[0]])
    recorder = None
    if config.record_video and config.video_output_dir:
        try:
            robot_file = str(ASSETS_ROOT_PATH / "robots" / "g1" / "g1.xml")
            recorder = _SolveRecorder(
                stem,
                Path(config.video_output_dir) / f"{stem}_opt.mp4",
                model_sources,
                robot_file,
                tracker,
                clip,
                tracked_bodies,
            )
        except Exception as error:  # noqa: BLE001 (recording must not kill solves)
            print(f"[ilqr] video disabled for {stem}: {error!r}", file=sys.stderr, flush=True)
            recorder = None
    started = time.perf_counter()
    try:
        xs, _us, _k, info = solve_motion(
            tracker,
            x0,
            clip.warm_start(model),
            config,
            watch=recorder,
            abort_pelvis_z=config.abort_pelvis_z,
            abort_patience=config.abort_patience,
            abort_warmup=config.abort_warmup,
        )
    finally:
        if recorder is not None:
            recorder.close()
    # Sliding windows commit whole steps, so the plan can overshoot the clip
    # by up to step-1 knots; keep the output frame-aligned with the source.
    xs = xs[: clip.frames]
    diagnostics = _solution_diagnostics(xs, clip)
    diagnostics["solve_seconds"] = round(time.perf_counter() - started, 1)
    if recorder is not None and not recorder.broken and recorder.out_path.exists():
        diagnostics["video"] = str(recorder.out_path)
    if info["aborted"]:
        diagnostics["aborted_reason"] = info["abort_reason"]
    return stem, xs.astype(np.float32), diagnostics


def _solution_diagnostics(xs: np.ndarray, clip: MotionClip) -> dict[str, Any]:
    nq = clip.qpos.shape[1]
    # Aborted plans are shorter than the clip; compare the overlapping prefix.
    count = min(len(xs), len(clip.qpos))
    plan, reference = xs[:count], clip.qpos[:count]
    stats = _deviation_stats(plan[:, 7:nq], reference[:, 7:nq], plan[:, :3], reference[:, :3])
    return {
        **stats,
        "root_path_len": _root_path_length(reference[:, :2]),
        "frames": int(len(xs)),
        "finite": bool(np.isfinite(xs).all()),
        "pelvis_z_min": float(xs[:, 2].min()),
        "pelvis_z_max": float(xs[:, 2].max()),
    }


def _root_path_length(reference_root_xy: np.ndarray) -> float:
    """Horizontal distance the source root travels, the drift scale."""
    if len(reference_root_xy) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(reference_root_xy, axis=0), axis=1)))


def _axis_stats(delta: np.ndarray, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_mean": float(delta.mean()),
        f"{prefix}_median": float(np.percentile(delta, 50)),
        f"{prefix}_p95": float(np.percentile(delta, 95)),
        f"{prefix}_max": float(delta.max()),
    }


def _deviation_stats(
    plan_joints: np.ndarray,
    reference_joints: np.ndarray,
    plan_root: np.ndarray,
    reference_root: np.ndarray,
) -> dict[str, Any]:
    """Mean/median/p95/max joint and root deviations of a plan vs its source.

    Root deviations are reported three ways: the legacy combined per-axis
    stats (``root_dev_*``), the per-frame horizontal magnitude
    (``root_horiz_dev_*``) and the vertical error (``root_z_dev_*``).  The
    split feeds the path-normalized acceptance gates: global x/y drift is
    invisible to SONIC's root-relative observations, while z fidelity keeps
    the plan's contact modes honest.
    """
    joint_delta = np.abs(plan_joints - reference_joints)
    root_delta = np.abs(plan_root - reference_root)
    horiz = np.linalg.norm(plan_root[:, :2] - reference_root[:, :2], axis=1)
    return {
        "joint_dev_mean": float(joint_delta.mean()),
        "joint_dev_median": float(np.percentile(joint_delta, 50)),
        "joint_dev_p95": float(np.percentile(joint_delta, 95)),
        "joint_dev_max": float(joint_delta.max()),
        "root_dev_mean": float(root_delta.mean()),
        "root_dev_median": float(np.percentile(root_delta, 50)),
        "root_dev_p95": float(np.percentile(root_delta, 95)),
        "root_dev_max": float(root_delta.max()),
        **_axis_stats(horiz, "root_horiz_dev"),
        **_axis_stats(root_delta[:, 2], "root_z_dev"),
    }


def _accept_solution(
    diagnostics: dict[str, Any],
    *,
    max_joint_deviation: float,
    max_root_deviation: float,
    max_joint_dev_p95: float,
    max_root_dev_p95: float,
    root_path_kappa: float = 0.0,
) -> str | None:
    """Return a rejection reason, or None when the plan is acceptable.

    Mean gates alone hide segment-level failures: a clip can track most frames
    well yet drift tens of centimetres for half of the motion (visually a
    failure).  The p95 gates reject exactly that.

    ``root_path_kappa > 0`` switches the root gates to the path-normalized
    form: horizontal drift is judged against ``max(floor, kappa * path)``
    where ``path`` is the horizontal distance the source root travels, while
    the vertical gate keeps the absolute floor (z fidelity bounds contact
    modes).  SONIC's observations are root-relative, so a horizontal drift
    that is small relative to the travelled path is training-neutral, yet
    the legacy absolute gates reject exactly those long traveling clips.
    """
    if not diagnostics["finite"]:
        return "non-finite states"
    if diagnostics["frames"] <= 1:
        return "degenerate plan"
    if "aborted_reason" in diagnostics:
        return f"aborted: {diagnostics['aborted_reason']}"
    if diagnostics["joint_dev_mean"] > max_joint_deviation:
        return f"mean joint deviation {diagnostics['joint_dev_mean']:.3f} rad"
    if diagnostics["joint_dev_p95"] > max_joint_dev_p95:
        return f"p95 joint deviation {diagnostics['joint_dev_p95']:.3f} rad"
    if root_path_kappa > 0.0:
        path = diagnostics.get("root_path_len", 0.0)
        allow_mean = max(max_root_deviation, root_path_kappa * path)
        allow_p95 = max(max_root_dev_p95, root_path_kappa * path)
        if diagnostics["root_horiz_dev_mean"] > allow_mean:
            return (
                f"mean horizontal root deviation {diagnostics['root_horiz_dev_mean']:.3f} m "
                f"(allow {allow_mean:.3f} = max({max_root_deviation}, {root_path_kappa}*{path:.1f}))"
            )
        if diagnostics["root_horiz_dev_p95"] > allow_p95:
            return (
                f"p95 horizontal root deviation {diagnostics['root_horiz_dev_p95']:.3f} m "
                f"(allow {allow_p95:.3f})"
            )
        if diagnostics["root_z_dev_mean"] > max_root_deviation:
            return f"mean vertical root deviation {diagnostics['root_z_dev_mean']:.3f} m"
        if diagnostics["root_z_dev_p95"] > max_root_dev_p95:
            return f"p95 vertical root deviation {diagnostics['root_z_dev_p95']:.3f} m"
        return None
    if diagnostics["root_dev_mean"] > max_root_deviation:
        return f"mean root deviation {diagnostics['root_dev_mean']:.3f} m"
    if diagnostics["root_dev_p95"] > max_root_dev_p95:
        return f"p95 root deviation {diagnostics['root_dev_p95']:.3f} m"
    return None


# ── Dataset pipeline. ────────────────────────────────────────────────────────


def _export_robot_npz(
    backend,
    model: mujoco.MjModel,
    xs: np.ndarray,
    joint_names: tuple[str, ...],
    output: Path,
) -> None:
    """Write the iLQR plan as a ``unilab_sonic_robot_v1`` clip.

    Joint arrays come from the plan's qpos/qvel; body arrays are rebuilt with
    the same ``materialize_reference_kinematics`` path that produced the
    source dataset.  ``materialize`` consumes actuator-order joint columns.
    """
    nq = model.nq
    quat = xs[:, 3:7] / np.linalg.norm(xs[:, 3:7], axis=1, keepdims=True)
    actuator_joint_ids = model.actuator_trnid[:, 0].astype(np.intp)
    qpos_ids = model.jnt_qposadr[actuator_joint_ids]
    qvel_ids = model.jnt_dofadr[actuator_joint_ids]
    joint_pos_act = xs[:, qpos_ids]
    joint_vel_act = xs[:, qvel_ids]
    kinematics = materialize_reference_kinematics(
        backend,
        root_pos=xs[:, :3],
        root_quat_wxyz=quat,
        root_lin_vel=xs[:, nq : nq + 3],
        # MuJoCo free-joint angular velocity is body-local; the source
        # pipeline stores the same convention, so pass it through unchanged.
        root_ang_vel=xs[:, nq + 3 : nq + 6],
        joint_pos=joint_pos_act,
        joint_vel=joint_vel_act,
        body_names=G1_SONIC_BODY_NAMES,
    )
    quat_w = _ensure_quaternion_continuity(kinematics.body_quat_w.astype(np.float64))
    quat_w = (quat_w / np.linalg.norm(quat_w, axis=-1, keepdims=True)).astype(np.float32)
    kinematics = type(kinematics)(
        kinematics.body_pos_w,
        quat_w,
        kinematics.body_lin_vel_w,
        kinematics.body_ang_vel_w,
    )
    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint))
        for joint in actuator_joint_ids
    ]
    to_npz_order = np.asarray([actuator_names.index(name) for name in joint_names], dtype=np.intp)
    _write_robot_npz(
        output,
        kinematics,
        joint_pos_act[:, to_npz_order].astype(np.float32),
        joint_vel_act[:, to_npz_order].astype(np.float32),
        joint_names,
        G1_SONIC_BODY_NAMES,
    )


def convert_ilqr_dataset(
    source: str | Path,
    output: str | Path,
    *,
    clips: list[str] | None = None,
    config: IlqrSolverConfig | None = None,
    dt: float = 0.005,
    sub: int = 4,
    tracked_bodies: tuple[str, ...] = DEFAULT_TRACKED_BODIES,
    max_joint_deviation: float = 0.15,
    max_root_deviation: float = 0.05,
    max_joint_dev_p95: float = 0.4,
    max_root_dev_p95: float = 0.15,
    root_path_kappa: float = 0.0,
    jobs: int = 1,
    overwrite: bool = False,
    skip_pack: bool = False,
) -> dict[str, Any]:
    if not np.isclose(sub * dt, 1.0 / _TARGET_FPS):
        raise ValueError(f"sub * dt must equal the {_TARGET_FPS} Hz motion period")
    config = config or IlqrSolverConfig()
    # The workers' prefix gate monitor judges with this run's acceptance
    # gates, so mirror them into the config that travels with each task.
    config = replace(
        config,
        max_joint_deviation=max_joint_deviation,
        max_joint_dev_p95=max_joint_dev_p95,
        max_root_deviation=max_root_deviation,
        max_root_dev_p95=max_root_dev_p95,
        root_path_kappa=root_path_kappa,
    )
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    robot_source = source / "robot_filtered"
    smpl_source = source / "smpl_filtered"
    pairs = resolve_sonic_pairs(
        str(robot_source), str(smpl_source), robot_suffix=".npz", smpl_suffix=".npz"
    )
    if clips is not None:
        wanted = set(clips)
        stems = {robot.stem for robot, _ in pairs}
        unknown = wanted - stems
        if unknown:
            raise ValueError(f"Unknown clip names: {sorted(unknown)}")
        pairs = [pair for pair in pairs if pair[0].stem in wanted]
    if not pairs:
        raise ValueError(f"No converted SONIC pairs found under {source}")
    pairs_by_stem = {robot.stem: (robot, smpl) for robot, smpl in pairs}

    robot_output = output / "robot_filtered"
    smpl_output = output / "smpl_filtered"
    robot_output.mkdir(parents=True, exist_ok=True)
    smpl_output.mkdir(parents=True, exist_ok=True)

    cfg, motion_cfg = _sonic_scene_cfg()
    model_file, fragments = _scene_sources(cfg)
    assert cfg.scene is not None
    backend = create_backend(
        "mujoco",
        cfg.scene,
        1,
        cfg.sim_dt,
        base_name=motion_cfg.anchor_body_name,
        add_body_sensors=True,
        **env_backend_kwargs(cfg),
    )
    try:
        solver_model, _ = _build_solver_model(model_file, fragments, dt, tracked_bodies)
        with np.load(pairs[0][0], allow_pickle=False) as first_robot:
            joint_names = tuple(str(name) for name in first_robot["joint_names"])
        manifest = {
            "format": ILQR_MANIFEST_FORMAT,
            "source": str(source),
            "robot_output": str(robot_output),
            "human_output": str(smpl_output),
            "fps": _TARGET_FPS,
            "dt": dt,
            "sub": sub,
            "tracked_bodies": list(tracked_bodies),
            "solver": {
                "horizon": config.horizon,
                "step": config.step,
                "iterations": config.iterations,
                "tolerance": config.tolerance,
                "abort_pelvis_z": config.abort_pelvis_z,
                "abort_patience": config.abort_patience,
                "abort_warmup": config.abort_warmup,
                "weights": {
                    "w_pos": config.weights.w_pos,
                    "w_rot": config.weights.w_rot,
                    "w_vel": config.weights.w_vel,
                    "w_ang": config.weights.w_ang,
                    "w_ctrl": config.weights.w_ctrl,
                    "w_root": list(config.weights.w_root),
                    "huber": config.weights.huber,
                },
            },
            "acceptance": {
                "max_joint_deviation": max_joint_deviation,
                "max_joint_dev_p95": max_joint_dev_p95,
                "max_root_deviation": max_root_deviation,
                "max_root_dev_p95": max_root_dev_p95,
                "root_path_kappa": root_path_kappa,
            },
            "clips": {},
        }
        entries: dict[str, Any] = cast(dict[str, Any], manifest["clips"])
        manifest_path = output / "ilqr_manifest.json"

        def _write_manifest() -> None:
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

        tasks = []
        for robot_path, _smpl_path in pairs:
            stem = robot_path.stem
            robot_out = robot_output / f"{stem}.npz"
            smpl_out = smpl_output / f"{stem}.npz"
            if robot_out.exists() and smpl_out.exists() and not overwrite:
                entries[stem] = {"status": "skipped"}
                continue
            if (robot_out.exists() or smpl_out.exists()) and not overwrite:
                raise FileExistsError(f"Incomplete iLQR output pair for {stem!r}")
            tasks.append(
                (stem, str(robot_path), config, dt, sub, (model_file, fragments), tracked_bodies)
            )
        _write_manifest()

        def _finalize(stem: str, xs: np.ndarray | None, diagnostics: dict[str, Any]) -> None:
            """Accept/reject a solved clip and land it on disk immediately."""
            robot_path, smpl_path = pairs_by_stem[stem]
            robot_out = robot_output / f"{stem}.npz"
            smpl_out = smpl_output / f"{stem}.npz"
            if xs is None:
                reason = diagnostics.get("error", "solver returned no plan")
            else:
                reason = _accept_solution(
                    diagnostics,
                    max_joint_deviation=max_joint_deviation,
                    max_root_deviation=max_root_deviation,
                    max_joint_dev_p95=max_joint_dev_p95,
                    max_root_dev_p95=max_root_dev_p95,
                    root_path_kappa=root_path_kappa,
                )
            if reason is None:
                assert xs is not None
                _export_robot_npz(backend, solver_model, xs, joint_names, robot_out)
                shutil.copy2(smpl_path, smpl_out)
                entry = {"status": "solved", **diagnostics}
            else:
                # Solved-only output: a failed solve lands nothing on disk, so
                # reruns retry it and the packed dataset keeps only
                # dynamics-resolved clips.
                details = {k: v for k, v in diagnostics.items() if k != "status"}
                entry = {"status": "fallback", "reason": reason, **details}
            entries[stem] = entry
            _write_manifest()
            print(f"[ilqr] {entry['status']}: {stem}", file=sys.stderr, flush=True)

        if jobs <= 1:
            for task in tasks:
                stem = task[0]
                try:
                    _, xs, diagnostics = _solve_task(task)
                except Exception as error:  # noqa: BLE001 (recorded per clip)
                    print(f"[ilqr] ERROR {stem}: {error!r}", file=sys.stderr, flush=True)
                    _finalize(stem, None, {"status": "error", "error": repr(error)})
                    continue
                print(f"[ilqr] solved {stem}: {diagnostics}", file=sys.stderr, flush=True)
                _finalize(stem, xs, diagnostics)
        else:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(_solve_task, task): task[0] for task in tasks}
                for future in as_completed(futures):
                    stem = futures[future]
                    try:
                        _, xs, diagnostics = future.result()
                    except Exception as error:  # noqa: BLE001 (recorded per clip)
                        print(f"[ilqr] ERROR {stem}: {error!r}", file=sys.stderr, flush=True)
                        _finalize(stem, None, {"status": "error", "error": repr(error)})
                        continue
                    print(f"[ilqr] solved {stem}: {diagnostics}", file=sys.stderr, flush=True)
                    _finalize(stem, xs, diagnostics)
    finally:
        backend.cleanup_scene_assets()

    packed = output / "packed"
    summary: dict[str, Any] = {"manifest": manifest, "packed": None}
    if skip_pack:
        return summary
    if packed.exists():
        if not overwrite:
            summary["packed"] = {"output": str(packed), "skipped": True}
            return summary
        shutil.rmtree(packed)
    summary["packed"] = pack_sonic_dataset(robot_output, smpl_output, packed)
    return summary


def reaudit_dataset(
    source: str | Path,
    output: str | Path,
    *,
    max_joint_deviation: float = 0.15,
    max_root_deviation: float = 0.05,
    max_joint_dev_p95: float = 0.4,
    max_root_dev_p95: float = 0.15,
    root_path_kappa: float = 0.0,
) -> dict[str, Any]:
    """Re-apply acceptance to an already-converted output without re-solving.

    Deviation statistics are recomputable from the landed NPZ pairs alone, so
    tightening the gates (or auditing a run produced with looser ones) is a
    pure file pass: failing clips, legacy fallback copies, and solved entries
    whose pair went missing are dropped so the output keeps only solved
    clips, and the manifest is rewritten.  Packed stores are NOT touched
    (delete and repack separately if clips flipped).
    """
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    robot_source = source / "robot_filtered"
    robot_output = output / "robot_filtered"
    pairs = resolve_sonic_pairs(
        str(robot_source), str(source / "smpl_filtered"), robot_suffix=".npz", smpl_suffix=".npz"
    )
    by_stem = {robot.stem: robot for robot, _ in pairs}
    manifest_path = output / "ilqr_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("clips", {})
    entries = manifest["clips"]
    manifest["acceptance"] = {
        "max_joint_deviation": max_joint_deviation,
        "max_joint_dev_p95": max_joint_dev_p95,
        "max_root_deviation": max_root_deviation,
        "max_root_dev_p95": max_root_dev_p95,
        "root_path_kappa": root_path_kappa,
        "reaudited": True,
    }
    flipped: list[str] = []

    def _drop_pair(stem: str, robot_out: Path) -> None:
        robot_out.unlink(missing_ok=True)
        (output / "smpl_filtered" / f"{stem}.npz").unlink(missing_ok=True)

    for robot_out in sorted(robot_output.glob("*.npz")):
        stem = robot_out.stem
        source_robot = by_stem.get(stem)
        if source_robot is None:
            continue
        with (
            np.load(source_robot, allow_pickle=False) as src,
            np.load(robot_out, allow_pickle=False) as plan,
        ):
            stats = _deviation_stats(
                plan["joint_pos"],
                src["joint_pos"],
                plan["body_pos_w"][:, 0],
                src["body_pos_w"][:, 0],
            )
            frames = int(np.asarray(plan["num_frames"]).reshape(-1)[0])
            finite = bool(np.isfinite(plan["joint_pos"]).all())
            path_len = _root_path_length(src["body_pos_w"][:, 0, :2])
        diagnostics = {
            **stats,
            "root_path_len": path_len,
            "frames": frames,
            "finite": finite,
        }
        entry = entries.get(stem, {})
        if entry.get("status") == "skipped":
            entry = {}
        # Reaudit only ever downgrades.  Fallback entries hold no files under
        # the solved-only policy; a file present here is a legacy source copy
        # from an older run, so drop it and keep the failed verdict (its
        # trivially zero file deviation says nothing about the failed solve).
        if entry.get("status") == "fallback":
            _drop_pair(stem, robot_out)
            entry.update({k: v for k, v in diagnostics.items() if k != "frames"})
            entries[stem] = entry
            continue
        # A genuinely solved plan always deviates from its source, so a
        # byte-identical output is a legacy fallback copy recorded as solved
        # and must be dropped.
        if robot_out.read_bytes() == source_robot.read_bytes():
            if entry.get("status") == "solved":
                flipped.append(stem)
            _drop_pair(stem, robot_out)
            entry = {
                "status": "fallback",
                "reason": "reaudit: output identical to source (dropped)",
                **{k: v for k, v in entry.items() if k not in ("status", "reason")},
                **diagnostics,
            }
            entries[stem] = entry
            continue
        reason = _accept_solution(
            diagnostics,
            max_joint_deviation=max_joint_deviation,
            max_root_deviation=max_root_deviation,
            max_joint_dev_p95=max_joint_dev_p95,
            max_root_dev_p95=max_root_dev_p95,
            root_path_kappa=root_path_kappa,
        )
        if reason is None:
            entry.update({"status": "solved", **diagnostics})
        else:
            if entry.get("status") == "solved":
                flipped.append(stem)
            _drop_pair(stem, robot_out)
            details = {k: v for k, v in entry.items() if k not in ("status", "reason")}
            entry = {"status": "fallback", "reason": f"reaudit: {reason}", **details, **diagnostics}
        entries[stem] = entry

    # Solved-only outputs hold a pair for every solved clip, so a solved
    # entry without files on disk is stale or corrupt and must downgrade.
    for stem, entry in list(entries.items()):
        if entry.get("status") == "solved" and not (robot_output / f"{stem}.npz").is_file():
            flipped.append(stem)
            entries[stem] = {
                "status": "fallback",
                "reason": "reaudit: solved entry has no output pair",
                **{k: v for k, v in entry.items() if k not in ("status", "reason")},
            }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"manifest": manifest, "flipped_to_fallback": flipped}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/lafan1"))
    parser.add_argument("--output", type=Path, default=Path("data/lafan1_ilqr"))
    parser.add_argument(
        "--clips",
        help="comma-separated clip stems (default: every converted pair)",
    )
    parser.add_argument(
        "--clips-file",
        type=Path,
        help="clip stems one per line (comments with '#'); for sharding large "
        "datasets where --clips would exceed the command-line length limit",
    )
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--sub", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--abort-pelvis-z",
        type=float,
        default=0.3,
        help="abort a clip whose committed plan stays below this pelvis height "
        "(negative disables the early abort)",
    )
    parser.add_argument(
        "--abort-patience",
        type=int,
        default=25,
        help="consecutive fallen/costly committed windows before aborting",
    )
    parser.add_argument(
        "--abort-ref-pelvis-z",
        type=float,
        default=0.45,
        help="a low plan only counts as fallen while the reference stands "
        "above this height (floor-motion clips track low by design); "
        "negative disables the reference gating",
    )
    parser.add_argument(
        "--abort-patience-fallen",
        type=int,
        default=8,
        help="patience applied while the committed window is fallen",
    )
    parser.add_argument(
        "--abort-warmup",
        type=int,
        default=50,
        help="committed windows sampled before the cost-abort baseline activates",
    )
    parser.add_argument(
        "--cost-abort-multiple",
        type=float,
        default=50.0,
        help="abort when committed window cost stays above this multiple of "
        "the clip's healthy baseline (non-positive disables)",
    )
    parser.add_argument(
        "--early-gate-reject",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="abort once the committed prefix makes the acceptance gates "
        "mathematically impossible to pass (provably safe: only clips that "
        "would be rejected at the end are cut early)",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="render the solve-time view (plan-state robot, reference ghost, "
        "window plan trails) to <output>/videos/<stem>_opt.mp4, one frame per "
        "committed knot; needs a GL backend (MUJOCO_GL=egl/osmesa/glfw)",
    )
    parser.add_argument(
        "--tracked-bodies",
        help="comma-separated tracked bodies (default: the dbm 11-body set; "
        "'sonic14' selects all SONIC bodies)",
    )
    parser.add_argument("--max-joint-deviation", type=float, default=0.15)
    parser.add_argument(
        "--max-joint-dev-p95",
        type=float,
        default=0.4,
        help="reject plans whose 95th-percentile joint deviation exceeds this",
    )
    parser.add_argument("--max-root-deviation", type=float, default=0.05)
    parser.add_argument(
        "--max-root-dev-p95",
        type=float,
        default=0.15,
        help="reject plans whose 95th-percentile root deviation exceeds this",
    )
    parser.add_argument(
        "--root-path-kappa",
        type=float,
        default=0.0,
        help="path-normalized root gates: horizontal drift is judged against "
        "max(floor, kappa * horizontal path length) while the vertical gate "
        "keeps the absolute floors; 0 keeps the legacy absolute gates",
    )
    parser.add_argument("--w-pos", type=float, default=100.0)
    parser.add_argument("--w-rot", type=float, default=100.0)
    parser.add_argument("--w-vel", type=float, default=5.0)
    parser.add_argument("--w-ang", type=float, default=1.0)
    parser.add_argument("--w-ctrl", type=float, default=1.0)
    parser.add_argument(
        "--w-root",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1000.0),
        metavar=("X", "Y", "Z"),
        help="pelvis position weights: the dbm flip default is loose in x and y, "
        "tight in z; raise x/y for translating clips (repeated jumps drift)",
    )
    parser.add_argument(
        "--reaudit",
        action="store_true",
        help="re-apply the acceptance gates to an existing output directory "
        "(no solving; failing clips and legacy fallback copies are removed so "
        "the output keeps only solved clips)",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-pack", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    tracked_bodies = DEFAULT_TRACKED_BODIES
    if args.tracked_bodies:
        tracked_bodies = (
            G1_SONIC_BODY_NAMES
            if args.tracked_bodies == "sonic14"
            else tuple(name.strip() for name in args.tracked_bodies.split(",") if name.strip())
        )
        unknown = [name for name in tracked_bodies if name not in G1_SONIC_BODY_NAMES]
        if unknown:
            raise SystemExit(f"Unknown tracked bodies: {unknown}")
    clips = None
    if args.clips and args.clips_file:
        raise SystemExit("use either --clips or --clips-file, not both")
    if args.clips:
        clips = [name.strip() for name in args.clips.split(",") if name.strip()]
    elif args.clips_file:
        lines = args.clips_file.read_text(encoding="utf-8").splitlines()
        clips = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    if args.reaudit:
        summary = reaudit_dataset(
            args.source,
            args.output,
            max_joint_deviation=args.max_joint_deviation,
            max_root_deviation=args.max_root_deviation,
            max_joint_dev_p95=args.max_joint_dev_p95,
            max_root_dev_p95=args.max_root_dev_p95,
            root_path_kappa=args.root_path_kappa,
        )
        manifest = summary["manifest"]
        statuses: dict[str, int] = {}
        for entry in manifest["clips"].values():
            statuses[entry["status"]] = statuses.get(entry["status"], 0) + 1
        print(
            json.dumps(
                {"clips": statuses, "flipped_to_fallback": summary["flipped_to_fallback"]},
                indent=2,
            )
        )
        return

    summary = convert_ilqr_dataset(
        args.source,
        args.output,
        clips=clips,
        config=IlqrSolverConfig(
            horizon=args.horizon,
            step=args.step,
            iterations=args.iterations,
            weights=IlqrCostWeights(
                w_pos=args.w_pos,
                w_rot=args.w_rot,
                w_vel=args.w_vel,
                w_ang=args.w_ang,
                w_ctrl=args.w_ctrl,
                w_root=tuple(args.w_root),
            ),
            abort_pelvis_z=(
                None
                if args.abort_pelvis_z is None or args.abort_pelvis_z < 0
                else args.abort_pelvis_z
            ),
            abort_patience=args.abort_patience,
            abort_ref_pelvis_z=(
                None
                if args.abort_ref_pelvis_z is None or args.abort_ref_pelvis_z < 0
                else args.abort_ref_pelvis_z
            ),
            abort_patience_fallen=args.abort_patience_fallen,
            abort_warmup=args.abort_warmup,
            cost_abort_multiple=args.cost_abort_multiple,
            early_gate_reject=args.early_gate_reject,
            record_video=args.record_video,
            video_output_dir=str(args.output / "videos") if args.record_video else None,
        ),
        dt=args.dt,
        sub=args.sub,
        tracked_bodies=tracked_bodies,
        max_joint_deviation=args.max_joint_deviation,
        max_root_deviation=args.max_root_deviation,
        max_joint_dev_p95=args.max_joint_dev_p95,
        max_root_dev_p95=args.max_root_dev_p95,
        root_path_kappa=args.root_path_kappa,
        jobs=args.jobs,
        overwrite=args.overwrite,
        skip_pack=args.skip_pack,
    )
    manifest = summary["manifest"]
    statuses = {}
    for entry in manifest["clips"].values():
        statuses[entry["status"]] = statuses.get(entry["status"], 0) + 1
    print(json.dumps({"clips": statuses, "packed": bool(summary["packed"])}, indent=2))


if __name__ == "__main__":
    main()
