"""Unit tests for the LAFAN1 iLQR dynamics-resolution tool."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
import pytest

from unilab.tasks.motion_tracking.g1.sonic_data import (
    _SONIC_ROBOT_NPZ_FORMAT,
    _SONIC_SMPL_NPZ_FORMAT,
    _TARGET_FPS,
    G1_POLICY_JOINT_NAMES,
)
from unilab.tasks.motion_tracking.g1.sonic_manager import G1_SONIC_BODY_NAMES
from unilab.tools import mocap2ilqr

_FRAMES = 25


@pytest.fixture(scope="module")
def scene() -> tuple[mujoco.MjModel, tuple[str, ...]]:
    cfg, _motion_cfg = mocap2ilqr._sonic_scene_cfg()
    model_file, fragments = mocap2ilqr._scene_sources(cfg)
    model, _ = mocap2ilqr._build_solver_model(
        model_file, fragments, dt=0.005, tracked_bodies=mocap2ilqr.DEFAULT_TRACKED_BODIES
    )
    return model, mocap2ilqr.DEFAULT_TRACKED_BODIES


def _standing_arrays(model: mujoco.MjModel, frames: int) -> dict[str, np.ndarray]:
    """The stand keyframe held for ``frames`` steps, with FK body arrays."""
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    qpos = np.tile(model.key_qpos[key], (frames, 1))
    data = mujoco.MjData(model)
    body_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in G1_SONIC_BODY_NAMES],
        dtype=np.intp,
    )
    pos = np.empty((frames, len(body_ids), 3), dtype=np.float32)
    quat = np.empty((frames, len(body_ids), 4), dtype=np.float32)
    for frame in range(frames):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        pos[frame] = data.xpos[body_ids]
        quat[frame] = data.xquat[body_ids]
    hinge = np.arange(1, model.njnt, dtype=np.intp)
    policy_from_model = np.asarray(
        [
            [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in hinge].index(name)
            for name in G1_POLICY_JOINT_NAMES
        ],
        dtype=np.intp,
    )
    qpos_ids = model.jnt_qposadr[hinge]
    joints = qpos[:, qpos_ids][:, policy_from_model]
    return {
        "joint_pos": joints.astype(np.float32),
        "joint_vel": np.zeros_like(joints, dtype=np.float32),
        "body_pos_w": pos,
        "body_quat_w": quat,
        "body_lin_vel_w": np.zeros_like(pos),
        "body_ang_vel_w": np.zeros_like(pos),
        "root_pos": qpos[:, :3],
        "root_quat": qpos[:, 3:7],
    }


def _write_source_pair(root: Path, stem: str, model: mujoco.MjModel, frames: int) -> None:
    arrays = _standing_arrays(model, frames)
    robot_dir = root / "robot_filtered"
    smpl_dir = root / "smpl_filtered"
    robot_dir.mkdir(parents=True, exist_ok=True)
    smpl_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        robot_dir / f"{stem}.npz",
        source_format=np.asarray(_SONIC_ROBOT_NPZ_FORMAT),
        fps=np.int32(_TARGET_FPS),
        num_frames=np.int32(frames),
        joint_names=np.asarray(G1_POLICY_JOINT_NAMES),
        body_names=np.asarray(G1_SONIC_BODY_NAMES),
        joint_pos=arrays["joint_pos"],
        joint_vel=arrays["joint_vel"],
        body_pos_w=arrays["body_pos_w"],
        body_quat_w=arrays["body_quat_w"],
        body_lin_vel_w=arrays["body_lin_vel_w"],
        body_ang_vel_w=arrays["body_ang_vel_w"],
    )
    np.savez(
        smpl_dir / f"{stem}.npz",
        source_format=np.asarray(_SONIC_SMPL_NPZ_FORMAT),
        fps=np.int32(_TARGET_FPS),
        num_frames=np.int32(frames),
        smpl_joints=np.zeros((frames, 24, 3), dtype=np.float32),
        smpl_root_quat=np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)).astype(np.float32),
    )


def test_solver_model_carries_frame_sensors_and_bounds(scene) -> None:
    model, tracked = scene
    assert model.opt.timestep == pytest.approx(0.005)
    for body in tracked:
        for name, width in (("pos", 3), ("quat", 4), ("linvel", 3), ("angvel", 3)):
            sensor = model.sensor(f"{body}_{name}")
            assert int(sensor.dim[0]) == width
    joint = model.actuator_trnid[:, 0]
    assert np.all(model.actuator_ctrllimited == 1)
    assert np.allclose(model.actuator_ctrlrange, model.jnt_range[joint])


def test_motion_clip_maps_model_joints_by_name(scene, tmp_path) -> None:
    model, tracked = scene
    stem = "map_check"
    _write_source_pair(tmp_path, stem, model, _FRAMES)
    clip = mocap2ilqr.MotionClip(tmp_path / "robot_filtered" / f"{stem}.npz", model, tracked)
    assert clip.qpos.shape == (_FRAMES, model.nq)
    assert clip.qvel.shape == (_FRAMES, model.nv)
    assert clip.features.shape == (_FRAMES, len(tracked), 13)
    # The pelvis is lifted slightly out of the floor for the solve.
    assert clip.qpos[0, 2] == pytest.approx(
        float(np.load(tmp_path / "robot_filtered" / f"{stem}.npz")["body_pos_w"][0, 0, 2])
        + mocap2ilqr.LIFT,
        abs=1e-9,
    )


def test_solve_motion_tracks_standing_pose(scene) -> None:
    model, tracked = scene
    clip = mocap2ilqr.MotionClip(_stand_clip(model, _FRAMES), model, tracked)
    tracker = mocap2ilqr.IlqrTracker(
        model,
        clip,
        mocap2ilqr.IlqrSolverConfig(horizon=6, step=3, iterations=2),
        sub=4,
        tracked_bodies=tracked,
    )
    x0 = np.concatenate([clip.qpos[0], clip.qvel[0]])
    xs, us, _k, info = mocap2ilqr.solve_motion(
        tracker,
        x0,
        clip.warm_start(model),
        mocap2ilqr.IlqrSolverConfig(horizon=6, step=3, iterations=2),
    )
    assert np.isfinite(xs).all() and np.isfinite(us).all()
    assert info["aborted"] is False
    # Sliding windows commit whole steps; the plan covers at least the clip.
    assert len(xs) >= _FRAMES
    xs = xs[:_FRAMES]
    assert np.abs(xs[:, :3] - clip.qpos[:, :3]).mean() < 0.1
    diagnostics = mocap2ilqr._solution_diagnostics(xs, clip)
    assert _accept(diagnostics) is None


def test_window_fallen_requires_standing_reference() -> None:
    fallen = mocap2ilqr._window_fallen
    # A real fall: the plan drops while the reference stands.
    assert fallen(0.2, 0.75, 0.3, 0.45) is True
    # Intentional floor motion: the reference lies low too (fall-and-get-up).
    assert fallen(0.2, 0.15, 0.3, 0.45) is False
    # A low crouching reference is below the standing threshold.
    assert fallen(0.2, 0.40, 0.3, 0.45) is False
    # The plan itself stands.
    assert fallen(0.75, 0.75, 0.3, 0.45) is False
    # Legacy mode: without reference gating any low plan counts as fallen.
    assert fallen(0.2, 0.15, 0.3, None) is True
    assert fallen(0.2, 0.15, None, 0.45) is False


def test_scene_trail_uses_knot_axis() -> None:
    """A (path, knot, 3) window plan renders all three planned polylines."""
    scene_mock = mujoco.MjvScene(mujoco.MjModel.from_xml_string("<mujoco/>"), 100)
    knots = np.linspace(0.0, 1.0, 7)
    points = np.zeros((3, 7, 3))
    points[:, :, 0] = knots
    mocap2ilqr._add_scene_trail(scene_mock, points, start=0, rgba=(1.0, 0.0, 0.0, 1.0))
    assert scene_mock.ngeom == 6


def _gate_monitor(clip, **kwargs) -> mocap2ilqr._PrefixGateMonitor:
    defaults = dict(
        max_joint_deviation=0.15,
        max_joint_dev_p95=0.4,
        max_root_deviation=0.05,
        max_root_dev_p95=0.15,
        root_path_kappa=0.0,
    )
    return mocap2ilqr._PrefixGateMonitor(clip, **(defaults | kwargs))


def test_prefix_gate_monitor_three_way(scene) -> None:
    model, tracked = scene
    clip = mocap2ilqr.MotionClip(_stand_clip(model, _FRAMES), model, tracked)

    def block(joint_delta: float, root_delta: float, knots: int) -> np.ndarray:
        xs = clip.qpos[:knots].copy()
        xs[:, 7:] += joint_delta
        xs[:, :3] += root_delta
        return xs

    # A well-tracking prefix is never doomed.
    assert _gate_monitor(clip).update(block(0.01, 0.005, 10), 0) is None

    # Mean-doom: the joint sum alone already exceeds the mean gate even if
    # every remaining frame were perfect (0.8 rad over 5 of 25 knots).
    monitor = _gate_monitor(clip)
    reason = monitor.update(block(0.8, 0.0, 5), 0)
    assert reason is not None and "mean joint deviation" in reason

    # p95-doom: enough entries over the p95 gate to blow the 5% budget while
    # the mean stays inside its gate.
    monitor = _gate_monitor(clip)
    reason = monitor.update(block(0.5, 0.0, 3), 0)
    assert reason is not None and "p95" in reason

    # Root-doom with the kappa-split gates: 3 of 25 frames 0.3 m off
    # horizontally is already 12% of the clip against a 5% budget.
    monitor = _gate_monitor(clip, root_path_kappa=0.0)
    reason = monitor.update(block(0.01, 0.3, 3), 0)
    assert reason is not None and "root" in reason

    # Overshoot knots past the clip length are not accumulated.
    monitor = _gate_monitor(clip)
    assert monitor.update(block(0.01, 0.005, 10), _FRAMES - 3) is None


def _stand_clip(model: mujoco.MjModel, frames: int) -> Path:
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="mocap2ilqr_stand_"))
    _write_source_pair(tmp, "stand", model, frames)
    return tmp / "robot_filtered" / "stand.npz"


def test_solution_diagnostics_handles_aborted_prefix(scene) -> None:
    model, tracked = scene
    clip = mocap2ilqr.MotionClip(_stand_clip(model, _FRAMES), model, tracked)
    partial = clip.qpos[:10].copy()  # an aborted plan shorter than the clip
    diagnostics = mocap2ilqr._solution_diagnostics(partial, clip)
    assert diagnostics["frames"] == 10
    assert diagnostics["joint_dev_mean"] == pytest.approx(0.0)
    assert "aborted" in _accept(diagnostics | {"aborted_reason": "pelvis z below 0.3 m"})


def _accept(diagnostics: dict, *, root_path_kappa: float = 0.0) -> str | None:
    return mocap2ilqr._accept_solution(
        diagnostics,
        max_joint_deviation=0.5,
        max_root_deviation=0.5,
        max_joint_dev_p95=0.4,
        max_root_dev_p95=0.15,
        root_path_kappa=root_path_kappa,
    )


def _plan_stats(
    *, horiz_p95: float, horiz_mean: float | None = None, z_p95: float = 0.0, path: float
) -> dict:
    horiz_mean = horiz_mean if horiz_mean is not None else horiz_p95 / 3
    return {
        "frames": 1000,
        "joint_dev_mean": 0.1,
        "joint_dev_p95": 0.2,
        "root_dev_mean": max(horiz_mean, z_p95),
        "root_dev_p95": max(horiz_p95, z_p95),
        "root_horiz_dev_mean": horiz_mean,
        "root_horiz_dev_p95": horiz_p95,
        "root_z_dev_mean": z_p95 / 3,
        "root_z_dev_p95": z_p95,
        "root_path_len": path,
        "finite": True,
    }


def test_solve_task_end_to_end_signature(scene, tmp_path) -> None:
    """Exercise the real worker entry (catches solve_motion signature drift)."""
    model, tracked = scene
    stem = "dance1_subject1"
    _write_source_pair(tmp_path, stem, model, _FRAMES)
    cfg, _motion_cfg = mocap2ilqr._sonic_scene_cfg()
    model_file, fragments = mocap2ilqr._scene_sources(cfg)
    config = mocap2ilqr.IlqrSolverConfig(horizon=6, step=3, iterations=2)
    task = (
        stem,
        str(tmp_path / "robot_filtered" / f"{stem}.npz"),
        config,
        0.005,
        4,
        (model_file, fragments),
        tracked,
    )
    result_stem, xs, diagnostics = mocap2ilqr._solve_task(task)
    assert result_stem == stem
    assert diagnostics["finite"] and diagnostics["frames"] == _FRAMES


def test_accept_solution_rejects_divergence() -> None:
    diverged = {
        "frames": 10,
        "joint_dev_mean": 0.1,
        "joint_dev_p95": 0.2,
        "root_dev_mean": 0.01,
        "root_dev_p95": 0.02,
        "finite": False,
    }
    assert "non-finite" in _accept(diverged)
    aborted = diverged | {
        "finite": True,
        "aborted_reason": "pelvis z below 0.3 m for 25 committed windows",
    }
    assert "aborted" in _accept(aborted)
    drifted = diverged | {"finite": True, "joint_dev_mean": 0.9}
    assert "joint deviation" in _accept(drifted)
    assert "root deviation" in _accept(drifted | {"joint_dev_mean": 0.1, "root_dev_mean": 0.9})


def test_accept_solution_rejects_low_mean_high_tail_drift() -> None:
    # A plan can track most frames well yet drift far for a large minority:
    # mean gates alone pass it, the p95 gates must reject it.
    sneaky = {
        "frames": 1000,
        "joint_dev_mean": 0.12,
        "joint_dev_p95": 1.03,
        "root_dev_mean": 0.04,
        "root_dev_p95": 0.61,
        "finite": True,
    }
    assert "p95 joint deviation" in _accept(sneaky)
    root_sneaky = sneaky | {"joint_dev_p95": 0.3, "root_dev_p95": 0.61}
    assert "p95 root deviation" in _accept(root_sneaky)


def test_accept_solution_path_normalized_root_gates() -> None:
    # A long traveling clip whose horizontal drift is small relative to the
    # travelled path: rejected by the legacy absolute gates, accepted once
    # the horizontal allowance scales with the path length.
    traveling = _plan_stats(horiz_p95=0.45, horiz_mean=0.11, path=10.0)
    assert "p95 root deviation" in _accept(traveling)
    assert _accept(traveling, root_path_kappa=0.10) is None  # allow max(0.15, 1.0)

    # kappa=0 must keep the legacy behaviour byte-for-byte.
    assert "p95 root deviation" in _accept(traveling, root_path_kappa=0.0)

    # An in-place clip (no path) gains nothing from the normalization: the
    # floors apply, and the jumped-elsewhere plans stay rejected.
    in_place = _plan_stats(horiz_p95=0.9, path=0.8)
    assert "p95 horizontal root deviation" in _accept(in_place, root_path_kappa=0.25)

    # Drift beyond even the scaled allowance is still rejected.
    greedy = _plan_stats(horiz_p95=2.5, path=10.0)
    assert "p95 horizontal root deviation" in _accept(greedy, root_path_kappa=0.10)

    # The vertical gate keeps the absolute floor under normalization: height
    # fidelity bounds contact modes and must not ride the path allowance.
    tall = _plan_stats(horiz_p95=0.1, z_p95=0.4, path=20.0)
    assert "p95 vertical root deviation" in _accept(tall, root_path_kappa=0.10)


def test_pipeline_falls_back_and_resumes(scene, tmp_path, monkeypatch) -> None:
    model, _tracked = scene
    source = tmp_path / "source"
    stem = "dance1_subject1"
    _write_source_pair(source, stem, model, _FRAMES)

    def _divergent(task):
        stem_, _, _, _, _, _, _ = task
        nan = np.full((3, model.nq + model.nv), np.nan, dtype=np.float32)
        return (
            stem_,
            nan,
            {
                "frames": 3,
                "joint_dev_mean": float("nan"),
                "root_dev_mean": float("nan"),
                "finite": False,
            },
        )

    monkeypatch.setattr(mocap2ilqr, "_solve_task", _divergent)
    output = tmp_path / "out"
    summary = mocap2ilqr.convert_ilqr_dataset(source, output, clips=[stem], skip_pack=True)
    entry = summary["manifest"]["clips"][stem]
    assert entry["status"] == "fallback"
    # Solved-only output: a failed solve lands nothing, so a rerun retries it.
    assert not (output / "robot_filtered" / f"{stem}.npz").exists()
    assert not (output / "smpl_filtered" / f"{stem}.npz").exists()
    manifest = json.loads((output / "ilqr_manifest.json").read_text())
    assert manifest["format"] == mocap2ilqr.ILQR_MANIFEST_FORMAT

    def _tracked(task):
        stem_, _, _, _, _, _, _ = task
        xs = np.zeros((3, model.nq + model.nv))
        xs[:, 3] = 1.0  # unit wxyz quaternion keeps the export finite
        return (
            stem_,
            xs,
            {
                "frames": 3,
                "joint_dev_mean": 0.0,
                "joint_dev_p95": 0.0,
                "root_dev_mean": 0.0,
                "root_dev_p95": 0.0,
                "finite": True,
            },
        )

    monkeypatch.setattr(mocap2ilqr, "_solve_task", _tracked)
    summary = mocap2ilqr.convert_ilqr_dataset(source, output, clips=[stem], skip_pack=True)
    assert summary["manifest"]["clips"][stem]["status"] == "solved"
    assert (output / "robot_filtered" / f"{stem}.npz").is_file()

    # A rerun without --overwrite skips existing solved pairs.
    monkeypatch.setattr(
        mocap2ilqr,
        "_solve_task",
        lambda task: pytest.fail("existing pairs must not be re-solved"),
    )
    summary = mocap2ilqr.convert_ilqr_dataset(source, output, clips=[stem], skip_pack=True)
    assert summary["manifest"]["clips"][stem]["status"] == "skipped"


def test_reaudit_only_downgrades_and_is_idempotent(scene, tmp_path) -> None:
    model, _tracked = scene
    source = tmp_path / "source"
    stem = "dance1_subject1"
    _write_source_pair(source, stem, model, _FRAMES)
    output = tmp_path / "out"
    (output / "robot_filtered").mkdir(parents=True)
    (output / "smpl_filtered").mkdir(parents=True)
    shutil.copy2(source / "smpl_filtered" / f"{stem}.npz", output / "smpl_filtered" / f"{stem}.npz")
    # A "solved" clip whose file is actually far off the source.
    with np.load(source / "robot_filtered" / f"{stem}.npz", allow_pickle=False) as src:
        drifted = {name: src[name] for name in src.files}
    drifted["joint_pos"] = drifted["joint_pos"] + 0.3
    np.savez(output / "robot_filtered" / f"{stem}.npz", **drifted)
    manifest = {
        "format": mocap2ilqr.ILQR_MANIFEST_FORMAT,
        "source": str(source),
        "clips": {stem: {"status": "solved"}},
    }
    (output / "ilqr_manifest.json").write_text(json.dumps(manifest))

    first = mocap2ilqr.reaudit_dataset(source, output)
    assert stem in first["flipped_to_fallback"]
    assert first["manifest"]["clips"][stem]["status"] == "fallback"
    # Solved-only output: the failing pair is removed, not source-copied.
    assert not (output / "robot_filtered" / f"{stem}.npz").exists()
    assert not (output / "smpl_filtered" / f"{stem}.npz").exists()

    # Re-running is a no-op: no files left to audit and the fallback verdict
    # must never be upgraded back to solved.
    second = mocap2ilqr.reaudit_dataset(source, output)
    assert second["flipped_to_fallback"] == []
    assert second["manifest"]["clips"][stem]["status"] == "fallback"

    # A manifest corrupted the other way (solved entry with no pair on disk)
    # is restored by the disk reconciliation pass.
    manifest["clips"][stem]["status"] = "solved"
    (output / "ilqr_manifest.json").write_text(json.dumps(manifest))
    third = mocap2ilqr.reaudit_dataset(source, output)
    assert stem in third["flipped_to_fallback"]
    assert third["manifest"]["clips"][stem]["status"] == "fallback"


def test_reaudit_removes_legacy_fallback_copies(scene, tmp_path) -> None:
    model, _tracked = scene
    source = tmp_path / "source"
    stem = "dance1_subject1"
    _write_source_pair(source, stem, model, _FRAMES)
    output = tmp_path / "out"
    (output / "robot_filtered").mkdir(parents=True)
    (output / "smpl_filtered").mkdir(parents=True)
    # An older run copied the source over both streams of a failed clip and
    # the manifest recorded it as solved; reaudit must drop the pair.
    shutil.copy2(
        source / "robot_filtered" / f"{stem}.npz", output / "robot_filtered" / f"{stem}.npz"
    )
    shutil.copy2(source / "smpl_filtered" / f"{stem}.npz", output / "smpl_filtered" / f"{stem}.npz")
    manifest = {
        "format": mocap2ilqr.ILQR_MANIFEST_FORMAT,
        "source": str(source),
        "clips": {stem: {"status": "solved"}},
    }
    (output / "ilqr_manifest.json").write_text(json.dumps(manifest))

    result = mocap2ilqr.reaudit_dataset(source, output)
    assert stem in result["flipped_to_fallback"]
    assert result["manifest"]["clips"][stem]["status"] == "fallback"
    assert not (output / "robot_filtered" / f"{stem}.npz").exists()
    assert not (output / "smpl_filtered" / f"{stem}.npz").exists()
