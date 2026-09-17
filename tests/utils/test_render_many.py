"""Tests for MuJoCo GL backend resolution in unilab.visualization.render_many."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import types

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="GitHub Actions runners do not provide stable EGL/GLFW rendering backends.",
)


def _reload_render_many(monkeypatch):
    monkeypatch.setitem(sys.modules, "mujoco", types.SimpleNamespace())
    sys.modules.pop("unilab.visualization.render_many", None)
    return importlib.import_module("unilab.visualization.render_many")


def test_resolve_gl_backend_uses_egl_when_probe_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)

    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "_egl_runtime_usable", lambda: True)

    assert render_many._resolve_gl_backend() == "egl"


def test_resolve_gl_backend_uses_osmesa_when_headless_and_egl_unavailable(monkeypatch) -> None:
    # Headless host (no DISPLAY): glfw cannot work, so software rendering wins.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)

    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "_egl_runtime_usable", lambda: False)

    assert render_many._resolve_gl_backend() == "osmesa"


def test_resolve_gl_backend_uses_glfw_when_display_present_and_egl_unavailable(monkeypatch) -> None:
    # A display is available: glfw can create an off-screen context.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")

    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "_egl_runtime_usable", lambda: False)

    assert render_many._resolve_gl_backend() == "glfw"


def test_resolve_gl_backend_preserves_explicit_safe_value(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("MUJOCO_GL", "osmesa")

    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "_egl_runtime_usable", lambda: False)

    assert render_many._resolve_gl_backend() == "osmesa"


def test_resolve_gl_backend_uses_glfw_on_windows_without_display(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)

    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "_egl_runtime_usable", lambda: False)

    assert render_many._resolve_gl_backend() == "glfw"


def test_resolve_gl_backend_rejects_linux_only_backend_on_windows(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("MUJOCO_GL", "osmesa")

    render_many = _reload_render_many(monkeypatch)

    assert render_many._resolve_gl_backend() == "glfw"


def test_egl_runtime_usable_sets_default_device_id(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)

    def _fake_run(cmd, env, check, stdout, stderr, timeout):
        assert cmd[0] == sys.executable
        assert env["MUJOCO_GL"] == "egl"
        assert env["MUJOCO_EGL_DEVICE_ID"] == "0"
        assert check is True
        assert stdout is subprocess.DEVNULL
        assert stderr is subprocess.DEVNULL
        assert timeout == 10
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(render_many.subprocess, "run", _fake_run)

    assert render_many._egl_runtime_usable() is True
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "0"


def test_egl_runtime_usable_returns_false_on_probe_failure(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)

    def _fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(render_many.subprocess, "run", _fake_run)

    assert render_many._egl_runtime_usable() is False


def _reload_render_many_with_geom_enums(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        types.SimpleNamespace(
            mjtGeom=types.SimpleNamespace(mjGEOM_PLANE=0, mjGEOM_HFIELD=1, mjGEOM_BOX=6),
        ),
    )
    sys.modules.pop("unilab.visualization.render_many", None)
    return importlib.import_module("unilab.visualization.render_many")


def test_replicable_terrain_geom_indices_selects_worldbody_box(monkeypatch) -> None:
    # The x2 wall-flip render twin declares the wall as a group-0 worldbody box
    # geom precisely so this selector picks it up and the grid renderer
    # replicates one wall per env cell. Lock that contract in.
    render_many = _reload_render_many_with_geom_enums(monkeypatch)

    model = types.SimpleNamespace(
        ngeom=4,
        # 0: floor plane (worldbody)  1: robot geom (body 5)
        # 2: wall box (worldbody)     3: group-2 worldbody box (non-default group)
        geom_group=np.array([0, 0, 0, 2], dtype=np.int32),
        geom_bodyid=np.array([0, 5, 0, 0], dtype=np.int32),
        geom_type=np.array([0, 6, 6, 6], dtype=np.int32),
    )

    indices = render_many._replicable_terrain_geom_indices(model)

    # Only the worldbody box wall (geom 2) is replicable: the plane is skipped,
    # the body-attached robot geom is skipped, and the non-group-0 geom is skipped.
    assert indices.tolist() == [2]


def test_offset_freejoint_object_qpos_handles_arbitrary_object_body(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)

    model = types.SimpleNamespace(
        nbody=4,
        body_jntadr=np.array([-1, 0, 1, -1], dtype=np.int32),
        body_jntnum=np.array([0, 1, 1, 0], dtype=np.int32),
        jnt_type=np.array([0, 0], dtype=np.int32),
        jnt_qposadr=np.array([0, 7], dtype=np.int32),
    )
    data = types.SimpleNamespace(qpos=np.zeros((14,), dtype=np.float32))

    shifted = render_many._offset_freejoint_object_qpos(
        model, data, np.array([1.5, -2.0], dtype=np.float32)
    )

    assert shifted == {2}
    assert data.qpos[0] == pytest.approx(0.0)
    assert data.qpos[1] == pytest.approx(0.0)
    assert data.qpos[7] == pytest.approx(1.5)
    assert data.qpos[8] == pytest.approx(-2.0)


def test_render_backend_usable_reflects_resolved_backend(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)

    seen: dict[str, str] = {}

    def _fake_probe(backend: str) -> bool:
        seen["backend"] = backend
        return backend == "egl"

    monkeypatch.setattr(render_many, "_gl_backend_runtime_usable", _fake_probe)

    monkeypatch.setenv("MUJOCO_GL", "egl")
    assert render_many.render_backend_usable() is True
    assert seen["backend"] == "egl"

    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    assert render_many.render_backend_usable() is False


def test_render_states_get_frames_skips_when_backend_unusable(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "render_backend_usable", lambda: False)

    frames = render_many.render_states_get_frames(
        [np.zeros((1, 8), dtype=np.float32)],
        "/no/such/model.xml",
        num_processes=4,
    )

    assert frames == []


def test_render_states_get_frames_tracking_skips_when_backend_unusable(monkeypatch) -> None:
    render_many = _reload_render_many(monkeypatch)
    monkeypatch.setattr(render_many, "render_backend_usable", lambda: False)

    frames = render_many.render_states_get_frames_tracking(
        [np.zeros((1, 8), dtype=np.float32)],
        "/no/such/model.xml",
    )

    assert frames == []


def test_render_states_get_frames_fails_fast_on_worker_init_error(monkeypatch) -> None:
    """A failing pool initializer must NOT respawn workers forever (issue #605).

    ProcessPoolExecutor raises BrokenProcessPool quickly instead of hanging, and
    render_states_get_frames degrades to an empty result + warning.
    """
    # Skip the EGL probe in spawned workers (they inherit MUJOCO_GL via os.environ).
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    render_many = _reload_render_many(monkeypatch)
    # Bypass the parent pre-flight so we exercise the pool's fail-fast path.
    monkeypatch.setattr(render_many, "render_backend_usable", lambda: True)

    frames = render_many.render_states_get_frames(
        [np.zeros((1, 8), dtype=np.float32)],
        "/nonexistent/model/path.xml",  # init_worker raises while loading this
        num_processes=2,
    )

    assert frames == []


def _reload_render_many_real():
    """Import render_many against the real mujoco bindings."""
    sys.modules.pop("unilab.visualization.render_many", None)
    return importlib.import_module("unilab.visualization.render_many")


_GHOST_TEST_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="root" pos="0 0 1">
      <freejoint name="root_free"/>
      <geom name="root_visual" type="capsule" fromto="0 0 0 0 0 0.2" size="0.05"
        contype="0" conaffinity="0"/>
      <geom name="root_collision" type="sphere" size="0.06"/>
      <body name="child" pos="0 0 0.2">
        <joint name="hinge_a" type="hinge" axis="0 1 0"/>
        <geom name="child_visual" type="box" size="0.05 0.05 0.05"
          contype="0" conaffinity="0"/>
        <body name="leaf" pos="0 0 0.1">
          <joint name="hinge_b" type="hinge" axis="1 0 0"/>
          <geom name="leaf_visual" type="sphere" size="0.03" contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_prepare_ghost_model_tints_visual_geoms_and_maps_joints() -> None:
    mujoco = pytest.importorskip("mujoco")
    render_many = _reload_render_many_real()
    model = mujoco.MjModel.from_xml_string(_GHOST_TEST_XML)

    ghost = render_many._prepare_ghost_model(model, ("hinge_a", "hinge_b"))

    def geom_rgba(name: str) -> np.ndarray:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert geom_id >= 0
        return model.geom_rgba[geom_id]

    np.testing.assert_allclose(geom_rgba("root_visual"), render_many.GHOST_RGBA)
    np.testing.assert_allclose(geom_rgba("child_visual"), render_many.GHOST_RGBA)
    np.testing.assert_allclose(geom_rgba("leaf_visual"), render_many.GHOST_RGBA)
    # Collision and world geoms are hidden so the overlay stays a clean shell.
    assert geom_rgba("root_collision")[3] == 0.0
    assert geom_rgba("floor")[3] == 0.0
    # The first free joint anchors the ghost root pose block.
    assert ghost["root_adr"] == 0
    hinge_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ("hinge_a", "hinge_b")
    ]
    np.testing.assert_array_equal(
        ghost["joint_adrs"], [model.jnt_qposadr[hinge_ids[0]], model.jnt_qposadr[hinge_ids[1]]]
    )
    assert ghost["vopt"].flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]


def test_prepare_ghost_model_rejects_unknown_joint() -> None:
    mujoco = pytest.importorskip("mujoco")
    render_many = _reload_render_many_real()
    model = mujoco.MjModel.from_xml_string(_GHOST_TEST_XML)

    with pytest.raises(ValueError, match="missing from the playback model"):
        render_many._prepare_ghost_model(model, ("hinge_a", "nope"))


def test_add_ghost_geoms_poses_twin_and_appends_dynamic_geoms() -> None:
    mujoco = pytest.importorskip("mujoco")
    render_many = _reload_render_many_real()
    model = mujoco.MjModel.from_xml_string(_GHOST_TEST_XML)
    ghost = render_many._prepare_ghost_model(model, ("hinge_a", "hinge_b"))
    scene = mujoco.MjvScene(model, 100)
    geoms_before = scene.ngeom

    row = np.array([1.5, -2.0, 0.9, 1.0, 0.0, 0.0, 0.0, 0.25, -0.5], dtype=np.float64)
    render_many._add_ghost_geoms(
        ghost, row, np.array([10.0, 4.0]), mujoco.MjvPerturb(), scene
    )

    # Only robot geoms are dynamic; the worldbody floor is never duplicated.
    assert scene.ngeom - geoms_before >= 3
    data = ghost["data"]
    np.testing.assert_allclose(data.qpos[0:3], [11.5, 2.0, 0.9])
    np.testing.assert_allclose(data.qpos[3:7], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(data.qpos[ghost["joint_adrs"]], [0.25, -0.5])
    # mj_kinematics ran: the root body world pose follows the offset ghost qpos.
    root_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "root")
    np.testing.assert_allclose(data.xpos[root_body], [11.5, 2.0, 0.9], atol=1e-6)


def test_render_states_get_frames_requires_paired_ghost_args() -> None:
    render_many = _reload_render_many_real()

    with pytest.raises(ValueError, match="ghost"):
        render_many.render_states_get_frames(
            [np.zeros((1, 8), dtype=np.float32)],
            "/no/such/model.xml",
            ghost_qpos_list=[np.zeros((1, 9), dtype=np.float32)],
        )


def test_render_states_get_frames_tracking_requires_paired_ghost_args() -> None:
    render_many = _reload_render_many_real()

    with pytest.raises(ValueError, match="ghost"):
        render_many.render_states_get_frames_tracking(
            [np.zeros((1, 8), dtype=np.float32)],
            "/no/such/model.xml",
            ghost_joint_names=("hinge_a",),
        )
