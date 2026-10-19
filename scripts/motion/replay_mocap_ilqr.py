# pyright: reportAttributeAccessIssue=false, reportReturnType=false
"""Compare an iLQR-converted SONIC motion against its source mocap in the viewer.

The converted motion drives the solid robot; the original mocap is drawn as a
translucent ghost anchored to the same frame sequence (or shifted by
``--ghost-offset`` for a side-by-side view).  Both streams play back
frame-locked, so any drift between the dynamically feasible plan and the
source retarget is directly visible.

Usage:
    uv run scripts/motion/replay_mocap_ilqr.py \
        --original data/lafan1/robot_filtered/dance1_subject1.npz \
        --ilqr data/lafan1_ilqr/robot_filtered/dance1_subject1.npz

    # Side-by-side instead of overlay
    uv run scripts/motion/replay_mocap_ilqr.py --original ... --ilqr ... \
        --ghost-offset 1.2 0

    # Headless sanity check (builds both robots, steps frames, prints the
    # deviation summary, never opens a window)
    uv run scripts/motion/replay_mocap_ilqr.py --original ... --ilqr ... --dry-run

Controls:
    Space: pause/resume
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH

GHOST_PREFIX = "ghost_"
GHOST_RGBA = (0.62, 0.30, 0.20, 0.35)


def load_robot_npz(path: str) -> dict[str, np.ndarray]:
    """Load a SONIC robot motion NPZ (original or iLQR-converted)."""
    with np.load(path, allow_pickle=False) as data:
        motion = {name: data[name] for name in data.files}
    for key in ("fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "joint_names"):
        if key not in motion:
            raise ValueError(f"{path}: missing SONIC robot key {key!r}")
    return motion


def build_comparison_model(scene_file: str, robot_file: str, ghost_offset: tuple[float, float]):
    """Compile the flat scene with a second, ghost-styled robot attached.

    Returns (model, qpos layout) where the layout maps each stream to the
    free-joint address plus per-joint qpos addresses of its robot copy.
    """
    spec = mujoco.MjSpec.from_file(scene_file)
    ghost_spec = mujoco.MjSpec.from_file(robot_file)
    frame = spec.worldbody.add_frame(pos=[ghost_offset[0], ghost_offset[1], 0.0])
    spec.attach(ghost_spec, prefix=GHOST_PREFIX, frame=frame)
    model = spec.compile()

    visual = (model.geom_contype == 0) & (model.geom_conaffinity == 0)
    for geom_id, body_id in enumerate(model.geom_bodyid):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id))
        if body_name is not None and body_name.startswith(GHOST_PREFIX):
            if visual[geom_id]:
                model.geom_rgba[geom_id] = GHOST_RGBA
            else:
                model.geom_rgba[geom_id, 3] = 0.0
    return model


def stream_qpos_layout(model: mujoco.MjModel, joint_names: list[str], *, ghost: bool):
    """Resolve (free-joint qpos address, joint qpos addresses) for one stream."""
    prefix = GHOST_PREFIX if ghost else ""
    free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free_joints) < 2:
        raise ValueError("comparison scene must contain two free-joint robots")
    free_id = int(free_joints[1] if ghost else free_joints[0])
    joint_adr = []
    for name in joint_names:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)
        if jnt_id < 0:
            raise ValueError(f"joint {prefix}{name!r} not found in the comparison scene")
        joint_adr.append(int(model.jnt_qposadr[jnt_id]))
    return int(model.jnt_qposadr[free_id]), joint_adr


def set_stream_frame(
    data: mujoco.MjData,
    motion: dict[str, np.ndarray],
    frame: int,
    free_adr: int,
    joint_adr: list[int],
    root_offset: np.ndarray,
) -> None:
    """Write one motion frame into its robot copy and run kinematics."""
    data.qpos[free_adr : free_adr + 3] = np.asarray(motion["body_pos_w"][frame, 0]) + root_offset
    data.qpos[free_adr + 3 : free_adr + 7] = motion["body_quat_w"][frame, 0]
    data.qpos[joint_adr] = motion["joint_pos"][frame]


def deviation_summary(
    original: dict[str, np.ndarray], converted: dict[str, np.ndarray]
) -> dict[str, float]:
    joint_delta = np.abs(converted["joint_pos"] - original["joint_pos"])
    root_delta = np.abs(converted["body_pos_w"][:, 0] - original["body_pos_w"][:, 0])
    return {
        "frames": float(len(joint_delta)),
        "joint_dev_mean_rad": float(joint_delta.mean()),
        "joint_dev_max_rad": float(joint_delta.max()),
        "root_dev_mean_m": float(root_delta.mean()),
        "root_dev_max_m": float(root_delta.max()),
    }


def replay(args) -> None:
    original = load_robot_npz(args.original)
    converted = load_robot_npz(args.ilqr)
    num_frames = int(original["joint_pos"].shape[0])
    if converted["joint_pos"].shape[0] != num_frames:
        raise ValueError(
            "stream frame counts differ: "
            f"{num_frames} (original) vs {converted['joint_pos'].shape[0]} (ilqr)"
        )
    fps = int(np.asarray(original["fps"]).reshape(-1)[0])
    if fps != int(np.asarray(converted["fps"]).reshape(-1)[0]):
        raise ValueError("stream fps differ; both must be the SONIC 50 Hz contract")
    joint_names = [str(name) for name in original["joint_names"].tolist()]
    if [str(name) for name in converted["joint_names"].tolist()] != joint_names:
        raise ValueError("streams use different joint orders")

    print(f"Original: {args.original}")
    print(f"iLQR:     {args.ilqr}")
    for key, value in deviation_summary(original, converted).items():
        print(f"  {key}: {value:.4f}" if key != "frames" else f"  {key}: {int(value)}")

    scene = args.model_file or str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
    robot = args.robot_file or str(ASSETS_ROOT_PATH / "robots" / "g1" / "g1.xml")
    model = build_comparison_model(scene, robot, tuple(args.ghost_offset))
    data = mujoco.MjData(model)
    solid_free, solid_joints = stream_qpos_layout(model, joint_names, ghost=False)
    ghost_free, ghost_joints = stream_qpos_layout(model, joint_names, ghost=True)
    solid_offset = np.zeros(3)
    ghost_offset = np.array([args.ghost_offset[0], args.ghost_offset[1], 0.0])

    def set_frame(frame: int) -> None:
        set_stream_frame(data, converted, frame, solid_free, solid_joints, solid_offset)
        set_stream_frame(data, original, frame, ghost_free, ghost_joints, ghost_offset)
        mujoco.mj_forward(model, data)

    if args.dry_run:
        for frame in range(0, num_frames, max(num_frames // 10, 1)):
            set_frame(frame)
        print("dry run OK: both robots driven through the clip")
        return

    paused = False
    frame = min(args.start_frame, num_frames - 1)

    def on_key(keycode: int) -> None:
        nonlocal paused
        if keycode == ord(" "):
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'} playback.")

    print("Opening viewer — close window or press Esc to quit. Controls: Space=pause")
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running():
            started = time.perf_counter()
            set_frame(frame)
            viewer.sync()
            print(f"\rFrame {frame + 1}/{num_frames}{' (paused)' if paused else '  '}", end="")
            if paused:
                time.sleep(0.05)
                continue
            frame += 1
            if frame >= num_frames:
                if args.loop:
                    frame = 0
                else:
                    print("\nPlayback finished.")
                    frame = num_frames - 1
                    paused = True
            time.sleep(max(0.0, 1.0 / (fps * args.speed) - (time.perf_counter() - started)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, help="source SONIC robot NPZ (mocap)")
    parser.add_argument("--ilqr", required=True, help="iLQR-converted SONIC robot NPZ")
    parser.add_argument(
        "--ghost-offset",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("X", "Y"),
        help="shift the mocap ghost in the ground plane (default: overlay in place)",
    )
    parser.add_argument("--model_file", help="scene XML (default: g1 scene_flat.xml)")
    parser.add_argument("--robot_file", help="robot XML used for the ghost (default: g1.xml)")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build both robots and step frames without opening the viewer",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if Path(args.original).resolve() == Path(args.ilqr).resolve():
        raise SystemExit("--original and --ilqr must differ")
    replay(args)


if __name__ == "__main__":
    main()
