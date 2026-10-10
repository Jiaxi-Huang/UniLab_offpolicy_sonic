#!/usr/bin/env python3
"""Convert paired SONIC CSV/PKL sources into aligned training NPZ files."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir

from unilab.assets.sonic import sonic_clip_is_filtered
from unilab.base import registry
from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.envs import ManagerBasedRlEnvCfg
from unilab.tasks.motion_tracking.g1.sonic_data import (
    SonicRawMotionLoader,
    resolve_sonic_pairs,
    write_sonic_npz_pair,
)
from unilab.tasks.motion_tracking.g1.sonic_manager import SonicMotionCommandCfg


def convert_sonic_dataset(
    robot_input: str | Path,
    smpl_input: str | Path,
    output: str | Path,
    *,
    max_clips: int | None = None,
    start_frame: int = 0,
    max_frames: int | None = None,
    overwrite: bool = False,
    exclude_filtered: bool = True,
) -> dict[str, Any]:
    """Convert all matched sources, optionally cropping for a smoke dataset.

    ``exclude_filtered`` drops clips whose name hits the upstream BONES-SEED
    blacklist (see ``SONIC_FILTERED_CLIP_KEYWORDS``) before any conversion
    work, so the converted dataset matches the gear_sonic training recipe.
    """

    if max_clips is not None and max_clips <= 0:
        raise ValueError("max_clips must be positive")
    root_dir = Path(__file__).parents[3]
    # The Hydra owner config tree lives inside the package (src/unilab/conf);
    # root_dir stays the repo root for BackendAdapter's relative asset paths.
    config_dir = root_dir / "src" / "unilab" / "conf" / "flashsac"
    registry.ensure_registries()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        composed = compose(config_name="config_sonic", overrides=["task=g1_sonic/mujoco"])
    cfg = cast(
        ManagerBasedRlEnvCfg,
        registry.materialize_env_config("G1SonicManager"),
    )
    apply_cfg_overrides(
        cfg,
        BackendAdapter(composed, root_dir=root_dir).build_task_env_cfg_override(),
    )
    motion_cfg = cfg.commands["motion"]
    if not isinstance(motion_cfg, SonicMotionCommandCfg):
        raise TypeError("G1SonicManager motion command has an incompatible configuration")
    if cfg.scene is None:
        raise ValueError("G1SonicManager requires a scene configuration")
    pairs = resolve_sonic_pairs(str(robot_input), str(smpl_input))
    filtered_clips = [robot.stem for robot, _ in pairs if sonic_clip_is_filtered(robot.stem)]
    if exclude_filtered and filtered_clips:
        pairs = [pair for pair in pairs if not sonic_clip_is_filtered(pair[0].stem)]
        print(
            f"Excluding {len(filtered_clips)} blacklisted clips (upstream BONES-SEED filter)",
            flush=True,
        )
    if max_clips is not None:
        pairs = pairs[:max_clips]
    output = Path(output).expanduser().resolve()
    robot_output = output / "robot_filtered"
    smpl_output = output / "smpl_filtered"
    robot_output.mkdir(parents=True, exist_ok=True)
    smpl_output.mkdir(parents=True, exist_ok=True)

    backend = create_backend(
        "mujoco",
        cfg.scene,
        1,
        cfg.sim_dt,
        base_name=motion_cfg.anchor_body_name,
        add_body_sensors=True,
        **env_backend_kwargs(cfg),
    )
    converted: list[str] = []
    skipped: list[str] = []
    try:
        for robot_source, smpl_source in pairs:
            robot_destination = robot_output / f"{robot_source.stem}.npz"
            smpl_destination = smpl_output / f"{smpl_source.stem}.npz"
            destinations_exist = (robot_destination.exists(), smpl_destination.exists())
            if all(destinations_exist) and not overwrite:
                skipped.append(robot_source.stem)
                continue
            if any(destinations_exist) and not all(destinations_exist) and not overwrite:
                raise FileExistsError(
                    f"Incomplete converted pair for {robot_source.stem!r}; "
                    "pass --overwrite to replace it"
                )
            loader = SonicRawMotionLoader(
                str(robot_source),
                str(smpl_source),
                backend=backend,
                body_names=motion_cfg.body_names,
                start_frame=start_frame,
                max_frames=max_frames,
            )
            write_sonic_npz_pair(loader, robot_destination, smpl_destination)
            converted.append(robot_source.stem)
    finally:
        backend.cleanup_scene_assets()

    summary = {
        "robot_output": str(robot_output),
        "smpl_output": str(smpl_output),
        "source_pairs": len(pairs),
        "converted": converted,
        "skipped": skipped,
        "start_frame": start_frame,
        "max_frames": max_frames,
        "filter_applied": bool(exclude_filtered and filtered_clips),
        "filtered_clip_count": len(filtered_clips) if exclude_filtered else 0,
        "filtered_clips": filtered_clips if exclude_filtered else [],
    }
    (output / "conversion_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-input",
        type=Path,
        default=Path("data/bones_seed_source/robot_filtered"),
        help="downloaded robot CSV file or directory",
    )
    parser.add_argument(
        "--smpl-input",
        type=Path,
        default=Path("data/bones_seed_source/smpl_filtered"),
        help="downloaded SMPL PKL file or directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sonic"),
        help="output root containing robot_filtered/ and smpl_filtered/",
    )
    parser.add_argument("--max-clips", type=int, help="convert only the first N matched clips")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, help="frames per clip after 50 Hz resampling")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-filtered",
        action="store_true",
        help="keep clips matching the upstream BONES-SEED blacklist (default: exclude)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        summary = convert_sonic_dataset(
            args.robot_input,
            args.smpl_input,
            args.output,
            max_clips=args.max_clips,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
            exclude_filtered=not args.include_filtered,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
