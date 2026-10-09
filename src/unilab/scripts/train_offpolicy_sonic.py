"""SONIC-specific FlashSAC training and playback entry."""

from __future__ import annotations

import datetime
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import hydra
from omegaconf import DictConfig

ROOT_DIR = Path.cwd()

from unilab.scripts.train_offpolicy import (
    build_failure_summary,
    build_run_dir_name,
    enable_faulthandler,
)
from uni_rl.ipc.dp_launcher import (
    UNILAB_DP_LOG_DIR,
    DpRankSupervisor,
    current_dp_rank,
    resolve_dp_topology,
    validate_dp_launchable,
)
from unilab.training import (
    apply_configured_training_seed,
    assert_offpolicy_task_choice_matches_algo,
    ensure_registries,
    get_log_root,
    should_run_playback,
)
from unilab.training.experiment import ExperimentTracker
from unilab.training.offpolicy_sonic import apply_rank_config, build_runner, play_offpolicy
from unilab.visualization.interactive_playback import default_device


def run(cfg: DictConfig) -> None:
    """Run SONIC through its isolated FlashSAC builder and play pipeline."""

    enable_faulthandler()
    ensure_registries()

    devices = resolve_dp_topology(cfg.training.devices)
    rank = current_dp_rank()
    rank_device = apply_rank_config(cfg)
    seed_info = apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    algo_name = cfg.algo.algo
    task_name = cfg.training.task_name
    assert_offpolicy_task_choice_matches_algo(cfg, algo_name=algo_name)

    if cfg.training.log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir_name = build_run_dir_name(
            timestamp,
            str(cfg.training.sim_backend),
            world_size=len(devices) if devices is not None else 1,
        )
        log_dir = str(get_log_root(ROOT_DIR, cfg) / task_name / run_dir_name)
    else:
        log_dir = cfg.training.log_dir
    if rank > 0:
        log_dir = os.environ[UNILAB_DP_LOG_DIR]

    supervisor: DpRankSupervisor | None = None
    if devices is not None and rank == 0 and len(devices) > 1:
        validate_dp_launchable(devices)
        supervisor = DpRankSupervisor(devices, log_dir)

    import torch

    tracker = None
    if not cfg.training.play_only and rank == 0:
        tracker = ExperimentTracker(
            root_dir=ROOT_DIR,
            log_dir=log_dir,
            algo_name=algo_name,
            task_name=task_name,
            sim_backend=cfg.training.sim_backend,
            training_cfg=cfg.training,
            full_cfg=cfg,
            device=default_device(torch, rank_device),
            seed_info=seed_info,
        )
        tracker.start()

    try:
        with supervisor if supervisor is not None else nullcontext():
            if not cfg.training.play_only:
                runner = None
                try:
                    runner = build_runner(algo_name, cfg, log_dir=log_dir)
                    runner.learn(
                        max_iterations=cfg.algo.max_iterations,
                        save_interval=cfg.algo.save_interval,
                        log_dir=log_dir,
                        logger_type=cfg.training.logger,
                    )
                    run_summary = getattr(runner, "last_run_summary", None)
                    if isinstance(run_summary, dict) and run_summary.get("status") not in (
                        None,
                        "completed",
                    ):
                        raise RuntimeError(
                            f"SONIC training ended with status={run_summary.get('status')!r}"
                        )
                    if tracker is not None:
                        tracker.update_summary(run_summary)
                except BaseException as exc:
                    if tracker is not None:
                        tracker.update_summary(
                            build_failure_summary(
                                exc,
                                getattr(runner, "last_run_summary", None),
                            )
                        )
                    raise
                finally:
                    if runner is not None:
                        runner.close()

            if rank == 0 and should_run_playback(
                play_only=cfg.training.play_only,
                no_play=cfg.training.no_play,
                play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
            ):
                print("@" * 50)
                play_video_path = play_offpolicy(algo_name, cfg)
                if tracker is not None:
                    tracker.log_video(play_video_path)
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    hydra.main(version_base="1.3", config_path="../conf/flashsac", config_name="config_sonic")(run)()
