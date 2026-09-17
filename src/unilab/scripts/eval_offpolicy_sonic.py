"""Run SONIC playback and write a JSON rollout/termination summary."""

from __future__ import annotations

import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig

from unilab.training import apply_configured_training_seed, ensure_registries
from unilab.training.offpolicy_sonic import apply_rank_config, play_offpolicy


@hydra.main(version_base="1.3", config_path="../conf/flashsac", config_name="config_sonic")
def main(cfg: DictConfig) -> None:
    ensure_registries()
    apply_rank_config(cfg)
    apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    configured = getattr(cfg.training, "play_stats_output", None)
    if configured:
        output = Path(str(configured))
    else:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_path = Path(str(cfg.algo.load_run))
        output = run_path.parent / f"play_stats_{stamp}.json"
    print(f"Writing playback statistics to {output}")
    play_offpolicy("flashsac", cfg, stats_output=output)
    print(f"Playback statistics written: {output}")


if __name__ == "__main__":
    main()
