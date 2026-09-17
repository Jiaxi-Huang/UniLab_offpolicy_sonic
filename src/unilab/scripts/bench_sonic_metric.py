"""Compare official and locally trained SONIC checkpoints clip by clip."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


from unilab.training import apply_configured_training_seed, ensure_registries
from unilab.training.offpolicy_sonic import apply_rank_config
from unilab.training.sonic_metrics import benchmark_sonic_models, write_sonic_benchmark


@hydra.main(
    version_base="1.3",
    config_path="../conf/flashsac",
    config_name="config_sonic_benchmark",
)
def main(cfg: DictConfig) -> None:
    ensure_registries()
    apply_rank_config(cfg)
    apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    result = benchmark_sonic_models(cfg)
    output = write_sonic_benchmark(result, to_absolute_path(str(cfg.benchmark.output)))

    release = result["models"]["sonic_release"]
    trained = result["models"]["trained"]
    print(f"Dataset: {result['dataset']} ({release['num_trials']} clips)")
    print("model          ep_len   survival   MPJPE-L   velocity   acceleration")
    for name, values in (("sonic_release", release), ("trained", trained)):
        metrics = [
            "n/a" if values[key] is None else f"{values[key]:.3f}"
            for key in (
                "mpjpe_l_mm",
                "velocity_distance_mm_per_frame",
                "acceleration_distance_mm_per_frame2",
            )
        ]
        print(
            f"{name:<14} {values['mean_ep_len']:>7.1f} "
            f"{values['survival_rate']:>10.3%} "
            f"{metrics[0]:>9} {metrics[1]:>10} {metrics[2]:>12}"
        )
    print(f"Wrote SONIC benchmark: {output}")


if __name__ == "__main__":
    main()
