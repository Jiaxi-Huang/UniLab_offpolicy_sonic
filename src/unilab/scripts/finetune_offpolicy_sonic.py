"""Fine-tune FlashSAC around a frozen official SONIC backbone.

Usage example::

    uv run scripts/finetune_offpolicy_sonic.py \
      algo.sonic.finetune_checkpoint=/path/to/sonic_release/last.pt
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unilab.scripts.train_offpolicy_sonic import run


@hydra.main(version_base="1.3", config_path="../conf/flashsac", config_name="config_sonic")
def main(cfg: DictConfig) -> None:
    if not cfg.algo.sonic.finetune_checkpoint:
        raise ValueError(
            "finetune_offpolicy_sonic requires "
            "algo.sonic.finetune_checkpoint=/path/to/official/last.pt"
        )
    cfg.algo.sonic.freeze_backbone = True
    cfg.training.play_only = False
    run(cfg)


if __name__ == "__main__":
    main()
