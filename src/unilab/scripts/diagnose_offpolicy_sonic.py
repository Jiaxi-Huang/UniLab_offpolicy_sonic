"""Print the resolved SONIC training contract without starting a learner."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


from uni_rl.algos.sonic.config import SonicModelConfig
from unilab.base.config_adapter import create_env
from uni_rl.utils.observations import get_obs_dims
from unilab.training import ensure_registries
from unilab.training.offpolicy_sonic import _model_config


@hydra.main(version_base="1.3", config_path="../conf/flashsac", config_name="config_sonic")
def main(cfg: DictConfig) -> None:
    ensure_registries()
    model: SonicModelConfig = _model_config(cfg.algo.sonic.model)
    model.validate_observation_contract()
    env = create_env(cfg, num_envs=1)
    try:
        obs_dim, critic_dim = get_obs_dims(env.obs_groups_spec)
        motion = cfg.env.commands.motion.params
        print(OmegaConf.to_yaml(cfg.algo.sonic.model, resolve=True), end="")
        print(f"resolved_obs_dim={obs_dim}")
        print(f"resolved_critic_obs_dim={critic_dim}")
        print(f"expected_actor_obs_dim={model.actor_obs_dim}")
        print(f"backbone_params={model.parameter_count():,}")
        print(f"num_future_frames={model.num_future_frames}")
        print(f"motion_sampling_mode={motion.sampling_mode}")
        print(
            f"termination_thresholds={ {name: getattr(motion, name) for name in ('anchor_pos_z_threshold', 'anchor_ori_threshold', 'ee_body_pos_z_threshold', 'foot_pos_threshold')} }"
        )
        print("training_action_scale=2.0 x effort_limit/stiffness")
        print("official_release_action_scale=0.25 x effort_limit/stiffness")
    finally:
        env.close()


if __name__ == "__main__":
    main()
