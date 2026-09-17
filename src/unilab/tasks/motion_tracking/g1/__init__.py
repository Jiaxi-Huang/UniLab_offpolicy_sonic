"""G1 motion profiles on the shared NumPy Manager-Based runtime."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

from .motion_box_loader import BoxMotionData, BoxMotionLoader
from .sonic_manager import G1SonicManagerCfg, G1SonicManagerEnv

G1_MOTION_TASKS = (
    "G1MotionTracking",
    "G1MotionTrackingSAC",
    "G1BoxTracking",
    "G1FlipTracking",
    "G1FlipTrackingSAC",
    "G1WBTObs",
    "G1WBTObs23Dof",
    "G1WBT23Dof",
    "G1WBT",
)

for _task_name in G1_MOTION_TASKS:
    registry.register_env_config(_task_name, ManagerBasedRlEnvCfg)
    registry.register_env(_task_name, make_manager_based_rl_env, sim_backend="mujoco")
    registry.register_env(_task_name, make_manager_based_rl_env, sim_backend="motrix")

# mjwarp is registered only for G1MotionTrackingSAC (benchmark scope, issue #1292);
# mujoco-warp + warp-lang remain optional deps and other motion tasks keep
# mujoco/motrix until their mjwarp paths are validated.
registry.register_env("G1MotionTrackingSAC", make_manager_based_rl_env, sim_backend="mjwarp")

# genesis/newton implement the motion-body-id capability since unisim-core 1.5.1
# (unilabsim/unisim#137); isaacgym/isaacsim join them since unisim-core 1.7.4
# fixed the subprocess body-state publish/reset paths (unilabsim/unisim#141,
# PR #145).
for _backend in ("genesis", "newton", "isaacgym", "isaacsim"):
    registry.register_env("G1MotionTrackingSAC", make_manager_based_rl_env, sim_backend=_backend)


__all__ = [
    "BoxMotionData",
    "BoxMotionLoader",
    "G1SonicManagerCfg",
    "G1SonicManagerEnv",
    "G1_MOTION_TASKS",
]
