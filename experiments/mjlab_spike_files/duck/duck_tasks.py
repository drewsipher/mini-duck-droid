"""Register the duck tracking task with mjlab so `train`/`play` can launch it by name.

mjlab's CLIs only import mjlab.tasks, so an out-of-tree task needs a wrapper that imports
this module first (see train_duck.py / play_duck.py). The motion file is given on the
command line: --env.commands.motion.motion-file <motion.npz>.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from duck_mjlab_cfg import duck_flat_tracking_env_cfg

TASK_ID = "Mjlab-Tracking-Flat-Open-Duck-Mini"


def duck_tracking_ppo_runner_cfg():
  cfg = unitree_g1_tracking_ppo_runner_cfg()  # BeyondMimic PPO recipe, unchanged
  cfg.experiment_name = "duck_tracking"
  cfg.logger = "tensorboard"  # no W&B account needed; pass --agent.logger wandb to opt in
  cfg.upload_model = False
  return cfg


register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=duck_flat_tracking_env_cfg(motion_file=""),
  play_env_cfg=duck_flat_tracking_env_cfg(motion_file="", play=True),
  rl_cfg=duck_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
