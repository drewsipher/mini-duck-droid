#!/usr/bin/env python3
"""A few real PPO iterations of BeyondMimic tracking on the duck, timed, then ONNX export.

  python duck_train_smoke.py --motion <motion.npz> --device cuda:0 --num-envs 1024 --iters 5 --log-dir <dir>

Mirrors mjlab.scripts.train.run_train without the CLI, W&B or registry pieces.
"""

from __future__ import annotations

import argparse
import subprocess
import time
import warnings
from dataclasses import asdict
from pathlib import Path

warnings.simplefilter("ignore")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--motion", required=True)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--num-envs", type=int, default=1024)
  ap.add_argument("--iters", type=int, default=5)
  ap.add_argument("--log-dir", required=True)
  a = ap.parse_args()

  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
  from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
  from mjlab.utils.torch import configure_torch_backends
  from duck_mjlab_cfg import duck_flat_tracking_env_cfg

  configure_torch_backends()
  env_cfg = duck_flat_tracking_env_cfg(motion_file=a.motion)
  env_cfg.scene.num_envs = a.num_envs
  env_cfg.commands["motion"].debug_vis = False

  rl_cfg = unitree_g1_tracking_ppo_runner_cfg()  # same PPO recipe as the G1 tracking task
  rl_cfg.experiment_name = "duck_tracking_smoke"
  rl_cfg.logger = "tensorboard"
  rl_cfg.upload_model = False
  rl_cfg.max_iterations = a.iters
  rl_cfg.save_interval = 10_000

  t0 = time.time()
  env = ManagerBasedRlEnv(cfg=env_cfg, device=a.device)
  venv = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
  log_dir = Path(a.log_dir)
  log_dir.mkdir(parents=True, exist_ok=True)
  runner = MotionTrackingOnPolicyRunner(venv, asdict(rl_cfg), str(log_dir), a.device, registry_name=None)
  print(f"[train] env+runner built in {time.time()-t0:.1f}s; num_envs={a.num_envs} steps/env/iter={rl_cfg.num_steps_per_env} "
        f"samples/iter={a.num_envs*rl_cfg.num_steps_per_env:,}")

  if a.device.startswith("cuda"):
    torch.cuda.reset_peak_memory_stats()
  # Time iterations one at a time so warm-up is visible.
  per_iter = []
  for i in range(a.iters):
    t1 = time.time()
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=(i == 0))
    if a.device.startswith("cuda"):
      torch.cuda.synchronize()
    per_iter.append(time.time() - t1)
    print(f"[train] iter {i}: {per_iter[-1]:.2f}s = {a.num_envs*rl_cfg.num_steps_per_env/per_iter[-1]:,.0f} samples/s")
  steady = per_iter[1:] or per_iter
  print(f"[train] steady state {sum(steady)/len(steady):.2f}s/iter = "
        f"{a.num_envs*rl_cfg.num_steps_per_env/(sum(steady)/len(steady)):,.0f} env-steps/s end-to-end")
  if a.device.startswith("cuda"):
    used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    print(f"[train] torch peak allocated {torch.cuda.max_memory_allocated()/2**20:,.0f} MiB, reserved "
          f"{torch.cuda.max_memory_reserved()/2**20:,.0f} MiB; nvidia-smi total used {used} MiB")

  t2 = time.time()
  runner.export_policy_to_onnx(str(log_dir / "exported"), filename="policy.onnx")
  import onnx
  m = onnx.load(str(log_dir / "exported" / "policy.onnx"))
  ins = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]
  outs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]
  print(f"[train] ONNX exported in {time.time()-t2:.1f}s: inputs {ins} outputs {outs}; "
        f"metadata keys {[p.key for p in m.metadata_props][:12]}")
  env.close()


if __name__ == "__main__":
  main()
