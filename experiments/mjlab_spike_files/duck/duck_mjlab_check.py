#!/usr/bin/env python3
"""Staged sanity check of the duck in mjlab: entity -> scene+sim -> tracking env with a dummy agent.

  python duck_mjlab_check.py --motion <motion.npz> [--device cpu|cuda:0] [--num-envs N]
                             [--steps K] [--agent zero|random] [--stage entity|sim|env]

Every stage prints what it built and the first exception it hits, so a failure is a
finding, not a crash. On CUDA it also reports env steps/s and peak VRAM.
"""

from __future__ import annotations

import argparse
import subprocess
import time
import traceback
import warnings

warnings.simplefilter("ignore")


def nvidia_smi_used_mib() -> str:
  try:
    out = subprocess.run(
      ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
      capture_output=True, text=True, timeout=5,
    ).stdout.strip()
    return out
  except Exception:
    return "?"


def stage_entity():
  from mjlab.entity import Entity
  from duck_mjlab_cfg import get_duck_robot_cfg

  e = Entity(get_duck_robot_cfg())
  m = e.compile()
  print(f"[entity] bodies={e.num_bodies} joints={e.num_joints} actuators={e.num_actuators} "
        f"nq={m.nq} nv={m.nv} nu={m.nu} ngeom={m.ngeom} nsensor={m.nsensor}")
  print("[entity] joints:", e.joint_names)
  print("[entity] bodies:", e.body_names)
  return e


def stage_sim(device: str, num_envs: int, steps: int):
  import torch
  import warp as wp
  from mjlab.scene import Scene, SceneCfg
  from mjlab.sim.sim import Simulation, SimulationCfg
  from mjlab.sim import MujocoCfg
  from mjlab.terrains import TerrainEntityCfg
  from duck_mjlab_cfg import get_duck_robot_cfg

  wp.config.log_level = wp.LOG_WARNING
  scene_cfg = SceneCfg(
    terrain=TerrainEntityCfg(terrain_type="plane"),
    num_envs=num_envs,
    extent=0.5,
    entities={"robot": get_duck_robot_cfg()},
  )
  t0 = time.time()
  scene = Scene(scene_cfg, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=num_envs, cfg=SimulationCfg(mujoco=MujocoCfg(timestep=0.002)), model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  scene.reset()
  sim.forward()
  print(f"[sim] built {num_envs} envs on {device} in {time.time()-t0:.1f}s; nworld={sim.data.nworld} "
        f"naconmax={getattr(sim.data, 'naconmax', '?')} njmax={getattr(sim.data, 'njmax', '?')}")
  robot = scene["robot"]
  # Entity.reset() does not write the keyframe (the env's events / motion command do), so
  # put every world at the home pose explicitly, then hold it with the XML servos.
  robot.write_root_state_to_sim(robot.data.default_root_state)
  robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
  sim.forward()
  print(f"[sim] root after keyframe write: {robot.data.root_link_pos_w[0].tolist()}")
  robot.set_joint_position_target(robot.data.default_joint_pos)
  t0 = time.time()
  for _ in range(steps):
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.mj_model.opt.timestep)
  if device.startswith("cuda"):
    torch.cuda.synchronize()
  dt = time.time() - t0
  z = robot.data.root_link_pos_w[:, 2]
  print(f"[sim] {steps} physics steps x {num_envs} envs in {dt:.2f}s = {steps*num_envs/dt:,.0f} env-steps/s; "
        f"root z after {steps*0.002:.2f}s: mean {z.mean():.3f} min {z.min():.3f} max {z.max():.3f} (home 0.150)")
  return sim


def stage_env(motion: str, device: str, num_envs: int, steps: int, agent: str, play: bool = False):
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from duck_mjlab_cfg import SERVO_JOINT_NAMES, duck_flat_tracking_env_cfg

  cfg = duck_flat_tracking_env_cfg(motion_file=motion, play=play)
  cfg.scene.num_envs = num_envs
  cfg.commands["motion"].debug_vis = False
  t0 = time.time()
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  print(f"[env] built in {time.time()-t0:.1f}s: num_envs={env.num_envs} step_dt={env.step_dt} "
        f"action_dim={env.action_manager.total_action_dim}")
  obs, _ = env.reset()
  for k, v in obs.items():
    print(f"[env] obs[{k}] shape {tuple(v.shape)}")
  act_dim = env.action_manager.total_action_dim
  if device.startswith("cuda"):
    torch.cuda.reset_peak_memory_stats()
  # One warm-up step so JIT/graph capture is not in the timing.
  env.step(torch.zeros(num_envs, act_dim, device=device))
  if device.startswith("cuda"):
    torch.cuda.synchronize()
  robot = env.scene["robot"]
  servo_ids = robot.find_joints(tuple(f"^{n}$" for n in SERVO_JOINT_NAMES), preserve_order=True)[0]
  motion_term = env.command_manager.get_term("motion")
  action_scale = cfg.actions["joint_pos"].scale
  if agent == "replay":
    env.reset()
  t0 = time.time()
  total_rew = 0.0
  n_term = 0
  term_reasons: dict[str, int] = {}
  for _ in range(steps):
    if agent == "zero":
      a = torch.zeros(num_envs, act_dim, device=device)
    elif agent == "random":
      a = 2 * torch.rand(num_envs, act_dim, device=device) - 1
    else:  # replay: position targets = reference joint angles of the current frame
      ref = motion_term.joint_pos[:, servo_ids]
      a = (ref - robot.data.default_joint_pos[:, servo_ids]) / action_scale
    obs, rew, term, trunc, extras = env.step(a)
    total_rew += float(rew.mean())
    n_term += int(term.sum())
    if int(term.sum()):
      for name in env.termination_manager.active_terms:
        n = int(env.termination_manager.get_term(name).sum())
        if n:
          term_reasons[name] = term_reasons.get(name, 0) + n
  metrics = {k: round(float(v.mean()), 4) for k, v in motion_term.metrics.items() if k.startswith("error_")}
  print(f"[env] final-step tracking errors (mean over envs): {metrics}")
  print(f"[env] termination reasons: {term_reasons}")
  if device.startswith("cuda"):
    torch.cuda.synchronize()
  dt = time.time() - t0
  print(f"[env] {steps} env steps x {num_envs} envs ({agent} agent) in {dt:.2f}s = {steps*num_envs/dt:,.0f} env-steps/s "
        f"({steps*num_envs*cfg.decimation/dt:,.0f} physics-steps/s); mean reward/step {total_rew/steps:.3f}; terminations {n_term}")
  if device.startswith("cuda"):
    print(f"[env] torch peak allocated {torch.cuda.max_memory_allocated()/2**20:,.0f} MiB, reserved "
          f"{torch.cuda.max_memory_reserved()/2**20:,.0f} MiB; nvidia-smi total used {nvidia_smi_used_mib()} MiB")
  for name, term in env.reward_manager._episode_sums.items() if hasattr(env.reward_manager, "_episode_sums") else []:
    pass
  env.close()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--motion", default="")
  ap.add_argument("--device", default="cpu")
  ap.add_argument("--num-envs", type=int, default=2)
  ap.add_argument("--steps", type=int, default=50)
  ap.add_argument("--agent", default="zero", choices=["zero", "random", "replay"])
  ap.add_argument("--play", action="store_true", help="play-mode cfg: start of clip, no pushes, no RSI noise")
  ap.add_argument("--stage", default="env", choices=["entity", "sim", "env"])
  a = ap.parse_args()

  for name, fn in (
    ("entity", lambda: stage_entity()),
    ("sim", lambda: stage_sim(a.device, a.num_envs, a.steps)),
    ("env", lambda: stage_env(a.motion, a.device, a.num_envs, a.steps, a.agent, a.play)),
  ):
    t0 = time.time()
    try:
      fn()
      print(f"[{name}] OK ({time.time()-t0:.1f}s)")
    except Exception as exc:  # noqa: BLE001
      print(f"[{name}] FAILED after {time.time()-t0:.1f}s: {type(exc).__name__}: {exc}")
      traceback.print_exc()
      break
    if name == a.stage:
      break


if __name__ == "__main__":
  main()
