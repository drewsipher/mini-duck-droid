#!/usr/bin/env python3
"""Convert an Open Duck Mini clip JSON into the NPZ that mjlab's tracking task loads.

Our format:   {"frames": [[t, {joint: rad}], ...], "base": [[[x,y,z],[w,x,y,z]], ...]?}
mjlab wants:  fps, joint_pos (T, nj), joint_vel (T, nj), body_pos_w (T, nb, 3),
              body_quat_w (T, nb, 4, wxyz), body_lin_vel_w (T, nb, 3), body_ang_vel_w (T, nb, 3)
  nj = every non-free joint of the robot spec in spec order (our 14 servo joints AND the
       10 backlash joints, which we write as 0),
  nb = every body of the robot spec except world, in spec (depth-first) order,
       exactly what mjlab's Entity.body_names returns.

Root trajectory: taken from the clip's "base" track when present; otherwise synthesised by
holding the home root pose and shifting z each frame so the lowest foot-mesh vertex stays
at the height it has in the first frame (foot pinning, no x/y/yaw motion). That is good
enough for episodic clips that start and end in the standing pose. Walking clips should
carry a measured base track (record_policy_clip.py) or be built through mjlab's own
Simulation the way scripts/csv_to_npz.py does.

Usage:
  python clip_to_mjlab_npz.py --xml <robot.xml> --clip <clip.json> --out <motion.npz>
"""

from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np


def _quat_mul(a, b):
  w1, x1, y1, z1 = a.T
  w2, x2, y2, z2 = b.T
  return np.stack(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    axis=-1,
  )


def _quat_conj(q):
  return q * np.array([1.0, -1.0, -1.0, -1.0])


def _ang_vel_from_quats(q, dt):
  """World-frame angular velocity of a wxyz quaternion sequence (T, 4) -> (T, 3)."""
  q = q / np.linalg.norm(q, axis=-1, keepdims=True)
  # Keep the sequence on one hemisphere so the finite difference is short.
  for i in range(1, len(q)):
    if np.dot(q[i], q[i - 1]) < 0:
      q[i] = -q[i]
  q_prev, q_next = q[:-2], q[2:]
  q_rel = _quat_mul(q_next, _quat_conj(q_prev))  # world-frame delta
  angle = 2.0 * np.arctan2(np.linalg.norm(q_rel[:, 1:], axis=-1), q_rel[:, 0])
  axis = q_rel[:, 1:] / np.maximum(np.linalg.norm(q_rel[:, 1:], axis=-1, keepdims=True), 1e-12)
  omega = axis * angle[:, None] / (2.0 * dt)
  return np.concatenate([omega[:1], omega, omega[-1:]], axis=0)


def _foot_min_z(model, data, foot_geom_ids):
  zs = []
  for gid in foot_geom_ids:
    mid = model.geom_dataid[gid]
    v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    verts = model.mesh_vert[v0 : v0 + nv]
    world = data.geom_xpos[gid] + verts @ data.geom_xmat[gid].reshape(3, 3).T
    zs.append(world[:, 2].min())
  return min(zs)


def convert(xml, clip_path, out, fps=50.0, home_z=0.15, foot_geoms=("left_foot_bottom_tpu", "right_foot_bottom_tpu")):
  model = mujoco.MjModel.from_xml_path(xml)
  data = mujoco.MjData(model)
  dt = 1.0 / fps

  clip = json.load(open(clip_path))
  frames = clip["frames"] if isinstance(clip, dict) else clip
  base_track = clip.get("base") if isinstance(clip, dict) else None
  t_in = np.array([f[0] for f in frames], dtype=np.float64)
  names = list(frames[0][1].keys())
  q_in = np.array([[f[1][n] for n in names] for f in frames], dtype=np.float64)

  # Regular grid at the output rate, linear interpolation (our clips are linear anyway).
  n_out = int(np.floor(t_in[-1] / dt + 1e-6)) + 1
  t_out = np.arange(n_out) * dt
  q_out = np.stack([np.interp(t_out, t_in, q_in[:, k]) for k in range(len(names))], axis=1)

  # Joint layout: every non-free joint in spec order; backlash joints stay at 0.
  jnt_ids = [j for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
  jnt_names = [model.joint(j).name for j in jnt_ids]
  free_j = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
  assert len(free_j) == 1 and free_j[0] == 0, "root freejoint must be the first joint"
  joint_pos = np.zeros((n_out, len(jnt_ids)))
  for k, n in enumerate(names):
    joint_pos[:, jnt_names.index(n)] = q_out[:, k]
  joint_vel = np.gradient(joint_pos, dt, axis=0)

  # Root trajectory.
  if base_track:
    bt = np.array([[*b[0], *b[1]] for b in base_track], dtype=np.float64)
    t_b = t_in[: len(bt)]
    root = np.stack([np.interp(t_out, t_b, bt[:, k]) for k in range(7)], axis=1)
    root_pos, root_quat = root[:, :3], root[:, 3:7]
    root_quat /= np.linalg.norm(root_quat, axis=-1, keepdims=True)
  else:
    root_pos = np.tile([0.0, 0.0, home_z], (n_out, 1))
    root_quat = np.tile([1.0, 0.0, 0.0, 0.0], (n_out, 1))
    foot_ids = [model.geom(g).id for g in foot_geoms]
    ref = None
    for i in range(n_out):
      data.qpos[:] = 0
      data.qpos[:3], data.qpos[3:7] = root_pos[i], root_quat[i]
      data.qpos[7:] = joint_pos[i]
      mujoco.mj_kinematics(model, data)
      z = _foot_min_z(model, data, foot_ids)
      ref = z if ref is None else ref
      root_pos[i, 2] += ref - z
  root_lin_vel = np.gradient(root_pos, dt, axis=0)
  root_ang_vel = _ang_vel_from_quats(root_quat.copy(), dt)

  # Forward kinematics with velocities for every body (spec order, world excluded).
  nb = model.nbody - 1
  body_pos = np.zeros((n_out, nb, 3))
  body_quat = np.zeros((n_out, nb, 4))
  body_lin = np.zeros((n_out, nb, 3))
  body_ang = np.zeros((n_out, nb, 3))
  vel6 = np.zeros(6)
  for i in range(n_out):
    data.qpos[:3], data.qpos[3:7], data.qpos[7:] = root_pos[i], root_quat[i], joint_pos[i]
    data.qvel[:3], data.qvel[3:6], data.qvel[6:] = root_lin_vel[i], root_ang_vel[i], joint_vel[i]
    mujoco.mj_forward(model, data)
    body_pos[i] = data.xpos[1:]
    body_quat[i] = data.xquat[1:]
    for b in range(1, model.nbody):
      mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, vel6, 0)
      body_ang[i, b - 1], body_lin[i, b - 1] = vel6[:3], vel6[3:]

  np.savez(
    out,
    fps=np.array([fps]),
    joint_pos=joint_pos.astype(np.float32),
    joint_vel=joint_vel.astype(np.float32),
    body_pos_w=body_pos.astype(np.float32),
    body_quat_w=body_quat.astype(np.float32),
    body_lin_vel_w=body_lin.astype(np.float32),
    body_ang_vel_w=body_ang.astype(np.float32),
  )
  print(f"wrote {out}: {n_out} frames @ {fps:g} Hz, {len(jnt_ids)} joints, {nb} bodies, base track {'from clip' if base_track else 'synthesised (foot-pinned)'}")
  print("joint order:", jnt_names)
  print("body order:", [model.body(b).name for b in range(1, model.nbody)])
  print("root z range: %.3f .. %.3f m" % (root_pos[:, 2].min(), root_pos[:, 2].max()))


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--xml", required=True)
  ap.add_argument("--clip", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--fps", type=float, default=50.0)
  a = ap.parse_args()
  convert(a.xml, a.clip, a.out, a.fps)
