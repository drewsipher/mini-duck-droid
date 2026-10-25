#!/usr/bin/env python3
"""Stage 2 gate: play a clip kinematically on the duck's MJCF and report what would stop it being
a usable reference: floor penetration, self-collision, joint-limit and servo-speed breaches, and
whether it starts and ends in the standing pose. Optionally renders the playback.

    cd sim && .venv/bin/python ../anim/clip_check.py ../tracker/data/clips/bow.json [--video out.mp4]
        [--max-vel 4.0] [--home-tol 0.15] [--json report.json]

The clip's "base" track (written by the Blender exporter) places the trunk; without it the trunk
is held at the home height and the lowest foot is dropped to the floor each frame, so penetration
then means "the feet are not level with each other", which is still worth knowing.
Exit code 0 = pass, 1 = fail, so it can gate a pipeline.
"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PG = os.path.join(HERE, "..", "sim", "train_pg", "playground", "open_duck_mini_v2")
XML = os.path.join(PG, "xmls", "scene_flat_terrain_backlash.xml")
if not os.path.exists(XML):  # a clone without the training repo: use the copy shipped in anim/model
    XML = os.path.join(HERE, "model", "scene_flat_terrain_backlash.xml")
FPS = 50


def load_frames(path):
    data = json.load(open(path))
    if isinstance(data, list):
        return data, None
    if "frames" in data:
        return data["frames"], data.get("base")
    sys.path.insert(0, os.path.join(HERE, "..", "sim", "train_pg"))
    raise SystemExit("keyed clips: export them from Blender or the puppet tool first")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--video")
    ap.add_argument("--max-vel", type=float, default=4.0, help="servo speed limit rad/s (deploy MAX_MOTOR_VEL)")
    ap.add_argument("--home-tol", type=float, default=0.15, help="rad: start/end pose must be this close to home")
    ap.add_argument("--penetration", type=float, default=0.003, help="m of floor penetration tolerated")
    ap.add_argument("--json")
    ap.add_argument("--loop", action="store_true", help="cyclic clip: end must match start, not home")
    a = ap.parse_args()
    frames, base_track = load_frames(a.clip)
    for p in (a.video, a.json):
        if p:
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    home = m.keyframe("home").qpos.copy()
    jid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(m.njnt)}
    names = list(frames[0][1].keys())
    qadr = np.array([m.jnt_qposadr[jid[n]] for n in names])
    lo = np.array([m.jnt_range[jid[n]][0] for n in names]); hi = np.array([m.jnt_range[jid[n]][1] for n in names])
    homeq = home[qadr]
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_geoms = [g for g in range(m.ngeom) if "foot" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")]
    Q = np.array([[f[1][n] for n in names] for f in frames])
    T = np.array([f[0] for f in frames])
    dt = np.diff(T) if len(T) > 1 else np.array([1 / FPS])
    vel = np.abs(np.diff(Q, axis=0)) / np.maximum(dt[:, None], 1e-6) if len(Q) > 1 else np.zeros((1, len(names)))
    rep = {"clip": a.clip, "frames": len(frames), "seconds": float(T[-1] - T[0]), "joints": names}
    # limits and speed
    over = np.clip(Q - hi, 0, None) + np.clip(lo - Q, 0, None)
    rep["limit_breach_rad"] = {n: float(over[:, i].max()) for i, n in enumerate(names) if over[:, i].max() > 1e-4}
    rep["peak_vel_rad_s"] = {n: round(float(vel[:, i].max()), 2) for i, n in enumerate(names)}
    rep["over_speed"] = {n: round(float(vel[:, i].max()), 2) for i, n in enumerate(names) if vel[:, i].max() > a.max_vel}
    # where it happens: clip frame indices (0 = first exported frame), the later frame of each too-fast step
    rep["over_speed_frames"] = {n: [int(k) + 1 for k in np.where(vel[:, i] > a.max_vel)[0]] for i, n in enumerate(names)
                                if vel[:, i].max() > a.max_vel}
    rep["limit_breach_frames"] = {n: [int(k) for k in np.where(over[:, i] > 1e-4)[0]] for i, n in enumerate(names)
                                  if over[:, i].max() > 1e-4}
    # pose at both ends
    rep["start_vs_home_rad"] = float(np.abs(Q[0] - homeq).max())
    rep["end_vs_home_rad"] = float(np.abs(Q[-1] - homeq).max())
    rep["end_vs_start_rad"] = float(np.abs(Q[-1] - Q[0]).max())
    # kinematic playback: penetration and self-collision
    worst_pen, self_hits, min_clear = 0.0, {}, 1e9
    pen_frames, self_frames = {}, {}                # frame index -> depth (m) / body pairs
    renderer = cam = writer = None
    if a.video:
        try:  # the video is a convenience: without imageio-ffmpeg or OpenGL the check still runs
            import imageio.v2 as imageio
            writer = imageio.get_writer(a.video, fps=25, codec="libx264", quality=7, macro_block_size=16, ffmpeg_params=["-pix_fmt", "yuv420p"])
            renderer = mujoco.Renderer(m, 400, 480)
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
            cam.distance, cam.azimuth, cam.elevation = 0.9, 140, -12
        except Exception as e:
            print(f"no video ({type(e).__name__}: {str(e).splitlines()[0]}); for video: pip install imageio imageio-ffmpeg", file=sys.stderr)
            if writer:
                writer.close()
            renderer = writer = None
            rep["video_error"] = str(e).splitlines()[0]
    for k, f in enumerate(frames):
        d.qpos[:] = home
        d.qpos[qadr] = Q[k]
        if base_track and k < len(base_track):
            d.qpos[0:3] = base_track[k][0]; d.qpos[3:7] = base_track[k][1]
        mujoco.mj_kinematics(m, d)
        if not base_track:                       # drop the duck so its lowest foot point touches the floor
            mujoco.mj_collision(m, d)
            lowest = min((d.contact[i].dist for i in range(d.ncon) if floor in (d.contact[i].geom1, d.contact[i].geom2)), default=None)
            if lowest is None:
                lowest = min(d.geom_xpos[g][2] - m.geom_rbound[g] for g in foot_geoms)
            d.qpos[2] -= lowest
            mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if floor in (g1, g2):
                worst_pen = max(worst_pen, -c.dist)
                if -c.dist > a.penetration:
                    pen_frames[k] = max(pen_frames.get(k, 0.0), float(-c.dist))
            else:
                b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g1]); b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g2])
                if c.dist < 0 and m.body_parentid[m.geom_bodyid[g1]] != m.geom_bodyid[g2] and m.body_parentid[m.geom_bodyid[g2]] != m.geom_bodyid[g1]:
                    key = f"{b1} x {b2}"
                    self_hits[key] = max(self_hits.get(key, 0.0), float(-c.dist))
                    self_frames.setdefault(k, set()).add(key)
        if writer and k % 2 == 0:
            renderer.update_scene(d, cam); writer.append_data(renderer.render())
    if writer:
        writer.close(); renderer.close()
    rep["floor_penetration_m"] = round(float(worst_pen), 4)
    rep["self_collisions_m"] = {k: round(v, 4) for k, v in self_hits.items()}
    rep["floor_penetration_frames"] = {int(k): round(v, 4) for k, v in sorted(pen_frames.items())}
    rep["self_collision_frames"] = {int(k): sorted(v) for k, v in sorted(self_frames.items())}
    fails = []
    if rep["limit_breach_rad"]: fails.append("joint limits")
    if rep["over_speed"]: fails.append(f"servo speed > {a.max_vel} rad/s")
    if worst_pen > a.penetration: fails.append("floor penetration")
    if self_hits: fails.append("self-collision")
    if a.loop:
        if rep["end_vs_start_rad"] > a.home_tol: fails.append("loop does not close")
    else:
        if rep["start_vs_home_rad"] > a.home_tol: fails.append("does not start at home")
        if rep["end_vs_home_rad"] > a.home_tol: fails.append("does not end at home")
    rep["fails"] = fails
    rep["pass"] = not fails
    print(json.dumps(rep, indent=1))
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1)
    if a.video and "video_error" not in rep:
        print("video", a.video)
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
