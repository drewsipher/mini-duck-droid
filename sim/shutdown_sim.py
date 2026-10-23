#!/usr/bin/env python3
"""How should the duck power down on a flat battery? Simulates a few shutdown sequences
from standing (servos 1.5x sim stiffness, 1.9 N.m max, like the real ones) and measures
how hard and which way he lands.  --video out.mp4 renders them side by side."""
import argparse

import mujoco
import numpy as np

from sim_test import PG

XML = "train_pg/playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash_collision.xml"
KP_SCALE, TORQUE = 1.5, 1.9
# crouch: more knee bend, hips and ankles each take half (leg order: yaw, roll, pitch, knee, ankle)
def crouch_delta(depth, head_down=0.0, hip_lean=0.0, ankle_lean=0.0):
    d = np.zeros(14)
    d[2], d[3], d[4] = -depth / 2 + hip_lean, depth, -depth / 2 + ankle_lean     # left leg
    d[11], d[12], d[13] = depth / 2 - hip_lean, depth, -depth / 2 + ankle_lean   # right leg
    d[5] = head_down                                          # neck pitch (+ = tip back... see sign check)
    return d


SEQS = {
    "Motors off while standing": dict(crouch=0.0, ramp=0.0),
    "Plain crouch, then off": dict(crouch=0.2, ramp=0.0),
    "Lean-back crouch, then off": dict(crouch=0.2, ramp=0.0, hip=0.55, ankle=-0.3),
    "Lean-back crouch, fade out": dict(crouch=0.2, ramp=2.0, hip=0.55, ankle=-0.3),
}


def simulate(cfg, render=None, label=""):
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    m.actuator_gainprm[:, 0] *= KP_SCALE
    m.actuator_biasprm[:, 1] *= KP_SCALE
    m.actuator_forcerange[:, :] = [-TORQUE, TORQUE]
    if cfg.get("knee_range"):                                   # what if the real knee bends further?
        for j in ("left_knee", "right_knee"):
            m.jnt_range[m.joint(j).id] = [-cfg["knee_range"], cfg["knee_range"]]
            m.actuator_ctrlrange[m.actuator(j).id] = [-cfg["knee_range"], cfg["knee_range"]]
    if cfg.get("limits"):                                       # measured real joint stops (json)
        import json
        for j, (a, b) in json.load(open(cfg["limits"])).items():
            m.jnt_range[m.joint(j).id] = [a, b]
            m.actuator_ctrlrange[m.actuator(j).id] = [a, b]
    kp0 = m.actuator_gainprm[:, 0].copy()
    home = m.keyframe("home").ctrl.copy()
    d.qpos[:] = m.keyframe("home").qpos
    d.ctrl[:] = home
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    target = np.clip(home + crouch_delta(cfg["crouch"], cfg.get("head", 0.0), cfg.get("hip", 0.0),
                                         cfg.get("ankle", 0.0)), lo, hi)
    target[[3, 12]] = np.minimum(target[[3, 12]], cfg.get("knee_max", 9.0))
    if "neck" in cfg:
        target[5] = cfg["neck"]
    if "head_pitch" in cfg:
        target[6] = cfg["head_pitch"]
    head = m.body("head_assembly").id
    trunk = m.body("trunk_assembly").id
    dt = m.opt.timestep
    T_SETTLE, T_CROUCH, T_HOLD = 1.0, 2.0 if cfg["crouch"] else 0.0, 0.5
    t_off = T_SETTLE + T_CROUCH + T_HOLD
    T_END = t_off + max(cfg["ramp"], 0) + 2.5
    frames, rec = [], {"head_z0": None, "trunk_z0": None, "peak_head_v": 0.0, "peak_trunk_v": 0.0}
    t = 0.0
    step = 0
    while t < T_END:
        if t < T_SETTLE:
            d.ctrl[:] = home
        elif t < T_SETTLE + T_CROUCH:
            a = (t - T_SETTLE) / T_CROUCH
            a = 0.5 - 0.5 * np.cos(np.pi * a)                       # smooth ease in/out
            d.ctrl[:] = home + a * (target - home)
        else:
            d.ctrl[:] = target
        if t >= t_off:                                              # fade kp, then off
            f = 0.0 if cfg["ramp"] <= 0 else max(0.0, 1 - (t - t_off) / cfg["ramp"])
            m.actuator_gainprm[:, 0] = kp0 * f
            m.actuator_biasprm[:, 1] = -kp0 * f
        mujoco.mj_step(m, d)
        if rec["head_z0"] is None and t >= t_off:
            rec["head_z0"], rec["trunk_z0"] = d.xpos[head, 2], d.xpos[trunk, 2]
            rec["tilt_at_off"] = float(np.degrees(np.arccos(np.clip(d.xmat[trunk].reshape(3, 3)[2, 2], -1, 1))))
        if t >= T_SETTLE:
            vh = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, head, vh, 0)
            vt = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, trunk, vt, 0)
            rec["peak_head_v"] = max(rec["peak_head_v"], np.linalg.norm(vh[3:]))
            rec["peak_trunk_v"] = max(rec["peak_trunk_v"], np.linalg.norm(vt[3:]))
        if render is not None and step % 20 == 0:                  # 500 Hz physics -> 25 fps
            frames.append(render(m, d, label, t, t_off, cfg))
        t += dt
        step += 1
    R = d.xmat[trunk].reshape(3, 3)
    up = R[:, 2]
    rec["final_tilt_deg"] = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
    fwd = R[:, 0]
    rec["fell_toward"] = ("forward" if fwd[2] < -0.3 else "backward" if fwd[2] > 0.3 else
                          "sideways" if rec["final_tilt_deg"] > 45 else "stayed upright-ish")
    rec["head_drop_cm"] = 100 * (rec["head_z0"] - d.xpos[head, 2])
    rec["head_final_cm"] = 100 * d.xpos[head, 2]
    rec["trunk_z_start_cm"] = 100 * rec["trunk_z0"]
    rec["trunk_final_cm"] = 100 * d.xpos[trunk, 2]
    rec["trunk_min_before_off_cm"] = rec["trunk_z0"] * 100
    return rec, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    a = ap.parse_args()
    render = None
    if a.video:
        from PIL import Image, ImageDraw, ImageFont
        W, H = 400, 360
        F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
        S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        rs = {}

        def render(m, d, label, t, t_off, cfg):
            r = rs.get(id(m))
            if r is None:
                r = rs[id(m)] = mujoco.Renderer(m, H, W)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0, 0, 0.12]
            cam.distance, cam.azimuth, cam.elevation = 0.8, 120, -10
            r.update_scene(d, cam)
            img = Image.fromarray(r.render())
            dr = ImageDraw.Draw(img)
            dr.rectangle([0, 0, W, 26], fill=(20, 20, 20)); dr.text((8, 4), label, font=F, fill=(255, 255, 255))
            if t < 1.0:
                st = "standing"
            elif t < t_off:
                st = "crouching" if cfg["crouch"] else "standing"
            elif cfg["ramp"] and t < t_off + cfg["ramp"]:
                st = f"motors fading: {100 * max(0, 1 - (t - t_off) / cfg['ramp']):.0f}%"
            else:
                st = "MOTORS OFF"
            dr.rectangle([0, H - 24, W, H], fill=(20, 20, 20)); dr.text((8, H - 21), f"{t:4.1f}s  {st}", font=S, fill=(255, 210, 90))
            return np.asarray(img)
    cols = []
    for name, cfg in SEQS.items():
        rec, frames = simulate(cfg, render, name)
        print(f"{name:32s} head peak {rec['peak_head_v']:.2f} m/s, trunk peak {rec['peak_trunk_v']:.2f} m/s, "
              f"head drop {rec['head_drop_cm']:.1f} cm (ends {rec['head_final_cm']:.1f} cm up), "
              f"tilt {rec['final_tilt_deg']:.0f} deg -> {rec['fell_toward']}")
        cols.append(frames)
    if a.video:
        import imageio.v2 as imageio
        n = max(len(c) for c in cols)
        cols = [c + [c[-1]] * (n - len(c)) for c in cols]
        top = [np.concatenate([cols[0][i], cols[1][i]], 1) for i in range(n)]
        bot = [np.concatenate([cols[2][i], cols[3][i]], 1) for i in range(n)]
        with imageio.get_writer(a.video, fps=25, codec="libx264", quality=7, macro_block_size=16,
                                ffmpeg_params=["-pix_fmt", "yuv420p"]) as w:
            for i in range(n):
                w.append_data(np.concatenate([top[i], bot[i]], 0))
        print("saved", a.video)


if __name__ == "__main__":
    main()
