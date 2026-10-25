"""Make a symmetric walking loop clip: the trunk travels forward, planted feet stay put and flat.

    sim/.venv/bin/python anim/make_walk.py out.json [--stride 0.10] [--period 50] [--lift 0.016] ...

The gait is planned in the world (trunk path, two foot paths half a cycle apart, the right foot
the mirror of the left) and the 14 joints come from IK on the MuJoCo model: each sole goes to its
planned point, flat and pointing forward. Import the clip into Blender with blender_duck.import_clip.

One cycle = `period` steps at 50 Hz; the clip has period+1 frames, the last one equals the first
with the trunk and feet moved forward by `stride` (the loop convention of clip_check --loop and the
trainer). Per foot: swing for `swing` of the cycle (lift and move forward one stride), stance for
the rest (fixed on the floor). The trunk moves at constant speed, sways toward the standing foot
and bobs up a little in single support; the head rolls with the sway.

The duck has no ankle roll, so a planted foot only stays flat when the trunk barely sways: the
sway is kept small (the balance comes from stepping, as in the upstream walks) and the feet sit
a little wider than home (0.091 m), where the hip roll that keeps them flat is close to zero.
"""
import argparse
import json
import math
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "model", "scene_flat_terrain_backlash.xml")
FPS = 50
LEGS = {"L": ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle"),
        "R": ("right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")}
PAD = {"L": "left_foot_bottom_tpu", "R": "right_foot_bottom_tpu"}
HEAD = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
FLAT = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])   # pad axes in the world: thickness up, length forward


def plan(a):
    """Per frame: trunk xyz, sole target xyz per foot, head roll."""
    n = a.period
    phase = np.arange(n + 1) / n
    trunk = np.stack([a.stride * phase,
                      a.sway * np.cos(2 * np.pi * (phase - (a.swing + 1) / 2)),          # toward the left foot mid left stance
                      a.height + a.bob * np.cos(4 * np.pi * (phase - a.swing / 2))], 1)    # up in single support
    feet = {}
    for side, off, y in (("L", 0.0, a.width), ("R", 0.5, -a.width)):
        mid = a.swing + (1 - a.swing) / 2                    # stance middle, in foot phase
        pts = []
        for ph in phase:
            c = math.floor(ph - off + 1e-9)
            s = ph - off - c
            land = a.stride * (c + off + mid) + a.foot_x    # where this cycle's stance is (foot under the hip at its middle)
            if s < a.swing:
                u = s / a.swing
                x = land - a.stride * 0.5 * (1 + math.cos(math.pi * u))
                z = a.lift * 0.5 * (1 - math.cos(2 * math.pi * u))
            else:
                x, z = land, 0.0
            pts.append((x, y, z + a.clear))
        feet[side] = np.array(pts)
    head_roll = a.head_roll * np.cos(2 * np.pi * (phase - (a.swing + 1) / 2))
    return phase, trunk, feet, head_roll


def solve(a):
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    d.qpos[:] = m.keyframe("home").qpos
    jid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(m.njnt)}
    phase, trunk, feet, head_roll = plan(a)
    out, worst = [], {"pos_m": 0.0, "tilt_rad": 0.0}
    for k in range(len(phase)):
        d.qpos[0:3] = trunk[k]
        d.qpos[3:7] = (1, 0, 0, 0)
        for j in HEAD:
            d.qpos[m.jnt_qposadr[jid[j]]] = 0.0
        d.qpos[m.jnt_qposadr[jid["head_roll"]]] = head_roll[k]
        for side, names in LEGS.items():
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, PAD[side])
            qa = np.array([m.jnt_qposadr[jid[j]] for j in names])
            da = np.array([m.jnt_dofadr[jid[j]] for j in names])
            lo, hi = m.jnt_range[[jid[j] for j in names]].T
            sole_local = np.array([a.sole, 0.0, 0.0])
            for _ in range(200):
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                R = d.geom_xmat[g].reshape(3, 3)
                p = d.geom_xpos[g] + R @ sole_local
                ep = feet[side][k] - p
                qe = np.zeros(4)
                mujoco.mju_mat2Quat(qe, (FLAT @ R.T).flatten())
                er = 2 * qe[1:] * np.sign(qe[0] or 1)
                jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jp, jr, p, m.geom_bodyid[g])
                J = np.vstack([jp[:, da], a.rot_w * jr[:, da]])
                e = np.concatenate([ep, a.rot_w * er])
                if np.abs(ep).max() < 1e-5 and np.abs(er).max() < 1e-4:
                    break
                dq = np.linalg.solve(J.T @ J + 1e-6 * np.eye(5), J.T @ e)
                d.qpos[qa] = np.clip(d.qpos[qa] + dq, lo, hi)
            worst["pos_m"] = max(worst["pos_m"], float(np.abs(ep).max()))
            worst["tilt_rad"] = max(worst["tilt_rad"], float(math.acos(min(1.0, R[2, 0]))))
        q = {j: round(float(d.qpos[m.jnt_qposadr[jid[j]]]), 4) for j in LEGS["L"] + HEAD + LEGS["R"]}
        out.append([round(k / FPS, 3), q])
    base = [[[round(float(x), 4) for x in t], [1.0, 0.0, 0.0, 0.0]] for t in trunk]
    return out, base, worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--period", type=int, default=50, help="steps per cycle at 50 Hz (both feet step once)")
    ap.add_argument("--stride", type=float, default=0.10, help="m the trunk (and each foot) moves per cycle")
    ap.add_argument("--swing", type=float, default=0.4, help="fraction of the cycle each foot is in the air")
    ap.add_argument("--lift", type=float, default=0.016, help="m the sole rises mid swing")
    ap.add_argument("--sway", type=float, default=0.004, help="m the trunk leans toward the standing foot")
    ap.add_argument("--bob", type=float, default=0.002, help="m the trunk rises in single support")
    ap.add_argument("--height", type=float, default=0.170, help="m trunk origin above the floor")
    ap.add_argument("--width", type=float, default=0.091, help="m from the centre line to each sole")
    ap.add_argument("--foot-x", type=float, default=-0.0164, help="m sole ahead of the trunk origin mid stance")
    ap.add_argument("--head-roll", type=float, default=0.06, help="rad the head rolls with the sway")
    ap.add_argument("--sole", type=float, default=-0.0046, help="pad-frame x of the sole face")
    ap.add_argument("--clear", type=float, default=0.0003, help="m the planted sole sits above the floor")
    ap.add_argument("--rot-w", type=float, default=0.05, help="m per rad: how much keeping the foot flat counts")
    a = ap.parse_args()
    frames, base, worst = solve(a)
    clip = {"frames": frames, "interp": "linear", "base": base,
            "authored_in": {"tool": "anim/make_walk.py", "fps": FPS, "loop": True,
                            "params": {k: v for k, v in vars(a).items() if k != "out"}, "ik_worst": worst}}
    with open(a.out, "w") as fh:
        json.dump(clip, fh)
    v = a.stride / (a.period / FPS)
    print(f"{a.out}: {len(frames)} frames, cycle {a.period / FPS:.2f} s, {v:.3f} m/s, IK worst {worst}")


if __name__ == "__main__":
    main()
