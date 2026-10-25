"""Make a symmetric walking loop clip: the trunk travels forward and waddles, planted feet stay put.

    sim/.venv/bin/python anim/make_walk.py out.json [--stride 0.10] [--period 50] [--sway 0.015] ...

The gait is planned in the world (trunk path, two foot paths half a cycle apart, the right foot
the mirror of the left) and the 14 joints come from whole-body IK on the MuJoCo model. Import the
clip into Blender with blender_duck.import_clip.

One cycle = `period` steps at 50 Hz; the clip has period+1 frames, the last one equals the first
with the trunk and feet moved forward by `stride` (the loop convention of clip_check --loop and the
trainer). Per foot: swing for `swing` of the cycle (lift and move forward one stride), stance for
the rest (fixed on the floor). The trunk moves forward at constant speed and bobs a little.

Sideways it waddles like the BD-X droids: the centre of mass swings `sway` toward the standing
foot. The duck has no ankle roll, so the IK gets there by rolling the trunk over the standing hip
(the far hip rises, which helps the swing foot clear) and by letting the standing foot roll a
little onto its inner edge; `--flat` sets how much keeping that foot flat counts against the sway.
The trunk's sideways position and roll are solved, not planned. The head undoes `head_level` of
the trunk roll. Last, each frame is raised so the lowest point of the feet just touches the floor.
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
    """Per frame: trunk x and z, centre-of-mass y, sole target xyz and stance flag per foot."""
    n = a.period
    phase = np.arange(n + 1) / n
    trunk_x = a.stride * phase
    trunk_z = a.height + a.bob * np.cos(4 * np.pi * (phase - a.swing / 2))          # up in single support
    com_y = a.sway * np.cos(2 * np.pi * (phase - (a.swing + 1) / 2))                  # over the left foot mid left stance
    feet, stance = {}, {}
    for side, off, y in (("L", 0.0, a.width), ("R", 0.5, -a.width)):
        mid = a.swing + (1 - a.swing) / 2                    # stance middle, in foot phase
        pts, st = [], []
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
            st.append(s >= a.swing or s == 0.0)
        feet[side], stance[side] = np.array(pts), st
    return phase, trunk_x, trunk_z, com_y, feet, stance


def solve(a):
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    d.qpos[:] = m.keyframe("home").qpos
    jid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(m.njnt)}
    phase, trunk_x, trunk_z, com_y, feet, stance = plan(a)
    legs = [jid[j] for side in "LR" for j in LEGS[side]]
    dofs = np.array([1, 3] + [m.jnt_dofadr[j] for j in legs])          # trunk y, trunk roll, the 10 leg joints
    qa = np.array([m.jnt_qposadr[j] for j in legs])
    lo, hi = m.jnt_range[legs].T
    pads = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, PAD[s]) for s in "LR"}
    sole_local = np.array([a.sole, 0.0, 0.0])
    mujoco.mj_kinematics(m, d)
    head_sign = -np.sign(d.joint("head_roll").xaxis[0])               # head roll that undoes a positive trunk roll
    frames, worst = [], {"pos_m": 0.0, "stance_roll_rad": 0.0, "trunk_roll_rad": 0.0, "com_y_m": 0.0}
    for k in range(len(phase)):
        d.qpos[0], d.qpos[2] = trunk_x[k], trunk_z[k]
        for j in HEAD:
            d.qpos[m.jnt_qposadr[jid[j]]] = 0.0
        for _ in range(300):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            rows, errs = [], []
            for s in "LR":
                g = pads[s]
                R = d.geom_xmat[g].reshape(3, 3)
                p = d.geom_xpos[g] + R @ sole_local
                qe = np.zeros(4)
                mujoco.mju_mat2Quat(qe, (FLAT @ R.T).flatten())
                er = 2 * qe[1:] * np.sign(qe[0] or 1)
                jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jp, jr, p, m.geom_bodyid[g])
                w = a.flat if stance[s][k] else a.swing_flat
                rows += [10 * jp[:, dofs], w * jr[:, dofs]]
                errs += [10 * (feet[s][k] - p), w * er]
            jc = np.zeros((3, m.nv))
            mujoco.mj_jacSubtreeCom(m, d, jc, 1)
            rows.append(jc[1:2, dofs])
            errs.append([com_y[k] - d.subtree_com[1][1]])
            reg = np.zeros((1, len(dofs))); reg[0, 1] = a.roll_reg                 # prefer a level trunk
            roll = 2 * math.atan2(d.qpos[4], d.qpos[3])
            rows.append(reg); errs.append([-a.roll_reg * roll])
            J, e = np.vstack(rows), np.concatenate(errs)
            dq = np.linalg.solve(J.T @ J + 1e-7 * np.eye(len(dofs)), J.T @ e)
            v = np.zeros(m.nv); v[dofs] = dq
            mujoco.mj_integratePos(m, d.qpos, v, 1.0)
            d.qpos[qa] = np.clip(d.qpos[qa], lo, hi)
            if np.abs(dq).max() < 1e-7:
                break
        d.qpos[m.jnt_qposadr[jid["head_roll"]]] = head_sign * a.head_level * roll
        # drop the duck so the lowest point of either pad sits `clear` above the floor
        mujoco.mj_kinematics(m, d)
        low = min((d.geom_xpos[g] + m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:][:m.mesh_vertnum[m.geom_dataid[g]]]
                   @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min() for g in pads.values())
        d.qpos[2] += a.clear - low
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        for s in "LR":
            R = d.geom_xmat[pads[s]].reshape(3, 3)
            p = d.geom_xpos[pads[s]] + R @ sole_local
            worst["pos_m"] = max(worst["pos_m"], float(np.abs((feet[s][k] - p)[:2]).max()))
            if stance[s][k]:
                worst["stance_roll_rad"] = max(worst["stance_roll_rad"], abs(float(math.asin(R[1, 0]))))
        worst["trunk_roll_rad"] = max(worst["trunk_roll_rad"], abs(roll))
        worst["com_y_m"] = max(worst["com_y_m"], abs(float(d.subtree_com[1][1])))
        q = {j: round(float(d.qpos[m.jnt_qposadr[jid[j]]]), 4) for j in LEGS["L"] + HEAD + LEGS["R"]}
        frames.append([round(k / FPS, 3), q, [[round(float(x), 4) for x in d.qpos[0:3]], [round(float(x), 4) for x in d.qpos[3:7]]]])
    worst = {k: round(v, 4) for k, v in worst.items()}
    return [f[:2] for f in frames], [f[2] for f in frames], worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--period", type=int, default=50, help="steps per cycle at 50 Hz (both feet step once)")
    ap.add_argument("--stride", type=float, default=0.10, help="m the trunk (and each foot) moves per cycle")
    ap.add_argument("--swing", type=float, default=0.4, help="fraction of the cycle each foot is in the air")
    ap.add_argument("--lift", type=float, default=0.016, help="m the sole rises mid swing")
    ap.add_argument("--sway", type=float, default=0.015, help="m the centre of mass swings toward the standing foot")
    ap.add_argument("--flat", type=float, default=0.03, help="m per rad: how much keeping the standing foot flat counts")
    ap.add_argument("--swing-flat", type=float, default=0.02, help="the same for the foot in the air")
    ap.add_argument("--roll-reg", type=float, default=0.005, help="m per rad: preference for a level trunk")
    ap.add_argument("--bob", type=float, default=0.002, help="m the trunk rises in single support")
    ap.add_argument("--height", type=float, default=0.170, help="m trunk origin above the floor (before the drop)")
    ap.add_argument("--width", type=float, default=0.091, help="m from the centre line to each sole")
    ap.add_argument("--foot-x", type=float, default=-0.0164, help="m sole ahead of the trunk origin mid stance")
    ap.add_argument("--head-level", type=float, default=0.5, help="fraction of the trunk roll the head undoes")
    ap.add_argument("--sole", type=float, default=-0.0046, help="pad-frame x of the sole face")
    ap.add_argument("--clear", type=float, default=0.0003, help="m the lowest foot point sits above the floor")
    a = ap.parse_args()
    frames, base, worst = solve(a)
    clip = {"frames": frames, "interp": "linear", "base": base,
            "authored_in": {"tool": "anim/make_walk.py", "fps": FPS, "loop": True,
                            "params": {k: v for k, v in vars(a).items() if k != "out"}, "ik_worst": worst}}
    with open(a.out, "w") as fh:
        json.dump(clip, fh)
    v = a.stride / (a.period / FPS)
    print(f"{a.out}: {len(frames)} frames, cycle {a.period / FPS:.2f} s, {v:.3f} m/s, worst {worst}")


if __name__ == "__main__":
    main()
