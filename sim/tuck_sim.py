"""Sit-back shutdown poses with the real joint limits: ease into the pose (motors on),
note what touches the floor, then motors off and see where he ends up."""
import sys, os, json, itertools
sys.path.insert(0, "/home/hermes/src/mini-duck-droid/sim"); os.chdir("/home/hermes/src/mini-duck-droid/sim")
import numpy as np, mujoco
XML = "train_pg/playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash_collision.xml"
LIM = json.load(open("/home/hermes/src/mini-duck-droid/joint_limits/real_limits_20261014.json"))

def build(kp_scale, torque):
    m = mujoco.MjModel.from_xml_path(XML)
    m.actuator_gainprm[:, 0] *= kp_scale; m.actuator_biasprm[:, 1] *= kp_scale
    m.actuator_forcerange[:] = [-torque, torque]
    for j, (a, b) in LIM.items():
        m.jnt_range[m.joint(j).id] = [a, b]; m.actuator_ctrlrange[m.actuator(j).id] = [a, b]
    return m

def pose(m, hip, knee, ankle, neck, head_pitch):
    """hip/knee/ankle are absolute left-leg angles (right leg mirrored for hip)."""
    t = m.keyframe("home").ctrl.copy()
    t[2], t[3], t[4] = hip, knee, ankle
    t[11], t[12], t[13] = -hip, knee, ankle
    t[5], t[6] = neck, head_pitch
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    return np.clip(t, lo, hi)

def run(p, kp_scale=1.5, torque=1.9, frames=None, render=None, T_CROUCH=3.0, bow=None, T_BOW=2.0):
    m = build(kp_scale, torque); d = mujoco.MjData(m)
    d.qpos[:] = m.keyframe("home").qpos; home = m.keyframe("home").ctrl.copy(); d.ctrl[:] = home
    tgt = pose(m, **p)
    trunk, head = m.body("trunk_assembly").id, m.body("head_assembly").id
    floor = m.geom("floor").id
    kp0 = m.actuator_gainprm[:, 0].copy()
    T1, T2, T3 = 1.0, 1.0 + T_CROUCH, 1.0 + T_CROUCH + 1.0
    if bow is not None:                                 # second stage: lean forward onto the chest, motors on
        tgt2 = pose(m, **dict(p, **bow)); T3 += T_BOW
    T_END = T3 + 3.0
    t, step, r = 0.0, 0, {"peak_head_v": 0.0}
    while t < T_END:
        if t < T1: d.ctrl[:] = home
        elif t < T2:
            a = (t - T1) / (T2 - T1); a = 0.5 - 0.5 * np.cos(np.pi * a); d.ctrl[:] = home + a * (tgt - home)
        elif bow is not None and t < T3 - 1.0:
            a = min(1.0, (t - T2) / T_BOW); a = 0.5 - 0.5 * np.cos(np.pi * a); d.ctrl[:] = tgt + a * (tgt2 - tgt)
        elif bow is not None: d.ctrl[:] = tgt2
        else: d.ctrl[:] = tgt
        if t >= T3 and "off" not in r:
            r["off"] = True; m.actuator_gainprm[:, 0] = 0; m.actuator_biasprm[:, 1] = 0
            r["touch_at_off"] = sorted({m.body(m.geom_bodyid[c.geom1 if c.geom2 == floor else c.geom2]).name
                                        for c in d.contact[:d.ncon] if floor in (c.geom1, c.geom2)})
            r["trunk_z_off"] = 100 * d.xpos[trunk, 2]
            r["pitch_off"] = pitch(d, trunk)
            r["holding_err"] = float(np.degrees(np.max(np.abs(d.qpos[7:][:0] if False else 0))))
        if t >= T1:
            v = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, head, v, 0)
            r["peak_head_v"] = max(r["peak_head_v"], float(np.linalg.norm(v[3:])))
            if t >= T3:
                r["peak_head_v_off"] = max(r.get("peak_head_v_off", 0.0), float(np.linalg.norm(v[3:])))
        if render is not None and step % 20 == 0:
            frames.append(render(m, d, t, T2, T3))
        mujoco.mj_step(m, d); t += m.opt.timestep; step += 1
    r["pitch_end"] = pitch(d, trunk)
    R = d.xmat[trunk].reshape(3, 3); r["tilt_end"] = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
    Rh = d.xmat[head].reshape(3, 3); face = -Rh[:, 2]
    r["face_deg"] = float(np.degrees(np.arcsin(np.clip(face[2], -1, 1))))   # + = looking up
    r["face_fwd"] = float(face[0])                                          # 1 = facing straight ahead, -1 = back
    r["head_up"] = float(Rh[2, 0])                                          # 1 = head upright, -1 = upside down
    r["head_cm"] = 100 * d.xpos[head, 2]; r["trunk_cm"] = 100 * d.xpos[trunk, 2]
    r["touch_end"] = sorted({m.body(m.geom_bodyid[c.geom1 if c.geom2 == floor else c.geom2]).name
                             for c in d.contact[:d.ncon] if floor in (c.geom1, c.geom2)})
    r["moving_end"] = float(np.linalg.norm(d.qvel[:6]))
    return r

def pitch(d, b):
    R = d.xmat[b].reshape(3, 3)
    return float(np.degrees(np.arctan2(R[2, 0], R[2, 2])))   # + = nose down? see sign check

if __name__ == "__main__":
    res = []
    for knee, ankle, hip, neck in itertools.product([1.73], [-1.18, -1.0, -0.8], [-0.2, 0.0, 0.29, -0.6, -1.0],
                                                     [-0.74, -0.45]):
        p = dict(hip=hip, knee=knee, ankle=ankle, neck=neck, head_pitch=min(1.1, -neck * 1.0 + 0.2))
        rows = [run(p, k, tq) for k, tq in [(1.2, 1.6), (1.5, 1.9), (1.8, 2.4)]]
        r = rows[1]
        hv = max(x["peak_head_v"] for x in rows)
        print(f"hip {hip:+.2f} knee {knee} ank {ankle:+.2f} neck {neck:+.2f} | at off: trunk {r['trunk_z_off']:4.1f}cm pitch {r['pitch_off']:+4.0f} "
              f"touch {[s[:10] for s in r['touch_at_off']]} | end: pitch {min(x['pitch_end'] for x in rows):+4.0f}..{max(x['pitch_end'] for x in rows):+4.0f} "
              f"head {r['head_cm']:4.1f}cm face {r['face_deg']:+4.0f} headv {hv:.2f}", flush=True)
        res.append(dict(p=p, rows=rows))
    json.dump(res, open("/tmp/claude-1001/-home-hermes-open-duck-mini/3cab2012-8b52-49f2-99b2-39034d3a243a/scratchpad/sit_scan.json", "w"))
