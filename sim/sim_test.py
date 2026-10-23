#!/usr/bin/env python3
"""Run the duck's walking policy in MuJoCo, headless, with scripted commands, and report
what the simulated duck actually does (distance moved, turn, feet, head).

    .venv/bin/python sim_test.py                     # standard test set
    .venv/bin/python sim_test.py --accel-bias 0      # without the +1.3 forward accel offset

Observation and control follow Open_Duck_Playground's mujoco_infer.py (the trainer's own
reference for running an .onnx in MuJoCo).
"""
import argparse
import os
import pickle

import mujoco
import numpy as np
import onnxruntime as ort

PG = "Open_Duck_Playground/playground/open_duck_mini_v2"
JOINTS = ["L hip_yaw", "L hip_roll", "L hip_pitch", "L knee", "L ankle", "neck", "head_pitch",
          "head_yaw", "head_roll", "R hip_yaw", "R hip_roll", "R hip_pitch", "R knee", "R ankle"]


class Sim:
    def __init__(self, onnx, xml, accel_bias=1.3, action_scale=0.25, phase_factor=1.0,
                 nb_steps=None, extra=None):
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.m.opt.timestep = 0.002
        self.d = mujoco.MjData(self.m)
        self.decim = 10
        self.policy = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
        self.accel_bias = accel_bias
        self.action_scale = action_scale
        self.phase_factor = phase_factor
        self.extra = extra or {}
        ref = pickle.load(open(f"{PG}/data/polynomial_coefficients.pkl", "rb"))
        first = next(iter(ref.values()))
        self.nb_steps = nb_steps or int(first["period"] * first["fps"])
        # A brain trained on a clip reference (REF_CLIP, 10-22) has the clip's gait clock, and
        # with REF_HYBRID=1 a clock whose period follows the command. Same env vars as training.
        self.ref = None
        if os.environ.get("REF_CLIP") and nb_steps is None and self.extra.get("use_ref", 1) != 0:   # use_ref=0: a brain from before the clips
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_pg"))
            from playground.common.clip_reference_motion import ClipReferenceMotion, HybridReferenceMotion
            if os.environ.get("REF_HYBRID", "0") == "1":
                self.ref = HybridReferenceMotion(os.environ["REF_CLIP"], None)
            else:
                self.ref = ClipReferenceMotion(os.environ["REF_CLIP"], None)
            self.nb_steps = self.ref.nb_steps_in_period
            print(f"[sim] gait clock from the reference clip: {self.nb_steps} steps"
                  + (" (hybrid: period follows the command)" if hasattr(self.ref, "period_for") else ""))
        self.phase_f = 0.0
        m = self.m
        self.act_names = [m.actuator(k).name for k in range(m.nu)]
        self.qadr = np.array([m.joint(n).qposadr[0] for n in self.act_names])
        self.vadr = np.array([m.joint(n).dofadr[0] for n in self.act_names])
        s = lambda n: m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)]
        self.gyro_a, self.acc_a = s("gyro"), s("accelerometer")
        self.default = m.keyframe("home").ctrl.copy()
        self.d.qpos[:] = m.keyframe("home").qpos
        self.d.ctrl[:] = self.default
        mujoco.mj_forward(m, self.d)
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.feet = [m.body("foot_assembly").id, m.body("foot_assembly_2").id]
        self.base = m.body("trunk_assembly").id if mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, "trunk_assembly") >= 0 else 1
        self.reset_policy_state()

    def reset_policy_state(self):
        n = self.m.nu
        self.a1 = np.zeros(n); self.a2 = np.zeros(n); self.a3 = np.zeros(n)
        self.targets = self.default.copy()
        self.prev_targets = self.default.copy()
        self.imitation_i = 0.0
        self.phase = np.zeros(2)

    def contacts(self):
        out = [0.0, 0.0]
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            for k, f in enumerate(self.feet):
                if (b1 == f and c.geom2 == self.floor) or (b2 == f and c.geom1 == self.floor):
                    out[k] = 1.0
        return out

    def obs(self, cmd):
        d = self.d
        gyro = d.sensordata[self.gyro_a:self.gyro_a + 3].copy()
        acc = d.sensordata[self.acc_a:self.acc_a + 3].copy()
        acc[0] += self.accel_bias
        if "acc_roll_deg" in self.extra:   # pretend the IMU is mounted tilted (about X)
            r = np.radians(self.extra["acc_roll_deg"])
            c, s_ = np.cos(r), np.sin(r)
            acc = np.array([acc[0], c * acc[1] - s_ * acc[2], s_ * acc[1] + c * acc[2]])
        q = d.qpos[self.qadr] - self.default
        if "joint_bias" in self.extra:     # policy sees joints offset from where they are
            q = q + self.extra["joint_bias"]
        return np.concatenate([gyro, acc, cmd, q, d.qvel[self.vadr] * 0.05,
                               self.a1, self.a2, self.a3, self.targets, self.contacts(),
                               self.phase]).astype(np.float32)

    def control(self, cmd):
        if self.ref is not None and hasattr(self.ref, "period_for"):     # hybrid: float clock, command-dependent period
            period = float(self.ref.period_for(float(cmd[0]), float(cmd[1]), float(cmd[2]), xp=np))
            self.phase_f = (self.phase_f + 0.02 * self.phase_factor / period) % 1.0
            self.imitation_i = self.phase_f * self.nb_steps
        else:
            self.imitation_i = (self.imitation_i + self.phase_factor) % self.nb_steps
        ang = self.imitation_i / self.nb_steps * 2 * np.pi
        self.phase = np.array([np.cos(ang), np.sin(ang)])
        o = self.obs(cmd)
        a = self.policy.run(None, {"obs": [o]})[0][0]
        self.a3, self.a2, self.a1 = self.a2, self.a1, a.copy()
        scale = np.full(len(a), self.action_scale)
        if self.extra.get("leg_action_scale"):      # LEG_ACTION_SCALE: the ten leg joints get their own scale
            scale[[0, 1, 2, 3, 4, 9, 10, 11, 12, 13]] = self.extra["leg_action_scale"]
        t = self.default + a * scale
        if self.extra.get("head_from_cmd"):      # head = home + head command, as trained
            t[5:9] = self.default[5:9] + np.asarray(cmd)[3:7]
        elif self.extra.get("head_add_cmd"):     # the duck runtime: policy head + head command
            t[5:9] = t[5:9] + np.asarray(cmd)[3:7]
        ankle = np.zeros(len(t), bool); ankle[[4, 13]] = True   # left_ankle, right_ankle
        if self.extra.get("lpf_hz"):             # policy trained with ACT_LPF_HZ: same first-order filter
            def _alpha(hz):
                return (1 / hz) / (0.02 + 1 / hz)
            alpha = np.full(len(t), _alpha(self.extra["lpf_hz"]))
            if self.extra.get("ankle_lpf_hz"):   # ANKLE_LPF_HZ: the ankles filtered less
                alpha[ankle] = _alpha(self.extra["ankle_lpf_hz"])
            t = alpha * self.prev_targets + (1 - alpha) * t
        step = np.full(len(t), self.extra.get("max_motor_vel", 5.24) * 0.02)
        if self.extra.get("ankle_max_vel"):      # ANKLE_MAX_VEL
            step[ankle] = self.extra["ankle_max_vel"] * 0.02
        t = np.clip(t, self.prev_targets - step, self.prev_targets + step)
        self.prev_targets = t.copy()
        self.targets = t
        ctrl = t.copy()
        if "joint_bias" in self.extra:     # the real joint sits at target - bias
            ctrl = ctrl - self.extra["joint_bias"]
        self.d.ctrl[:] = ctrl
        for _ in range(self.decim):
            mujoco.mj_step(self.m, self.d)
        return o, a

    def base_pose(self):
        q = self.d.qpos[:7]
        w, x, y, z = q[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        up = 1 - 2 * (x * x + y * y)          # z component of the body's z axis
        return q[:3].copy(), yaw, np.degrees(np.arccos(np.clip(up, -1, 1)))


def run(sim, cmd, seconds, settle=2.0):
    zero = np.zeros(7)
    for _ in range(int(settle * 50)):
        sim.control(zero)
    p0, y0, _ = sim.base_pose()
    obs, acts, tilt_max, feet = [], [], 0.0, []
    yaw_total, y_prev = 0.0, y0
    for _ in range(int(seconds * 50)):
        o, a = sim.control(np.array(cmd, dtype=float))
        obs.append(o); acts.append(a)
        _, y_now, tilt = sim.base_pose()
        yaw_total += (y_now - y_prev + np.pi) % (2 * np.pi) - np.pi
        y_prev = y_now
        tilt_max = max(tilt_max, tilt)
        feet.append(o[97:99])
    p1, y1, _ = sim.base_pose()
    dp = p1 - p0
    c, s = np.cos(-y0), np.sin(-y0)
    fwd, left = c * dp[0] - s * dp[1], s * dp[0] + c * dp[1]
    dyaw = np.degrees(yaw_total)
    obs = np.array(obs); feet = np.array(feet)
    head = np.degrees(np.median(obs[:, 13 + 5:13 + 9], 0))
    legs = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
    motion = np.degrees(np.abs(np.diff(obs[:, 13:27][:, legs], axis=0))).sum(1).mean()
    return dict(fwd=fwd / seconds, left=left / seconds, yaw=dyaw / seconds, tilt_max=tilt_max,
                fell=tilt_max > 45, feet=feet.mean(0), head=head, motion=motion)


TESTS = [("stand", [0, 0, 0]), ("forward", [0.15, 0, 0]), ("backward", [-0.15, 0, 0]),
         ("step left", [0, 0.2, 0]), ("step right", [0, -0.2, 0]),
         ("turn left", [0, 0, 1.0]), ("turn right", [0, 0, -1.0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="BEST_WALK_ONNX_2.onnx")
    ap.add_argument("--xml", default=f"{PG}/xmls/scene_flat_terrain.xml")
    ap.add_argument("--accel-bias", type=float, default=1.3)
    ap.add_argument("--action-scale", type=float, default=0.25)
    ap.add_argument("--phase-factor", type=float, default=1.0)
    ap.add_argument("--nb-steps", type=int, default=None)
    ap.add_argument("--acc-roll-deg", type=float, default=None)
    ap.add_argument("--seconds", type=float, default=6.0)
    a = ap.parse_args()
    extra = {}
    if a.acc_roll_deg is not None:
        extra["acc_roll_deg"] = a.acc_roll_deg
    print(f"xml {a.xml.split('/')[-1]}  accel_bias {a.accel_bias}  action_scale {a.action_scale}  "
          f"phase_factor {a.phase_factor}  extra {extra}")
    for name, cmd in TESTS:
        sim = Sim(a.onnx, a.xml, a.accel_bias, a.action_scale, a.phase_factor, a.nb_steps, extra)
        r = run(sim, cmd + [0, 0, 0, 0], a.seconds)
        print(f"  {name:10s} cmd {cmd}  ->  fwd {r['fwd']:+.3f} m/s  left {r['left']:+.3f} m/s  "
              f"turn {r['yaw']:+6.1f} deg/s  max tilt {r['tilt_max']:4.1f}{' FELL' if r['fell'] else ''}  "
              f"feet {r['feet'][0]:.2f}/{r['feet'][1]:.2f}  leg motion {r['motion']:.2f}  "
              f"head(n,p,y,r) {np.round(r['head'], 1)}")
    print(f"  (nb_steps_in_period {sim.nb_steps})")


if __name__ == "__main__":
    main()
