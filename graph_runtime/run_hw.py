#!/usr/bin/env python3
"""Run ONE tracking skill package on the real duck (or in MuJoCo with the same code) and log every tick.

    on the Pi:   DUCK_RUNTIME_ALLOW_HARDWARE=1 python run_hw.py --package skills/stand/v1 --seconds 60 --log logs/stand_held.jsonl --hardware
    on the box:  MUJOCO_GL=egl sim/.venv/bin/python graph_runtime/run_hw.py --package skills/stand/v1 --seconds 10 --log /tmp/x.jsonl \
                     [--sim-push 0.3,180,3]      # test the pipeline: 0.3 m/s thrown backward every 3 s

Solo runner, no motion graph: the package's policy drives the backend directly with the training
contract (sim/eval_tracking.py, sim/fusion/fusion_sim.py): obs = gyro, accelerometer, q - home,
dq x 0.05, the previous two actions, filtered targets - home, clip phase [t/T, cos, sin], the clip's
next joint targets - home and velocities x 0.05; target = clip(k + 1) + action_scale x action, then the
8 Hz low-pass and the 4 rad/s rate clamp. A loop clip wraps (the motion graph engine cannot run loop
clips yet, which is why this exists). ref.json in the package (tools/clip_ref_table.py) holds the clip
table so the Pi needs only numpy and onnxruntime.

Pushes are given by hand. A push shows in the log as an accelerometer spike; tools/hw_report.py finds
them, measures peak tilt, settle time and drift, and compares runs.

Log: {"meta": ...}, then one JSON object per tick (50 Hz): t, wall, gyro, acc, q, dq, action, targets,
contacts, tilt (accelerometer estimate, low-passed), then {"summary": ...}. Typing a note + Enter writes
{"mark": text}; 'q' + Enter or Ctrl-C stops.

Safety: the hardware backend starts soft, goes to the home pose, then stiffens (about 2 s); on exit it
softens, then turns the servos off. Tilt above --fall-deg (45) for 0.2 s softens the servos and stops.
Hold the duck for the first run of any new skill.
"""
from __future__ import annotations

import argparse
import json
import os
import select
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from graph_runtime.backends.base import SensorFrame      # noqa: E402

DT = 0.02


class TrackingSkill:
    """One package's policy, run as in training. Needs skill.json, policy.onnx, ref.json."""

    def __init__(self, package: str):
        p = Path(package)
        self.skill = json.load(open(p / "skill.json"))
        self.ref = json.load(open(p / "ref.json"))
        pol = self.skill["policy"]
        self.scale = float(pol["action_scale"])
        self.lpf_hz = float(pol.get("lpf_hz", 8.0))
        self.max_step = float(pol.get("max_vel_rad_s", 4.0)) * DT
        self.alpha = (1.0 / self.lpf_hz) / (DT + 1.0 / self.lpf_hz) if self.lpf_hz > 0 else 0.0
        self.input_name = pol.get("input_name", "obs")
        self.home = np.asarray(self.ref["home"], float)
        self.q = np.asarray(self.ref["q"], float)
        self.qd = np.asarray(self.ref["qd"], float)
        self.n = int(self.ref["cycle_steps"])
        self.loop = bool(self.ref["loop"])
        self.vel_scale = float(self.ref.get("dof_vel_scale", 0.05))
        names = [b["name"] if isinstance(b, dict) else b for b in self.skill["obs_spec"].get("layout", self.skill["obs_spec"]["names"])]
        expected = ["gyro", "accelerometer", "joint_pos_rel", "joint_vel_scaled", "prev_action_1", "prev_action_2",
                    "motor_targets_rel", "phase", "ref_joint_pos_next_rel", "ref_joint_vel_next_scaled"]
        if names != expected:
            raise SystemExit(f"{package}: obs layout {names} is not the tracking contract {expected}")
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(p / "policy.onnx"), providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.k = 0
        self.a1 = np.zeros(14); self.a2 = np.zeros(14)
        self.targets = self.q[0].copy()

    def idx(self, k):
        return k % self.n if self.loop else min(k, self.n - 1)

    def phase(self):
        f = (self.k % self.n) / self.n if self.loop else self.idx(self.k) / max(self.n - 1, 1)
        return np.array([f, np.cos(2 * np.pi * f), np.sin(2 * np.pi * f)])

    @property
    def done(self):
        return (not self.loop) and self.k >= self.n - 1

    def step(self, frame: SensorFrame):
        kn = self.idx(self.k + 1)
        obs = np.concatenate([frame.gyro, frame.acc, frame.q - self.home, frame.dq * self.vel_scale, self.a1, self.a2,
                              self.targets - self.home, self.phase(), self.q[kn] - self.home, self.qd[kn] * self.vel_scale]).astype(np.float32)
        a = self.sess.run(None, {self.input_name: obs[None, :]})[0][0].astype(float)
        raw = self.q[kn] + self.scale * a
        new = self.alpha * self.targets + (1 - self.alpha) * raw
        self.targets = np.clip(new, self.targets - self.max_step, self.targets + self.max_step)
        self.a2, self.a1 = self.a1, a
        self.k += 1
        return self.targets.copy(), a


def make_backend(a):
    if a.hardware:
        from graph_runtime.backends.hardware import HardwareBackend     # noqa: PLC0415
        return HardwareBackend(duck_config_path=a.duck_config, serial_port=a.serial, allow=True)
    from graph_runtime.backends.mujoco import DEFAULT_XML, MujocoBackend  # noqa: PLC0415
    return MujocoBackend(xml=str(DEFAULT_XML), torque_nm=1.86, seed=a.seed)


def stdin_line():
    try:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.readline().rstrip("\n")
    except (OSError, ValueError):
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--log", required=True, help="jsonl path")
    ap.add_argument("--hardware", action="store_true", help="drive the real duck (needs DUCK_RUNTIME_ALLOW_HARDWARE=1)")
    ap.add_argument("--duck-config", default=None)
    ap.add_argument("--serial", default="/dev/ttyACM0")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fall-deg", type=float, default=45.0)
    ap.add_argument("--sim-push", default=None, help="SIM ONLY: 'dv,deg,every_s' (deg: 0 thrown forward, 180 backward, 90 left)")
    a = ap.parse_args()

    skill = TrackingSkill(a.package)
    be = make_backend(a)
    out = Path(a.log)
    out.parent.mkdir(parents=True, exist_ok=True)
    f = out.open("w")
    f.write(json.dumps({"meta": {"package": str(Path(a.package).resolve()), "skill": skill.skill.get("name"), "backend": be.name,
                                 "action_scale": skill.scale, "seconds": a.seconds, "sim_push": a.sim_push,
                                 "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": os.uname().nodename}}) + "\n")
    push = None
    if a.sim_push and not a.hardware:
        dv, deg, every = (float(x) for x in a.sim_push.split(","))
        push = (dv, np.radians(deg), max(1, int(round(every / DT))))

    print(f"{skill.skill.get('name')} ({'hardware' if a.hardware else 'sim'}), action scale {skill.scale}, loop {skill.loop}, "
          f"{a.seconds:.0f} s; type a note + Enter to mark, 'q' + Enter or Ctrl-C to stop", flush=True)
    t0, n, tilt_f, peak_tilt, fallen_for, stopped = time.time(), 0, 0.0, 0.0, 0, None
    try:
        while n * DT < a.seconds:
            if push and n and n % push[2] == 0:
                d = be.d
                yaw = np.arctan2(d.xmat[1][3], d.xmat[1][0])
                d.qvel[0] += push[0] * np.cos(push[1] + yaw); d.qvel[1] += push[0] * np.sin(push[1] + yaw)
                f.write(json.dumps({"mark": f"sim push {push[0]:g} m/s at {np.degrees(push[1]):.0f} deg", "t": round(n * DT, 3)}) + "\n")
            fr = be.read()
            targets, act = skill.step(fr)
            be.write_targets(targets)
            n += 1
            tilt_now = fr.tilt_deg if fr.tilt_deg is not None else SensorFrame.tilt_from_accel(fr.acc)
            tilt_f = 0.8 * tilt_f + 0.2 * float(tilt_now)
            peak_tilt = max(peak_tilt, tilt_f)
            f.write(json.dumps({"t": round(n * DT, 3), "wall": round(time.time() - t0, 4),
                                "gyro": [round(float(x), 4) for x in fr.gyro], "acc": [round(float(x), 4) for x in fr.acc],
                                "q": [round(float(x), 4) for x in fr.q], "dq": [round(float(x), 3) for x in fr.dq],
                                "action": [round(float(x), 3) for x in act], "targets": [round(float(x), 4) for x in targets],
                                "contacts": [float(x) for x in fr.contacts], "tilt": round(tilt_f, 2),
                                "height": None if fr.height_m is None else round(float(fr.height_m), 4)}) + "\n")
            fallen_for = fallen_for + 1 if tilt_f > a.fall_deg else 0
            if fallen_for >= 10:
                stopped = f"tilt {tilt_f:.0f} deg for 0.2 s"
                print(f"[{n * DT:.2f}s] FALL: {stopped}; softening", flush=True)
                break
            if skill.done:
                stopped = "clip finished"
                break
            line = stdin_line()
            if line is not None:
                if line.strip().lower() == "q":
                    stopped = "operator"
                    break
                if line.strip():
                    f.write(json.dumps({"mark": line.strip(), "t": round(n * DT, 3)}) + "\n")
                    print(f"[{n * DT:.2f}s] mark: {line.strip()}", flush=True)
            if n % 250 == 0:
                print(f"[{n * DT:5.1f}s] tilt {tilt_f:5.1f} deg  loop {(time.time() - t0) / (n * DT):.2f}x real time", flush=True)
    except KeyboardInterrupt:
        stopped = "interrupted"
        print("interrupted", flush=True)
    finally:
        summary = {"summary": {"seconds": round(n * DT, 2), "ticks": n, "wall_s": round(time.time() - t0, 2),
                               "stopped": stopped or "time up", "peak_tilt_deg": round(peak_tilt, 1)}}
        f.write(json.dumps(summary) + "\n")
        f.close()
        try:
            be.set_stiffness(0.1)
            time.sleep(0.3)
        finally:
            be.close()
        print("summary:", json.dumps(summary["summary"]))
        print("log", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
