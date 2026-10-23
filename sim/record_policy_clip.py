#!/usr/bin/env python3
"""Record what a brain does in the simulator as a clip (10-22): one gait cycle, phase-averaged
over several cycles of its gait clock, in the same format tracker/ saves, so a learned gait
(s2a's turning) can be the imitation reference for the next brain, or be played on the duck.

    .venv/bin/python record_policy_clip.py <onnx> out.json --cmd 0,0,1.0 [--seconds 8] [--settle 2]
        [--torque 1.86] [--cutoff 8] [--max-motor-vel 4] [--leg-action-scale 0.25]

Prints the measured base speeds (forward / sideways m/s, yaw deg/s) and how repeatable the
cycles were (mean spread across cycles in degrees). Nothing is written when the duck falls.
"""
import argparse
import json

import numpy as np

from sim_test import PG, Sim

XML = f"{PG}/xmls/scene_flat_terrain_backlash.xml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("out")
    ap.add_argument("--cmd", default="0,0,1.0", help="vx,vy,wz")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--settle", type=float, default=2.0, help="seconds discarded at the start")
    ap.add_argument("--torque", type=float, default=1.86)
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--max-motor-vel", type=float, default=4.0)
    ap.add_argument("--leg-action-scale", type=float, default=0.0)
    ap.add_argument("--shift", type=int, default=3, help="clip frame k = pose after acting on clock k-shift; the MJX env "
                    "scores step k later than the recorder stamps it (step order + action delay): measured on s2a, shift 3 gives the highest imitation reward")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    cmd = np.array([float(x) for x in a.cmd.split(",")] + [0, 0, 0, 0])
    extra = {"head_from_cmd": True, "lpf_hz": a.cutoff, "max_motor_vel": a.max_motor_vel}
    if a.leg_action_scale:
        extra["leg_action_scale"] = a.leg_action_scale
    s = Sim(a.onnx, XML, 0.0, extra=extra)
    s.m.actuator_forcerange[:, :] = [-a.torque, a.torque]
    names = [s.m.actuator(k).name for k in range(s.m.nu)]
    n = s.nb_steps
    cycles = {k: [] for k in range(n)}          # clock index -> list of joint vectors
    poses, yaws = [], []
    steps = int(a.seconds * 50)
    for t in range(steps):
        s.control(cmd)
        if s.base_pose()[2] > 45:                # tilt in degrees, as render_video's FALLEN
            print(f"fell at {t / 50:.1f} s, nothing saved")
            return
        if t >= a.settle * 50:
            k = (int(round(s.imitation_i)) + a.shift) % n
            cycles[k].append(s.d.qpos[s.qadr].copy())
            poses.append(s.d.qpos[:3].copy())
            w, qx, qy, qz = s.d.qpos[3:7]
            yaws.append(np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
    poses, yaws = np.array(poses), np.unwrap(np.array(yaws))
    dur = len(poses) / 50
    # speeds in the body's heading frame, from the start-to-end displacement
    dp = poses[-1] - poses[0]
    mean_yaw = yaws.mean()
    fwd = np.cos(mean_yaw) * dp[0] + np.sin(mean_yaw) * dp[1]
    side = -np.sin(mean_yaw) * dp[0] + np.cos(mean_yaw) * dp[1]
    yaw_rate = np.degrees(yaws[-1] - yaws[0]) / dur
    q = np.array([np.mean(cycles[k], axis=0) for k in range(n)])
    spread = np.degrees(np.mean([np.std(cycles[k], axis=0).mean() for k in range(n)]))
    print(f"cmd {cmd[:3]}: fwd {fwd / dur:+.3f} m/s  side {side / dur:+.3f} m/s  yaw {yaw_rate:+.0f} deg/s  "
          f"over {dur:.1f} s, {len(cycles[0])} cycles of {n} steps ({n / 50:.2f} s), cycle spread {spread:.2f} deg")
    legs = [i for i, nm in enumerate(names) if "hip" in nm or "knee" in nm or "ankle" in nm]
    print("  leg joint swing (deg):", {names[i]: round(float(np.degrees(q[:, i].max() - q[:, i].min())), 1) for i in legs})
    if a.no_save:
        return
    frames = [[round(k / 50, 3), {nm: round(float(q[k, i]), 4) for i, nm in enumerate(names)}] for k in range(n)]
    frames.append([round(n / 50, 3), dict(frames[0][1])])          # closes the loop
    meta = {"source": a.onnx, "cmd": [float(c) for c in cmd[:3]], "measured": {"vx": fwd / dur, "vy": side / dur, "yaw_deg_s": yaw_rate},
            "period": n / 50, "cycles": len(cycles[0]), "spread_deg": spread}
    json.dump({"frames": frames, "interp": "linear", "recorded_from_policy": meta}, open(a.out, "w"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
