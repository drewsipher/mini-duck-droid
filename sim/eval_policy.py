#!/usr/bin/env python3
"""Score walking policies on the sim duck with real-servo torque (1.86 N.m), so a new
policy can be compared with BEST_WALK_ONNX_2 before it goes near the real duck.

    .venv/bin/python eval_policy.py BEST_WALK_ONNX_2.onnx new.onnx [--torque 1.86]

Per policy:
  - head pose standing still (deg from home; the old policy holds it turned right)
  - which single-axis commands start stepping from a standstill (y = steps)
  - speed reached for each command, and falls
  - body sway (roll / pitch, deg/s) walking forward, and standing jitter
  - pushes while standing: how many of 8 it survives
"""
import argparse
import importlib.util

import mujoco
import numpy as np

from sim_test import PG, Sim, run

spec = importlib.util.spec_from_file_location("sc", "smooth_cmp_lib.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)
XML = f"{PG}/xmls/scene_flat_terrain_backlash.xml"
START = [("fwd", [0.15, 0, 0]), ("back", [-0.15, 0, 0]), ("left", [0, 0.11, 0]),
         ("right", [0, -0.11, 0]), ("turnL", [0, 0, 1.0]), ("turnR", [0, 0, -1.0])]


HEAD_FROM_CMD = False
LPF_HZ = 0.0          # --cutoff: the low-pass the policy was trained with (ACT_LPF_HZ)
MAX_MOTOR_VEL = 5.24  # --max-motor-vel: MAX_MOTOR_VEL it was trained with
ANKLE_LPF_HZ = 0.0    # --ankle-cutoff / --ankle-max-vel: ANKLE_LPF_HZ / ANKLE_MAX_VEL (0 = same as the rest)
ANKLE_MAX_VEL = 0.0
LEG_ACTION_SCALE = 0.0  # --leg-action-scale: LEG_ACTION_SCALE it was trained with (0 = the default 0.25)


def make(onnx, torque):
    s = Sim(onnx, XML, 0.0, extra={"head_from_cmd": HEAD_FROM_CMD, "head_add_cmd": not HEAD_FROM_CMD,
                                   "lpf_hz": LPF_HZ, "max_motor_vel": MAX_MOTOR_VEL,
                                   "ankle_lpf_hz": ANKLE_LPF_HZ, "ankle_max_vel": ANKLE_MAX_VEL,
                                   "leg_action_scale": LEG_ACTION_SCALE})
    s.m.actuator_forcerange[:, :] = [-torque, torque]
    return s


def push_test(onnx, torque, n=8, force=15.0):
    """Shove the standing duck from n directions for 0.1 s and see if it stays up."""
    ok = 0
    for k in range(n):
        s = make(onnx, torque)
        for _ in range(100):
            s.control(np.zeros(7))
        ang = 2 * np.pi * k / n
        trunk = s.base
        for t in range(150):
            if t < 5:
                s.d.xfrc_applied[trunk, :3] = [force * np.cos(ang), force * np.sin(ang), 0]
            else:
                s.d.xfrc_applied[trunk, :3] = 0
            s.control(np.zeros(7))
            if s.base_pose()[2] > 45:
                break
        else:
            ok += 1
    return ok


def head_test(onnx, torque):
    """Stand still while the head is swung around (yaw within the duck's +-20 deg clamp):
    how many leg steps does it take and does it fall?"""
    s = make(onnx, torque)
    for _ in range(100):
        s.control(np.zeros(7))
    moves = [[0.6, 0, 0, 0], [0, 0, 0.35, 0], [0, 0, -0.35, 0], [0, 0, 0, 0.5], [0, 0, 0, -0.5],
             [-0.3, 0.5, 0, 0], [0, -0.6, 0.3, 0.3], [0, 0, 0, 0]]
    feet, fell = [], False
    for m in moves:
        for _ in range(50):
            o, _ = s.control(np.array([0, 0, 0] + m, float))
            feet.append(o[97:99].mean())
            fell |= s.base_pose()[2] > 45
    return 1 - float(np.mean(feet)), fell


def evaluate(onnx, torque):
    out = {}
    s = make(onnx, torque)
    obs = []
    for _ in range(250):
        o, _ = s.control(np.zeros(7))
        obs.append(o)
    obs = np.array(obs[50:], dtype=np.float64)
    out["head"] = np.degrees(np.median(obs[:, 18:22], 0))
    out["stand_jit"] = sc.bands(obs[:, 0:2])[1].mean()
    starts, speeds, fell = [], {}, False
    for name, cmd in START:
        r = run(make(onnx, torque), cmd + [0, 0, 0, 0], 6)
        starts.append(name if r["feet"].mean() < 0.95 else name.upper().join(["(", ")"]))
        speeds[name] = r["yaw"] if "turn" in name else (r["left"] if name in ("left", "right") else r["fwd"])
        fell |= r["fell"]
    out["starts"], out["speeds"], out["fell"] = starts, speeds, fell
    s = make(onnx, torque)
    o_fwd = []
    for _ in range(100):
        s.control(np.zeros(7))
    for _ in range(400):
        o, _ = s.control(np.array([0.15, 0, 0, 0, 0, 0, 0], float))
        o_fwd.append(o)
    lo, hi = sc.bands(np.array(o_fwd[100:], dtype=np.float64)[:, 0:2])
    out["fwd_sway"], out["fwd_jit"] = lo, hi
    out["push"] = push_test(onnx, torque)
    out["head_lift"], out["head_fell"] = head_test(onnx, torque)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", nargs="+")
    ap.add_argument("--torque", type=float, default=1.86)
    ap.add_argument("--head-from-cmd", action="store_true", help="policy trained with HEAD_FROM_CMD=1")
    ap.add_argument("--cutoff", type=float, default=0.0, help="low-pass cutoff Hz the policy was trained with (ACT_LPF_HZ)")
    ap.add_argument("--max-motor-vel", type=float, default=5.24, help="MAX_MOTOR_VEL the policy was trained with")
    ap.add_argument("--ankle-cutoff", type=float, default=0.0, help="ANKLE_LPF_HZ the policy was trained with (0 = same as --cutoff)")
    ap.add_argument("--ankle-max-vel", type=float, default=0.0, help="ANKLE_MAX_VEL the policy was trained with (0 = same as --max-motor-vel)")
    ap.add_argument("--leg-action-scale", type=float, default=0.0, help="LEG_ACTION_SCALE the policy was trained with (0 = default)")
    a = ap.parse_args()
    global HEAD_FROM_CMD, LPF_HZ, MAX_MOTOR_VEL, ANKLE_LPF_HZ, ANKLE_MAX_VEL, LEG_ACTION_SCALE
    HEAD_FROM_CMD, LPF_HZ, MAX_MOTOR_VEL = a.head_from_cmd, a.cutoff, a.max_motor_vel
    ANKLE_LPF_HZ, ANKLE_MAX_VEL, LEG_ACTION_SCALE = a.ankle_cutoff, a.ankle_max_vel, a.leg_action_scale
    print(f"sim duck, backlash model, servo torque limit {a.torque} N.m, action low-pass {a.cutoff or 'off'} Hz"
          + (f", ankles {a.ankle_cutoff:g} Hz / {a.ankle_max_vel:g} rad/s" if a.ankle_cutoff or a.ankle_max_vel else "")
          + (f", leg action scale {a.leg_action_scale:g}" if a.leg_action_scale else ""))
    for p in a.onnx:
        r = evaluate(p, a.torque)
        sp = r["speeds"]
        print(f"\n{p}")
        print(f"  head at rest (neck, pitch, yaw, roll) deg: {np.round(r['head'], 1)}")
        print(f"  starts stepping from standstill: {' '.join(r['starts'])}   (bracketed = just stands)")
        print(f"  speed: fwd {sp['fwd']:+.3f}  back {sp['back']:+.3f}  left {sp['left']:+.3f}  right {sp['right']:+.3f} m/s"
              f"  turn {sp['turnL']:+.0f}/{sp['turnR']:+.0f} deg/s  {'FELL' if r['fell'] else 'no falls'}")
        print(f"  walking sway roll/pitch {r['fwd_sway'][0]:.1f}/{r['fwd_sway'][1]:.1f}  jitter {r['fwd_jit'][0]:.1f}/{r['fwd_jit'][1]:.1f}"
              f"   standing jitter {r['stand_jit']:.2f}   15 N pushes survived {r["push"]}/8")
        print(f"  head swung while standing: feet off the floor {100 * r['head_lift']:.0f}% of the time"
              f"{'  FELL' if r['head_fell'] else ''}")


if __name__ == "__main__":
    main()
