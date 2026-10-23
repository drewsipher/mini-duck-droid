#!/usr/bin/env python3
"""Find how far each joint really bends. Motors go limp; you move each joint gently to its
mechanical stops by hand (both ways) while this records the furthest angle seen.

Run ON THE DUCK with the walk program stopped, duck on the stand (or lying down):
    ~/.virtualenvs/open-duck-mini-runtime/bin/python ~/joint_limits_probe.py
Ctrl+C when done: prints the table and saves ~/joint_limits_<time>.json.
Angles are in the same frame as the walk script and the simulator (calibration offsets removed).
"""
import json
import os
import sys
import time

import numpy as np
from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.rustypot_position_hwi import HWI

SIM_RANGE = {"hip_yaw": (-0.52, 0.52), "hip_roll": (-0.44, 0.44), "left_hip_pitch": (-1.22, 0.52),
             "right_hip_pitch": (-0.52, 1.22), "knee": (-1.57, 1.57), "ankle": (-1.57, 1.57),
             "neck_pitch": (-0.35, 1.13), "head_pitch": (-0.79, 0.79), "head_yaw": (-2.79, 2.79),
             "head_roll": (-0.52, 0.52)}


def sim_range(name):
    if name in SIM_RANGE:
        return SIM_RANGE[name]
    return SIM_RANGE[name.split("_", 1)[1]]


def main():
    hwi = HWI(DuckConfig())
    names = list(hwi.joints.keys())
    print("Hold the duck: motors going limp.")
    hwi.turn_off()
    time.sleep(0.3)
    lo = np.full(len(names), np.inf)
    hi = np.full(len(names), -np.inf)
    print("Move each joint gently to its stop both ways (knees, hips, ankles, neck, head).")
    print("Ctrl+C when done.\n")
    try:
        while True:
            p = hwi.get_present_positions()
            if p is not None:
                p = np.asarray(p, dtype=float)
                lo, hi = np.minimum(lo, p), np.maximum(hi, p)
                sys.stdout.write("\r" + " ".join(f"{n.replace('left_', 'L').replace('right_', 'R')[:9]}"
                                                 f" {np.degrees(v):5.0f}" for n, v in zip(names, p)) + "  ")
                sys.stdout.flush()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    print("\n\njoint               real min   real max  |  sim min  sim max   (radians)")
    out = {}
    for n, a, b in zip(names, lo, hi):
        s0, s1 = sim_range(n)
        print(f"{n:18s} {a:9.2f} {b:9.2f}  | {s0:8.2f} {s1:8.2f}")
        out[n] = [round(float(a), 3), round(float(b), 3)]
    path = os.path.expanduser(f"~/joint_limits_{time.strftime('%Y%m%d-%H%M%S')}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
