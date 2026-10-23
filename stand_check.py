#!/usr/bin/env python3
"""Stand the duck in its walking start pose (no AI) and report body tilt and joint tracking.

Run on the duck, from ~/Open_Duck_Mini_Runtime/scripts:
    ~/.virtualenvs/open-duck-mini-runtime/bin/python ~/stand_check.py

It moves to the same standing pose the walk uses (same strength), then once a second prints:
  - body tilt from the IMU (+pitch = leaning BACK, +roll = leaning to his RIGHT)
  - each leg joint: where it was told to be vs where it actually is
Hold him lightly with his feet on the floor; let him take his own weight if he can.

Balance keys (no Enter needed) shift BOTH ankles together, tipping his body over his feet:
    f = 0.5 deg more forward     b = 0.5 deg more back     0 = back to the walk pose
Find the shift where he stands on his own, then press q: it shows the ankle offsets that
would make that his standing pose and asks before saving them to ~/duck_config.json.
Ctrl+C also stops it. The motors then go limp, so be holding him.
"""
import math
import select
import sys
import termios
import time
import tty

import numpy as np

from mini_bdx_runtime.duck_config import DuckConfig, save_config_fields
from mini_bdx_runtime.raw_imu import Imu
from mini_bdx_runtime.rustypot_position_hwi import HWI

LEGS = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"]
ANKLES = ["left_ankle", "right_ankle"]
STEP = 0.5          # degrees per key press
FORWARD = +1        # + on the ankle joint tips the body forward over the feet (checked in the URDF)


def key():
    """One pending keypress, or None."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def main():
    cfg = DuckConfig(ignore_default=True)
    hwi = HWI(cfg)
    imu = Imu(50, upside_down=cfg.imu_upside_down,
              pitch_trim=float(cfg.imu_trim.get("pitch", 0.0)), roll_trim=float(cfg.imu_trim.get("roll", 0.0)))
    kps = [30] * 14
    kps[5:9] = [8, 8, 8, 8]                      # head softer, as in the walk
    hwi.set_kps(kps)
    hwi.turn_on()                                # same standing pose as the walk's start
    names = list(hwi.joints.keys())
    want = np.array([hwi.init_pos[n] for n in names])
    print("\nStanding. Put him on the floor, hold lightly. f / b shift the ankles, q to finish.\n")
    shift = 0.0
    old_term = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        while True:
            t_end = time.time() + 1.0
            quit_ = False
            while time.time() < t_end:
                k = key()
                if k in ("f", "b", "0"):
                    shift = 0.0 if k == "0" else shift + (STEP if k == "f" else -STEP)
                    pose = dict(hwi.init_pos)
                    for a in ANKLES:
                        pose[a] += FORWARD * math.radians(shift)
                    hwi.set_position_all(pose)
                    print(f"  ankle shift now {shift:+.1f} deg ({'forward' if shift > 0 else 'back' if shift < 0 else 'walk pose'})")
                elif k == "q":
                    quit_ = True
                    break
                time.sleep(0.02)
            if quit_:
                break
            d = imu.get_data()
            got = hwi.get_present_positions()
            if d is None or got is None:
                print("  (no data yet)")
                continue
            ax, ay, az = d["accelero"]
            back = math.degrees(math.atan2(ax, az))     # +x is forward: leaning back puts gravity on +x
            right = math.degrees(math.atan2(ay, az))    # +y is left: leaning right lifts it
            target = want.copy()
            for a in ANKLES:
                target[names.index(a)] += FORWARD * math.radians(shift)
            err = np.degrees(got - target)
            print(f"tilt: {'BACK' if back > 0 else 'FWD '} {abs(back):4.1f} deg   "
                  f"{'RIGHT' if right > 0 else 'LEFT '} {abs(right):4.1f} deg   ankle shift {shift:+.1f}")
            for side in ("left", "right"):
                cells = []
                for j in LEGS:
                    i = names.index(f"{side}_{j}")
                    cells.append(f"{j} {math.degrees(want[i]):+4.0f} err {err[i]:+5.1f}")
                print(f"  {side:5s} " + " | ".join(cells))
            print("        (each joint: target angle, then error in degrees: + means past the target)")
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
        hwi.turn_off()
        print("\nMotors off.")

    if shift:
        # Same physical pose in the walk: the walk commands init_pos + offset, so fold the shift
        # into each ankle's offset (radians).
        new = dict(cfg.joints_offset)
        for a in ANKLES:
            new[a] = round(new[a] + FORWARD * math.radians(shift), 4)
        print(f"\nAnkle shift {shift:+.1f} deg. Offsets in radians (degrees):")
        for a in ANKLES:
            print(f"  {a:12s} {cfg.joints_offset[a]:+.4f} ({math.degrees(cfg.joints_offset[a]):+.1f})"
                  f"  ->  {new[a]:+.4f} ({math.degrees(new[a]):+.1f})")
        if input("Save these to ~/duck_config.json? (y/N) ").strip().lower() == "y":
            backup = save_config_fields({"joints_offsets": new})
            print(f"Saved. Backup: {backup}")
        else:
            print("Not saved.")


if __name__ == "__main__":
    main()
