#!/usr/bin/env python3
"""Check how the IMU is mounted in the assembled duck and which imu_upside_down setting it needs.

    python3 imu_mount_check.py

Step 1: hold the duck upright and still, press Enter.
Step 2: tip it nose-down about 30 degrees, hold still, press Enter.
It works out which way each chip axis points in the duck and prints the setting to use.
"""
import statistics
import sys
import time

import adafruit_bno055
import board
import busio

# The two mountings the runtime understands (see mini_bdx_runtime/imu.py axis_remap):
# chip axis -> direction in the duck
MOUNTS = {
    False: {"x": "left", "y": "back", "z": "up"},
    True: {"x": "right", "y": "back", "z": "down"},
}
OPPOSITE = {"up": "down", "down": "up", "front": "back", "back": "front", "left": "right", "right": "left"}


def connect():
    i2c = busio.I2C(board.SCL, board.SDA)
    for addr in (0x28, 0x29):
        for _ in range(3):
            try:
                return adafruit_bno055.BNO055_I2C(i2c, address=addr)
            except (ValueError, OSError, RuntimeError):
                time.sleep(0.5)
    sys.exit("No IMU found at 0x28 or 0x29. Check its four wires (VIN 1, GND 9, SDA 3, SCL 5).")


def gravity(imu):
    """Median accelerometer reading over about a second. + on an axis means it points up."""
    xs, ys, zs = [], [], []
    while len(xs) < 40:
        try:
            a = imu.acceleration
            if a and None not in a:
                xs.append(a[0]); ys.append(a[1]); zs.append(a[2])
        except OSError:
            pass
        time.sleep(0.02)
    return {"x": statistics.median(xs), "y": statistics.median(ys), "z": statistics.median(zs)}


def dominant(g):
    axis = max(g, key=lambda k: abs(g[k]))
    return axis, g[axis] > 0


def main():
    imu = connect()
    time.sleep(0.5)
    input("Step 1: hold the duck UPRIGHT and still, then press Enter ")
    up = gravity(imu)
    input("Step 2: tip it NOSE-DOWN about 30 degrees, hold still, then press Enter ")
    tilt = gravity(imu)

    print(f"\n  upright:   x {up['x']:+6.2f}  y {up['y']:+6.2f}  z {up['z']:+6.2f}")
    print(f"  nose-down: x {tilt['x']:+6.2f}  y {tilt['y']:+6.2f}  z {tilt['z']:+6.2f}")

    up_axis, up_pos = dominant(up)
    if abs(up[up_axis]) < 8:
        sys.exit("\nThe upright reading isn't clear (no axis near 9.8). Hold it still and run again.")
    found = {up_axis: "up" if up_pos else "down"}
    # Tipping nose-down raises the back: the axis pointing back gains a positive reading.
    change = {k: tilt[k] - up[k] for k in "xyz" if k != up_axis}
    fb_axis = max(change, key=lambda k: abs(change[k]))
    if abs(change[fb_axis]) < 2.5:
        sys.exit("\nThe tilt was too small to read. Tip it further (30 degrees or more) and run again.")
    found[fb_axis] = "back" if change[fb_axis] > 0 else "front"
    last = ({"x", "y", "z"} - set(found)).pop()
    # right-handed axes: work out the third from the other two
    vec = {"up": (0, 0, 1), "down": (0, 0, -1), "front": (1, 0, 0), "back": (-1, 0, 0),
           "left": (0, 1, 0), "right": (0, -1, 0)}
    name = {v: k for k, v in vec.items()}
    order = "xyz"
    i, j = order.index(fb_axis), order.index(up_axis)
    a, b = (vec[found[fb_axis]], vec[found[up_axis]]) if (j - i) % 3 == 1 else (vec[found[up_axis]], vec[found[fb_axis]])
    cross = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
    found[last] = name[cross]

    print("\n  Measured: chip " + ", ".join(f"+{k.upper()} points {found[k]}" for k in "xyz"))
    for setting, mount in MOUNTS.items():
        if mount == found:
            print(f"\n  RESULT: set  \"imu_upside_down\": {str(setting).lower()}  in ~/duck_config.json")
            return
    print("\n  RESULT: this mounting matches NEITHER setting the runtime supports:")
    for setting, mount in MOUNTS.items():
        print(f"    imu_upside_down {str(setting).lower():5s}: " + ", ".join(f"+{k.upper()} {v}" for k, v in mount.items()))
    print("  The board needs turning so it matches one of those lines.")


if __name__ == "__main__":
    main()
