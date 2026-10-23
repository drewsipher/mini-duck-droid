#!/usr/bin/env python3
"""Find out how your BNO055 board must sit in the duck, using gravity. Run on the Pi.

    python3 imu_axes_check.py

Hold the IMU the way it will be mounted in the duck and follow three prompts:
  1. duck standing upright        2. duck tipped onto its FACE        3. duck tipped onto its LEFT side
It reads the chip's raw axes (no remapping) and tells you, in plain words, which way the chip's
X, Y and Z point relative to the duck. Then it switches the chip to each of the runtime's two
mounting settings (imu_upside_down false / true) and checks which one gives the axes the walking
policy was trained with: X forward, Y to the duck's left, Z up.
Read-only for your config: it only changes the IMU chip's temporary settings, which reset at power-off.
Needs the Adafruit BNO055 library (already on the Duck image). Wiring: VIN 3.3 V, GND, SDA, SCL.
"""
import time

import adafruit_bno055
import board
import busio

G = 9.81
R = adafruit_bno055
SETTINGS = {  # exactly what mini_bdx_runtime/imu.py writes
    "imu_upside_down: false": (R.AXIS_REMAP_Y, R.AXIS_REMAP_X, R.AXIS_REMAP_Z,
                               R.AXIS_REMAP_NEGATIVE, R.AXIS_REMAP_POSITIVE, R.AXIS_REMAP_POSITIVE),
    "imu_upside_down: true": (R.AXIS_REMAP_Y, R.AXIS_REMAP_X, R.AXIS_REMAP_Z,
                              R.AXIS_REMAP_NEGATIVE, R.AXIS_REMAP_NEGATIVE, R.AXIS_REMAP_NEGATIVE),
}
RAW = (R.AXIS_REMAP_X, R.AXIS_REMAP_Y, R.AXIS_REMAP_Z, R.AXIS_REMAP_POSITIVE, R.AXIS_REMAP_POSITIVE,
       R.AXIS_REMAP_POSITIVE)
POSES = [
    ("upright", "Hold the IMU exactly as it will sit in the duck, with the duck standing upright."),
    ("face", "Now tip the duck forward onto its FACE: the side that faces the duck's front points at the floor."),
    ("left", "Now tip the duck onto its LEFT side: the side that faces the duck's left points at the floor."),
]
# what an axis that is level, pointing up, or pointing down reads (accelerometer sees +g when pointing up)
EXPECT = {"upright": (0, 0, G), "face": (-G, 0, 0), "left": (0, -G, 0)}  # in duck axes X fwd, Y left, Z up


def read(imu, remap, n=20):
    imu.axis_remap = remap
    time.sleep(0.15)
    samples = []
    for _ in range(n * 3):
        try:
            a = imu.acceleration
        except OSError:  # the Pi's I2C occasionally garbles a read; skip it
            a = None
        if a and None not in a:
            samples.append(a)
            if len(samples) == n:
                break
        time.sleep(0.02)
    # median per axis, so one corrupted reading can't skew the result
    return [sorted(s[k] for s in samples)[len(samples) // 2] if samples else 0.0 for k in range(3)]


def dominant(v):
    i = max(range(3), key=lambda k: abs(v[k]))
    return i, (1 if v[i] > 0 else -1), abs(v[i])


def main():
    i2c = busio.I2C(board.SCL, board.SDA)
    while not i2c.try_lock():
        pass
    found = i2c.scan()
    i2c.unlock()
    addr = 0x28 if 0x28 in found else 0x29 if 0x29 in found else None
    if addr is None:
        print("No BNO055 on the I2C bus (looked for 0x28 and 0x29). Check VIN, GND, SDA, SCL.")
        return
    if addr == 0x29:
        print("Note: IMU answers at 0x29. The walking code expects 0x28: tie the board's ADR pin to GND.")
    imu = adafruit_bno055.BNO055_I2C(i2c, address=addr)
    imu.mode = adafruit_bno055.ACCONLY_MODE
    raw, out = {}, {name: {} for name in SETTINGS}
    for key, text in POSES:
        input(f"\n{text}\nHold it still and press Enter… ")
        raw[key] = read(imu, RAW)
        for name, remap in SETTINGS.items():
            out[name][key] = read(imu, remap)
        print("  chip axes read  X {:+6.2f}  Y {:+6.2f}  Z {:+6.2f}  m/s²".format(*raw[key]))
    imu.axis_remap = RAW

    # Plain-words answer from the raw readings
    print("\nMEASURED - how you held it this time (a result, not an instruction):")
    names, sides = "XYZ", {}
    up_i, up_s, up_m = dominant(raw["upright"])
    dn_i, dn_s, dn_m = dominant(raw["face"])     # the axis pointing at the floor reads -g
    lf_i, lf_s, lf_m = dominant(raw["left"])
    if min(up_m, dn_m, lf_m) < 0.7 * G or len({up_i, dn_i, lf_i}) < 3:
        print("  The three poses did not give three clean, different axes. Hold each pose closer to")
        print("  90 degrees, keep it still, and run again.")
        return
    sides[up_i] = "UP" if up_s > 0 else "DOWN"
    sides[dn_i] = "to the duck's BACK" if dn_s > 0 else "to the duck's FRONT"
    sides[lf_i] = "to the duck's RIGHT" if lf_s > 0 else "to the duck's LEFT"
    for i in range(3):
        print(f"  chip +{names[i]} was pointing {sides[i]}")

    # Which runtime setting produces X forward, Y left, Z up?
    print("\nRuntime setting check (wanted: X forward, Y left, Z up):")
    winner = None
    for name in SETTINGS:
        ok = all(all(abs(o - e) < 0.35 * G for o, e in zip(out[name][k], EXPECT[k])) for k in EXPECT)
        print(f"  {name:24s} {'MATCHES' if ok else 'does not match'}")
        if ok:
            winner = name
    if winner:
        print(f"\nPut  \"{winner.split(':')[0]}\": {winner.split(': ')[1]}  in ~/duck_config.json and keep this mounting.")
    else:
        print("\nNeither setting fits the way you held it. TARGET - rotate the board until it matches")
        print("one of these lines, then run this check again:")
        print("  imu_upside_down false:  chip +X to the duck's LEFT,  +Y to the BACK, +Z UP")
        print("  imu_upside_down true:   chip +X to the duck's RIGHT, +Y to the BACK, +Z DOWN")


if __name__ == "__main__":
    main()
