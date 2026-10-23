#!/usr/bin/env python3
"""Measure one servo's real strength by pressing the foot down onto a kitchen scale.

Run ON THE DUCK, with the walk program stopped and the duck on its stand, feet hanging:
    python3 ~/strength_test.py                 # right ankle (ID 14)
    python3 ~/strength_test.py --id 24         # left ankle
    python3 ~/strength_test.py --id 13         # right knee (any joint whose part can press on the scale)

1. All ten leg servos hold where they are (same strength setting as walking, P=30).
2. Put the scale under the foot. Jog the joint with  j / k  (0.5 deg per press) until the foot
   JUST touches the scale. Tare (zero) the scale. Press Enter.
   HOLD HIM DOWN: press firmly on his body (not the leg) for the whole test, or the foot just
   lifts him off the stand - a servo can push 3-4 kg through the toe.
3. The joint then presses further in 1-degree steps. At each step it waits a moment, then
   you type the scale reading in grams and press Enter (or just Enter to stop).
   It stops by itself at 15 degrees, if the servo gets hot, or draws too much current.
4. Measure the lever: the horizontal distance from the joint's axis (centre of the servo horn)
   to the point of the foot touching the scale, in cm. Type it in.
It saves ~/strength_<id>_<time>.csv and prints the torque at each step. Ctrl+C = all motors off.
Needs: pip install pyserial (already on the duck).
"""
import argparse
import csv
import os
import select
import sys
import termios
import time
import tty

import serial

LEG_IDS = [10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
NAMES = {10: "right_hip_yaw", 11: "right_hip_roll", 12: "right_hip_pitch", 13: "right_knee", 14: "right_ankle",
         20: "left_hip_yaw", 21: "left_hip_roll", 22: "left_hip_pitch", 23: "left_knee", 24: "left_ankle"}
P_COEF, TORQUE, ACCEL, GOAL, TORQUE_LIMIT, STATE = 21, 40, 41, 42, 48, 56
PING, READ, WRITE = 0x01, 0x02, 0x03
STEPS_PER_DEG = 4096 / 360
MA_PER_UNIT = 6.5
MAX_PRESS_DEG = 15
MAX_TEMP = 55
MAX_MA = 2500


class Bus:
    def __init__(self, port):
        self.port = port

    @staticmethod
    def packet(sid, instr, params=b""):
        body = bytes([sid, len(params) + 2, instr]) + bytes(params)
        return b"\xff\xff" + body + bytes([~sum(body) & 0xFF])

    def txn(self, sid, instr, params=b"", wait=0.05):
        self.port.reset_input_buffer()
        self.port.write(self.packet(sid, instr, params))
        buf, end = bytearray(), time.monotonic() + wait
        while time.monotonic() < end:
            n = self.port.in_waiting
            if not n:
                time.sleep(0.0005)
                continue
            buf += self.port.read(n)
            i = buf.find(b"\xff\xff")
            if i >= 0 and len(buf) >= i + 4 and len(buf) >= i + 4 + buf[i + 3]:
                pkt = buf[i + 2:i + 4 + buf[i + 3]]
                if pkt[0] == sid and pkt[-1] == (~sum(pkt[:-1]) & 0xFF):
                    return bytes(pkt[3:-1])
                return None
        return None

    def read(self, sid, addr, n):
        for _ in range(3):
            d = self.txn(sid, READ, bytes([addr, n]))
            if d is not None and len(d) >= n:
                return d
        raise IOError(f"servo {sid} did not answer a read of address {addr}")

    def write(self, sid, addr, *data):
        for _ in range(3):
            if self.txn(sid, WRITE, bytes([addr, *data])) is not None:
                return
        raise IOError(f"servo {sid} did not confirm a write to address {addr}")

    def write16(self, sid, addr, value):
        self.write(sid, addr, value & 0xFF, value >> 8)

    def state(self, sid):
        d = self.read(sid, STATE, 15)   # 56 pos, 58 speed, 60 load, 62 volt, 63 temp, 65 status, 69 current
        load = d[4] | d[5] << 8
        load = -(load & 0x3FF) if load & 0x400 else (load & 0x3FF)
        cur = d[13] | d[14] << 8
        return {"pos": d[0] | d[1] << 8, "load": load / 10.0, "volt": d[6] / 10, "temp": d[7],
                "ma": (cur & 0x7FFF) * MA_PER_UNIT}

    def goal(self, sid, pos):
        self.write16(sid, GOAL, int(round(pos)))

    def off(self, sid):
        try:
            self.write(sid, TORQUE, 0)
        except IOError as e:
            print(f"  warning: {e}")


def key(timeout=0.05):
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.read(1)
    return None


def averaged(bus, sid, n=6):
    rows = [bus.state(sid) for _ in range(n)]
    return {k: sum(r[k] for r in rows) / n for k in rows[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", type=int, default=14)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--p", type=int, default=30, help="servo P coefficient (the walk uses 30)")
    a = ap.parse_args()
    if a.id not in LEG_IDS:
        sys.exit(f"--id must be one of {LEG_IDS}")
    bus = Bus(serial.Serial(a.port, 1_000_000, timeout=0))
    held = []
    try:
        for sid in LEG_IDS:                       # hold every leg servo where it is
            st = bus.state(sid)
            bus.write16(sid, TORQUE_LIMIT, 1000)
            bus.write(sid, P_COEF, a.p)
            bus.write(sid, ACCEL, 20)
            bus.goal(sid, st["pos"])
            bus.write(sid, TORQUE, 1)
            held.append(sid)
        sid = a.id
        pos = bus.state(sid)["pos"]
        print(f"\nTesting {NAMES[sid]} (ID {sid}), P={a.p}. All leg servos holding.")
        print("Put the scale under the foot. j / k jog 0.5 deg. When the foot JUST touches,")
        print("(and keep a hand pressing his body down firmly the whole time)")
        print("tare the scale and press Enter.\n")
        old = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        direction = 0
        try:
            while True:
                k = key()
                if k in ("j", "k"):
                    step = (1 if k == "k" else -1) * 0.5 * STEPS_PER_DEG
                    direction = 1 if step > 0 else -1
                    pos += step
                    bus.goal(sid, pos)
                    print(f"  jogged {'+' if step > 0 else '-'}0.5 deg", flush=True)
                elif k in ("\n", "\r"):
                    break
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        if direction == 0:
            print("No jog direction yet. Which way presses INTO the scale? Type + or - : ", end="")
            direction = 1 if input().strip().startswith("+") else -1
        contact = pos
        base = averaged(bus, sid)
        print(f"\nContact set. Volts {base['volt']:.1f}. Pressing in 1-degree steps "
              f"({'+' if direction > 0 else '-'} direction).")
        rows = []
        for press in range(1, MAX_PRESS_DEG + 1):
            target = contact + direction * press * STEPS_PER_DEG
            bus.goal(sid, target)
            time.sleep(1.0)
            s = averaged(bus, sid)
            behind = (target - s["pos"]) * direction / STEPS_PER_DEG   # how far it is short of the target
            print(f"  step {press:2d} deg: servo is {behind:4.1f} deg short of target, load {s['load']:5.1f}%, "
                  f"{s['ma']:5.0f} mA, {s['temp']:.0f} C, {s['volt']:.1f} V")
            g = input("    scale reading in grams (Enter = stop): ").strip()
            if not g:
                break
            if rows and float(g) <= rows[-1]["grams"] * 1.05 and behind < 2:
                print("    ! the reading stopped rising but the servo isn't straining (it's reaching its target),")
                print("      so something else is giving way - is he lifting off the stand? Hold him down harder.")
            rows.append(dict(press_deg=press, grams=float(g), short_deg=round(behind, 2), load_pct=s["load"],
                             ma=round(s["ma"]), temp=s["temp"], volt=s["volt"]))
            if s["temp"] >= MAX_TEMP or s["ma"] >= MAX_MA:
                print("  stopping: servo hot or drawing a lot of current")
                break
        bus.goal(sid, contact)                    # back off the scale
        time.sleep(0.5)
        if not rows:
            print("No readings.")
            return
        arm = float(input("\nLever: horizontal distance from the joint axis to the contact point, in cm: "))
        print(f"\n{'press':>6s} {'grams':>7s} {'torque N.m':>11s} {'short deg':>10s} {'load %':>7s} {'mA':>6s}")
        for r in rows:
            r["arm_cm"] = arm
            r["torque_nm"] = round(r["grams"] / 1000 * 9.81 * arm / 100, 3)
            print(f"{r['press_deg']:6d} {r['grams']:7.0f} {r['torque_nm']:11.3f} {r['short_deg']:10.1f} "
                  f"{r['load_pct']:7.1f} {r['ma']:6.0f}")
        path = os.path.expanduser(f"~/strength_{sid}_{time.strftime('%Y%m%d-%H%M%S')}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved {path}. Peak torque {max(r['torque_nm'] for r in rows):.2f} N.m.")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        print("Hold the duck: leg motors going limp.")
        for s_ in held:
            bus.off(s_)


if __name__ == "__main__":
    main()
