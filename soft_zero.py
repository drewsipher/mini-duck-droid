#!/usr/bin/env python3
"""Move each servo to its zero position, one at a time, gently, watching the current draw.

For each servo it shows where it is and how far it will move, then waits for you:
    Enter = go     s = skip this one     q = quit (all motors off)
It moves slowly at reduced strength. It cuts that motor and reports an error if:
  - the current stays above the limit (default 500 mA), or
  - the servo flags overheating / overcurrent / overload, or
  - it stops making progress before reaching zero (blocked or binding).
Servos that reached zero keep holding, softly, while you do the next one. At the end,
Enter releases them all and restores full strength.

Zero = step 2048 plus the joint's offset in ~/duck_config.json (all 0 until find_soft_offsets.py).
Hold the duck in the air or on a stand so the legs can move freely.

    python3 soft_zero.py                     # all 14, right leg, left leg, neck and head
    python3 soft_zero.py --ids 32 33         # just these
    python3 soft_zero.py --strength 20 --max-ma 400
    python3 soft_zero.py --release           # motors off and full strength restored, nothing moves
Needs: pip install pyserial
"""
import argparse
import json
import math
import os
import sys
import time

import serial

JOINTS = [  # (id, name) in the order to work through
    (10, "right_hip_yaw"), (11, "right_hip_roll"), (12, "right_hip_pitch"), (13, "right_knee"),
    (14, "right_ankle"), (20, "left_hip_yaw"), (21, "left_hip_roll"), (22, "left_hip_pitch"),
    (23, "left_knee"), (24, "left_ankle"), (30, "neck_pitch"), (31, "head_pitch"),
    (32, "head_yaw"), (33, "head_roll"),
]
TORQUE, ACCEL, GOAL, TORQUE_LIMIT, STATE = 40, 41, 42, 48, 56
PING, READ, WRITE = 0x01, 0x02, 0x03
ZERO = 2048
STEPS_PER_DEG = 4096 / 360
MA_PER_UNIT = 6.5          # STS3215 present-current unit
FAULTS = {2: "overheating", 3: "overcurrent", 5: "overload"}   # bit 0 (voltage) is only a warning
ARRIVED = round(1.5 * STEPS_PER_DEG)


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

    def ping(self, sid):
        return self.txn(sid, PING, wait=0.02) is not None

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
        d = self.read(sid, STATE, 15)            # 56 pos, 58 speed, 60 load, 62 volt, 63 temp, 65 status, 69 current
        cur = d[13] | d[14] << 8
        return {"pos": d[0] | d[1] << 8, "volt": d[6] / 10, "temp": d[7], "status": d[9],
                "ma": (cur & 0x7FFF) * MA_PER_UNIT}

    def release(self, sid):
        """Motor off and full strength restored (strength is RAM: it would also reset at power-up)."""
        try:
            self.write(sid, TORQUE, 0)
            self.write16(sid, TORQUE_LIMIT, 1000)
        except IOError as e:
            print(f"  warning: {e}")


class Trip(Exception):
    pass


def deg(steps):
    return steps / STEPS_PER_DEG


def zero_targets():
    """Zero step per joint: 2048 plus duck_config.json's offset (radians, same convention as the runtime)."""
    offsets = {}
    try:
        offsets = json.load(open(os.path.expanduser("~/duck_config.json"))).get("joints_offsets", {})
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {name: ZERO + round(offsets.get(name, 0.0) * 4096 / (2 * math.pi)) for _, name in JOINTS}


def faults(status):
    return [txt for bit, txt in FAULTS.items() if status >> bit & 1]


def move_to_zero(bus, sid, goal, a):
    st = bus.state(sid)
    bus.write16(sid, TORQUE_LIMIT, a.strength * 10)
    bus.write(sid, ACCEL, 10)                                   # gentle start and stop
    bus.write16(sid, GOAL, st["pos"])                           # hold where it is first: no jump
    bus.write(sid, TORQUE, 1)
    speed = round(a.speed * STEPS_PER_DEG)
    bus.write(sid, GOAL, goal & 0xFF, goal >> 8, 0, 0, speed & 0xFF, speed >> 8)

    peak, high, best, best_t = 0.0, 0, abs(st["pos"] - goal), time.monotonic()
    timeout = time.monotonic() + abs(st["pos"] - goal) / speed + 6
    while True:
        time.sleep(0.02)
        st = bus.state(sid)
        peak = max(peak, st["ma"])
        high = high + 1 if st["ma"] > a.max_ma else 0
        if high >= 5:                                           # about 0.1 s above the limit
            raise Trip(f"current {st['ma']:.0f} mA, above the {a.max_ma} mA limit", st, peak)
        if faults(st["status"]):
            raise Trip("servo reports " + ", ".join(faults(st["status"])), st, peak)
        err = abs(st["pos"] - goal)
        if err <= ARRIVED:
            break
        if err < best - 3:
            best, best_t = err, time.monotonic()
        elif time.monotonic() - best_t > 1.5:
            raise Trip(f"stopped {deg(st['pos'] - goal):+.1f} deg short of zero: blocked or binding", st, peak)
        if time.monotonic() > timeout:
            raise Trip("took too long to reach zero", st, peak)

    hold = []                                                   # watch it settle for half a second
    for _ in range(25):
        time.sleep(0.02)
        st = bus.state(sid)
        hold.append(st["ma"])
        peak = max(peak, st["ma"])
    holding = sorted(hold)[len(hold) // 2]
    if holding > a.max_ma:
        raise Trip(f"holding current {holding:.0f} mA at zero, above the {a.max_ma} mA limit", st, peak)
    return st, peak, holding


def ask(prompt):
    try:
        return input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return "q"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--ids", type=int, nargs="+", help="only these IDs (default: all 14)")
    ap.add_argument("--strength", type=int, default=30, help="percent of full strength while moving (default 30)")
    ap.add_argument("--speed", type=float, default=12, help="degrees per second (default 12)")
    ap.add_argument("--max-ma", type=int, default=500, help="current limit in mA (default 500)")
    ap.add_argument("--release", action="store_true", help="motors off and full strength back, then exit")
    a = ap.parse_args()
    a.strength = max(5, min(100, a.strength))

    joints = [(i, n) for i, n in JOINTS if not a.ids or i in a.ids]
    try:
        bus = Bus(serial.Serial(a.port, 1_000_000, timeout=0))
    except serial.SerialException as e:
        sys.exit(f"Can't open {a.port}: {e}")
    if a.release:
        for sid, name in joints:
            bus.release(sid)
        print("Motors off, full strength restored.")
        return

    targets = zero_targets()
    missing = [sid for sid, _ in joints if not bus.ping(sid)]
    if missing:
        print(f"Not answering: {missing}. They'll be skipped.")
    print(f"Strength {a.strength} %, speed {a.speed:g} deg/s, current limit {a.max_ma} mA. "
          "Hold the duck in the air.\nCtrl+C at any time turns every motor off.\n")

    holding, results = [], []
    try:
        for sid, name in joints:
            if sid in missing:
                results.append((sid, name, "not answering"))
                continue
            while True:
                st = bus.state(sid)
                goal = targets[name]
                move = deg(goal - st["pos"])
                note = "   <-- big move, check it's free" if abs(move) > 45 else ""
                print(f"ID {sid:2d} {name:16s} at {deg(st['pos'] - ZERO):+6.1f} deg, "
                      f"will turn {move:+6.1f} deg to zero.{note}")
                c = ask("  Enter = go, s = skip, q = quit: ")
                if c == "q":
                    raise KeyboardInterrupt
                if c == "s":
                    results.append((sid, name, "skipped"))
                    break
                try:
                    st, peak, hold = move_to_zero(bus, sid, goal, a)
                except Trip as t:
                    msg, st, peak = t.args
                    bus.release(sid)
                    print(f"  ERROR: {msg}. Motor {sid} turned off. "
                          f"(position {deg(st['pos'] - ZERO):+.1f} deg, peak {peak:.0f} mA, {st['temp']} C)")
                    c = ask("  r = retry, s = skip, q = quit: ")
                    if c == "r":
                        continue
                    results.append((sid, name, f"ERROR: {msg}"))
                    if c == "q":
                        raise KeyboardInterrupt
                    break
                holding.append(sid)
                print(f"  OK: at {deg(st['pos'] - goal):+.1f} deg from zero, "
                      f"peak {peak:.0f} mA, holding {hold:.0f} mA, {st['temp']} C. Holding softly.")
                results.append((sid, name, f"ok (peak {peak:.0f} mA)"))
                break
    except KeyboardInterrupt:
        print("\nStopping: all motors off.")
        for sid, _ in joints:
            bus.release(sid)
        holding = []

    print("\nSummary:")
    for sid, name, r in results:
        print(f"  ID {sid:2d} {name:16s} {r}")
    if holding:
        ask("\nPress Enter to release all motors (they go limp) and restore full strength: ")
        for sid in holding:
            bus.release(sid)
        print("Released.")


if __name__ == "__main__":
    main()
