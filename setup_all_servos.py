#!/usr/bin/env python3
"""Set up all 14 Open Duck Mini v2 servos, one at a time.

For each servo it: finds it (new ones are ID 1), applies the same settings as the official
configure_motor.py (position mode, P=32, I=0, D=0, acceleration 0), clears the stored offset,
gives it its joint ID, locks the saved settings, moves it to true zero (step 2048) and tells
you which ID to write on it. Then it waits while you fit the horn and swap in the next servo.

    python3 setup_all_servos.py                    # all 14, starting with ID 10
    python3 setup_all_servos.py --start-at 22      # resume from a joint
    python3 setup_all_servos.py --port /dev/ttyACM1

Bench supply about 7.2 V, adapter jumper on B, USB to this computer. ONE servo on the bus.
Needs: pip install pyserial
"""
import argparse
import csv
import datetime
import sys
import time

import serial

JOINTS = [  # order to work through: right leg, left leg, neck and head
    (10, "right_hip_yaw"), (11, "right_hip_roll"), (12, "right_hip_pitch"), (13, "right_knee"),
    (14, "right_ankle"), (20, "left_hip_yaw"), (21, "left_hip_roll"), (22, "left_hip_pitch"),
    (23, "left_knee"), (24, "left_ankle"), (30, "neck_pitch"), (31, "head_pitch"),
    (32, "head_yaw"), (33, "head_roll"),
]
ID, P, D, I, OFFSET, MODE = 5, 21, 22, 23, 31, 33          # saved settings (EEPROM)
TORQUE, ACCEL, GOAL, LOCK, POS, MAX_ACCEL = 40, 41, 42, 55, 56, 85
PING, READ, WRITE = 0x01, 0x02, 0x03
ZERO = 2048
LOG = "servo_setup_log.csv"


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

    def ping(self, sid, wait=0.015):
        return self.txn(sid, PING, wait=wait) is not None

    def scan(self):
        found = []
        for sid in range(254):
            if sid % 32 == 0:
                print(f"\r  looking for servos… {sid * 100 // 253:3d}%", end="", flush=True)
            if self.ping(sid):
                found.append(sid)
        print("\r" + " " * 40 + "\r", end="")
        return found

    def read(self, sid, addr, n=1):
        for _ in range(3):
            d = self.txn(sid, READ, bytes([addr, n]))
            if d is not None and len(d) >= n:
                return d[0] if n == 1 else d[0] | d[1] << 8
        raise IOError(f"servo {sid} did not answer a read of address {addr}")

    def write(self, sid, addr, *data, eeprom=False):
        for _ in range(3):
            if self.txn(sid, WRITE, bytes([addr, *data])) is not None:
                if eeprom:
                    time.sleep(0.02)  # let the EEPROM write finish
                return
        raise IOError(f"servo {sid} did not confirm a write to address {addr}")


def offset_value(raw):  # sign-magnitude, bit 11 is the sign
    return -(raw & 0x7FF) if raw & 0x800 else raw & 0x7FF


def banner(*lines):
    width = max(len(l) for l in lines) + 4
    print("\n" + "=" * width)
    for l in lines:
        print(f"  {l}")
    print("=" * width)


def ask(prompt):
    """Enter = go, s = skip this joint, q = quit."""
    try:
        return input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return "q"


def find_one(bus, target, done):
    """Return the ID of the single servo on the bus, or None to retry."""
    found = bus.scan()
    if not found:
        print("  No servo answered. Check it is plugged in and the supply output is on.\n"
              "  Also check only ONE new servo is connected: two new servos both answer as ID 1\n"
              "  and drown each other out.")
        return None
    if len(found) > 1:
        print(f"  More than one servo answered ({found}). Leave only ONE connected.")
        return None
    sid = found[0]
    other = {i: n for i, n in done.items() if i != target}
    if sid in other:
        print(f"  This servo is already set up as ID {sid} ({other[sid]}) earlier in this run.")
        if ask("  Re-do it as this joint instead? [y/N] ") != "y":
            return None
    elif sid not in (1, target):
        print(f"  Found a servo with ID {sid}, not a new one (ID 1).")
        if ask(f"  Use it anyway and turn it into ID {target}? [y/N] ") != "y":
            return None
    return sid


def setup(bus, sid, target, name):
    old_offset = offset_value(bus.read(sid, OFFSET, 2))
    print(f"  Found ID {sid}. Stored offset {old_offset}. Configuring…")
    bus.write(sid, TORQUE, 0)
    bus.write(sid, LOCK, 0)                         # unlock: EEPROM writes now stick
    bus.write(sid, MODE, 0, eeprom=True)            # position mode
    bus.write(sid, MAX_ACCEL, 0, eeprom=True)
    bus.write(sid, ACCEL, 0)
    bus.write(sid, P, 32, eeprom=True)
    bus.write(sid, I, 0, eeprom=True)
    bus.write(sid, D, 0, eeprom=True)
    bus.write(sid, OFFSET, 0, 0, eeprom=True)
    if sid != target:
        bus.txn(sid, WRITE, bytes([ID, target]))    # the reply may come from either ID; check below
        time.sleep(0.05)
        if not bus.ping(target, wait=0.05):
            raise IOError(f"after changing the ID, nothing answers as ID {target}")
    bus.write(target, LOCK, 1)                      # lock the saved settings again

    checks = {"ID": (target, target), "mode": (bus.read(target, MODE), 0), "P": (bus.read(target, P), 32),
              "I": (bus.read(target, I), 0), "D": (bus.read(target, D), 0),
              "offset": (offset_value(bus.read(target, OFFSET, 2)), 0)}
    bad = {k: v for k, (v, want) in checks.items() if v != want}
    if bad:
        raise IOError(f"read-back does not match: {bad}")

    bus.write(target, ACCEL, 20)                    # gentle move to zero
    bus.write(target, GOAL, ZERO & 0xFF, ZERO >> 8, 0, 0, 600 & 0xFF, 600 >> 8)
    bus.write(target, TORQUE, 1)
    pos = None
    for _ in range(80):
        time.sleep(0.05)
        pos = bus.read(target, POS, 2)
        if abs(pos - ZERO) <= 3:
            break
    bus.write(target, ACCEL, 0)
    if abs(pos - ZERO) > 3:
        raise IOError(f"did not reach zero (position {pos}). Is it free to turn?")
    return old_offset, pos


def log_row(row):
    new = False
    try:
        open(LOG).close()
    except FileNotFoundError:
        new = True
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "id", "joint", "found_as_id", "old_offset", "final_position"])
        w.writerow(row)


def verify_chain(bus, done):
    banner("All done. Optional final check:",
           "turn the supply off, chain ALL the servos together, turn it on.")
    if ask("Press Enter to check the whole chain, or q to finish: ") == "q":
        return
    found = set(bus.scan())
    for sid, name in JOINTS:
        if sid not in found:
            print(f"  ID {sid:2d} {name:16s} MISSING")
            continue
        off = offset_value(bus.read(sid, OFFSET, 2))
        pos = bus.read(sid, POS, 2)
        flag = "ok" if off == 0 else "OFFSET NOT 0"
        print(f"  ID {sid:2d} {name:16s} offset {off:5d}  position {pos:4d}  {flag}")
    extra = found - {s for s, _ in JOINTS}
    if extra:
        print(f"  Unexpected IDs on the bus: {sorted(extra)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--start-at", type=int, default=JOINTS[0][0], help="joint ID to begin with")
    args = ap.parse_args()
    ids = [j[0] for j in JOINTS]
    if args.start_at not in ids:
        sys.exit(f"--start-at must be one of {ids}")
    todo = JOINTS[ids.index(args.start_at):]

    with serial.Serial(args.port, 1_000_000, timeout=0) as port:
        bus = Bus(port)
        done = {}
        for n, (target, name) in enumerate(todo, 1):
            banner(f"Servo {n} of {len(todo)}: {name.replace('_', ' ')}  ->  ID {target}",
                   "Connect ONE servo (no horn yet).")
            while True:
                a = ask("Press Enter when it's connected (s = skip, q = quit): ")
                if a == "q":
                    print("Stopped. Resume later with --start-at", target)
                    return
                if a == "s":
                    break
                sid = find_one(bus, target, done)
                if sid is None:
                    continue
                try:
                    old_offset, pos = setup(bus, sid, target, name)
                except IOError as e:
                    print(f"  PROBLEM: {e}. Fix it and press Enter to try again.")
                    continue
                done[target] = name
                log_row([datetime.datetime.now().isoformat(timespec="seconds"), target, name, sid,
                         old_offset, pos])
                banner(f"LABEL THIS SERVO:  ID {target}  ({name.replace('_', ' ')})",
                       f"Offset cleared, holding at zero (position {pos}).",
                       "Fit the horn now, lined up as in the guide photo.",
                       "Then turn the supply output off and swap in the next servo.")
                break
        print(f"\nSet up {len(done)} servo(s). A record is in {LOG}.")
        verify_chain(bus, done)


if __name__ == "__main__":
    main()
