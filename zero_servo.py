#!/usr/bin/env python3
"""Clear a Feetech STS servo's stored position offset and move it to true zero (step 2048).

Run it after configure_motor.py has given the servo its ID, and BEFORE fitting the horn:
    python3 zero_servo.py --id 10
    python3 zero_servo.py --id 10 11 12          # several servos on the bus, one after another
    python3 zero_servo.py --id 10 --check        # only read and report, change nothing

What it does for each ID:
  1. motor off, unlock the saved settings
  2. write offset = 0 into saved memory (EEPROM), then lock it again
  3. read the offset back to prove it saved
  4. motor on, move to step 2048 (the same zero configure_motor.py and the playground use)
  5. report where it ended up; the horn goes on now
Needs: pip install pyserial
"""
import argparse
import sys
import time

import serial

TORQUE_ENABLE, ACCEL, GOAL, LOCK, OFFSET, PRESENT_POS = 40, 41, 42, 55, 31, 56
PING, READ, WRITE = 0x01, 0x02, 0x03
ZERO = 2048


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
        d = self.txn(sid, READ, bytes([addr, n]))
        if d is None or len(d) < n:
            raise IOError(f"servo {sid} did not answer a read of address {addr}")
        return d[0] if n == 1 else d[0] | d[1] << 8

    def write(self, sid, addr, data):
        if self.txn(sid, WRITE, bytes([addr]) + bytes(data)) is None:
            raise IOError(f"servo {sid} did not confirm a write to address {addr}")


def offset_value(raw):  # sign-magnitude, bit 11 is the sign
    return -(raw & 0x7FF) if raw & 0x800 else raw & 0x7FF


def zero(bus, sid, check_only):
    if bus.txn(sid, PING) is None:
        print(f"ID {sid}: no answer. Is it plugged in and powered?")
        return False
    before_off = offset_value(bus.read(sid, OFFSET, 2))
    before_pos = bus.read(sid, PRESENT_POS, 2)
    print(f"ID {sid}: stored offset {before_off}, position {before_pos}")
    if check_only:
        return True

    bus.write(sid, TORQUE_ENABLE, [0])
    bus.write(sid, LOCK, [0])            # unlock so the EEPROM write is kept after power-off
    bus.write(sid, OFFSET, [0, 0])
    time.sleep(0.05)                     # give the EEPROM write time to finish
    bus.write(sid, LOCK, [1])
    after_off = offset_value(bus.read(sid, OFFSET, 2))
    if after_off != 0:
        print(f"ID {sid}: offset reads {after_off} after writing 0. Stopping; nothing else changed.")
        return False

    bus.write(sid, ACCEL, [20])          # gentle start and stop for this one move
    bus.write(sid, GOAL, [ZERO & 0xFF, ZERO >> 8, 0, 0, 0x58, 0x02])  # step 2048 at 600 steps/s
    bus.write(sid, TORQUE_ENABLE, [1])
    for _ in range(60):                  # wait up to 3 s for it to arrive
        time.sleep(0.05)
        pos = bus.read(sid, PRESENT_POS, 2)
        if abs(pos - ZERO) <= 3:
            break
    bus.write(sid, ACCEL, [0])           # back to what configure_motor.py set
    ok = abs(pos - ZERO) <= 3
    print(f"ID {sid}: offset now 0, position {pos} "
          f"({'at zero, holding. Fit the horn now.' if ok else 'did NOT reach 2048, check it is free to turn'})")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", type=int, nargs="+", required=True)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--check", action="store_true", help="only report offset and position")
    args = ap.parse_args()
    with serial.Serial(args.port, 1_000_000, timeout=0) as port:
        bus = Bus(port)
        results = [zero(bus, sid, args.check) for sid in args.id]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
