#!/usr/bin/env python3
"""Ping Feetech STS servos through a Waveshare Bus Servo Adapter (jumper on B, USB).
Read-only: it never writes to a servo or moves it.
Usage: python servo_ping.py PORT        e.g. COM5, /dev/ttyACM0, /dev/cu.usbmodem1101
       python servo_ping.py PORT --scan to try every ID (0-253)
Needs: pip install pyserial"""
import sys, time
import serial

def packet(sid, instr, params=b""):
    body = bytes([sid, len(params) + 2, instr]) + params
    return b"\xff\xff" + body + bytes([~sum(body) & 0xFF])

def transact(port, sid, instr, params=b""):
    out = packet(sid, instr, params)
    port.reset_input_buffer()
    port.write(out)
    port.flush()
    time.sleep(0.003)
    data = port.read(64)
    if data.startswith(out):          # some adapters echo what they send
        data = data[len(out):]
    i = data.find(b"\xff\xff")
    if i < 0 or len(data) < i + 6:
        return None
    rid, ln, err = data[i + 2], data[i + 3], data[i + 4]
    payload = data[i + 5:i + 5 + ln - 2]
    chk = data[i + 3 + ln] if len(data) > i + 3 + ln else None
    if rid != sid or chk != (~sum(data[i + 2:i + 3 + ln]) & 0xFF):
        return None
    return err, payload

def read(port, sid, addr, n):
    r = transact(port, sid, 0x02, bytes([addr, n]))
    return None if r is None else r[1]

def report(port, sid):
    model = read(port, sid, 3, 2)
    pos = read(port, sid, 56, 2)
    volt = read(port, sid, 62, 1)
    temp = read(port, sid, 63, 1)
    vmax = read(port, sid, 14, 1)
    vmin = read(port, sid, 15, 1)
    off = read(port, sid, 31, 2)
    def u16(b): return None if b is None else b[0] | b[1] << 8
    def offset(b):  # sign-magnitude, bit 11 is the sign
        if b is None: return "?"
        v = u16(b); mag = v & 0x7FF
        return -mag if v & 0x800 else mag
    print(f"ID {sid}: model {u16(model)}  position {u16(pos)} / 4095  "
          f"supply {volt[0] / 10 if volt else '?'} V  temp {temp[0] if temp else '?'} C  "
          f"voltage limits {vmin[0] / 10 if vmin else '?'}-{vmax[0] / 10 if vmax else '?'} V  "
          f"stored offset {offset(off)}")

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ids = range(254) if "--scan" in sys.argv else [1]
    with serial.Serial(sys.argv[1], 1_000_000, timeout=0.02) as port:
        found = 0
        for sid in ids:
            if transact(port, sid, 0x01) is not None:
                found += 1
                report(port, sid)
        if not found:
            print("No servo answered. Check servo power, the jumper is on B, and the cable.",
                  "" if "--scan" in sys.argv else "Try --scan in case its ID is not 1.")

if __name__ == "__main__":
    main()
