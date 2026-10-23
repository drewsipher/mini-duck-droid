#!/usr/bin/env python3
"""Rolling wave through all 14 duck servos, then all back to zero at once, logging power draw.

    python3 wave_power_test.py                     # 40 degree wave, 3 rolls, fastest snap back
    python3 wave_power_test.py --amplitude 60 --cycles 5
    python3 wave_power_test.py --return-speed 800   # gentler snap back (steps/s, 0 = fastest)

BENCH ONLY, servos loose with horns free to swing. Bench supply about 7.2 V.
The snap back pulls all 14 motors at once: set the supply's current limit to what you are
happy to see drawn (5 A is a good start) and watch whether the supply drops into CC mode.
Ctrl+C at any time switches every motor off. Each sample is saved to power_log.csv.
Needs: pip install pyserial
"""
import argparse
import csv
import math
import sys
import time

import serial

JOINTS = [(10, "right_hip_yaw"), (11, "right_hip_roll"), (12, "right_hip_pitch"), (13, "right_knee"),
          (14, "right_ankle"), (20, "left_hip_yaw"), (21, "left_hip_roll"), (22, "left_hip_pitch"),
          (23, "left_knee"), (24, "left_ankle"), (30, "neck_pitch"), (31, "head_pitch"),
          (32, "head_yaw"), (33, "head_roll")]
TORQUE, ACCEL, GOAL, PRESENT = 40, 41, 42, 56
READ, WRITE, SYNC_WRITE, BROADCAST = 0x02, 0x03, 0x83, 0xFE
ZERO, STEPS_PER_DEG = 2048, 4096 / 360
ERRORS = {0: "voltage", 1: "sensor", 2: "temperature", 3: "current", 4: "angle", 5: "overload"}


class Bus:
    def __init__(self, port):
        self.port = port

    @staticmethod
    def packet(sid, instr, params=b""):
        body = bytes([sid, len(params) + 2, instr]) + bytes(params)
        return b"\xff\xff" + body + bytes([~sum(body) & 0xFF])

    def txn(self, sid, instr, params=b"", wait=0.03):
        self.port.reset_input_buffer()
        self.port.write(self.packet(sid, instr, params))
        if sid == BROADCAST:
            time.sleep(0.002)
            return None
        buf, end = bytearray(), time.monotonic() + wait
        while time.monotonic() < end:
            n = self.port.in_waiting
            if not n:
                time.sleep(0.0003)
                continue
            buf += self.port.read(n)
            i = buf.find(b"\xff\xff")
            if i >= 0 and len(buf) >= i + 4 and len(buf) >= i + 4 + buf[i + 3]:
                pkt = buf[i + 2:i + 4 + buf[i + 3]]
                if pkt[0] == sid and pkt[-1] == (~sum(pkt[:-1]) & 0xFF):
                    return bytes(pkt[3:-1])
                return None
        return None

    def sync_write(self, addr, per_servo):
        """One packet, every servo acts on it at the same moment. per_servo: {id: bytes}"""
        length = len(next(iter(per_servo.values())))
        params = bytes([addr, length]) + b"".join(bytes([sid]) + bytes(d) for sid, d in per_servo.items())
        self.txn(BROADCAST, SYNC_WRITE, params)

    def goals(self, targets, speeds):
        self.sync_write(GOAL, {sid: [int(p) & 0xFF, int(p) >> 8, 0, 0, int(speeds[sid]) & 0xFF,
                                      int(speeds[sid]) >> 8] for sid, p in targets.items()})

    def all_torque(self, on):
        self.txn(BROADCAST, WRITE, bytes([TORQUE, 1 if on else 0]))

    def all_accel(self, value):
        self.txn(BROADCAST, WRITE, bytes([ACCEL, value]))

    def status(self, sid):
        d = self.txn(sid, READ, bytes([PRESENT, 15]))
        if d is None or len(d) < 15:
            return None
        return {"pos": d[0] | d[1] << 8, "volt": d[6] / 10, "errors": d[9],
                "ma": ((d[13] | d[14] << 8) & 0x7FFF) * 6.5}


class Recorder:
    def __init__(self, bus, ids, path):
        self.bus, self.ids, self.t0 = bus, ids, time.monotonic()
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["t", "phase", "total_mA", "total_W", "min_V"] + [f"mA_{i}" for i in ids])
        self.stats = {}
        self.errors = set()
        self.missing = set()

    def sample(self, phase):
        mas, volts = {}, []
        for sid in self.ids:
            st = self.bus.status(sid)
            if st is None:
                self.missing.add(sid)
                mas[sid] = 0
                continue
            mas[sid] = st["ma"]
            volts.append(st["volt"])
            for bit, name in ERRORS.items():
                if st["errors"] >> bit & 1:
                    self.errors.add((sid, name))
        total = sum(mas.values())
        vmin = min(volts) if volts else 0
        watts = total / 1000 * (sum(volts) / len(volts) if volts else 0)
        t = time.monotonic() - self.t0
        self.w.writerow([f"{t:.3f}", phase, f"{total:.0f}", f"{watts:.2f}", vmin] + [f"{mas[i]:.0f}" for i in self.ids])
        s = self.stats.setdefault(phase, {"n": 0, "sum": 0.0, "peak": 0.0, "vmin": 99.0, "wpeak": 0.0})
        s["n"] += 1
        s["sum"] += total
        s["peak"] = max(s["peak"], total)
        s["wpeak"] = max(s["wpeak"], watts)
        s["vmin"] = min(s["vmin"], vmin) if volts else s["vmin"]
        print(f"\r  {phase:10s} t={t:5.1f}s  total {total / 1000:5.2f} A  {watts:5.1f} W  "
              f"lowest supply {vmin:4.1f} V   ", end="", flush=True)
        return total

    def close(self):
        self.f.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--amplitude", type=float, default=40, help="degrees either side of zero (max 90)")
    ap.add_argument("--cycles", type=float, default=3, help="how many times the wave rolls through")
    ap.add_argument("--period", type=float, default=2.0, help="seconds per roll")
    ap.add_argument("--return-speed", type=int, default=0, help="snap-back speed, steps/s, 0 = fastest")
    ap.add_argument("--log", default="power_log.csv")
    args = ap.parse_args()
    amp = min(abs(args.amplitude), 90) * STEPS_PER_DEG

    with serial.Serial(args.port, 1_000_000, timeout=0) as port:
        bus = Bus(port)
        present = {sid: bus.status(sid) for sid, _ in JOINTS}
        ids = [sid for sid, _ in JOINTS if present[sid] is not None]
        missing = [sid for sid, _ in JOINTS if present[sid] is None]
        if missing:
            print(f"Not answering: {missing}. Carrying on with {len(ids)} servos.")
        if not ids:
            sys.exit("No servos answered.")
        rec = Recorder(bus, ids, args.log)
        try:
            print(f"Idle, motors off:")
            for _ in range(5):
                rec.sample("idle")
            # hold where they are, then switch on (no jump), then ease to zero
            bus.all_accel(30)
            bus.goals({s: present[s]["pos"] for s in ids}, {s: 400 for s in ids})
            bus.all_torque(True)
            bus.goals({s: ZERO for s in ids}, {s: 400 for s in ids})
            end = time.monotonic() + 1.0
            print("\nHolding at zero:")
            while time.monotonic() < end:
                rec.sample("hold_zero")

            print("\nRolling wave:")
            bus.all_accel(0)
            lag = 2 * math.pi / len(ids)          # one full wave spread across the servos
            w = 2 * math.pi / args.period
            duration = args.cycles * args.period
            start = last_send = time.monotonic()
            prev = {s: ZERO for s in ids}
            while True:
                t = time.monotonic() - start
                if t >= duration:
                    break
                ramp = min(1.0, t / args.period)  # grow the wave over the first roll
                targets = {s: ZERO + ramp * amp * math.sin(w * t - i * lag) for i, s in enumerate(ids)}
                dt = max(0.02, time.monotonic() - last_send)
                speeds = {s: max(60, min(3400, abs(targets[s] - prev[s]) / dt * 1.5)) for s in ids}
                bus.goals(targets, speeds)
                last_send, prev = time.monotonic(), targets
                rec.sample("wave")

            # freeze where the wave left them (spread across the whole range), let them settle
            settle = time.monotonic() + 0.6
            while time.monotonic() < settle:
                rec.sample("frozen")

            print("\nAll back to zero at once:")
            bus.goals({s: ZERO for s in ids}, {s: args.return_speed for s in ids})
            end = time.monotonic() + 1.5
            while time.monotonic() < end:
                rec.sample("snap_back")

            print("\nHolding at zero again:")
            end = time.monotonic() + 1.0
            while time.monotonic() < end:
                rec.sample("hold_end")
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            bus.all_torque(False)
            bus.all_accel(0)
            rec.close()

        print("\n\nMotors off.\n")
        print(f"  {'phase':10s} {'average':>9s} {'peak':>9s} {'peak W':>8s} {'lowest V':>9s}")
        for phase, s in rec.stats.items():
            print(f"  {phase:10s} {s['sum'] / s['n'] / 1000:8.2f}A {s['peak'] / 1000:8.2f}A "
                  f"{s['wpeak']:7.1f}W {s['vmin']:8.1f}V")
        print("\n  Totals add each servo's own current reading. The servos sample their current,")
        print("  so very short spikes can be missed: the bench supply display is the real check.")
        if rec.errors:
            print("  Servo warnings seen:", ", ".join(f"ID {s} {n}" for s, n in sorted(rec.errors)))
        if rec.missing:
            print("  Servos that missed a reply at least once:", sorted(rec.missing))
        print(f"  Every sample is in {args.log}")


if __name__ == "__main__":
    main()
