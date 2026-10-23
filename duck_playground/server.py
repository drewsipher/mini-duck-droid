#!/usr/bin/env python3
"""Duck Playground: a web page for playing with the 14 duck servos on the bench.

Run on the Pi (servo adapter on /dev/ttyACM0, bench supply on):
    python3 server.py                 # then open http://bdxv2.local:8090 on a phone or laptop
    python3 server.py --demo          # pretend servos, for trying it without hardware

Modes: hands-on dials, follow the leader, rolling wave, record and replay, STOP.
Safety: bench only (servos loose, horns free). Angles are kept within +/-150 degrees.
If no browser has checked in for 4 seconds, every motor switches off.
Only changes the servos' working memory (targets, torque on/off); saved settings are untouched.
Do not run it at the same time as another program that uses the servo adapter.
"""
import argparse
import json
import math
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

JOINTS = [(10, "right hip yaw"), (11, "right hip roll"), (12, "right hip pitch"), (13, "right knee"),
          (14, "right ankle"), (20, "left hip yaw"), (21, "left hip roll"), (22, "left hip pitch"),
          (23, "left knee"), (24, "left ankle"), (30, "neck pitch"), (31, "head pitch"),
          (32, "head yaw"), (33, "head roll")]
IDS = [j[0] for j in JOINTS]
TORQUE, ACCEL, GOAL, PRESENT = 40, 41, 42, 56
READ, WRITE, SYNC_WRITE, BROADCAST = 0x02, 0x03, 0x83, 0xFE
STEPS_PER_DEG = 4096 / 360
LIMIT_DEG = 150
ERRORS = {0: "voltage", 1: "sensor", 2: "too hot", 3: "current", 4: "angle", 5: "overload"}
HERE = Path(__file__).resolve().parent


def to_steps(deg):
    deg = max(-LIMIT_DEG, min(LIMIT_DEG, deg))
    return int(round(2048 + deg * STEPS_PER_DEG))


def to_deg(steps):
    return (steps - 2048) / STEPS_PER_DEG


# ------------------------------------------------------------------ serial bus
class Bus:
    def __init__(self, port):
        self.port = port

    @staticmethod
    def packet(sid, instr, params=b""):
        body = bytes([sid, len(params) + 2, instr]) + bytes(params)
        return b"\xff\xff" + body + bytes([~sum(body) & 0xFF])

    def txn(self, sid, instr, params=b"", wait=0.02):
        self.port.reset_input_buffer()
        self.port.write(self.packet(sid, instr, params))
        if sid == BROADCAST:
            time.sleep(0.001)
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

    def status(self, sid):
        d = self.txn(sid, READ, bytes([PRESENT, 15]))
        if d is None or len(d) < 15:
            return None
        load = d[4] | d[5] << 8
        return {"steps": d[0] | d[1] << 8, "load": (load & 0x3FF) / 10, "volt": d[6] / 10, "temp": d[7],
                "err": d[9], "ma": ((d[13] | d[14] << 8) & 0x7FFF) * 6.5}

    def torque(self, sid, on):
        self.txn(sid, WRITE, bytes([TORQUE, 1 if on else 0]), wait=0.01)

    def all_off(self):
        self.txn(BROADCAST, WRITE, bytes([TORQUE, 0]))

    def accel_all(self, v):
        self.txn(BROADCAST, WRITE, bytes([ACCEL, v]))

    def goals(self, goals):
        """goals: {id: (steps, speed)} sent as ONE packet so every servo starts together."""
        if not goals:
            return
        params = bytearray([GOAL, 6])
        for sid, (st, sp) in goals.items():
            st, sp = int(st), int(sp)
            params += bytes([sid, st & 0xFF, st >> 8, 0, 0, sp & 0xFF, sp >> 8])
        self.txn(BROADCAST, SYNC_WRITE, params)


class FakePort:
    """Pretend servos for --demo. A servo with its motor off drifts as if someone turned it."""

    def __init__(self):
        self.s = {i: {"pos": 2048.0 + random.randint(-300, 300), "goal": 2048, "speed": 600, "torque": 0,
                      "ph": random.random() * 6, "vel": 0.0} for i in IDS}
        self.out, self.t = bytearray(), time.monotonic()

    def _tick(self):
        now = time.monotonic()
        dt, self.t = now - self.t, now
        for v in self.s.values():
            before = v["pos"]
            if v["torque"]:
                step = (v["speed"] or 3400) * dt
                v["pos"] += max(-step, min(step, v["goal"] - v["pos"]))
            else:
                v["pos"] = 2048 + 700 * math.sin(now * 0.7 + v["ph"])
            v["vel"] = (v["pos"] - before) / dt if dt else 0

    @property
    def in_waiting(self):
        return len(self.out)

    def read(self, n):
        d, self.out = bytes(self.out[:n]), self.out[n:]
        return d

    def reset_input_buffer(self):
        self.out.clear()

    def write(self, b):
        self._tick()
        sid, instr, p = b[2], b[4], b[5:-1]
        if instr == SYNC_WRITE:
            for k in range(2, len(p), 7):
                v = self.s[p[k]]
                v["goal"], v["speed"] = p[k + 1] | p[k + 2] << 8, p[k + 5] | p[k + 6] << 8
            return
        if sid == BROADCAST:
            if p[0] == TORQUE:
                for v in self.s.values():
                    v["torque"] = p[1]
            return
        v = self.s.get(sid)
        if v is None:
            return
        data = b""
        if instr == READ:
            pos = int(v["pos"])
            moving = v["torque"] and abs(v["vel"]) > 20
            load = (random.randint(120, 260) if moving else random.randint(15, 40)) if v["torque"] else 0
            cur = int(abs(v["vel"]) / 25 + 3) if v["torque"] else 1
            regs = bytes([pos & 255, pos >> 8, 0, 0, load & 255, load >> 8, 70, 24, 0, 0, 0, 0, 0,
                          cur & 255, cur >> 8])
            data = regs[:p[1]]
        elif instr == WRITE and p[0] == TORQUE:
            v["torque"] = p[1]
        self.out += Bus.packet(sid, 0, data)


# ------------------------------------------------------------------ the duck
class Duck:
    def __init__(self, bus):
        self.bus = bus
        self.lock = threading.Lock()
        self.s = {i: {"steps": None, "load": 0, "volt": 0, "temp": 0, "err": 0, "ma": 0, "ok": False,
                      "torque": False, "target": None} for i in IDS}
        self.mode, self.leader, self.opposite = "free", None, False
        self.wave = {"amp": 45, "period": 2.5, "t0": 0}
        self.frames, self.rec_t0, self.play = [], 0, None
        self.pending = {}          # goals to send this loop
        self.last_client = time.monotonic()
        self.rot = 0
        bus.all_off()
        bus.accel_all(40)

    # ---- commands (called from web requests)
    def command(self, c):
        a = c.get("action")
        with self.lock:
            if a == "stop":
                self._all_off()
                self.mode = "free"
            elif a == "set":
                if self.mode != "free":
                    self._all_off()
                    self.mode = "free"
                sid = int(c["id"])
                self._hold_and_on([sid])
                self.s[sid]["target"] = to_steps(float(c["deg"]))
                self.pending[sid] = (self.s[sid]["target"], int(c.get("speed", 900)))
            elif a == "torque":
                sid = int(c["id"])
                if c["on"]:
                    self._hold_and_on([sid])
                else:
                    self.bus.torque(sid, False)
                    self.s[sid]["torque"] = False
            elif a == "centre":
                self.mode = "free"
                self._hold_and_on(IDS)
                for sid in IDS:
                    self.s[sid]["target"] = 2048
                    self.pending[sid] = (2048, 500)
            elif a == "follow":
                self.mode, self.leader = "follow", int(c["leader"])
                self.opposite = bool(c.get("opposite"))
                self.bus.torque(self.leader, False)
                self.s[self.leader]["torque"] = False
                self._hold_and_on([i for i in IDS if i != self.leader])
            elif a == "wave":
                self.mode = "wave"
                self.wave = {"amp": max(10, min(90, float(c.get("amp", 45)))),
                             "period": max(1.2, min(6, float(c.get("period", 2.5)))), "t0": time.monotonic()}
                self._hold_and_on(IDS)
            elif a == "record":
                self._all_off()
                self.mode, self.frames, self.rec_t0 = "record", [], time.monotonic()
            elif a == "play":
                if not self.frames:
                    return {"ok": False, "msg": "Nothing recorded yet"}
                self.mode = "play"
                self.play = {"t0": None, "loop": bool(c.get("loop")), "approach_until": time.monotonic() + 3}
                self._hold_and_on(IDS)
                first = self.frames[0][1]
                for sid, st in first.items():
                    self.s[sid]["target"] = st
                    self.pending[sid] = (st, 500)
            else:
                return {"ok": False, "msg": f"unknown action {a}"}
        return {"ok": True}

    def _hold_and_on(self, ids):
        hold = {}
        for sid in ids:
            if not self.s[sid]["torque"]:
                here = self.s[sid]["steps"] if self.s[sid]["steps"] is not None else 2048
                hold[sid] = (here, 500)
                self.s[sid]["target"] = here
        self.bus.goals(hold)
        for sid in hold:
            self.bus.torque(sid, True)
            self.s[sid]["torque"] = True

    def _all_off(self):
        self.bus.all_off()
        for v in self.s.values():
            v["torque"], v["target"] = False, None
        self.pending.clear()
        self.play = None

    # ---- the loop
    def run(self):
        while True:
            t = time.monotonic()
            try:
                with self.lock:
                    self._step()
            except Exception as e:  # keep going; a missed packet is not fatal
                print("loop error:", e, flush=True)
            time.sleep(max(0.0, 0.02 - (time.monotonic() - t)))

    def _read(self, sid):
        st = self.bus.status(sid)
        v = self.s[sid]
        if st is None:
            v["ok"] = False
            return
        v.update(st)
        v["ok"] = True

    def _step(self):
        now = time.monotonic()
        if now - self.last_client > 4 and any(v["torque"] for v in self.s.values()):
            self._all_off()
            self.mode = "free"
            print("no browser for 4 s: motors off", flush=True)

        # which servos to read this loop: the important ones plus a few in rotation
        if self.mode == "record":
            read_ids = IDS
        else:
            pri = [self.leader] if self.mode == "follow" else []
            rest = [i for i in IDS if i not in pri]
            n = 5
            read_ids = pri + [rest[(self.rot + k) % len(rest)] for k in range(n)]
            self.rot = (self.rot + n) % len(rest)
        for sid in read_ids:
            self._read(sid)

        if self.mode == "follow" and self.s[self.leader]["steps"] is not None:
            deg = to_deg(self.s[self.leader]["steps"])
            if self.opposite:
                deg = -deg
            for sid in IDS:
                if sid != self.leader:
                    st = to_steps(deg)
                    if self.s[sid]["target"] != st:
                        self.s[sid]["target"] = st
                        self.pending[sid] = (st, 2000)
        elif self.mode == "wave":
            w = self.wave
            t = now - w["t0"]
            ramp = min(1.0, t / w["period"])
            omega = 2 * math.pi / w["period"]
            for k, sid in enumerate(IDS):
                st = to_steps(ramp * w["amp"] * math.sin(omega * t - k * 2 * math.pi / len(IDS)))
                peak_speed = w["amp"] * STEPS_PER_DEG * omega * 1.4
                self.s[sid]["target"] = st
                self.pending[sid] = (st, max(80, min(3000, peak_speed)))
        elif self.mode == "record":
            if all(self.s[i]["steps"] is not None for i in IDS):
                self.frames.append((now - self.rec_t0, {i: self.s[i]["steps"] for i in IDS}))
            if now - self.rec_t0 > 60:
                self.mode = "free"
        elif self.mode == "play" and self.play:
            p = self.play
            if p["t0"] is None:  # moving to the first pose
                first = self.frames[0][1]
                close = all(self.s[i]["steps"] is None or abs(self.s[i]["steps"] - first[i]) < 40 for i in IDS)
                if close or now > p["approach_until"]:
                    p["t0"] = now
            else:
                t = now - p["t0"]
                if t > self.frames[-1][0]:
                    if p["loop"]:
                        p["t0"] = now
                        t = 0
                    else:
                        self.mode, self.play = "free", None
                if self.play:
                    i = max(0, min(len(self.frames) - 1, next((k for k, f in enumerate(self.frames) if f[0] >= t),
                                                              len(self.frames) - 1)))
                    j = min(i + 3, len(self.frames) - 1)
                    for sid, st in self.frames[i][1].items():
                        ahead = self.frames[j][1][sid]
                        dt = max(0.05, self.frames[j][0] - self.frames[i][0])
                        speed = max(80, min(3000, abs(ahead - st) / dt * 1.3))
                        self.s[sid]["target"] = st
                        self.pending[sid] = (st, speed)

        if self.pending:
            self.bus.goals({sid: g for sid, g in self.pending.items() if self.s[sid]["torque"]})
            self.pending.clear()

    def snapshot(self):
        with self.lock:
            self.last_client = time.monotonic()
            servos, amps, watts = [], 0.0, 0.0
            for sid, name in JOINTS:
                v = self.s[sid]
                a = v["ma"] / 1000 if v["ok"] else 0
                amps += a
                watts += a * v["volt"]
                servos.append({"id": sid, "name": name, "ok": v["ok"], "torque": v["torque"],
                               "deg": None if v["steps"] is None else round(to_deg(v["steps"]), 1),
                               "target": None if v["target"] is None else round(to_deg(v["target"]), 1),
                               "load": v["load"], "ma": round(v["ma"]), "volt": v["volt"], "temp": v["temp"],
                               "errors": [txt for b, txt in ERRORS.items() if v["err"] >> b & 1]})
            volts = [v["volt"] for v in self.s.values() if v["ok"]]
            return {"mode": self.mode, "leader": self.leader, "opposite": self.opposite, "servos": servos,
                    "amps": round(amps, 2), "watts": round(watts, 1),
                    "volts": round(min(volts), 1) if volts else None,
                    "recorded": round(self.frames[-1][0], 1) if self.frames else 0,
                    "rec_live": round(time.monotonic() - self.rec_t0, 1) if self.mode == "record" else None,
                    "wave": {k: self.wave[k] for k in ("amp", "period")}}


# ------------------------------------------------------------------ web server
def make_handler(duck):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self.send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            if self.path.startswith("/api/state"):
                return self.send(200, json.dumps(duck.snapshot()).encode())
            self.send(404, b'{"error":"not found"}')

        def do_POST(self):
            if self.path != "/api/cmd":
                return self.send(404, b'{"error":"not found"}')
            n = int(self.headers.get("Content-Length", 0))
            try:
                cmd = json.loads(self.rfile.read(min(n, 10000)) or b"{}")
                res = duck.command(cmd)
            except Exception as e:
                res = {"ok": False, "msg": str(e)}
            self.send(200, json.dumps(res).encode())
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--http", type=int, default=8090)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        port = FakePort()
    else:
        import serial
        port = serial.Serial(args.port, 1_000_000, timeout=0)
    duck = Duck(Bus(port))
    threading.Thread(target=duck.run, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", args.http), make_handler(duck))
    print(f"Duck Playground on http://0.0.0.0:{args.http}  ({'demo' if args.demo else args.port})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with duck.lock:
            duck.bus.all_off()
        print("motors off, bye", flush=True)


if __name__ == "__main__":
    main()
