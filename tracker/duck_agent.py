#!/usr/bin/env python3
"""Duck agent: the puppet tool's hands on the real duck. Runs ON THE PI.

    python3 ~/duck_agent.py                 # servo adapter on /dev/ttyACM0, HTTP on :8091
    python3 ~/duck_agent.py --demo          # pretend servos, no hardware

The puppet page (tracker/server.py on the PC) talks to it through /duck/api/...:
    GET  /api/state      every joint: angle, torque on/off, target, load, volts, temp, errors
    POST /api/cmd        {"action": ...}, see Duck.command
    GET  /api/clip       the last recording, [[t, {joint: rad}], ...]

Angles are in the runtime / simulator frame (radians, calibration offsets from
~/duck_config.json removed), the same numbers the walk policy and MuJoCo use, so a clip
recorded here plays back on the 3D model and in the simulator without conversion.

Safety: if no page has asked for the state for 5 s, every motor switches off (the duck goes
limp, so keep it on its cable). Only the servos' working memory is written (targets, torque,
P gain); their saved settings are never touched. Do not run it together with the walk
program: both need the servo adapter.
"""
import argparse
import json
import math
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Same order as the runtime's HWI and the simulator's actuators.
JOINTS = [("left_hip_yaw", 20), ("left_hip_roll", 21), ("left_hip_pitch", 22), ("left_knee", 23),
          ("left_ankle", 24), ("neck_pitch", 30), ("head_pitch", 31), ("head_yaw", 32), ("head_roll", 33),
          ("right_hip_yaw", 10), ("right_hip_roll", 11), ("right_hip_pitch", 12), ("right_knee", 13),
          ("right_ankle", 14)]
NAMES = [j[0] for j in JOINTS]
ID_OF = dict(JOINTS)
NAME_OF = {i: n for n, i in JOINTS}
# The runtime's standing pose (HWI.init_pos), radians in the runtime frame.
HOME = {"left_hip_yaw": 0.002, "left_hip_roll": 0.053, "left_hip_pitch": -0.63, "left_knee": 1.368,
        "left_ankle": -0.784, "neck_pitch": 0.0, "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0,
        "right_hip_yaw": -0.003, "right_hip_roll": -0.065, "right_hip_pitch": 0.635, "right_knee": 1.379,
        "right_ankle": -0.796}
# Simulator joint ranges (rad): the hard limits the page may ask for.
RANGE = {"hip_yaw": (-0.52, 0.52), "hip_roll": (-0.44, 0.44), "left_hip_pitch": (-1.22, 0.52),
         "right_hip_pitch": (-0.52, 1.22), "knee": (-1.57, 1.57), "ankle": (-1.57, 1.57),
         "neck_pitch": (-0.35, 1.13), "head_pitch": (-0.79, 0.79), "head_yaw": (-2.79, 2.79),
         "head_roll": (-0.52, 0.52)}
P_GAIN, TORQUE, ACCEL, GOAL, PRESENT = 21, 40, 41, 42, 56
READ, WRITE, SYNC_WRITE, BROADCAST = 0x02, 0x03, 0x83, 0xFE
STEPS_PER_RAD = 4096 / (2 * math.pi)
ERRORS = {0: "voltage", 1: "sensor", 2: "too hot", 3: "current", 4: "angle", 5: "overload"}
CONFIG = Path.home() / "duck_config.json"


def joint_range(name):
    for key, r in RANGE.items():
        if name == key or name.endswith(key):
            return r
    return (-1.57, 1.57)


def load_config():
    try:
        cfg = json.loads(CONFIG.read_text())
    except Exception:
        cfg = {}
    offsets = {n: float(cfg.get("joints_offsets", {}).get(n, 0.0)) for n in NAMES}
    limits = {}
    for n in NAMES:
        lo, hi = joint_range(n)
        deg = cfg.get("joint_limits_deg", {}).get(n)
        if deg:
            lo, hi = max(lo, math.radians(deg[0])), min(hi, math.radians(deg[1]))
        limits[n] = (lo, hi)
    return offsets, limits


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

    def p_gain(self, sid, p):
        self.txn(sid, WRITE, bytes([P_GAIN, max(0, min(254, int(p)))]), wait=0.01)

    def goals(self, goals):
        """goals: {id: (steps, speed)} in ONE packet so every servo starts together."""
        if not goals:
            return
        params = bytearray([GOAL, 6])
        for sid, (st, sp) in goals.items():
            st, sp = int(st), int(sp)
            params += bytes([sid, st & 0xFF, st >> 8, 0, 0, sp & 0xFF, sp >> 8])
        self.txn(BROADCAST, SYNC_WRITE, params)


class FakePort:
    """Pretend servos for --demo: a motor that is off drifts as if a hand moved it."""

    def __init__(self):
        self.s = {i: {"pos": 2048.0 + random.randint(-200, 200), "goal": 2048, "speed": 600, "torque": 0,
                      "ph": random.random() * 6, "vel": 0.0} for i in ID_OF.values()}
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
                v["pos"] += 150 * math.sin(now * 0.5 + v["ph"]) * dt
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
            load = random.randint(20, 120) if v["torque"] else 0
            cur = int(abs(v["vel"]) / 25 + 3) if v["torque"] else 1
            regs = bytes([pos & 255, pos >> 8, 0, 0, load & 255, load >> 8, 74, 31, 0, 0, 0, 0, 0,
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
        self.offsets, self.limits = load_config()
        self.s = {n: {"steps": None, "load": 0, "volt": 0, "temp": 0, "err": 0, "ma": 0, "ok": False,
                      "torque": False, "target": None, "kp": 32,
                      # clay mode: holds where you leave it, gives way when you push
                      "clay": None, "last_steps": None, "still_since": 0.0} for n in NAMES}
        self.mode = "idle"            # idle | record | play
        self.frames, self.rec_t0 = [], 0.0
        self.play = None
        self.timers = {}              # joint -> monotonic time to hold it again
        self.pending = {}             # goals for this loop: joint -> (steps, speed)
        self.last_client = time.monotonic()
        self.message = ""
        bus.all_off()
        bus.accel_all(0)

    # ---- conversions: runtime radians <-> servo steps
    def to_steps(self, name, rad):
        lo, hi = self.limits[name]
        rad = max(lo, min(hi, rad))
        return int(round(2048 + (rad + self.offsets[name]) * STEPS_PER_RAD))

    def to_rad(self, name, steps):
        return (steps - 2048) / STEPS_PER_RAD - self.offsets[name]

    def rad(self, name):
        st = self.s[name]["steps"]
        return None if st is None else self.to_rad(name, st)

    # ---- commands
    def command(self, c):
        a = c.get("action")
        names = c.get("joints") or NAMES
        names = [n for n in names if n in self.s]
        with self.lock:
            self.last_client = time.monotonic()
            if a == "stop":
                self._all_off()
                self.mode, self.message = "idle", "all motors off"
            elif a == "hold":                       # torque on where it is
                self.timers = {n: t for n, t in self.timers.items() if n not in names}
                self._clay_off(names)
                self._hold_and_on(names)
            elif a == "clay":                       # weak resistance, stays where you put it
                kp = int(c.get("kp", 16))
                self.timers = {n: t for n, t in self.timers.items() if n not in names}
                self._hold_and_on(names)
                for n in names:
                    v = self.s[n]
                    v["clay"] = {"hold": max(2, min(32, kp)), "soft": max(2, kp // 4), "state": "hold",
                                 "band": 20}            # steps of give before it follows (about 1.8 deg)
                    v["last_steps"], v["still_since"] = v["steps"], time.monotonic()
                    self.bus.p_gain(ID_OF[n], v["clay"]["hold"])
                    v["kp"] = v["clay"]["hold"]
            elif a == "free":                       # torque off; optional timer to hold again
                self._clay_off(names)
                for n in names:
                    self.bus.torque(ID_OF[n], False)
                    self.s[n]["torque"], self.s[n]["target"] = False, None
                secs = float(c.get("seconds") or 0)
                for n in names:
                    if secs > 0:
                        self.timers[n] = time.monotonic() + secs
                    else:
                        self.timers.pop(n, None)
            elif a == "set":                        # one joint to an angle (rad), optional duration
                n = c["joint"]
                self._clay_off([n])
                self._hold_and_on([n])
                self._move({n: float(c["rad"])}, float(c.get("seconds") or 0))
            elif a == "pose":                       # several joints to angles, over `seconds`
                pose = {n: float(r) for n, r in c.get("pose", {}).items() if n in self.s}
                self._clay_off(list(pose))
                self._hold_and_on(list(pose))
                self._move(pose, float(c.get("seconds") or 1.0))
            elif a == "home":
                self._clay_off(NAMES)
                self._hold_and_on(NAMES)
                self._move(HOME, float(c.get("seconds") or 2.0))
            elif a == "kp":
                kp = int(c.get("kp", 32))
                for n in names:
                    self.bus.p_gain(ID_OF[n], kp)
                    self.s[n]["kp"] = kp
            elif a == "record":
                self.mode, self.frames, self.rec_t0 = "record", [], time.monotonic()
                self.message = "recording"
            elif a == "record_stop":
                if self.mode == "record":
                    self.mode = "idle"
                self.message = f"recorded {self.frames[-1][0]:.1f} s" if self.frames else "nothing recorded"
            elif a == "play":
                frames = c.get("frames") or self.frames
                if not frames:
                    return {"ok": False, "msg": "nothing to play"}
                self._clay_off(NAMES)
                self.play = {"frames": [(float(t), {n: float(r) for n, r in f.items() if n in self.s})
                                        for t, f in frames],
                             "t0": None, "loop": bool(c.get("loop")), "speed": float(c.get("speed") or 1.0),
                             "approach_until": time.monotonic() + 3}
                self.mode = "play"
                first = self.play["frames"][0][1]
                self._hold_and_on(list(first))
                self._move(first, 1.5)
                self.message = "playing"
            elif a == "play_stop":
                if self.mode == "play":
                    self.mode, self.play = "idle", None
                self.message = "stopped"
            else:
                return {"ok": False, "msg": f"unknown action {a}"}
        return {"ok": True}

    def _move(self, pose, seconds):
        """Send goals so every joint arrives together after `seconds` (0 = as fast as it can)."""
        for n, rad in pose.items():
            st = self.to_steps(n, rad)
            here = self.s[n]["steps"] if self.s[n]["steps"] is not None else st
            speed = 0 if seconds <= 0 else max(40, min(3400, abs(st - here) / seconds))
            self.s[n]["target"] = st
            self.pending[n] = (st, speed)

    def _hold_and_on(self, names):
        hold = {}
        for n in names:
            if not self.s[n]["torque"]:
                here = self.s[n]["steps"] if self.s[n]["steps"] is not None else 2048
                hold[ID_OF[n]] = (here, 500)
                self.s[n]["target"] = here
        self.bus.goals(hold)
        for n in names:
            if not self.s[n]["torque"]:
                self.bus.torque(ID_OF[n], True)
                self.s[n]["torque"] = True

    def _clay_off(self, names):
        """Leave clay mode: back to the normal gain, target stays where the joint is."""
        for n in names:
            v = self.s[n]
            if v["clay"]:
                v["clay"] = None
                self.bus.p_gain(ID_OF[n], 32)
                v["kp"] = 32

    def _clay_step(self, now):
        for n in NAMES:
            v, cl = self.s[n], self.s[n]["clay"]
            if not cl or v["steps"] is None or not v["torque"]:
                continue
            here, tgt = v["steps"], v["target"] if v["target"] is not None else v["steps"]
            moved = v["last_steps"] is not None and abs(here - v["last_steps"]) > 3
            v["last_steps"] = here
            if moved:
                v["still_since"] = now
            if cl["state"] == "hold":
                if abs(here - tgt) > cl["band"]:          # pushed past the give: let it follow
                    cl["state"] = "follow"
                    self.bus.p_gain(ID_OF[n], cl["soft"])
                    v["kp"] = cl["soft"]
                    v["target"] = here
                    self.pending[n] = (here, 0)
            else:                                          # following the hand
                if here != tgt:
                    v["target"] = here
                    self.pending[n] = (here, 0)
                if now - v["still_since"] > 0.5:           # let go of: lock where it is
                    cl["state"] = "hold"
                    self.bus.p_gain(ID_OF[n], cl["hold"])
                    v["kp"] = cl["hold"]

    def _all_off(self):
        self.bus.all_off()
        for v in self.s.values():
            v["torque"], v["target"] = False, None
            if v["clay"]:
                v["clay"] = None
                v["kp"] = 32
        self.pending.clear()
        self.timers.clear()
        self.play = None

    # ---- the loop
    def run(self):
        while True:
            t = time.monotonic()
            try:
                with self.lock:
                    self._step()
            except Exception as e:
                print("loop error:", e, flush=True)
            time.sleep(max(0.0, 0.02 - (time.monotonic() - t)))

    def _step(self):
        now = time.monotonic()
        if now - self.last_client > 5 and any(v["torque"] for v in self.s.values()):
            self._all_off()
            self.mode, self.message = "idle", "no page for 5 s: motors off"
            print(self.message, flush=True)

        for n in NAMES:                      # read every servo every loop (about 25 Hz)
            st = self.bus.status(ID_OF[n])
            v = self.s[n]
            if st is None:
                v["ok"] = False
                continue
            v.update(st)
            v["ok"] = True

        for n, t in list(self.timers.items()):   # "free for N s, then hold"
            if now >= t:
                self._hold_and_on([n])
                del self.timers[n]
        self._clay_step(now)

        if self.mode == "record":
            if all(self.s[n]["steps"] is not None for n in NAMES):
                self.frames.append((round(now - self.rec_t0, 3), {n: round(self.rad(n), 4) for n in NAMES}))
            if now - self.rec_t0 > 120:
                self.mode, self.message = "idle", "recording stopped at 2 min"
        elif self.mode == "play" and self.play:
            p = self.play
            fr = p["frames"]
            if p["t0"] is None:               # moving to the first pose
                first = fr[0][1]
                close = all(self.s[n]["steps"] is None or abs(self.s[n]["steps"] - self.to_steps(n, r)) < 40
                            for n, r in first.items())
                if close or now > p["approach_until"]:
                    p["t0"] = now
            else:
                t = (now - p["t0"]) * p["speed"]
                if t > fr[-1][0]:
                    if p["loop"]:
                        p["t0"], t = now, 0.0
                    else:
                        self.mode, self.play, self.message = "idle", None, "played"
                if self.play:
                    i = min(range(len(fr)), key=lambda k: abs(fr[k][0] - t))
                    j = min(i + 3, len(fr) - 1)
                    dt = max(0.05, (fr[j][0] - fr[i][0]) / p["speed"])
                    for n, r in fr[i][1].items():
                        st, ahead = self.to_steps(n, r), self.to_steps(n, fr[j][1][n])
                        self.s[n]["target"] = st
                        self.pending[n] = (st, max(60, min(3400, abs(ahead - st) / dt * 1.3)))

        if self.pending:
            self.bus.goals({ID_OF[n]: g for n, g in self.pending.items() if self.s[n]["torque"]})
            self.pending.clear()

    def snapshot(self):
        with self.lock:
            self.last_client = time.monotonic()
            joints = []
            for n in NAMES:
                v = self.s[n]
                joints.append({"name": n, "id": ID_OF[n], "ok": v["ok"], "torque": v["torque"],
                               "rad": None if v["steps"] is None else round(self.rad(n), 4),
                               "target": None if v["target"] is None else round(self.to_rad(n, v["target"]), 4),
                               "kp": v["kp"], "clay": v["clay"]["state"] if v["clay"] else None,
                               "load": v["load"], "ma": round(v["ma"]), "volt": v["volt"],
                               "temp": v["temp"], "hold_in": round(self.timers[n] - time.monotonic(), 1) if n in self.timers else None,
                               "errors": [txt for b, txt in ERRORS.items() if v["err"] >> b & 1]})
            volts = [v["volt"] for v in self.s.values() if v["ok"]]
            return {"mode": self.mode, "message": self.message, "joints": joints,
                    "volts": round(min(volts), 1) if volts else None,
                    "amps": round(sum(v["ma"] for v in self.s.values() if v["ok"]) / 1000, 2),
                    "recorded": round(self.frames[-1][0], 1) if self.frames else 0,
                    "rec_live": round(time.monotonic() - self.rec_t0, 1) if self.mode == "record" else None,
                    "play_t": None if not self.play or self.play["t0"] is None else round((time.monotonic() - self.play["t0"]) * self.play["speed"], 2),
                    "limits": {n: [round(lo, 3), round(hi, 3)] for n, (lo, hi) in self.limits.items()},
                    "home": HOME}


# ------------------------------------------------------------------ web server
def make_handler(duck):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/api/state"):
                return self.send(200, json.dumps(duck.snapshot()).encode())
            if self.path.startswith("/api/clip"):
                with duck.lock:
                    return self.send(200, json.dumps(duck.frames).encode())
            self.send(404, b'{"error":"not found"}')

        def do_POST(self):
            if self.path != "/api/cmd":
                return self.send(404, b'{"error":"not found"}')
            n = int(self.headers.get("Content-Length", 0))
            try:
                cmd = json.loads(self.rfile.read(min(n, 4_000_000)) or b"{}")
                res = duck.command(cmd)
            except Exception as e:
                res = {"ok": False, "msg": str(e)}
            self.send(200, json.dumps(res).encode())
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--http", type=int, default=8091)
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
    print(f"Duck agent on http://0.0.0.0:{args.http}  ({'demo' if args.demo else args.port})", flush=True)
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
