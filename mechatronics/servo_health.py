#!/usr/bin/env python3
"""Read-only health read of the Open Duck Mini's 14 STS3215 servos. Runs ON THE PI.

    ~/.virtualenvs/open-duck-mini-runtime/bin/python ~/servo_health.py            # real bus
    python3 servo_health.py --fake                                                 # no hardware

Prints ONE JSON line on stdout (see FIELDS below) and nothing else; diagnostics go to stderr.
Exit codes: 0 ok, 2 bad arguments, 3 the serial port is busy (refused, nothing touched),
4 the port could not be opened, 5 no servo answered.

Safety: this script only ever sends PING (0x01) and READ (0x02) instructions. `Bus.packet`
refuses anything else, so no edit can turn it into a writer by accident. It never enables
torque, never sends a goal, never writes EEPROM. If `/dev/ttyACM0` is held by another
process (the walk program, the puppet agent) it exits before opening the port.

Register map (STS3215 / SMS_STS memory table). The ones marked [v] were checked against the
other scripts in this project (servo_ping.py, duck_agent.py, setup_all_servos.py,
strength_test.py, battery_guard.py); the others come from Feetech's memory table and the
LeRobot feetech table and should be confirmed on the first real read (the values read back
should look like their documented defaults: 70, 80, 40, 1000, 44, 47, 32, 0, 0, 500...).
  EEPROM  0x03 model [v]  0x05 id [v]  0x09/0x0B min/max angle limit  0x0D max temperature
          0x0E max input voltage [v]  0x0F min input voltage [v]  0x10 max torque
          0x13 unloading condition  0x14 LED alarm condition  0x15 P [v] 0x16 D [v] 0x17 I [v]
          0x1C protection current  0x1F position offset [v]  0x21 mode [v]
          0x22 protective torque  0x23 protection time  0x24 overload torque
          0x26 over-current protection time
  SRAM    0x28 torque enable [v]  0x29 acceleration [v]  0x2A goal position [v]
          0x30 torque limit [v]  0x38 present position [v]  0x3A present speed [v]
          0x3C present load [v]  0x3E present voltage (0.1 V) [v]  0x3F present temperature (C) [v]
          0x41 status flags [v]  0x42 moving  0x45 present current (6.5 mA units) [v]
Status / alarm bits: 0 voltage, 1 sensor, 2 temperature, 3 current, 4 angle, 5 overload.

FIELDS of the JSON line:
  t, source="servo_health", host, port, fake, duration_s, missing (joints that did not answer),
  voltage {joint: V}, temps {joint: C}, currents {joint: A}, loads {joint: % signed},
  positions {joint: rad, runtime frame: (steps-2048)*2pi/4096 - joints_offsets[joint]},
  positions_steps, speeds {joint: rad/s}, torque_on {joint: bool}, moving {joint: bool},
  status {joint: [flag names]}, status_raw {joint: int}, limits {joint: {min_v, max_v, max_temp,
  max_torque_pct, protection_current_a, unload_condition, led_alarm, protective_torque_pct,
  protection_time_s, overload_torque_pct, overcurrent_time_s, angle_min_steps, angle_max_steps,
  p, i, d, offset_steps, mode}}, ids, model, firmware, summary {v_min, v_max, t_max,
  t_max_joint, i_total_a, flags {flag: [joints]}, voltage_flag_explained {joint: text},
  unload_on_voltage}.
"""
import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

JOINTS = [("left_hip_yaw", 20), ("left_hip_roll", 21), ("left_hip_pitch", 22), ("left_knee", 23),
          ("left_ankle", 24), ("neck_pitch", 30), ("head_pitch", 31), ("head_yaw", 32), ("head_roll", 33),
          ("right_hip_yaw", 10), ("right_hip_roll", 11), ("right_hip_pitch", 12), ("right_knee", 13),
          ("right_ankle", 14)]
NAMES = [j[0] for j in JOINTS]
ID_OF = dict(JOINTS)

PING, READ = 0x01, 0x02
ALLOWED_INSTRUCTIONS = (PING, READ)          # the whole point of this script
FLAGS = {0: "voltage", 1: "sensor", 2: "temperature", 3: "current", 4: "angle", 5: "overload"}
STEPS_PER_RAD = 4096 / (2 * math.pi)
MA_PER_UNIT = 6.5
CONFIG = Path.home() / "duck_config.json"
BUSY_PATTERNS = ("duck_agent.py", "v2_rl_walk", "mujoco_infer", "servo_playground", "strength_test",
                 "wave_power_test", "joint_limits_probe", "head_puppet", "check_motors")


# ------------------------------------------------------------------ port-busy check
def who_holds(dev):
    """List of 'pid name' strings for processes that have `dev` open. Empty = free (as far as
    we can see). Uses fuser, then a /proc scan, then process names as a last resort."""
    holders = []
    try:
        r = subprocess.run(["fuser", dev], capture_output=True, text=True, timeout=5)
        pids = [p for p in (r.stdout + r.stderr).replace(dev + ":", "").split() if p.isdigit()]
        for p in pids:
            try:
                name = Path(f"/proc/{p}/cmdline").read_bytes().replace(b"\0", b" ").decode().strip()[:80]
            except OSError:
                name = "?"
            holders.append(f"{p} {name}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    if not holders:
        try:
            real = os.path.realpath(dev)
            for p in os.listdir("/proc"):
                if not p.isdigit() or int(p) == os.getpid():
                    continue
                fd_dir = f"/proc/{p}/fd"
                try:
                    for fd in os.listdir(fd_dir):
                        try:
                            if os.path.realpath(os.path.join(fd_dir, fd)) == real:
                                cmd = Path(f"/proc/{p}/cmdline").read_bytes().replace(b"\0", b" ").decode().strip()[:80]
                                holders.append(f"{p} {cmd}")
                                break
                        except OSError:
                            continue
                except OSError:
                    continue      # not our process; fuser (above) would have seen it if it could
        except OSError:
            pass
    if not holders:
        try:
            r = subprocess.run(["pgrep", "-af", "|".join(BUSY_PATTERNS)], capture_output=True, text=True, timeout=5)
            for line in r.stdout.splitlines():
                if "pgrep" in line or "servo_health" in line:
                    continue
                holders.append(line.strip()[:90])
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return holders


# ------------------------------------------------------------------ serial bus (read-only)
class Bus:
    def __init__(self, port):
        self.port = port

    @staticmethod
    def packet(sid, instr, params=b""):
        if instr not in ALLOWED_INSTRUCTIONS:
            raise PermissionError(f"servo_health is read-only; instruction {instr:#x} refused")
        body = bytes([sid, len(params) + 2, instr]) + bytes(params)
        return b"\xff\xff" + body + bytes([~sum(body) & 0xFF])

    def txn(self, sid, instr, params=b"", wait=0.03):
        """Returns (error_byte, params) of the status packet, or None."""
        self.port.reset_input_buffer()
        self.port.write(self.packet(sid, instr, params))
        buf, end = bytearray(), time.monotonic() + wait
        while time.monotonic() < end:
            n = self.port.in_waiting
            if not n:
                time.sleep(0.0003)
                continue
            buf += self.port.read(n)
            i = buf.find(b"\xff\xff")
            if i >= 0 and len(buf) >= i + 4 and len(buf) >= i + 4 + buf[i + 3]:
                pkt = buf[i + 2:i + 4 + buf[i + 3]]          # id, len, err, params..., checksum
                if pkt[0] == sid and pkt[-1] == (~sum(pkt[:-1]) & 0xFF):
                    return pkt[2], bytes(pkt[3:-1])
                return None
        return None

    def ping(self, sid):
        for _ in range(2):
            if self.txn(sid, PING, wait=0.02) is not None:
                return True
        return False

    def read(self, sid, addr, n):
        for _ in range(3):
            r = self.txn(sid, READ, bytes([addr, n]))
            if r is not None and len(r[1]) >= n:
                return r[1][:n]
        return None


class FakePort:
    """--fake: answers reads from an invented register image, so the whole decode path runs
    without hardware. Values mimic the real duck just after a charge: 8.3 V on every servo,
    which is above the (assumed) 8.0 V max-voltage register, so every joint shows the
    'voltage' flag, like the real one does."""

    def __init__(self):
        import random
        rnd = random.Random(7)
        self.regs = {}
        for k, (name, sid) in enumerate(JOINTS):
            r = bytearray(80)
            r[0], r[1] = 3, 9                       # firmware 3.9
            r[3], r[4] = 9, 3                       # model 777 (0x0309)
            r[5], r[6], r[7], r[8] = sid, 0, 0, 1
            r[9:11] = (0).to_bytes(2, "little")     # min angle
            r[11:13] = (4095).to_bytes(2, "little") # max angle
            r[13], r[14], r[15] = 70, 80, 40        # max temp, max V (8.0), min V (4.0)
            r[16:18] = (1000).to_bytes(2, "little") # max torque
            r[18], r[19], r[20] = 12, 44, 47        # phase, unload cond, LED alarm
            r[21], r[22], r[23] = 32, 0, 0          # P D I
            r[24:26] = (16).to_bytes(2, "little")
            r[26], r[27] = 1, 1
            r[28:30] = (500).to_bytes(2, "little")  # protection current (x6.5 mA)
            r[30] = 1
            r[31:33] = (0).to_bytes(2, "little")    # offset
            r[33], r[34], r[35], r[36] = 0, 20, 200, 80
            r[37], r[38], r[39] = 10, 200, 10
            r[40], r[41] = 0, 0                     # torque off, accel
            pos = 2048 + rnd.randint(-400, 400)
            r[42:44] = pos.to_bytes(2, "little")
            r[48:50] = (1000).to_bytes(2, "little")
            r[55] = 1
            r[56:58] = pos.to_bytes(2, "little")
            r[58:60] = (0).to_bytes(2, "little")
            r[60:62] = (0).to_bytes(2, "little")
            r[62] = 83                              # 8.3 V
            r[63] = 31 + (k % 5) + (9 if name == "left_knee" else 0)
            r[65] = 0x01                            # voltage flag
            r[66] = 0
            r[69:71] = (rnd.randint(0, 3)).to_bytes(2, "little")
            self.regs[sid] = r
        self.out = bytearray()

    @property
    def in_waiting(self):
        return len(self.out)

    def read(self, n):
        d, self.out = bytes(self.out[:n]), self.out[n:]
        return d

    def reset_input_buffer(self):
        self.out.clear()

    def write(self, b):
        sid, instr, p = b[2], b[4], b[5:-1]
        r = self.regs.get(sid)
        if r is None:
            return
        data = b""
        if instr == READ:
            data = bytes(r[p[0]:p[0] + p[1]])
        body = bytes([sid, len(data) + 2, r[65]]) + data
        self.out += b"\xff\xff" + body + bytes([~sum(body) & 0xFF])

    def close(self):
        pass


# ------------------------------------------------------------------ decoding
def u16(b, i):
    return b[i] | b[i + 1] << 8


def sign_mag(v, bit):
    return -(v & ((1 << bit) - 1)) if v & (1 << bit) else v


def load_offsets():
    try:
        cfg = json.loads(CONFIG.read_text())
        return {n: float(cfg.get("joints_offsets", {}).get(n, 0.0)) for n in NAMES}, True
    except Exception:
        return {n: 0.0 for n in NAMES}, False


def read_servo(bus, sid):
    """Three reads per servo: header (0x00, 9 B), EEPROM protections (0x09, 31 B), SRAM state
    (0x28, 31 B: torque enable .. present current). Returns None if the servo is silent."""
    head = bus.read(sid, 0x00, 9)
    eep = bus.read(sid, 0x09, 31)
    st = bus.read(sid, 0x28, 31)
    if head is None and eep is None and st is None:
        return None
    out = {}
    if head is not None:
        out["firmware"] = f"{head[0]}.{head[1]}"
        out["model"] = u16(head, 3)
        out["id_read"] = head[5]
    if eep is not None:
        e = eep        # e[0] is address 0x09
        out["limits"] = {
            "angle_min_steps": u16(e, 0), "angle_max_steps": u16(e, 2),
            "max_temp": e[4], "max_v": e[5] / 10, "min_v": e[6] / 10,
            "max_torque_pct": u16(e, 7) / 10,
            "phase": e[9], "unload_condition": e[10], "led_alarm": e[11],
            "p": e[12], "d": e[13], "i": e[14],
            "min_startup_force": u16(e, 15), "cw_dead": e[17], "ccw_dead": e[18],
            "protection_current_a": round(u16(e, 19) * MA_PER_UNIT / 1000, 3),
            "angular_resolution": e[21],
            "offset_steps": sign_mag(u16(e, 22), 11),
            "mode": e[24], "protective_torque_pct": e[25], "protection_time_s": e[26] / 100,
            "overload_torque_pct": e[27], "speed_p": e[28], "overcurrent_time_s": e[29] / 100,
            "speed_i": e[30],
        }
    if st is not None:
        s = st         # s[0] is address 0x28
        load_raw = u16(s, 20)
        cur_raw = u16(s, 29)
        out["state"] = {
            "torque_on": bool(s[0]), "accel": s[1], "goal_steps": u16(s, 2),
            "torque_limit_pct": u16(s, 8) / 10, "lock": s[15],
            "pos_steps": u16(s, 16),
            "speed_steps_s": sign_mag(u16(s, 18), 15),
            "load_pct": (-(load_raw & 0x3FF) if load_raw & 0x400 else (load_raw & 0x3FF)) / 10,
            "volt": s[22] / 10, "temp": s[23], "status": s[25], "moving": bool(s[26]),
            "current_a": round(sign_mag(cur_raw, 15) * MA_PER_UNIT / 1000, 4),
        }
    return out


def flags_of(raw):
    return [name for bit, name in FLAGS.items() if raw >> bit & 1]


def build_record(per_joint, offsets, offsets_found, port_name, fake, t0):
    rec = {"t": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
           "source": "servo_health", "host": socket.gethostname(), "port": port_name, "fake": fake,
           "offsets_from_config": offsets_found,
           "voltage": {}, "temps": {}, "currents": {}, "loads": {}, "positions": {}, "positions_steps": {},
           "speeds": {}, "torque_on": {}, "moving": {}, "status": {}, "status_raw": {}, "limits": {},
           "ids": dict(JOINTS), "model": {}, "firmware": {}, "missing": []}
    for n in NAMES:
        d = per_joint.get(n)
        if not d:
            rec["missing"].append(n)
            continue
        if "model" in d:
            rec["model"][n] = d["model"]
            rec["firmware"][n] = d["firmware"]
        if "limits" in d:
            rec["limits"][n] = d["limits"]
        s = d.get("state")
        if s:
            rec["voltage"][n] = s["volt"]
            rec["temps"][n] = s["temp"]
            rec["currents"][n] = s["current_a"]
            rec["loads"][n] = s["load_pct"]
            rec["positions_steps"][n] = s["pos_steps"]
            rec["positions"][n] = round((s["pos_steps"] - 2048) / STEPS_PER_RAD - offsets[n], 4)
            rec["speeds"][n] = round(s["speed_steps_s"] / STEPS_PER_RAD, 3)
            rec["torque_on"][n] = s["torque_on"]
            rec["moving"][n] = s["moving"]
            rec["status_raw"][n] = s["status"]
            rec["status"][n] = flags_of(s["status"])
    # summary and the voltage-flag verdict
    volts, temps = rec["voltage"], rec["temps"]
    flags = {}
    for n, fl in rec["status"].items():
        for f in fl:
            flags.setdefault(f, []).append(n)
    explained, unload_on_v = {}, None
    for n in rec["status"]:
        lim = rec["limits"].get(n)
        if not lim or n not in volts:
            continue
        v = volts[n]
        if "voltage" in rec["status"][n]:
            if v > lim["max_v"]:
                explained[n] = f"{v:.1f} V is above the servo's max_v register {lim['max_v']:.1f} V"
            elif v < lim["min_v"]:
                explained[n] = f"{v:.1f} V is below the servo's min_v register {lim['min_v']:.1f} V"
            else:
                explained[n] = (f"flag set but {v:.1f} V is inside [{lim['min_v']:.1f}, {lim['max_v']:.1f}] V: "
                                "latched from earlier, or a dip the register sampled")
        unload_on_v = bool(lim["unload_condition"] & 1)
    tmax_j = max(temps, key=temps.get) if temps else None
    rec["summary"] = {
        "answered": len(NAMES) - len(rec["missing"]),
        "v_min": min(volts.values()) if volts else None, "v_max": max(volts.values()) if volts else None,
        "t_max": temps[tmax_j] if tmax_j else None, "t_max_joint": tmax_j,
        "i_total_a": round(sum(rec["currents"].values()), 3) if rec["currents"] else None,
        "any_torque_on": any(rec["torque_on"].values()) if rec["torque_on"] else None,
        "flags": flags, "voltage_flag_explained": explained,
        "unload_on_voltage": unload_on_v,
        "limits_identical": len({json.dumps(l, sort_keys=True) for l in rec["limits"].values()}) <= 1,
    }
    rec["duration_s"] = round(time.monotonic() - t0, 3)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--fake", action="store_true", help="invented servos, no hardware, no port check")
    ap.add_argument("--force", action="store_true", help=argparse.SUPPRESS)   # deliberately undocumented: skips nothing, kept for symmetry; busy port always refuses
    args = ap.parse_args()
    t0 = time.monotonic()
    offsets, offsets_found = load_offsets()

    if args.fake:
        port = FakePort()
    else:
        holders = who_holds(args.port)
        if holders:
            print(json.dumps({"error": "port busy", "port": args.port, "holders": holders,
                              "hint": "stop nothing; try again when the walk program / puppet agent is not running"}),
                  file=sys.stderr)
            return 3
        try:
            import serial
        except ImportError:
            print(json.dumps({"error": "pyserial missing; run with ~/.virtualenvs/open-duck-mini-runtime/bin/python"}),
                  file=sys.stderr)
            return 4
        try:
            port = serial.Serial(args.port, 1_000_000, timeout=0, exclusive=True)
        except Exception as e:      # noqa: BLE001
            print(json.dumps({"error": f"could not open {args.port}: {e}"}), file=sys.stderr)
            return 4

    try:
        bus = Bus(port)
        per_joint = {}
        for n, sid in JOINTS:
            if not bus.ping(sid):
                continue
            d = read_servo(bus, sid)
            if d:
                per_joint[n] = d
    finally:
        try:
            port.close()
        except Exception:       # noqa: BLE001
            pass
    if not per_joint:
        print(json.dumps({"error": "no servo answered", "port": args.port}), file=sys.stderr)
        return 5
    rec = build_record(per_joint, offsets, offsets_found, args.port, args.fake, t0)
    print(json.dumps(rec, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
