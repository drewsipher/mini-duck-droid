"""
Low-battery failsafe for the walk script (v2_rl_walk_mujoco.py).

  below warn_v (7.0 V)  -> "LOW BATTERY" banner on the web page + a beep every 30 s
  below crit_v (6.8 V)  -> twice in a row (or once below hard_v, 6.5 V): stops walking,
                           stands 1.5 s, crouches slowly (3 s) into a tuck (SHUTDOWN_POSES),
                           motors off, saves the recording, powers the Pi off

The Pi has no voltage sensor, so the voltage comes from the servos' own voltage
register. The control loop's serial library (rustypot) can't read that register, so
the guard briefly hands the bus over: drop rustypot, read 3 servos with pyserial,
reopen rustypot. During that gap no new targets are sent and the servos hold their
last ones. It reads every 5 s while paused, every 20 s while standing still (no drive
command for 1.5 s), and while walking only if there has been no standing-still read
for 90 s. Every read is logged to ~/battery_log.csv with how long the handoff took.

The tuck pose was picked in the simulator with his measured joint stops: he tips forward onto
his chest when the motors go limp (head lands at ~0.4-0.6 m/s, against ~1.4 m/s if the motors
cut while standing).

Testing without a flat battery:  echo 6.9 > ~/.duck_fake_volts   (warning)
                                 echo 6.7 > ~/.duck_fake_volts   (shutdown)
                                 rm ~/.duck_fake_volts           (back to real readings)
With "poweroff": false in the config it does everything except power the Pi off.

duck_config.json, all optional:
  "battery_guard": {"enabled": true, "warn_v": 7.0, "crit_v": 6.8, "hard_v": 6.5,
                    "poweroff": true, "shutdown_pose": "chin_up"}
"""
import gc
import math
import os
import shlex
import subprocess
import time

import numpy as np

FAKE_FILE = os.path.expanduser("~/.duck_fake_volts")
LOG_FILE = os.path.expanduser("~/battery_log.csv")
VOLT_ADDR = 62                          # STS3215 present voltage, 0.1 V units
READ_IDS = (14, 24, 30)                 # right ankle, left ankle, neck: any servo sees the bus voltage

DEFAULTS = dict(
    enabled=True,
    warn_v=7.0, crit_v=6.8, hard_v=6.5, confirm=2,
    period_paused_s=5.0, period_idle_s=20.0, max_gap_walking_s=90.0, idle_after_s=1.5,
    beep_every_s=30.0,
    settle_s=1.5, crouch_s=3.0, hold_s=1.0,
    # Which pose to go down in (see SHUTDOWN_POSES): "chin_up", "head_down" or "lean_back".
    shutdown_pose="chin_up",
    # "lean_back" only: home pose + these (radians): deeper knee bend 0.2, hips and ankles take
    # half of it, then hips lean 0.55 and the ankles tip 0.3. Knees capped at 1.55 rad.
    crouch_depth=0.2, crouch_hip_lean=0.55, crouch_ankle=-0.3, knee_max=1.55,
    poweroff=True, poweroff_cmd="sudo -n /usr/sbin/poweroff",
)


# Tuck poses picked in the simulator with the duck's measured joint stops
# (~/src/mini-duck-droid/sim/render_tuck.py, joint_limits/real_limits_20261014.json). Knees, ankles and
# head pitch sit ~3 deg short of the stops. Limp, he can't stay sitting: he tips forward onto his chest.
# Never backward in 70-105 simulated variations of stops, servo strength and pose errors.
#   chin_up:   head forward, face forward while crouched; head lands ~0.6 m/s (worst 0.74)
#   head_down: head tucked down; gentlest, head lands ~0.4 m/s (worst 0.47), ends upside down under him
# Left-leg angles (radians, walk-script frame); the right hip is mirrored.
SHUTDOWN_POSES = {
    "chin_up":   dict(hip=0.0, knee=1.68, ankle=-1.13, neck=-0.3, head_pitch=1.08),
    "head_down": dict(hip=-0.4, knee=1.68, ankle=-1.13, neck=-0.69, head_pitch=-0.6),
}


def _packet(sid, instr, params=b""):
    body = bytes([sid, len(params) + 2, instr]) + bytes(params)
    return b"\xff\xff" + body + bytes([~sum(body) & 0xFF])


def _read_byte(port, sid, addr, wait=0.01):
    """One STS3215 register byte over a raw pyserial port, or None."""
    port.reset_input_buffer()
    port.write(_packet(sid, 0x02, bytes([addr, 1])))
    buf, end = bytearray(), time.monotonic() + wait
    while time.monotonic() < end:
        n = port.in_waiting
        if not n:
            time.sleep(0.0002)
            continue
        buf += port.read(n)
        i = buf.find(b"\xff\xff")
        if i >= 0 and len(buf) >= i + 4 and len(buf) >= i + 4 + buf[i + 3]:
            pkt = buf[i + 2:i + 4 + buf[i + 3]]
            if pkt[0] == sid and pkt[-1] == (~sum(pkt[:-1]) & 0xFF) and len(pkt) >= 4:
                return pkt[3]
            return None
    return None


class BatteryGuard:
    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULTS)
        self.cfg.update(cfg or {})
        c = self.cfg
        self.enabled = bool(c["enabled"])
        self.state = "ok"                  # ok | warn | critical
        self.voltage = None
        self.critical_t = None
        self._low_count = 0
        self._last_read_t = None
        self._last_idle_read_t = None
        self._last_beep_t = None
        self._active_t = None              # last time a drive command was non-zero
        self.last_handoff_ms = None
        self.beep_due = False

    # ---------------------------------------------------------------- reading
    def _read_bus(self, hwi):
        """Brief bus handoff: rustypot out, raw pyserial read, rustypot back in."""
        import rustypot
        import serial
        t0 = time.monotonic()
        vals = []
        try:
            hwi.io = None                  # drop the rustypot handle -> releases the port
            try:
                port = serial.Serial(hwi.usb_port, 1_000_000, timeout=0)
            except serial.SerialException:  # handle not freed yet: a full gc is slow, so only now
                gc.collect()
                port = serial.Serial(hwi.usb_port, 1_000_000, timeout=0)
            t_open = time.monotonic()
            with port:
                for sid in READ_IDS:
                    b = _read_byte(port, sid, VOLT_ADDR)
                    if b:
                        vals.append(b / 10.0)
        except Exception as e:  # noqa: BLE001 - never let the guard break the loop
            print(f"[battery] read failed: {e}")
        finally:
            t_read = time.monotonic()
            if hwi.io is None:
                try:
                    hwi.io = rustypot.feetech(hwi.usb_port, 1000000)
                except Exception:  # noqa: BLE001 - one retry; losing the bus is fatal
                    time.sleep(0.05)
                    hwi.io = rustypot.feetech(hwi.usb_port, 1000000)
        t1 = time.monotonic()
        self.last_handoff_ms = (t1 - t0) * 1000
        try:
            self.last_split = f"open {1000*(t_open-t0):.0f} read {1000*(t_read-t_open):.0f} reopen {1000*(t1-t_read):.0f}"
        except NameError:
            self.last_split = ""
        return float(np.median(vals)) if vals else None

    def _read(self, hwi):
        try:
            with open(FAKE_FILE) as f:
                v = float(f.read().strip())
            self.last_handoff_ms = 0.0
            return v
        except (OSError, ValueError):
            pass
        return self._read_bus(hwi)

    def poll(self, now, hwi, paused, drive):
        """Call once per loop tick. Reads the voltage when it's due and safe, updates
        the state, and returns the new voltage (or None if no read this tick)."""
        if not self.enabled or self.state == "critical":
            return None
        self._maybe_beep(now)
        if max(abs(float(x)) for x in drive) > 0.02:
            self._active_t = now
        standing = self._active_t is None or now - self._active_t >= self.cfg["idle_after_s"]
        since = math.inf if self._last_read_t is None else now - self._last_read_t
        since_idle = math.inf if self._last_idle_read_t is None else now - self._last_idle_read_t
        if paused:
            due = since >= self.cfg["period_paused_s"]
        elif standing:
            due = since >= self.cfg["period_idle_s"]
        else:
            due = since_idle >= self.cfg["max_gap_walking_s"] and since >= self.cfg["max_gap_walking_s"]
        if not due:
            return None
        self.last_split = ""
        v = self._read(hwi)
        if self.last_split and os.environ.get("DUCK_BATT_DEBUG"):
            print(f"[battery] handoff {self.last_handoff_ms:.0f} ms ({self.last_split})")
        self._last_read_t = now
        if paused or standing:
            self._last_idle_read_t = now
        if v is None:
            return None
        self.voltage = v
        self._update_state(v, now)
        self._log(now, v, "paused" if paused else "standing" if standing else "walking")
        return v

    # ---------------------------------------------------------------- state
    def _update_state(self, v, now):
        c = self.cfg
        if v <= c["hard_v"]:
            self._low_count = c["confirm"]
        elif v <= c["crit_v"]:
            self._low_count += 1
        else:
            self._low_count = 0
        if self._low_count >= c["confirm"]:
            self.state = "critical"
            self.critical_t = now
            print(f"[battery] {v:.2f} V: FLAT. Stopping, crouching, shutting down.")
        elif v <= c["warn_v"] and self.state == "ok":
            self.state = "warn"            # latches: a resting pack creeps back up
            print(f"[battery] {v:.2f} V: LOW BATTERY")
            self._last_beep_t = None

    def _maybe_beep(self, now):
        self.beep_due = False
        if self.state != "warn":
            return
        if self._last_beep_t is None or now - self._last_beep_t >= self.cfg["beep_every_s"]:
            self._last_beep_t = now
            self.beep_due = True

    @property
    def message(self):
        if self.state == "critical":
            return "🔋 BATTERY FLAT: crouching and shutting down"
        if self.state == "warn":
            return f"🔋 LOW BATTERY {self.voltage:.2f} V: land him soon"
        return ""

    def _log(self, now, v, mode):
        try:
            new = not os.path.exists(LOG_FILE)
            with open(LOG_FILE, "a") as f:
                if new:
                    f.write("time,volts,mode,state,handoff_ms\n")
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{v:.2f},{mode},{self.state},"
                        f"{self.last_handoff_ms:.1f}\n")
        except OSError:
            pass

    # ---------------------------------------------------------------- shutdown
    def crouch_target(self, init_pos):
        """Home pose + the lean-back crouch. Joint order: left hip yaw, roll, pitch, knee,
        ankle, neck pitch, head pitch, head yaw, head roll, right hip yaw ... ankle."""
        c = self.cfg
        t = np.array(init_pos, dtype=float).copy()
        p = SHUTDOWN_POSES.get(c["shutdown_pose"])
        if p is not None:
            t[2], t[3], t[4] = p["hip"], p["knee"], p["ankle"]
            t[11], t[12], t[13] = -p["hip"], p["knee"], p["ankle"]
            t[5], t[6] = p["neck"], p["head_pitch"]
            return t
        dep, lean, ank = c["crouch_depth"], c["crouch_hip_lean"], c["crouch_ankle"]
        t[2] += -dep / 2 + lean
        t[3] += dep
        t[4] += -dep / 2 + ank
        t[11] += dep / 2 - lean
        t[12] += dep
        t[13] += -dep / 2 + ank
        t[3] = min(t[3], c["knee_max"])
        t[12] = min(t[12], c["knee_max"])
        return t

    def crouch(self, hwi, start, init_pos, names, freq, make_action_dict):
        """Ease from `start` to the crouch over crouch_s, then hold hold_s."""
        target = self.crouch_target(init_pos)
        start = np.array(start, dtype=float)
        n = max(1, int(self.cfg["crouch_s"] * freq))
        for k in range(1, n + 1):
            a = 0.5 - 0.5 * math.cos(math.pi * k / n)
            hwi.set_position_all(make_action_dict(start + a * (target - start), names))
            time.sleep(1.0 / freq)
        time.sleep(self.cfg["hold_s"])
        # Check he actually got there (logged so a crouch that didn't happen shows up).
        try:
            got = np.array(hwi.get_present_positions(), dtype=float)
            err = np.degrees(np.abs(got - target))
            worst = int(np.argmax(err))
            msg = (f"crouch reached: worst joint {names[worst]} off by {err[worst]:.1f} deg, "
                   f"knees {np.degrees(got[3]):.0f}/{np.degrees(got[12]):.0f} deg")
        except Exception as e:  # noqa: BLE001
            msg = f"crouch check failed: {e}"
        print(f"[battery] {msg}")
        try:
            with open(LOG_FILE, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},,crouch,{msg.replace(',', ';')},\n")
        except OSError:
            pass

    def poweroff(self):
        if not self.cfg["poweroff"]:
            print("[battery] poweroff disabled in config: leaving the Pi on")
            return
        print("[battery] powering the Pi off. UNPLUG THE BATTERY.")
        try:
            r = subprocess.run(shlex.split(self.cfg["poweroff_cmd"]), capture_output=True, text=True,
                               timeout=10)
            if r.returncode != 0:
                print(f"[battery] poweroff failed ({r.stderr.strip()}). Add a sudo rule:\n"
                      "  echo \"$USER ALL=(root) NOPASSWD: /usr/sbin/poweroff\" | "
                      "sudo tee /etc/sudoers.d/duck-poweroff")
        except Exception as e:  # noqa: BLE001
            print(f"[battery] poweroff failed: {e}")
