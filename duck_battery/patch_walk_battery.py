#!/usr/bin/env python3
"""Hook battery_guard.py (low-battery warning + crouch-and-shutdown) into the walk script.

Run on the duck, with battery_guard.py copied next to the walk script:
    python3 patch_walk_battery.py ~/Open_Duck_Mini_Runtime/scripts/v2_rl_walk_mujoco.py
Turn it off without un-patching: "battery_guard": {"enabled": false} in ~/duck_config.json.
"""
import sys

path = sys.argv[1]
s = open(path).read()
if "battery_guard" in s:
    sys.exit("already patched")

EDITS = [
    # 1. import
    ("""from mini_bdx_runtime.rustypot_position_hwi import HWI
""", """from mini_bdx_runtime.rustypot_position_hwi import HWI
try:
    from battery_guard import BatteryGuard   # low-battery failsafe (scripts/battery_guard.py)
except ImportError:
    BatteryGuard = None
"""),
    # 2. create the guard next to the battery monitor
    ("""        self.battery_mon = (
            BatteryMonitor(getattr(self.duck_config, "battery", {}))
            if BatteryMonitor is not None else None
        )
""", """        self.battery_mon = (
            BatteryMonitor(getattr(self.duck_config, "battery", {}))
            if BatteryMonitor is not None else None
        )
        guard_cfg = self.duck_config.json_config.get("battery_guard", {})
        self.battery_guard = (
            BatteryGuard(guard_cfg)
            if BatteryGuard is not None and guard_cfg.get("enabled", True) else None
        )
        self._battery_poweroff = False
        print(f"battery_guard = {self.battery_guard.cfg if self.battery_guard else None}")
"""),
    # 3. the guard does the voltage reads now; show its warning on the web page
    ("""        allow_batt = self.paused and self.duck_config.battery_servo_read
""", """        allow_batt = (self.paused and self.duck_config.battery_servo_read
                      and self.battery_guard is None)
"""),
    ("""            sounds=self._sound_names, uptime_s=now - self._start_t,
""", """            sounds=self._sound_names, uptime_s=now - self._start_t,
            message=(self.battery_guard.message if self.battery_guard else ""),
"""),
    # 4. check the battery every tick (before the paused short-cut, so it works paused too)
    ("""                self._publish_telemetry(t)

                if self.paused:
""", """                self._publish_telemetry(t)

                if self.battery_guard is not None:
                    raw_commands = self._battery_tick(raw_commands, t)
                    if self._battery_poweroff:
                        break

                if self.paused:
"""),
    # 5. methods
    ("""    def run(self):
""", """    def _battery_tick(self, raw_commands, now):
        \"\"\"Low-battery failsafe (scripts/battery_guard.py). Returns the commands to use.\"\"\"
        g = self.battery_guard
        v = g.poll(now, self.hwi, paused=self.paused, drive=raw_commands[:3])
        if v is not None and self.battery_mon is not None:
            from mini_bdx_runtime.battery import estimate_percent
            self.battery_mon._cache = {
                "voltage": v,
                "percent": estimate_percent(v, self.battery_mon.v_min, self.battery_mon.v_max),
                "charging": self.battery_mon.charge.update(now, v),
            }
        if g.beep_due and self.duck_config.speaker:
            try:
                self.sounds.play("beep1.wav")
            except Exception:  # noqa: BLE001
                pass
        if g.state != "critical":
            return raw_commands
        # Flat: stand still for settle_s (walking only), then crouch, motors off, power off.
        if self.recorder.state == "playing":
            self.recorder.state = "idle"
        if not self.paused and not self.fall_detector.fallen and now - g.critical_t < g.cfg["settle_s"]:
            return [0.0] * len(raw_commands)
        self._battery_shutdown()
        return [0.0] * len(raw_commands)

    def _battery_shutdown(self):
        g = self.battery_guard
        print("[battery] FLAT: crouching, motors off, shutting down")
        if self.duck_config.speaker:
            try:
                self.sounds.play("beep2.wav")
            except Exception:  # noqa: BLE001
                pass
        if not self.fall_detector.fallen:
            g.crouch(self.hwi, self.motor_targets, self.init_pos, list(self.hwi.joints.keys()),
                     self.control_freq, make_action_dict)
        self.hwi.turn_off()
        print("[battery] motors off")
        for name in ("antennas", "eyes", "projector"):
            if getattr(self.duck_config, name, False):
                try:
                    getattr(self, name).stop()
                except Exception:  # noqa: BLE001
                    pass
        if self.scanner is not None:
            self.scanner.stop()
        self.feet_contacts.stop()
        self._battery_poweroff = True

    def run(self):
"""),
    # 6. after the loop has saved the recording: power off
    ("""        print("TURNING OFF")
""", """        print("TURNING OFF")
        if self._battery_poweroff:
            if self.web_server is not None:
                self.web_server.stop()
            self.battery_guard.poweroff()
"""),
]

for old, new in EDITS:
    if s.count(old) != 1:
        sys.exit(f"could not find exactly one copy of:\n{old}")
    s = s.replace(old, new)
open(path, "w").write(s)
print("patched", path)
