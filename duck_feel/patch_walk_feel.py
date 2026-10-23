#!/usr/bin/env python3
"""Live "feel" controls for the walk script (web UI Tune panel -> Feel):
leg / head servo stiffness, per-joint gains on the policy's motion (hip pitch, knee,
ankle), smoothing of the motor targets, and a live ankle lean trim (shifts both ankle
offsets, invisible to the policy, like editing joints_offsets).

Run on the duck:
    python3 patch_walk_feel.py ~/Open_Duck_Mini_Runtime/scripts/v2_rl_walk_mujoco.py
Saved values live in ~/duck_config.json under "feel"; the ankle trim is saved into
joints_offsets.
"""
import sys

path = sys.argv[1]
s = open(path).read()
if "_feel" in s:
    sys.exit("already patched")

EDITS = [
    # 1. state, read from the config at start-up
    ("""        self.velocity_clip = bool(self.duck_config.velocity_clip)
""", """        self.velocity_clip = bool(self.duck_config.velocity_clip)

        # Live "feel" controls (web Tune panel). Saved under "feel" in duck_config.
        f = dict(FEEL_DEFAULTS)
        f.update(self.duck_config.json_config.get("feel", {}))
        self._feel = f
        self._feel_kp_dirty = False
        self._feel_kp_t = 0.0
        self._feel_ankle_base = {j: float(self.hwi.joints_offsets[j]) for j in ("left_ankle", "right_ankle")}
        self._feel_ankle_trim_deg = 0.0
        self._feel_smoothed = None
        print(f"feel = {self._feel}")
"""),
    # 2. start(): servo stiffness from the config
    ("""        kps = [self.pid[0]] * 14
        kds = [self.pid[2]] * 14

        # lower head kps
        kps[5:9] = [8, 8, 8, 8]
""", """        feel = dict(FEEL_DEFAULTS)
        feel.update(self.duck_config.json_config.get("feel", {}))
        kps = [int(feel["leg_kp"])] * 14
        kds = [self.pid[2]] * 14

        # lower head kps
        kps[5:9] = [int(feel["head_kp"])] * 4
"""),
    # 3. web settings
    ("""                elif key == "governor_enabled":
""", """                elif key in FEEL_RANGES:
                    lo, hi = FEEL_RANGES[key]
                    v = float(np.clip(float(v), lo, hi))
                    if key == "ankle_trim_deg":
                        self._set_ankle_trim(v)
                    else:
                        self._feel[key] = v
                        if key in ("leg_kp", "head_kp"):
                            self._feel_kp_dirty = True
                elif key == "governor_enabled":
"""),
    ("""        backup = save_config_fields(fields)
        print(f"[walk] SAVED walk tuning {fields} (backup {backup})")
""", """        fields["feel"] = {k: round(float(v), 3) for k, v in self._feel.items()}
        if self._feel_ankle_trim_deg:
            offs = dict(self.duck_config.json_config.get("joints_offsets", {}))
            for j in ("left_ankle", "right_ankle"):
                offs[j] = round(float(self.hwi.joints_offsets[j]), 4)
            fields["joints_offsets"] = offs
            self.duck_config.json_config["joints_offsets"] = offs
            self._feel_ankle_base = {j: float(self.hwi.joints_offsets[j]) for j in ("left_ankle", "right_ankle")}
            self._feel_ankle_trim_deg = 0.0      # saved into the offsets: the trim starts again from 0
        backup = save_config_fields(fields)
        print(f"[walk] SAVED walk tuning {fields} (backup {backup})")
"""),
    ("""        self.governor.smooth = float(g["smooth"])
        print(f"[walk] RESET""", """        self.governor.smooth = float(g["smooth"])
        self._feel = dict(FEEL_DEFAULTS)
        self._feel_kp_dirty = True
        self._set_ankle_trim(0.0)
        print(f"[walk] RESET"""),
    # 4. helpers
    ("""    def _governor_config_dict(self):
""", """    def _set_ankle_trim(self, deg):
        \"\"\"Shift both ankle offsets by `deg` from their saved values (+ = lean forward).\"\"\"
        self._feel_ankle_trim_deg = float(deg)
        for j, base in self._feel_ankle_base.items():
            self.hwi.joints_offsets[j] = base + np.radians(deg)

    def _apply_feel_kp(self, now):
        \"\"\"Write changed servo stiffness, at most 5 times a second (it's a bus write).\"\"\"
        if not self._feel_kp_dirty or now - self._feel_kp_t < 0.2:
            return
        kps = [int(round(self._feel["leg_kp"]))] * 14
        kps[5:9] = [int(round(self._feel["head_kp"]))] * 4
        try:
            self.hwi.set_kps(kps)
            print(f"[feel] stiffness legs {kps[0]} head {kps[5]}")
        except Exception as e:  # noqa: BLE001
            print(f"[feel] stiffness write failed: {e}")
        self._feel_kp_dirty = False
        self._feel_kp_t = now

    def _governor_config_dict(self):
"""),
    # 5. telemetry for the sliders
    ("""                "governor": self._governor_config_dict(),
            },""", """                "governor": self._governor_config_dict(),
                **{k: round(float(v), 3) for k, v in self._feel.items()},
                "ankle_trim_deg": round(float(self._feel_ankle_trim_deg), 2),
            },"""),
    # 6. loop: stiffness writes (also while paused)
    ("""                self._consume_web_settings()
""", """                self._consume_web_settings()
                self._apply_feel_kp(t)
"""),
    # 7. loop: joint gains + smoothing on the policy's targets
    ("""                self.motor_targets = self.init_pos + action * self.action_scale
""", """                delta = action * self.action_scale
                fe = self._feel
                delta[[2, 11]] *= fe["hip_gain"]
                delta[[3, 12]] *= fe["knee_gain"]
                delta[[4, 13]] *= fe["ankle_gain"]
                self.motor_targets = self.init_pos + delta
                if fe["smoothing"] > 0:
                    if self._feel_smoothed is None:
                        self._feel_smoothed = self.motor_targets.copy()
                    a = fe["smoothing"]
                    self._feel_smoothed = a * self._feel_smoothed + (1 - a) * self.motor_targets
                    self.motor_targets = self._feel_smoothed.copy()
                else:
                    self._feel_smoothed = None
"""),
    # 8. defaults + ranges
    ("""class RLWalk:""", """# Live "feel" controls (web Tune panel). Defaults = how the stock runtime behaves.
FEEL_DEFAULTS = {"leg_kp": 30, "head_kp": 8, "hip_gain": 1.0, "knee_gain": 1.0,
                 "ankle_gain": 1.0, "smoothing": 0.0}
FEEL_RANGES = {"leg_kp": (8, 32), "head_kp": (2, 32), "hip_gain": (0.5, 1.5),
               "knee_gain": (0.5, 1.5), "ankle_gain": (0.5, 2.0), "smoothing": (0.0, 0.8),
               "ankle_trim_deg": (-8.0, 8.0)}


class RLWalk:"""),
]

for old, new in EDITS:
    if s.count(old) != 1:
        sys.exit(f"could not find exactly one copy of:\n{old}")
    s = s.replace(old, new)
open(path, "w").write(s)
print("patched", path)
