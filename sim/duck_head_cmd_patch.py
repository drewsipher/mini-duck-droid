#!/usr/bin/env python3
"""Patch the duck's walk script for policies trained with HEAD_FROM_CMD=1, and add an
automatic head pose.

With "head_from_command": true in ~/duck_config.json, the head target becomes
home + head command (the policy's head outputs are ignored), set before the slew
clamp, as in training. Without the flag the script behaves exactly as before.

With "auto_head": true (or a dict of gain overrides), the head also follows the drive
command, eased: forward/back pitch it forward/back, turning yaws it into the turn,
side-stepping tilts it toward the step. Your own head commands add on top. The default
gains are the ones tested in sim (~/src/mini-duck-droid/sim/autohead.py).

Run on the duck:  python3 duck_head_cmd_patch.py ~/Open_Duck_Mini_Runtime/scripts/v2_rl_walk_mujoco.py
"""
import sys

path = sys.argv[1]
s = open(path).read()
if "head_from_command" in s:
    sys.exit("already patched")

old_target = """                self.motor_targets = self.init_pos + action * self.action_scale
"""
new_target = """                self.motor_targets = self.init_pos + action * self.action_scale
                if self.head_from_command:
                    # Policy trained with HEAD_FROM_CMD: head = home + head command only.
                    self.motor_targets[5:9] = (
                        np.array(self.init_pos)[5:9] + np.array(self.last_commands[3:])
                    )
"""
old_head = """                head_motor_targets = self.last_commands[3:] + self.motor_targets[5:9]
                self.motor_targets[5:9] = head_motor_targets
"""
new_head = """                if not self.head_from_command:
                    head_motor_targets = self.last_commands[3:] + self.motor_targets[5:9]
                    self.motor_targets[5:9] = head_motor_targets
"""
old_init = """        self.motor_targets = np.array(self.init_pos.copy())
"""
new_init = """        self.motor_targets = np.array(self.init_pos.copy())
        # duck_config "head_from_command": true for policies trained with HEAD_FROM_CMD=1.
        self.head_from_command = bool(self.duck_config.json_config.get("head_from_command", False))
        print(f"head_from_command = {self.head_from_command}")
"""
old_gov = """                self._apply_stability_governor(self.last_commands)
"""
new_gov = """                self._apply_stability_governor(self.last_commands)
                if self.auto_head is not None:
                    self.last_commands = self._with_auto_head(self.last_commands)
"""
old_def = """    def run(self):
"""
new_def = """    def _with_auto_head(self, commands):
        \"\"\"Add the eased automatic head pose (see duck_config "auto_head") to the head commands.
        Joint signs: +neck/+head pitch tip the head back, +yaw looks left, +roll tilts left.\"\"\"
        g = self.auto_head
        vx, vy, wz = float(commands[0]), float(commands[1]), float(commands[2])
        fx = vx / 0.15
        target = np.array([
            g["neck_fwd"] * max(fx, 0.0) + g["neck_back"] * max(-fx, 0.0),
            g["pitch_fwd"] * max(fx, 0.0) + g["pitch_back"] * max(-fx, 0.0),
            g["yaw_turn"] * wz,
            g["roll_side"] * vy / 0.2,
        ])
        dt = 1.0 / self.control_freq
        self._auto_head_pose += dt / (g["tau"] + dt) * (target - self._auto_head_pose)
        out = list(commands)
        for k in range(4):
            out[3 + k] = float(out[3 + k]) + float(self._auto_head_pose[k])
        return out

    def run(self):
"""
old_init2 = """        print(f"head_from_command = {self.head_from_command}")
"""
new_init2 = """        print(f"head_from_command = {self.head_from_command}")
        auto = self.duck_config.json_config.get("auto_head", False)
        self.auto_head = None
        if auto:
            self.auto_head = dict(neck_fwd=-0.20, neck_back=0.15, pitch_fwd=-0.12, pitch_back=0.12,
                                  yaw_turn=0.30, roll_side=0.25, tau=0.35)
            if isinstance(auto, dict):
                self.auto_head.update({k: float(v) for k, v in auto.items()})
        self._auto_head_pose = np.zeros(4)
        print(f"auto_head = {self.auto_head}")
"""
for old, new in ((old_target, new_target), (old_head, new_head), (old_init, new_init),
                 (old_init2, new_init2), (old_gov, new_gov), (old_def, new_def)):
    if s.count(old) != 1:
        sys.exit(f"could not find exactly one copy of:\n{old}")
    s = s.replace(old, new)
open(path, "w").write(s)
print("patched", path)
