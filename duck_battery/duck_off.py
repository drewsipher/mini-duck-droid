#!/usr/bin/env python3
"""Motors off, gently: ease from wherever he is into the shutdown tuck (same pose as the
low-battery failsafe), then turn the torque off. Run with the walk script stopped:
    ~/.virtualenvs/open-duck-mini-runtime/bin/python ~/Open_Duck_Mini_Runtime/scripts/duck_off.py
"""
import json
import os

from battery_guard import BatteryGuard
from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.rustypot_position_hwi import HWI


def make_action_dict(targets, names):
    return dict(zip(names, targets))


def main():
    cfg = json.load(open(os.path.expanduser("~/duck_config.json"))).get("battery_guard", {})
    guard = BatteryGuard(cfg)
    hwi = HWI(DuckConfig())
    names = list(hwi.joints.keys())
    start = hwi.get_present_positions()
    print(f"easing into the '{guard.cfg['shutdown_pose']}' tuck...")
    guard.crouch(hwi, start, list(hwi.init_pos.values()), names, 50, make_action_dict)
    hwi.turn_off()
    print("motors off")


if __name__ == "__main__":
    main()
