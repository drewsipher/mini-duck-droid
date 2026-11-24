#!/usr/bin/env python3
"""Write ref.json into a skill package: the clip's joint targets and velocities on the 50 Hz control grid,
exactly as the tracking task saw them (sim/train_pg/playground/common/clip_tracking_reference.py), so a
runtime without MuJoCo (the Pi) can build the tracking observation and the residual's base.

    cd sim && PYTHONPATH=train_pg train_pg/.venv/bin/python ../tools/clip_ref_table.py ../skills/stand/v1 [--loop]

ref.json: {"joints": [...actuator order...], "home": [14], "dt": 0.02, "loop": bool, "cycle_steps": n,
           "q": [[14] x n], "qd": [[14] x n]}   (one cycle for a loop clip, the whole clip otherwise)
"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim", "train_pg"))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
import recipe_v1 as R                                                # noqa: E402
from playground.common.clip_tracking_reference import ClipTrackingReference   # noqa: E402

SCENE_XML = os.path.join(HERE, "..", "sim", "train_pg", R.SCENE_XML)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("package")
    ap.add_argument("--loop", action="store_true", default=None, help="default: skill.json's 'loop'")
    a = ap.parse_args()
    skill = json.load(open(os.path.join(a.package, "skill.json")))
    loop = bool(skill.get("loop", False)) if a.loop is None else a.loop
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    ref = ClipTrackingReference(os.path.join(a.package, "clip.json"), m, loop=loop, horizon_s=4.0, quiet=True)
    n = int(ref.cycle_steps) if loop else int(ref.n)
    out = {"joints": list(ref.actuator_names), "home": [float(x) for x in ref.home_ctrl], "dt": R.CTRL_DT, "loop": loop,
           "cycle_steps": n, "duration_s": float(ref.duration), "dof_vel_scale": R.DOF_VEL_SCALE,
           "q": np.asarray(ref.q[:n]).round(6).tolist(), "qd": np.asarray(ref.qd[:n]).round(6).tolist()}
    path = os.path.join(a.package, "ref.json")
    json.dump(out, open(path, "w"))
    print(f"wrote {path}: {n} steps, loop {loop}, joints {len(out['joints'])}")


if __name__ == "__main__":
    main()
