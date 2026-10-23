#!/usr/bin/env python3
"""Placeholder clips to exercise the pipeline until real animations are authored in Blender.
They only move the head, so the trunk reference (home pose, feet planted) stays valid.

    python3 anim/make_test_clips.py            # writes tracker/data/clips/test_bow.json, test_look.json
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tracker", "data", "clips")
FPS = 50
HOME = json.load(open(os.path.join(HERE, "duck_model.json")))["home"]["joints"]


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def bump(t, t0, t1, hold=0.0):
    """0 -> 1 over [t0, t0+ramp], hold, back to 0 by t1."""
    ramp = (t1 - t0 - hold) / 2
    if t < t0 + ramp:
        return smooth((t - t0) / ramp)
    if t < t1 - ramp:
        return 1.0
    return smooth((t1 - t) / ramp)


def clip(seconds, fn, name):
    frames = []
    for k in range(int(seconds * FPS) + 1):
        t = k / FPS
        pose = dict(HOME)
        for j, v in fn(t).items():
            pose[j] = HOME[j] + v
        frames.append([round(t, 3), {j: round(v, 4) for j, v in pose.items()}])
    path = os.path.join(OUT, name + ".json")
    json.dump({"frames": frames, "interp": "linear", "authored_in": {"tool": "make_test_clips.py", "placeholder": True}}, open(path, "w"))
    print("wrote", path, len(frames), "frames")


def bow(t):            # head bows down and comes back, 2.5 s
    a = bump(t, 0.2, 2.3, hold=0.6)
    return {"neck_pitch": 0.45 * a, "head_pitch": 0.35 * a}


def look(t):           # look left, then right, then centre, 3 s
    a = bump(t, 0.1, 1.5, hold=0.5) - bump(t, 1.5, 2.9, hold=0.5)
    return {"head_yaw": 0.6 * a, "head_roll": 0.15 * a, "head_pitch": -0.1 * abs(a)}


if __name__ == "__main__":
    clip(2.5, bow, "test_bow")
    clip(3.0, look, "test_look")
