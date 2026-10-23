"""Every clip in tracker/data/clips/ is in the shared format the puppet tool, Blender, the
checker and the trainer agree on: {"frames": [[t, {joint: rad}], ...]} at 50 Hz, 14 joints.
"""
import glob
import json
import os

import pytest

from conftest import ACTUATOR_ORDER, CLIPS

CLIP_FILES = sorted(glob.glob(os.path.join(CLIPS, "*.json")))


@pytest.mark.parametrize("path", CLIP_FILES, ids=[os.path.basename(p) for p in CLIP_FILES])
def test_clip_format(path, joint_ranges):
    data = json.load(open(path))
    if isinstance(data, dict) and "keys" in data and "frames" not in data:
        pytest.skip("keyed clip from anim.js, not yet sampled to frames")
    frames = data if isinstance(data, list) else data["frames"]
    assert frames, "empty clip"
    times = [f[0] for f in frames]
    assert all(b > a for a, b in zip(times, times[1:])), "timestamps not strictly increasing"
    assert abs(times[0]) < 1e-6, "clip does not start at t = 0"
    dts = [b - a for a, b in zip(times, times[1:])]
    if dts:                                       # puppet recordings carry real timestamps (one 18 ms step seen)
        assert max(abs(d - 0.02) for d in dts) <= 5e-3, f"not sampled at 50 Hz: dt range {min(dts):.4f}..{max(dts):.4f}"
    for t, pose in frames:
        assert set(pose) == set(ACTUATOR_ORDER), f"frame {t}: joints {sorted(pose)}"
        for j, q in pose.items():
            lo, hi = joint_ranges[j]
            assert lo - 0.05 <= q <= hi + 0.05, f"{os.path.basename(path)} t={t}: {j}={q} outside {lo:.2f}..{hi:.2f}"
    if isinstance(data, dict):                  # anim.js writes "smooth" for keyframed clips, recordings "linear"
        assert data.get("interp", "linear") in ("linear", "smooth"), data.get("interp")
        if data.get("base"):
            assert len(data["base"]) == len(frames)


def test_clips_folder_has_clips():
    assert CLIP_FILES, f"no clips in {CLIPS}"
