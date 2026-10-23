"""anim/blender_duck.py inside headless Blender: import a clip onto anim/duck.blend, export it
again, and compare in pytest. The round trip must be exact to 1e-6 rad on all 14 joints.
"""
import json
import os
import subprocess

import pytest

from conftest import ANIM, CLIPS, DUCK_BLEND, ACTUATOR_ORDER, require_blender, load_clip_frames

pytestmark = pytest.mark.slow

SCRIPT = """
import sys, json, bpy
sys.path.insert(0, {anim!r})
import blender_duck as bd
arm = bpy.data.objects[bd.ARM]
act = bd.import_clip({clip!r}, "roundtrip_test")
info = bd.export_clip({out!r}, meta={{"test": "roundtrip"}})
json.dump({{"export": info, "frame_start": bpy.context.scene.frame_start, "frame_end": bpy.context.scene.frame_end,
            "ik_legs": float(arm["ik_legs"]), "action": act.name, "bones": len(arm.pose.bones)}}, open({meta!r}, "w"))
"""


def blender_roundtrip(clip, tmp_path):
    blender = require_blender()
    out, meta, script = tmp_path / "out.json", tmp_path / "meta.json", tmp_path / "rt.py"
    script.write_text(SCRIPT.format(anim=ANIM, clip=clip, out=str(out), meta=str(meta)))
    r = subprocess.run([blender, "-b", DUCK_BLEND, "--python", str(script)], capture_output=True, text=True, timeout=240)
    assert out.exists(), f"Blender did not export (exit {r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    return json.load(open(out)), json.load(open(meta)), r


@pytest.fixture(scope="module", params=["s2a_turn_left.json", "test_bow.json"])
def roundtrip(request, tmp_path_factory):
    clip = os.path.join(CLIPS, request.param)
    if not os.path.exists(clip):
        pytest.skip(f"{request.param} not in tracker/data/clips")
    exported, meta, proc = blender_roundtrip(clip, tmp_path_factory.mktemp("bl"))
    return clip, exported, meta, proc


def test_blender_runs_headless():
    blender = require_blender()
    r = subprocess.run([blender, "-b", "--version"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "Blender" in r.stdout, r.stderr[-500:]


def test_roundtrip_exact(roundtrip):
    clip, exported, meta, _ = roundtrip
    src = load_clip_frames(clip)
    dst = exported["frames"]
    assert len(dst) == len(src), f"{len(src)} frames in, {len(dst)} out"
    worst, worst_j = 0.0, None
    for (ts, ps), (td, pd) in zip(src, dst):
        assert abs(ts - td) < 1e-6
        for j in ACTUATOR_ORDER:
            d = abs(ps[j] - pd[j])
            if d > worst:
                worst, worst_j = d, j
    assert worst < 1e-6, f"max abs joint diff {worst:.3e} rad on {worst_j}"
    assert set(dst[0][1]) == set(ACTUATOR_ORDER)


def test_roundtrip_metadata(roundtrip):
    clip, exported, meta, _ = roundtrip
    n = len(exported["frames"])
    assert exported["interp"] == "linear"
    if "base" in exported:                   # written only when the trunk bone is animated (10-23 change)
        assert len(exported["base"]) == n, "one base pose per frame"
        for pos, quat in exported["base"]:
            assert len(pos) == 3 and len(quat) == 4
            assert abs(sum(q * q for q in quat) - 1) < 1e-3
    assert exported["authored_in"]["tool"] == "blender"
    assert exported["authored_in"]["fps"] == 50
    assert exported["authored_in"]["off_axis_leak"] < 1e-5, "joint rotation leaked off the hinge axis"
    assert meta["ik_legs"] == 0.0, "import must switch leg IK off so FK keys play unchanged"
    assert meta["frame_end"] - meta["frame_start"] + 1 == n
    assert meta["bones"] == 14 + 1 + 2       # 14 joints, base, two foot IK targets


def test_roundtrip_base_is_standing(roundtrip):
    """A clip without a base track keeps the trunk where the rig's home pose put it."""
    clip, exported, _, _ = roundtrip
    src = json.load(open(clip))
    if isinstance(src, dict) and src.get("base"):
        pytest.skip("clip carries its own base track")
    if "base" not in exported:
        pytest.skip("exporter omits the base track when the trunk is not animated (clip_check then plants the feet)")
    z = [b[0][2] for b in exported["base"]]
    assert 0.12 < min(z) and max(z) < 0.22, f"base z range {min(z):.3f}..{max(z):.3f}"
