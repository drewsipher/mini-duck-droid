"""anim/dump_mjcf.py: the MJCF -> JSON dump the Blender rig is built from.

Regenerates the dump into a temp file and checks the invariants the rig, the clips and the
trainer all rely on (actuator order, 14-joint home pose, compiled meshes, finite rest geometry),
then checks the committed anim/duck_model.json still matches the MJCF.
"""
import json
import math
import os

import pytest

from conftest import ACTUATOR_ORDER, ANIM, MODEL_JSON, PY, SIM, run

DUMP = os.path.join(ANIM, "dump_mjcf.py")


@pytest.fixture(scope="module")
def dumped(tmp_path_factory):
    out = tmp_path_factory.mktemp("model") / "duck_model.json"
    r = run([PY, DUMP, str(out)], cwd=SIM, timeout=120)
    assert r.returncode == 0, f"dump_mjcf.py failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    assert out.exists()
    return json.load(open(out))


def test_actuator_order(dumped):
    assert dumped["actuators"] == ACTUATOR_ORDER
    assert dumped["actuator_joints"] == ACTUATOR_ORDER, "every actuator drives the joint of the same name"


def test_counts(dumped):
    joints = [j for b in dumped["bodies"] for j in b["joints"]]
    assert len(dumped["bodies"]) == 17
    assert len(joints) == 25                      # 1 free base + 14 actuated + 10 passive backlash
    assert sum(j["name"].endswith("_backlash") for j in joints) == 10
    assert sum(j["type"] == "free" for j in joints) == 1
    assert sum(len(b["geoms"]) for b in dumped["bodies"]) == 46
    assert len(dumped["actuators"]) == 14
    assert joints[0]["type"] == "free", "the first joint in tree order is the floating base"


def test_home_pose_has_14_joints(dumped):
    home = dumped["home"]
    assert set(home["joints"]) == set(ACTUATOR_ORDER)
    assert len(home["joints"]) == 14
    assert len(home["base_pos"]) == 3 and len(home["base_quat"]) == 4
    assert abs(sum(q * q for q in home["base_quat"]) - 1) < 1e-6
    assert 0.1 < home["base_pos"][2] < 0.3, "home pose base height should be a standing duck (~0.15 m)"


def test_home_inside_joint_ranges(dumped):
    ranges = {j["name"]: j["range"] for b in dumped["bodies"] for j in b["joints"] if j["range"]}
    for name, q in dumped["home"]["joints"].items():
        lo, hi = ranges[name]
        assert lo - 1e-6 <= q <= hi + 1e-6, f"{name} home {q} outside MJCF range {ranges[name]}"


def test_meshes_non_empty(dumped):
    meshes = dumped["meshes"]
    assert meshes, "no compiled meshes in the dump"
    used = {g["mesh"] for b in dumped["bodies"] for g in b["geoms"]}
    assert used <= set(meshes), f"geoms reference meshes missing from the dump: {used - set(meshes)}"
    for name, md in meshes.items():
        assert len(md["verts"]) >= 4, name
        assert len(md["faces"]) >= 4, name
        n = len(md["verts"])
        assert all(0 <= i < n for f in md["faces"] for i in f), f"{name}: face index out of range"
        assert all(len(v) == 3 and all(math.isfinite(x) for x in v) for v in md["verts"]), name


def test_geom_rest_centroids_finite(dumped):
    for b in dumped["bodies"]:
        for g in b["geoms"]:
            c = g["rest_world_centroid"]
            assert len(c) == 3 and all(math.isfinite(x) for x in c), f"{b['name']}:{g['name']} centroid {c}"
            assert max(abs(x) for x in c) < 0.6, f"{b['name']}:{g['name']} centroid {c} is not near the duck"
            zlo, zhi = g["rest_world_zrange"]
            assert math.isfinite(zlo) and math.isfinite(zhi) and zlo <= zhi


def test_every_actuated_joint_is_a_limited_hinge(dumped):
    by_name = {j["name"]: j for b in dumped["bodies"] for j in b["joints"]}
    for a in ACTUATOR_ORDER:
        j = by_name[a]
        assert j["type"] == "hinge", a
        assert j["range"] and j["range"][0] < j["range"][1], a
        assert abs(sum(x * x for x in j["axis"]) - 1) < 1e-6, f"{a} hinge axis not unit length"


def test_committed_dump_matches_mjcf(dumped):
    """anim/duck_model.json is what duck.blend was built from; it must not drift from the MJCF."""
    if not os.path.exists(MODEL_JSON):
        pytest.skip("anim/duck_model.json not committed")
    committed = json.load(open(MODEL_JSON))
    assert committed["actuators"] == dumped["actuators"]
    assert [b["name"] for b in committed["bodies"]] == [b["name"] for b in dumped["bodies"]]
    for k, v in dumped["home"]["joints"].items():
        assert abs(committed["home"]["joints"][k] - v) < 1e-9, f"home {k} drifted"
    for bc, bd in zip(committed["bodies"], dumped["bodies"]):
        assert abs(bc["mass"] - bd["mass"]) < 1e-9, f"mass of {bc['name']} changed in the MJCF: regenerate duck_model.json"
        assert all(abs(x - y) < 1e-9 for x, y in zip(bc["xpos"], bd["xpos"])), f"rest pose of {bc['name']} changed"
