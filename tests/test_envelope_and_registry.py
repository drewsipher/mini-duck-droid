"""Cross-team data files that other files must agree with:

  mechatronics/envelope.json   deploy ranges inside the MJCF ranges, home inside deploy, hash of the XML
  tracker/data/tools.json      every dashboard tool points at a script that exists; GPU flag only on training
"""
import hashlib
import json
import os

import pytest

from conftest import ACTUATOR_ORDER, ROOT, SIM, require

ENVELOPE = os.path.join(ROOT, "mechatronics", "envelope.json")
TOOLS_JSON = os.path.join(ROOT, "tracker", "data", "tools.json")
MODEL_XML = os.path.join(SIM, "train_pg", "playground", "open_duck_mini_v2", "xmls", "open_duck_mini_v2_backlash.xml")

pytestmark = pytest.mark.external


def val(x):
    """Envelope entries are {value, unit, status, source} records; joints are plain."""
    return x["value"] if isinstance(x, dict) and "value" in x else x


@pytest.fixture(scope="module")
def envelope():
    require(ENVELOPE, "mechatronics/envelope.json")
    return json.load(open(ENVELOPE))


def test_envelope_has_14_joints_with_deploy_ranges(envelope):
    joints = envelope["joints"]
    assert set(joints) == set(ACTUATOR_ORDER)
    for name, j in joints.items():
        lo, hi = j["deploy_range_rad"]
        mlo, mhi = j["mjcf_range_rad"]
        assert lo < hi, name
        assert mlo - 1e-6 <= lo and hi <= mhi + 1e-6, f"{name}: deploy range {lo:.3f}..{hi:.3f} outside MJCF {mlo:.3f}..{mhi:.3f}"
        assert lo <= j["home_rad"] <= hi, f"{name}: home {j['home_rad']} outside its deploy range {lo:.3f}..{hi:.3f}"
        assert j["home_inside"] is True, name
        assert j["max_velocity_rad_s"] > 0


def test_envelope_mjcf_ranges_match_the_model(envelope, joint_ranges, home):
    for name, j in envelope["joints"].items():
        for a, b in zip(j["mjcf_range_rad"], joint_ranges[name]):
            assert abs(a - b) < 1e-3, f"{name}: envelope says MJCF range {j['mjcf_range_rad']}, model says {joint_ranges[name]}"
        assert abs(j["home_rad"] - home[name]) < 1e-3, f"{name}: envelope home {j['home_rad']} vs keyframe {home[name]}"


def test_envelope_hash_is_the_training_xml(envelope):
    rm = envelope["robot_model"]
    h = hashlib.sha256(open(MODEL_XML, "rb").read()).hexdigest()
    assert val(rm["mjcf_sha256"]) == h, "envelope.robot_model.mjcf_sha256 is not the sha256 of open_duck_mini_v2_backlash.xml (packages use that)"
    assert os.path.exists(os.path.join(ROOT, val(rm["mjcf"]))), "envelope.robot_model.mjcf path does not exist"


def test_envelope_headline_numbers(envelope):
    assert abs(val(envelope["torque"]["deploy_limit_nm"]) - 1.86) < 1e-6
    assert val(envelope["velocity"]["max_motor_velocity_rad_s"]) == 4.0
    assert 30 <= val(envelope["fall"]["tilt_deg"]) <= 90
    assert 0.05 <= val(envelope["fall"]["min_base_height_m"]) <= 0.14
    assert 1.4 <= val(envelope["robot_model"]["total_mass_kg"]) <= 2.3
    legend = set(envelope.get("status_legend", []))
    if legend:                                   # every record says how much to trust it
        for section in ("torque", "velocity", "fall", "thermal", "voltage"):
            for k, rec in envelope[section].items():
                if isinstance(rec, dict) and "value" in rec:
                    assert rec.get("status") in legend, f"{section}.{k}: status {rec.get('status')!r} not in the legend"


# ---------------------------------------------------------------- dashboard tool registry

@pytest.fixture(scope="module")
def tools():
    require(TOOLS_JSON, "tracker/data/tools.json")
    t = json.load(open(TOOLS_JSON))
    items = t if isinstance(t, list) else t.get("tools", t)
    return list(items.values()) if isinstance(items, dict) else items


def test_tools_json_commands_point_at_existing_scripts(tools):
    assert tools, "no tools registered"
    missing = []
    for tool in tools:
        cwd = os.path.normpath(os.path.join(ROOT, tool.get("cwd", ".")))
        assert os.path.isdir(cwd), f"{tool['id']}: cwd {tool.get('cwd')} does not exist"
        for tok in tool["command"]:
            if tok.startswith("{") or "=" in tok or tok.startswith("-"):
                continue
            if tok.endswith((".py", ".sh")):
                p = tok if os.path.isabs(tok) else os.path.join(cwd, tok)
                if not os.path.exists(p):
                    missing.append(f"{tool['id']}: {tok} (cwd {tool.get('cwd')})")
    assert not missing, "dashboard tools whose script is missing:\n" + "\n".join(missing)


def test_tools_json_gpu_flag_only_on_training(tools):
    for tool in tools:
        cmd = " ".join(tool["command"])
        trains = any(k in cmd for k in ("curriculum.sh", "train_skill.sh", "train_pg/playground", "runner.py"))
        assert bool(tool.get("gpu")) == trains, f"{tool['id']}: gpu={tool.get('gpu')} for command {cmd!r}"


def test_tools_json_ids_unique_and_named(tools):
    ids = [t["id"] for t in tools]
    assert len(ids) == len(set(ids)), f"duplicate tool ids: {ids}"
    for t in tools:
        assert t.get("name") and t.get("group"), t["id"]
        for a in t.get("args", []):
            assert "name" in a and "type" in a, f"{t['id']}: arg without name/type: {a}"
            assert "{" + a["name"] + "}" in " ".join(t["command"]) or a.get("env"), f"{t['id']}: arg {a['name']} never used in the command"
