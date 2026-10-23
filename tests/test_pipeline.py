"""End-to-end smoke of the M1 chain in simulation, with the tools that exist tonight:

    clip (test_bow) -> stage 2 check -> skill package (episodic) -> G1 battery (1 seed x 1 episode)
    -> validate -> graph_runtime loads it

The policy is a placeholder (the walk brain or a null policy), so the battery may legitimately
report a FAIL (exit 2): what this test guards is that every hand-off in the chain reads what the
previous step wrote. Skips while a link is missing.
"""
import json
import os
import shutil
import sys

import pytest

from conftest import ANIM, CLIPS, PY, ROOT, SIM, TOOLS, WALK_ONNX, require, run

pytestmark = [pytest.mark.slow, pytest.mark.external]

CLIP = os.path.join(CLIPS, "test_bow.json")
CLIP_CHECK = os.path.join(ANIM, "clip_check.py")
SKILL_PACKAGE = os.path.join(TOOLS, "skill_package.py")
BATTERY = os.path.join(TOOLS, "battery.py")


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    for p, what in ((CLIP, "tracker/data/clips/test_bow.json"), (SKILL_PACKAGE, "tools/skill_package.py"),
                    (BATTERY, "tools/battery.py"), (WALK_ONNX, "sim/BEST_WALK_ONNX_2.onnx")):
        require(p, what)
    root = tmp_path_factory.mktemp("chain")
    state = {"root": root}
    # 1. stage 2 gate
    rep = root / "check.json"
    r = run([PY, CLIP_CHECK, CLIP, "--json", str(rep)], cwd=SIM)
    state["check"] = (r.returncode, json.load(open(rep)))
    # 2. package
    pkg = root / "skills" / "bow" / "v1"
    r = run([PY, SKILL_PACKAGE, "create", "--name", "bow", "--type", "episodic", "--onnx", WALK_ONNX, "--clip", CLIP,
             "--out", str(pkg), "--version", "1", "--timeout", "4", "--entry-from", "walk", "--exit-to", "walk",
             "--clip-source", "anim/make_test_clips.py", "--run", "placeholder: deployed walk brain"], cwd=SIM)
    state["create"] = r
    state["pkg"] = str(pkg)
    # 3. battery, smallest possible
    if r.returncode == 0:
        state["battery"] = run([PY, BATTERY, str(pkg), "--seeds", "1", "--episodes", "1", "--no-video", "--no-status"], cwd=SIM, timeout=900)
    return state


def test_clip_passes_stage2(chain):
    rc, rep = chain["check"]
    assert rc == 0 and rep["pass"], rep["fails"]


def test_package_created_from_the_checked_clip(chain):
    r = chain["create"]
    assert r.returncode == 0, f"create failed:\n{r.stdout[-1500:]}\n{r.stderr[-800:]}"
    skill = json.load(open(os.path.join(chain["pkg"], "skill.json")))
    assert skill["type"] == "episodic" and skill["timeout_s"] == 4
    assert skill["entry"]["from"] == ["walk"] and skill["exit"]["to"] == ["walk"]
    clip = json.load(open(os.path.join(chain["pkg"], "clip.json")))
    src = json.load(open(CLIP))
    assert clip["frames"] == src["frames"], "the package clip must be the checked clip, byte for byte in content"


def test_battery_runs_and_reports_on_the_package(chain):
    r = chain.get("battery")
    if r is None:
        pytest.skip("package was not created")
    assert r.returncode in (0, 2), f"battery tool error (exit {r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-1500:]}"
    rep = json.load(open(os.path.join(chain["pkg"], "report", "battery.json")))
    assert rep["skill"] == "bow" and rep["type"] == "episodic"
    assert rep["overall"].count("/") == 1
    assert isinstance(rep["passed"], bool)
    assert rep["tests"], "an episodic battery must have at least one test cell"
    cell = next(iter(rep["tests"].values()))
    assert "episodes" in cell and len(cell["episodes"]) == 1
    assert "fell" in cell["episodes"][0]
    if rep.get("mean_joint_err_rad") is not None:
        assert 0 <= rep["mean_joint_err_rad"] < 3.2


def test_package_validates_after_battery(chain):
    if chain.get("battery") is None:
        pytest.skip("package was not created")
    r = run([PY, SKILL_PACKAGE, "validate", chain["pkg"]], cwd=SIM)
    assert r.returncode == 0, r.stdout[-1500:]


def test_runtime_loads_the_packaged_skill(chain):
    if chain.get("battery") is None:
        pytest.skip("package was not created")
    require(os.path.join(ROOT, "graph_runtime", "packages.py"), "graph_runtime")
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from graph_runtime.packages import PackageError, load_package
    try:
        pkg = load_package(chain["pkg"])
    except PackageError as e:
        pytest.fail(f"graph_runtime refuses the package skill_package.py created and battery.py scored: {e}")
    assert pkg.type == "episodic" and pkg.timeout_s == 4.0
    assert abs(pkg.clip.duration - 2.5) < 0.03
    assert pkg.allows_entry_from("walk") and pkg.allows_exit_to("walk")
