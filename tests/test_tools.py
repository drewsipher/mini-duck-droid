"""Tools the other teams are building tonight. Each test skips with a clear reason while its
tool is absent, so the suite stays green and the skip list in tests/last_run.md shows what is
still to come. No test here may start training or import JAX (the GPU is busy all night).

  tools/skill_package.py   validate | create | import | list | hash-model | status
  tools/battery.py         G1 sim battery: multi-seed rollouts, ratios, report/battery.json
  tools/sim2sim.py         G2: same ONNX through the runtime's own policy step
  experiments/log.py       append-only experiment log
  graph_runtime/           motion graph runtime (packages, policies, graph, engine), own tests in graph_runtime/tests
  sim/train_skill.sh, sim/eval_tracking.py     (checked statically only)
  mechatronics/servo_health.py --fake
  tracker/server.py        GET /api/jobs on :8430
"""
import glob
import hashlib
import json
import os
import py_compile
import re
import shutil
import stat
import sys
import urllib.error
import urllib.request

import pytest

from conftest import CLIPS, PY, ROOT, SIM, TOOLS, WALK_ONNX, require, run

pytestmark = pytest.mark.external

SKILL_PACKAGE = os.path.join(TOOLS, "skill_package.py")
BATTERY = os.path.join(TOOLS, "battery.py")
SIM2SIM = os.path.join(TOOLS, "sim2sim.py")
SCHEMA = os.path.join(ROOT, "skills", "schema.json")
EXP_LOG = os.path.join(ROOT, "experiments", "log.py")
SERVO_HEALTH = os.path.join(ROOT, "mechatronics", "servo_health.py")
TRAIN_SKILL = os.path.join(SIM, "train_skill.sh")
EVAL_TRACKING = os.path.join(SIM, "eval_tracking.py")
GRAPH_RUNTIME = os.path.join(ROOT, "graph_runtime")
GRAPH_TESTS = os.path.join(GRAPH_RUNTIME, "tests")
MODEL_XML = os.path.join(SIM, "train_pg", "playground", "open_duck_mini_v2", "xmls", "open_duck_mini_v2_backlash.xml")
JOBS_URL = "http://localhost:8430/api/jobs"
LIBRARY_PACKAGES = sorted(glob.glob(os.path.join(ROOT, "skills", "*", "v*")))
# the shape tools/battery.py writes (graph_runtime.battery_passed reads "passed")
PASSED_BATTERY = {"passed": True, "overall": "50/50", "overall_ratio": 1.0, "seeds": 5, "episodes": 10, "tests": {}}


def helps(tool, *sub):
    """`tool [sub] --help` must exit 0 and print usage."""
    r = run([PY, tool, *sub, "--help"], cwd=SIM, timeout=120)
    assert r.returncode == 0, f"{os.path.relpath(tool, ROOT)} {' '.join(sub)} --help exit {r.returncode}\n{r.stderr[-1500:]}"
    assert "usage" in (r.stdout + r.stderr).lower()
    return r.stdout + r.stderr


def skill_package(*args, **kw):
    require(SKILL_PACKAGE, "tools/skill_package.py")
    return run([PY, SKILL_PACKAGE, *args], cwd=SIM, timeout=300, **kw)


@pytest.fixture(scope="module")
def created_package(tmp_path_factory):
    """A walk package built by `skill_package.py create` from the deployed brain and a recorded clip."""
    require(SKILL_PACKAGE, "tools/skill_package.py")
    clip = os.path.join(CLIPS, "s2a_fwd.json")
    if not (os.path.exists(clip) and os.path.exists(WALK_ONNX)):
        pytest.skip("s2a_fwd.json or BEST_WALK_ONNX_2.onnx missing")
    out = tmp_path_factory.mktemp("skills") / "walk" / "v1"
    r = skill_package("create", "--name", "walk", "--type", "locomotion", "--onnx", WALK_ONNX, "--clip", clip,
                      "--out", str(out), "--version", "1", "--commands", "vx:-0.15,0.15", "vy:-0.2,0.2", "wz:-1,1",
                      "--entry-from", "bow", "--exit-to", "bow")
    assert r.returncode == 0, f"create failed:\n{r.stdout[-2000:]}\n{r.stderr[-1000:]}"
    return str(out)


def copy_package(src, tmp_path, name="copy"):
    dst = tmp_path / name / "v1"
    shutil.copytree(src, dst)
    return str(dst)


# ---------------------------------------------------------------- skill packages

def test_skill_package_help():
    require(SKILL_PACKAGE, "tools/skill_package.py")
    text = helps(SKILL_PACKAGE)
    for sub in ("validate", "create", "import", "list"):
        assert sub in text, f"skill_package.py --help does not mention '{sub}'"


def test_skill_package_create_writes_the_layout(created_package):
    for f in ("skill.json", "clip.json", "policy.onnx", "report"):
        assert os.path.exists(os.path.join(created_package, f)), f"create did not write {f}"
    skill = json.load(open(os.path.join(created_package, "skill.json")))
    assert skill["name"] == "walk" and skill["type"] == "locomotion" and skill["version"] == 1
    assert skill["status"] == "draft"
    assert skill["commands"] == {"vx": [-0.15, 0.15], "vy": [-0.2, 0.2], "wz": [-1, 1]}
    assert skill["envelope"]["torque_nm"] == 1.86 and skill["envelope"]["fall_tilt_deg"] == 45
    assert skill["obs_spec"]["size"] == 101, "the deployed walk brain takes a 101-wide observation"
    assert re.fullmatch(r"[0-9a-f]{64}", skill["robot_model_hash"])
    assert skill["hashes"]["policy_onnx"] == hashlib.sha256(open(WALK_ONNX, "rb").read()).hexdigest()
    assert "bow" in skill["neighbours"]


def test_skill_package_draft_validates_without_battery_only(created_package):
    r = skill_package("validate", "--no-battery", created_package)
    assert r.returncode == 0, r.stdout[-1500:]
    r = skill_package("validate", created_package)
    assert r.returncode == 1, "a draft without report/battery.json must not validate for loading"
    assert "battery" in r.stdout.lower()


def test_skill_package_refuses_missing_policy(created_package, tmp_path):
    d = copy_package(created_package, tmp_path)
    os.remove(os.path.join(d, "policy.onnx"))
    r = skill_package("validate", "--no-battery", d)
    assert r.returncode == 1, f"validate accepted a package without policy.onnx:\n{r.stdout[-1500:]}"
    assert "policy" in r.stdout.lower(), "the refusal should name the missing part"


def test_skill_package_refuses_missing_clip_and_skill_json(created_package, tmp_path):
    d = copy_package(created_package, tmp_path)
    os.remove(os.path.join(d, "clip.json"))
    r = skill_package("validate", "--no-battery", d)
    assert r.returncode == 1 and "clip" in r.stdout.lower(), r.stdout[-1500:]
    os.remove(os.path.join(d, "skill.json"))
    r = skill_package("validate", "--no-battery", d)
    assert r.returncode == 1 and "skill.json" in r.stdout.lower(), r.stdout[-1500:]


def test_skill_package_refuses_schema_violations(created_package, tmp_path):
    d = copy_package(created_package, tmp_path)
    p = os.path.join(d, "skill.json")
    skill = json.load(open(p))
    skill["type"] = "dance"                                 # not one of the four skill types
    del skill["envelope"]["fall_tilt_deg"]                  # required envelope field
    json.dump(skill, open(p, "w"))
    r = skill_package("validate", "--no-battery", d)
    assert r.returncode == 1, "validate accepted type 'dance' and a missing fall_tilt_deg"
    out = r.stdout.lower()
    assert "type" in out and "fall_tilt_deg" in out, r.stdout[-1500:]


def test_skill_package_hash_model_is_the_mjcf_sha256(created_package):
    r = skill_package("hash-model")
    assert r.returncode == 0, r.stderr[-800:]
    h = r.stdout.strip().split()[-1]
    assert re.fullmatch(r"[0-9a-f]{64}", h), r.stdout
    skill = json.load(open(os.path.join(created_package, "skill.json")))
    assert skill["robot_model_hash"] == h, "create must stamp the same hash hash-model prints"
    assert h == hashlib.sha256(open(MODEL_XML, "rb").read()).hexdigest(), \
        "robot_model_hash should be the sha256 of open_duck_mini_v2_backlash.xml (skills/schema.json)"


def test_skill_package_list_runs():
    r = skill_package("list")
    assert r.returncode == 0, r.stderr[-1500:]


def test_schema_json_lists_the_plan_fields():
    require(SCHEMA, "skills/schema.json")
    schema = json.load(open(SCHEMA))
    required = set(schema["required"])
    for f in ("name", "type", "version", "commands", "entry", "exit", "neighbours", "timeout_s",
              "envelope", "robot_model_hash", "actuator_model_version", "recipe_version", "seed"):
        assert f in required, f"skill-library-plan.md section 4 field {f!r} not required by the schema"
    assert set(schema["properties"]["type"]["enum"]) == {"locomotion", "overlay", "episodic", "recovery"}


@pytest.mark.parametrize("pkg", LIBRARY_PACKAGES, ids=[os.path.relpath(p, os.path.join(ROOT, "skills")) for p in LIBRARY_PACKAGES])
def test_library_package_validates(pkg):
    """Every package already in skills/ must pass the validator (drafts may lack a battery)."""
    r = skill_package("validate", "--no-battery", pkg)
    assert r.returncode == 0, f"{os.path.relpath(pkg, ROOT)} is invalid:\n{r.stdout[-1500:]}"


def test_library_has_packages():
    require(os.path.join(ROOT, "skills"), "skills/")
    if not LIBRARY_PACKAGES:
        pytest.skip("no skills/<name>/v<N> packages yet")


@pytest.mark.parametrize("pkg", LIBRARY_PACKAGES, ids=[os.path.relpath(p, os.path.join(ROOT, "skills")) for p in LIBRARY_PACKAGES])
def test_library_reports_are_ratios(pkg):
    """Framework 4.3: results are ratios, never 'works'. Every report in the library says k of n."""
    rep_dir = os.path.join(pkg, "report")
    checked = 0
    b = os.path.join(rep_dir, "battery.json")
    if os.path.exists(b):
        rep = json.load(open(b))
        assert re.fullmatch(r"\d+/\d+", str(rep.get("overall", ""))), f"battery.json overall must be 'k/n', got {rep.get('overall')!r}"
        k, n = (int(x) for x in rep["overall"].split("/"))
        assert 0 <= k <= n and n > 0
        assert isinstance(rep.get("passed"), bool)
        for name, t in rep.get("tests", {}).items():
            assert isinstance(t.get("success"), int) and isinstance(t.get("total"), int), f"{name}: no success/total"
            assert 0 <= t["success"] <= t["total"]
        assert sum(t["total"] for t in rep["tests"].values()) == n, "overall n must be the sum of the test cells"
        checked += 1
    s = os.path.join(rep_dir, "sim2sim.json")
    if os.path.exists(s):
        rep = json.load(open(s))
        assert rep.get("match") in (True, False, None)
        if rep.get("match") is None:     # skipped (no runtime for this obs_spec yet): it must say why
            assert rep.get("notes"), "a skipped sim2sim.json must carry notes"
        else:
            assert isinstance(rep.get("rms_target_diff_rad"), (int, float)) and isinstance(rep.get("threshold_rad"), (int, float))
        checked += 1
    h = os.path.join(rep_dir, "hardware.json")
    if os.path.exists(h):
        rep = json.load(open(h))
        assert isinstance(rep.get("trials"), int) and rep["trials"] > 0, "hardware.json needs trials (n)"
        assert 0 <= float(rep.get("ratio", -1)) <= 1, "hardware.json needs ratio in 0..1"
        assert rep.get("date") and rep.get("setup"), "hardware.json needs date and setup (held/gantry/mats)"
        checked += 1
    if not checked:
        pytest.skip("draft package without reports yet")


# ---------------------------------------------------------------- battery and sim-to-sim

def test_battery_help_mentions_seeds_and_episodes():
    require(BATTERY, "tools/battery.py")
    text = helps(BATTERY).lower()
    assert "seed" in text, "battery.py must take a seed count (G1 is multi-seed)"
    assert "episode" in text


@pytest.mark.slow
def test_battery_one_episode_writes_ratios(created_package, tmp_path):
    require(BATTERY, "tools/battery.py")
    d = copy_package(created_package, tmp_path, "batt")
    r = run([PY, BATTERY, d, "--seeds", "1", "--episodes", "1", "--tests", "standing,forward", "--no-video", "--no-status"],
            cwd=SIM, timeout=600)
    assert r.returncode in (0, 2), f"battery tool error (exit {r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-1500:]}"
    rep_path = os.path.join(d, "report", "battery.json")
    assert os.path.exists(rep_path), "battery did not write report/battery.json"
    rep = json.load(open(rep_path))
    text = json.dumps(rep).lower()
    assert "seed" in text and "tests" in rep, f"report keys {sorted(rep)}"
    for name in ("standing", "forward"):
        assert name in rep["tests"], f"no result for the {name} test: {sorted(rep['tests'])}"
        t = rep["tests"][name]
        ok = next((t[k] for k in ("success", "ok", "passed") if k in t), None)
        n = next((t[k] for k in ("total", "n") if k in t), None)
        assert isinstance(ok, int) and isinstance(n, int) and 0 <= ok <= n == 1, f"{name}: ratio not reported as k of n: {t}"
        assert all(k in t for k in ("ratio", "episodes")), f"{name}: per-episode records missing: {sorted(t)}"
    assert rep["overall"].count("/") == 1, f"overall must read 'k/n', got {rep['overall']!r}"
    assert "thresholds" in rep and "randomisation" in rep and "robot_model_hash" in rep
    assert ("standing" in r.stdout) and ("/1" in r.stdout or "1/1" in r.stdout or "0/1" in r.stdout), "ratios must be printed as k/n"
    assert isinstance(rep["passed"], bool) and rep["passed"] == (r.returncode == 0), "exit code and report disagree on pass"
    r2 = skill_package("validate", d)
    assert r2.returncode == 0, f"a package with a battery report should validate fully:\n{r2.stdout[-1500:]}"
    # the runtime must read the real report the same way the battery tool meant it
    if os.path.exists(os.path.join(GRAPH_RUNTIME, "packages.py")):
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        from graph_runtime.packages import battery_passed
        assert battery_passed(rep) == rep["passed"], "graph_runtime.battery_passed disagrees with battery.json['passed']"


def test_sim2sim_help():
    require(SIM2SIM, "tools/sim2sim.py")
    helps(SIM2SIM)


@pytest.mark.slow
def test_sim2sim_short_run_writes_report(created_package, tmp_path):
    """G2 tool: the upstream runtime code drives the package in MuJoCo and is compared with
    sim_test.Sim in lockstep; the report must carry the RMS, the threshold and the verdict."""
    require(SIM2SIM, "tools/sim2sim.py")
    d = copy_package(created_package, tmp_path, "s2s")
    r = run([PY, SIM2SIM, d, "--seconds", "2", "--settle", "1", "--no-video", "--no-status"], cwd=SIM, timeout=600)
    assert r.returncode in (0, 2), f"sim2sim tool error (exit {r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-1500:]}"
    rep_path = os.path.join(d, "report", "sim2sim.json")
    assert os.path.exists(rep_path), "sim2sim did not write report/sim2sim.json"
    rep = json.load(open(rep_path))
    assert isinstance(rep["match"], bool) and rep["match"] == (r.returncode == 0)
    rms, thr = rep["rms_target_diff_rad"], rep["threshold_rad"]
    assert 0 <= rms < 1.0 and 0 < thr <= 0.1, (rms, thr)
    assert rep["match"] == (rms < thr and not rep["free_run"]["runtime_fell"] and not rep["free_run"]["sim_test_fell"]), \
        "match must follow the documented rule (RMS below threshold and neither stack fell)"
    assert len(rep["per_joint_rms_rad"]) == 14
    assert "RMS" in r.stdout


# ---------------------------------------------------------------- experiment log

def test_experiment_log_list_runs_without_a_log():
    require(EXP_LOG, "experiments/log.py")
    r = run([PY, EXP_LOG, "list", "--json"], cwd=ROOT, timeout=60)
    assert r.returncode == 0, r.stderr[-800:]


def test_experiment_log_parser_reads_the_entry_shape():
    """`add` writes only to experiments/log.md (never from a test); parse() is checked on text."""
    require(EXP_LOG, "experiments/log.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("duck_exp_log", EXP_LOG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    text = ("# Experiment log\n\n## 2026-10-23T01:02:03-04:00 walk/v1 battery\n"
            "- Hypothesis: the brain walks under the deployed filters\n- Command: tools/battery.py skills/walk/v1\n"
            "- Evidence: overall 280/400\n  backward 0/50\n- **Decision:** repackage with --lpf 0\n")
    entries = mod.parse(text)
    assert len(entries) == 1
    e = entries[0]
    assert e["title"] == "walk/v1 battery" and e["when"].startswith("2026-10-23T")
    assert set(mod.FIELDS) <= set(e["fields"]), e["fields"]
    assert "backward 0/50" in e["fields"]["evidence"], "continuation lines must stay inside the bullet"
    assert e["fields"]["decision"] == "repackage with --lpf 0"

def test_experiment_log_help():
    require(EXP_LOG, "experiments/log.py")
    helps(EXP_LOG)


def test_experiment_log_md_has_the_four_fields():
    md = require(os.path.join(ROOT, "experiments", "log.md"), "experiments/log.md")
    text = open(md).read().lower()
    assert text.strip(), "log.md is empty"
    for field in ("hypothesis", "command", "evidence", "decision"):
        assert field in text, f"no '{field}' field anywhere in experiments/log.md"


# ---------------------------------------------------------------- motion graph runtime

@pytest.fixture(scope="module")
def gr():
    """Import graph_runtime from the project root (it must not need JAX or the training tree)."""
    require(os.path.join(GRAPH_RUNTIME, "__init__.py"), "graph_runtime package")
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import graph_runtime.packages as packages
    return packages


@pytest.fixture(scope="module")
def gr_package(created_package, tmp_path_factory):
    """The created package made loadable by graph_runtime as of tonight: timeout_s null instead of
    skill_package.py's 0.0 for locomotion, and a passing battery report. The raw contract is
    checked separately in test_graph_runtime_loads_created_package."""
    d = tmp_path_factory.mktemp("gr") / "walk" / "v1"
    shutil.copytree(created_package, d)
    p = d / "skill.json"
    skill = json.load(open(p))
    if not skill.get("timeout_s"):
        skill["timeout_s"] = None
    json.dump(skill, open(p, "w"), indent=1)
    json.dump(PASSED_BATTERY, open(d / "report" / "battery.json", "w"))
    return str(d)


def test_graph_runtime_own_tests_pass():
    require(GRAPH_TESTS, "graph_runtime/tests")
    r = run([PY, "-m", "pytest", GRAPH_TESTS, "-q", "--no-header", "-p", "no:cacheprovider"], cwd=ROOT, timeout=900)
    summary = next((l for l in reversed(r.stdout.splitlines()) if " passed" in l or " failed" in l or "no tests" in l), r.stdout[-200:])
    failed = [l for l in r.stdout.splitlines() if l.startswith("FAILED ")]
    assert r.returncode in (0, 5), f"graph_runtime's own tests: {summary.strip()}\n" + "\n".join(failed[:20]) + f"\n{r.stderr[-800:]}"
    if r.returncode == 5:
        pytest.skip("graph_runtime/tests collected no tests yet")


def test_graph_runtime_imports_without_training_libraries():
    require(os.path.join(GRAPH_RUNTIME, "__init__.py"), "graph_runtime package")
    code = ("import sys; import graph_runtime, graph_runtime.packages, graph_runtime.policies, graph_runtime.graph; "
            "bad=[m for m in sys.modules if m.split('.')[0] in ('jax','jaxlib','torch','mujoco_playground','brax','mujoco')]; "
            "print('clean' if not bad else 'heavy', bad)")
    r = run([PY, "-c", code], cwd=ROOT, timeout=120)
    assert r.returncode == 0, f"import graph_runtime failed:\n{r.stderr[-1500:]}"
    assert r.stdout.startswith("clean"), f"graph_runtime pulls in simulator/training libraries on import (it must run on the Pi): {r.stdout}"


def test_graph_runtime_loads_created_package(gr, created_package, tmp_path):
    """Contract: what `skill_package.py create` writes (plus a battery report) is what
    `graph_runtime.load_package` accepts. Same team, two modules."""
    d = copy_package(created_package, tmp_path)
    json.dump(PASSED_BATTERY, open(os.path.join(d, "report", "battery.json"), "w"))
    pkg = gr.load_package(d)
    assert pkg.name == "walk" and pkg.type == "locomotion" and pkg.version == 1


def test_graph_runtime_package_fields(gr, gr_package):
    pkg = gr.load_package(gr_package)
    assert pkg.name == "walk" and pkg.type == "locomotion" and pkg.version == 1
    assert pkg.obs_spec["size"] == 101
    assert pkg.clip.duration > 0.4
    assert pkg.allows_entry_from("bow") and pkg.allows_exit_to("bow") and not pkg.allows_entry_from("hop")
    assert pkg.policy_path.exists() and pkg.battery_passed


def test_graph_runtime_refuses_incomplete_package(gr, gr_package, tmp_path):
    d = copy_package(gr_package, tmp_path)
    os.remove(os.path.join(d, "policy.onnx"))
    with pytest.raises(gr.PackageError, match="incomplete|missing"):
        gr.load_package(d)
    d2 = copy_package(gr_package, tmp_path, "nobattery")
    shutil.rmtree(os.path.join(d2, "report"))
    with pytest.raises(gr.PackageError):
        gr.load_package(d2)
    d3 = copy_package(gr_package, tmp_path, "emptyonnx")
    open(os.path.join(d3, "policy.onnx"), "w").close()
    with pytest.raises(gr.PackageError, match="empty"):
        gr.load_package(d3)


def test_graph_runtime_refuses_wrong_clip_joints(gr, gr_package, tmp_path):
    d = copy_package(gr_package, tmp_path)
    p = os.path.join(d, "clip.json")
    clip = json.load(open(p))
    for t, pose in clip["frames"]:
        pose.pop("left_knee")
    json.dump(clip, open(p, "w"))
    with pytest.raises(gr.PackageError, match="joint"):
        gr.load_package(d)


def test_graph_runtime_null_policy_package_round_trip(gr, gr_package, tmp_path):
    """write_package with no ONNX writes a zero policy that onnxruntime can run."""
    import numpy as np
    import onnxruntime as ort
    skill = json.load(open(os.path.join(gr_package, "skill.json")))
    skill.update({"name": "sit", "type": "recovery", "timeout_s": 2.0, "obs_spec": {"names": ["joint_pos_rel"], "size": 14},
                  "entry": {"from": ["*"], "phase": [0, 1], "window_s": 0.3}, "exit": {"to": ["walk"]}})
    clip = json.load(open(os.path.join(gr_package, "clip.json")))
    d = gr.write_package(tmp_path / "sit" / "v1", skill, clip, PASSED_BATTERY)
    pkg = gr.load_package(d)
    assert pkg.type == "recovery" and pkg.battery_passed
    sess = ort.InferenceSession(str(pkg.policy_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.run(None, {inp.name: np.ones((1, 14), np.float32)})[0]
    assert out.shape[-1] == 14 and np.all(out == 0), "the null policy must output 14 zeros"


def test_graph_runtime_honours_package_head_mode(gr, tmp_path):
    """Contract: the head mode the package declares is the one the runtime applies. skill_package.py
    writes runtime.head_mode in {add, from_cmd, policy}; graph_runtime reads a policy block with
    {add, from_command, policy}. A HEAD_FROM_CMD brain packaged with --head-mode from_cmd must not be
    run with head_mode 'add' on the robot."""
    clip = os.path.join(CLIPS, "s2a_fwd.json")
    out = tmp_path / "walk_fc" / "v1"
    r = skill_package("create", "--name", "walk", "--type", "locomotion", "--onnx", WALK_ONNX, "--clip", clip,
                      "--out", str(out), "--version", "1", "--head-mode", "from_cmd", "--lpf", "6")
    assert r.returncode == 0, r.stdout[-1500:]
    json.dump(PASSED_BATTERY, open(out / "report" / "battery.json", "w"))
    p = out / "skill.json"
    skill = json.load(open(p))
    skill["timeout_s"] = skill["timeout_s"] or None        # sidestep the timeout_s mismatch tested elsewhere
    json.dump(skill, open(p, "w"))
    pkg = gr.load_package(out)
    assert pkg.policy.head_mode in ("from_command", "from_cmd"), \
        f"package says runtime.head_mode=from_cmd, runtime will apply {pkg.policy.head_mode!r}"
    lpf = getattr(pkg.policy, "lpf_hz", None)
    assert lpf == 6.0, f"package says runtime.lpf_hz=6, runtime will apply {lpf!r}"


@pytest.mark.parametrize("pkg_dir", LIBRARY_PACKAGES, ids=[os.path.relpath(p, os.path.join(ROOT, "skills")) for p in LIBRARY_PACKAGES])
def test_graph_runtime_loads_library_packages(gr, pkg_dir):
    if not os.path.exists(os.path.join(pkg_dir, "report", "battery.json")):
        pytest.skip("draft without a battery report")
    pkg = gr.load_package(pkg_dir)
    assert pkg.name == os.path.basename(os.path.dirname(pkg_dir))


# ---------------------------------------------------------------- training entry points (never run)

def test_train_skill_sh_is_valid_bash():
    p = require(TRAIN_SKILL, "sim/train_skill.sh")
    r = run(["bash", "-n", p], cwd=SIM, timeout=30)
    assert r.returncode == 0, r.stderr
    assert os.stat(p).st_mode & stat.S_IXUSR, "train_skill.sh is not executable"
    assert "set -" in open(p).read(), "no set -e/-u/-o pipefail in train_skill.sh"


def test_eval_tracking_compiles():
    p = require(EVAL_TRACKING, "sim/eval_tracking.py")
    py_compile.compile(p, doraise=True)              # static only: importing it may touch the GPU
    text = open(p).read()
    assert '"""' in text[:600] or "'''" in text[:600], "eval_tracking.py should start with a usage docstring"


# ---------------------------------------------------------------- mechatronics

def test_servo_health_fake_reports_14_servos():
    require(SERVO_HEALTH, "mechatronics/servo_health.py")
    r = run([PY, SERVO_HEALTH, "--fake"], cwd=ROOT, timeout=120)
    assert r.returncode == 0, f"servo_health.py --fake exit {r.returncode}:\n{r.stderr[-1500:]}"
    lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
    assert len(lines) == 1, f"stdout must be exactly one JSON line, got {len(lines)} lines"
    rec = json.loads(lines[0])
    assert rec["fake"] is True and rec["missing"] == []
    for key in ("temps", "voltage", "positions", "status"):
        assert len(rec[key]) == 14, f"{key} has {len(rec[key])} entries, not 14"
    assert rec["summary"]["answered"] == 14
    assert all(0 < v < 100 for v in rec["temps"].values())
    assert all(5 < v < 10 for v in rec["voltage"].values())


# ---------------------------------------------------------------- dashboard jobs API

def test_jobs_api():
    try:
        with urllib.request.urlopen(JOBS_URL, timeout=3) as resp:
            body = resp.read().decode()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            pytest.skip("tracker/server.py has no /api/jobs yet (404)")
        raise
    except (urllib.error.URLError, OSError) as e:
        pytest.skip(f"tracker not running on :8430 ({e})")
    data = json.loads(body)
    assert isinstance(data, (list, dict)), "GET /api/jobs should return JSON"
    jobs = data.get("jobs", data) if isinstance(data, dict) else data
    if isinstance(jobs, list):
        for j in jobs:
            assert isinstance(j, dict) and ("id" in j or "name" in j), f"job entry without id/name: {j}"
