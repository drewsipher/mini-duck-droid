"""Shared fixtures for the Open Duck Mini skill-library test suite.

Run from the project root:   sim/.venv/bin/python -m pytest tests -q
or                           ./run_tests.sh          (also writes tests/last_run.md)

Rules baked in here
- MUJOCO_GL=egl so MuJoCo renders headless.
- Nothing in this suite may touch the GPU (training runs overnight): JAX is pinned to CPU
  and CUDA devices are hidden for every subprocess the tests start.
- REF_CLIP / REF_HYBRID are removed so sim_test.Sim uses the deployed brain's own gait clock.
"""
import json
import os
import shutil
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(ROOT, "sim")
ANIM = os.path.join(ROOT, "anim")
CLIPS = os.path.join(ROOT, "tracker", "data", "clips")
TOOLS = os.path.join(ROOT, "tools")
VENV_PY = os.path.join(SIM, ".venv", "bin", "python")
PY = VENV_PY if os.path.exists(VENV_PY) else sys.executable
XML = os.path.join(SIM, "train_pg", "playground", "open_duck_mini_v2", "xmls", "scene_flat_terrain_backlash.xml")
WALK_ONNX = os.path.join(SIM, "BEST_WALK_ONNX_2.onnx")
BLENDER = os.environ.get("BLENDER", os.path.expanduser("~/apps/blender-5.2.2-linux-x64/blender"))
DUCK_BLEND = os.path.join(ANIM, "duck.blend")
MODEL_JSON = os.path.join(ANIM, "duck_model.json")

ACTUATOR_ORDER = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
                  "neck_pitch", "head_pitch", "head_yaw", "head_roll",
                  "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["JAX_PLATFORMS"] = "cpu"            # never let anything here grab the training GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
for _k in ("REF_CLIP", "REF_HYBRID"):
    os.environ.pop(_k, None)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes more than a few seconds (Blender, long rollouts)")
    config.addinivalue_line("markers", "external: needs a tool or service another team is building; skips when absent")


def run(cmd, cwd=SIM, timeout=300, env=None):
    """Run a command, return CompletedProcess with text output; never raises on exit code."""
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=e)


def load_clip_frames(path):
    data = json.load(open(path))
    return data if isinstance(data, list) else data["frames"]


@pytest.fixture(scope="session")
def model():
    """The committed anim/duck_model.json (home pose, actuator order, joint ranges)."""
    if not os.path.exists(MODEL_JSON):
        pytest.skip(f"{MODEL_JSON} missing: run anim/dump_mjcf.py")
    return json.load(open(MODEL_JSON))


@pytest.fixture(scope="session")
def home(model):
    return dict(model["home"]["joints"])


@pytest.fixture(scope="session")
def joint_ranges(model):
    out = {}
    for b in model["bodies"]:
        for j in b["joints"]:
            if j["range"]:
                out[j["name"]] = tuple(j["range"])
    return out


@pytest.fixture
def write_clip(tmp_path):
    """write_clip(name, frames) -> path, in the puppet/simulator clip format at 50 Hz."""
    def _w(name, frames, **extra):
        p = tmp_path / f"{name}.json"
        json.dump({"frames": frames, "interp": "linear", **extra}, open(p, "w"))
        return str(p)
    return _w


def require(path, what):
    if not os.path.exists(path):
        pytest.skip(f"{what} not present yet: {os.path.relpath(path, ROOT)}")
    return path


def require_blender():
    if not (os.path.exists(BLENDER) and os.access(BLENDER, os.X_OK)):
        pytest.skip(f"Blender not found at {BLENDER} (set BLENDER=...)")
    if not os.path.exists(DUCK_BLEND):
        pytest.skip("anim/duck.blend not built")
    return BLENDER


# ----------------------------------------------------------------------------------------------
# last_run.md / last_run.json reporter, for the dashboard. Written when DUCK_TESTS_LAST_RUN is set
# (run_tests.sh sets it) so an ad-hoc partial run never overwrites the full report.
# ----------------------------------------------------------------------------------------------
_RESULTS = []
_EXTRA_SECONDS = {}          # setup/teardown time per file (module fixtures such as the Blender run)
_T0 = time.time()


def pytest_runtest_logreport(report):
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        _RESULTS.append(report)
    else:
        f = report.nodeid.split("::")[0]
        _EXTRA_SECONDS[f] = _EXTRA_SECONDS.get(f, 0.0) + getattr(report, "duration", 0.0)


def _outcome(r):
    if r.outcome == "skipped":
        return "skipped"
    if r.outcome == "failed":
        return "failed"
    return "passed"


def pytest_sessionfinish(session, exitstatus):
    target = os.environ.get("DUCK_TESTS_LAST_RUN")
    if not target:
        return
    files = {}
    failures = []
    skips = []
    for r in _RESULTS:
        f = r.nodeid.split("::")[0]
        d = files.setdefault(f, {"passed": 0, "failed": 0, "skipped": 0, "seconds": 0.0})
        o = _outcome(r)
        d[o] += 1
        d["seconds"] += getattr(r, "duration", 0.0)
        if o == "failed":
            failures.append((r.nodeid, str(r.longrepr)))
        elif o == "skipped":
            reason = r.longrepr[2] if isinstance(r.longrepr, tuple) else str(r.longrepr)
            skips.append((r.nodeid, reason.replace("Skipped: ", "")))
    for f, extra in _EXTRA_SECONDS.items():
        if f in files:
            files[f]["seconds"] += extra
    tot = {k: sum(d[k] for d in files.values()) for k in ("passed", "failed", "skipped")}
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    wall = time.time() - _T0
    lines = [f"# Test run {now}", "",
             f"**{tot['passed']} passed, {tot['failed']} failed, {tot['skipped']} skipped** "
             f"in {wall:.0f} s (pytest exit {int(exitstatus)}).", "",
             "| File | Pass | Fail | Skip | Seconds |", "|---|---|---|---|---|"]
    for f, d in sorted(files.items()):
        lines.append(f"| {f} | {d['passed']} | {d['failed']} | {d['skipped']} | {d['seconds']:.1f} |")
    lines.append(f"| **total** | **{tot['passed']}** | **{tot['failed']}** | **{tot['skipped']}** | **{wall:.1f}** |")
    slow = sorted(_RESULTS, key=lambda r: -getattr(r, "duration", 0.0))[:8]
    lines += ["", "## Slowest tests", ""]
    for r in slow:
        lines.append(f"- {getattr(r, 'duration', 0.0):.1f} s  {r.nodeid}")
    if failures:
        lines += ["", "## Failures", ""]
        for nid, rep in failures:
            short = "\n".join(rep.strip().splitlines()[-25:])
            lines += [f"### {nid}", "", "```", short, "```", ""]
    if skips:
        lines += ["", "## Skipped (tools not built yet, or marked slow)", ""]
        for nid, reason in skips:
            lines.append(f"- {nid}: {reason}")
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    with open(target, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.splitext(target)[0] + ".json", "w") as fh:
        json.dump({"time": now, "wall_seconds": wall, "exit": int(exitstatus), "totals": tot, "files": files,
                   "failures": [{"test": n, "output": o[-4000:]} for n, o in failures],
                   "skipped": [{"test": n, "reason": r} for n, r in skips]}, fh, indent=1)
