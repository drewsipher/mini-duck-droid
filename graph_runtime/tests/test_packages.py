"""Package loading: refusal rules, clip model, null policy file."""
import json
import shutil

import numpy as np
import pytest

from graph_runtime.packages import HOME, JOINTS, Clip, PackageError, battery_passed, load_package
from graph_runtime.policies import ClipPolicy, DeployFilter, make_policy
from graph_runtime.tests.conftest import PROJECT, pose_clip, skill, write_json, WALK_OBS


def test_loads_complete_package(skill_root):
    pkg = load_package(skill_root / "bow" / "v1")
    assert pkg.name == "bow" and pkg.type == "episodic" and pkg.version == 1
    assert pkg.clip.duration == pytest.approx(1.0)
    assert pkg.battery_passed
    assert pkg.policy.kind == "null"
    assert pkg.allows_entry_from("walk") and not pkg.allows_entry_from("sit")


@pytest.mark.parametrize("missing", ["skill.json", "clip.json", "policy.onnx", "report/battery.json"])
def test_refuses_missing_part(skill_root, missing):
    d = skill_root / "bow" / "v1"
    (d / missing).unlink()
    with pytest.raises(PackageError, match="incomplete"):
        load_package(d)


def test_refuses_empty_policy_file(skill_root):
    d = skill_root / "bow" / "v1"
    (d / "policy.onnx").write_bytes(b"")
    with pytest.raises(PackageError, match="empty"):
        load_package(d)


def test_refuses_garbage_onnx_for_onnx_policy(skill_root):
    d = skill_root / "walk" / "v1"
    s = json.loads((d / "skill.json").read_text())
    s["policy"] = {"kind": "onnx"}
    s["obs_spec"] = {"names": WALK_OBS, "size": 101}
    write_json(d / "skill.json", s)
    (d / "policy.onnx").write_bytes(b"not an onnx file")
    with pytest.raises(PackageError, match="loadable ONNX"):
        load_package(d)


def test_refuses_unknown_type_and_obs_names(skill_root):
    d = skill_root / "bow" / "v1"
    s = json.loads((d / "skill.json").read_text())
    s["type"] = "dance-ish"
    write_json(d / "skill.json", s)
    with pytest.raises(PackageError, match="type must be one of"):
        load_package(d)
    s["type"] = "episodic"
    s["obs_spec"] = {"names": ["gyro", "secret_sauce"], "size": 10}
    write_json(d / "skill.json", s)
    with pytest.raises(PackageError, match="unknown to the runtime"):
        load_package(d)


def test_refuses_obs_size_mismatch_for_onnx(skill_root):
    d = skill_root / "walk" / "v1"
    s = json.loads((d / "skill.json").read_text())
    s["policy"] = {"kind": "onnx"}
    s["obs_spec"] = {"names": WALK_OBS, "size": 100}      # blocks add up to 101
    write_json(d / "skill.json", s)
    shutil.copy(PROJECT / "sim" / "BEST_WALK_ONNX_2.onnx", d / "policy.onnx")
    with pytest.raises(PackageError, match="add up to 101"):
        load_package(d)


def test_real_walk_onnx_matches_layout(skill_root):
    d = skill_root / "walk" / "v1"
    s = json.loads((d / "skill.json").read_text())
    s["policy"] = {"kind": "onnx", "head_mode": "add", "max_vel_rad_s": 4.0}
    s["obs_spec"] = {"names": WALK_OBS, "size": 101}
    write_json(d / "skill.json", s)
    shutil.copy(PROJECT / "sim" / "BEST_WALK_ONNX_2.onnx", d / "policy.onnx")
    pkg = load_package(d)
    pol = make_policy(pkg)
    assert pol.kind == "onnx"


def test_refuses_bad_clip_joints(skill_root):
    d = skill_root / "bow" / "v1"
    c = json.loads((d / "clip.json").read_text())
    del c["frames"][0][1]["left_knee"]
    write_json(d / "clip.json", c)
    with pytest.raises(PackageError, match="joints wrong"):
        load_package(d)


def test_episodic_needs_timeout(skill_root):
    d = skill_root / "bow" / "v1"
    s = json.loads((d / "skill.json").read_text())
    s["timeout_s"] = None
    write_json(d / "skill.json", s)
    with pytest.raises(PackageError, match="timeout_s"):
        load_package(d)


def test_battery_rules():
    assert battery_passed({"passed": True})
    assert not battery_passed({"passed": False})
    assert battery_passed({"trials": 10, "passes": 9, "threshold": 0.9})
    assert not battery_passed({"trials": 10, "passes": 8, "threshold": 0.9})
    assert not battery_passed({"pending": True})
    assert not battery_passed("nope")


def test_clip_sampling_and_phase():
    a, b = HOME.copy(), HOME.copy()
    b[3] += 1.0
    clip = Clip.from_dict(pose_clip([a, b], 2.0))
    assert np.allclose(clip.sample(-1.0), a)
    assert np.allclose(clip.sample(1.0)[3], a[3] + 0.5)
    assert np.allclose(clip.sample(5.0), b)
    assert clip.phase(0.5) == pytest.approx(0.25) and clip.phase(9.0) == 1.0


def test_null_policy_plays_clip_and_reports_phase(skill_root):
    pkg = load_package(skill_root / "bow" / "v1")
    pol = make_policy(pkg)
    assert isinstance(pol, ClipPolicy)
    from graph_runtime.policies import HandOff, PolicyContext
    from graph_runtime.tests.conftest import FakeBackend
    be = FakeBackend()
    pol.reset(HandOff((np.zeros(14),) * 3, HOME.copy(), np.zeros(7), 0.0))
    ts = []
    for _ in range(60):
        t, a = pol.step(PolicyContext(be.read(), np.zeros(7), (np.zeros(14),) * 3, HOME.copy(), 0.0))
        ts.append(t)
    assert pol.done and pol.phase == 1.0
    assert max(t[5] for t in ts) == pytest.approx(0.4, abs=0.02)     # bow pose reached mid clip
    assert np.allclose(ts[-1], HOME, atol=1e-6)


TRACKING_OBS = ["gyro", "accelerometer", "joint_pos_rel", "joint_vel_scaled", "prev_action_1", "prev_action_2",
                "motor_targets_rel", "phase", "ref_joint_pos_next_rel", "ref_joint_vel_next_scaled"]
SMOKE_ONNX = sorted((PROJECT / "sim" / "train_pg" / "runs_skills" / "bow" / "smoke").glob("*.onnx")) if (
    PROJECT / "sim" / "train_pg" / "runs_skills" / "bow" / "smoke").exists() else []


def test_tracking_layout_blocks_match_recipe(skill_root):
    """The tracking recipe's observation: t = k * dt at step k, reference at t + dt, residual action."""
    from graph_runtime.policies import OBS_BLOCKS, HandOff, PolicyContext, OnnxPolicy
    from graph_runtime.onnx_minimal import write_zero_policy
    from graph_runtime.tests.conftest import FakeBackend
    d = skill_root / "bow" / "v1"
    s = json.loads((d / "skill.json").read_text())
    s["policy"] = {"kind": "onnx", "action_mode": "residual_on_reference", "lpf_hz": 8.0, "max_vel_rad_s": 4.0, "head_mode": "policy"}
    s["obs_spec"] = {"obs_size": 107, "layout": [{"name": n, "size": OBS_BLOCKS[n][0], "offset": 0} for n in TRACKING_OBS]}
    write_json(d / "skill.json", s)
    write_zero_policy(d / "policy.onnx", 107, 14)                     # zero residual = the clip, open loop
    pkg = load_package(d)
    assert pkg.obs_spec["names"] == TRACKING_OBS and int(pkg.obs_spec["size"]) == 107
    pol = OnnxPolicy(pkg)
    be = FakeBackend()
    a0 = np.full(14, 0.3)
    pol.reset(HandOff((a0, a0 * 0.5, a0 * 0.1), HOME.copy(), np.zeros(7), 0.0))
    ctx = PolicyContext(be.read(), np.zeros(7), (a0, a0 * 0.5, a0 * 0.1), HOME.copy(), 0.0)
    obs = pol.observe(ctx)
    assert obs.shape == (107,)
    assert np.allclose(obs[34:48], a0) and np.allclose(obs[48:62], a0 * 0.5)             # prev_action_1, prev_action_2
    assert np.allclose(obs[62:76], 0.0)                                                   # motor_targets_rel at home
    assert np.allclose(obs[76:79], [0.0, 1.0, 0.0])                                       # phase at k = 0
    assert np.allclose(obs[79:93], pkg.clip.sample(0.02) - HOME)                          # next frame
    assert np.allclose(obs[93:107], pkg.clip.velocity(0.02) * 0.05)
    t1, a = pol.step(ctx)                                                                 # k: 0 -> 1
    assert np.allclose(a, 0.0)
    alpha = (1 / 8) / (0.02 + 1 / 8)
    assert np.allclose(t1, alpha * HOME + (1 - alpha) * pkg.clip.sample(0.02), atol=1e-6)  # lpf toward clip(dt)
    obs2 = pol.observe(PolicyContext(be.read(), np.zeros(7), (a, a0, a0 * 0.5), t1, 0.02))
    assert obs2[76] == pytest.approx(0.02 / pkg.clip.duration)                           # phase t/T at k = 1
    for _ in range(60):
        t1, a = pol.step(PolicyContext(be.read(), np.zeros(7), (a, a0, a0 * 0.5), t1, 0.0))
    assert pol.done and np.allclose(t1, HOME, atol=0.02)


@pytest.mark.skipif(not SMOKE_ONNX, reason="Controls agent's smoke ONNX not present")
def test_controls_smoke_onnx_runs_through_the_engine(skill_root):
    """Plumbing only: the Controls agent's 107-input tracking ONNX loads, observes and acts in the engine."""
    from graph_runtime.engine import Engine
    from graph_runtime.graph import build_graph
    from graph_runtime.tests.conftest import FakeBackend, graph_cfg
    d = skill_root / "bow" / "v1"
    s = json.loads((d / "skill.json").read_text())
    s["policy"] = {"kind": "onnx", "action_mode": "residual_on_reference", "lpf_hz": 8.0, "max_vel_rad_s": 4.0, "head_mode": "policy"}
    s["obs_spec"] = {"names": TRACKING_OBS, "size": 107}
    write_json(d / "skill.json", s)
    shutil.copy(SMOKE_ONNX[-1], d / "policy.onnx")
    eng = Engine(build_graph(graph_cfg(skill_root)), FakeBackend())
    eng.run(0.2)
    assert eng.trigger("bow")
    infos = eng.run(2.0)
    assert any(i.active == "bow" for i in infos)
    assert all(np.all(np.isfinite(i.targets)) for i in infos)
    assert np.max(np.abs(np.diff([i.targets for i in infos], axis=0))) <= 4.0 * eng.dt + 1e-9     # rate limit held


def test_deploy_filter_matches_sim_test_formula():
    f = DeployFilter(8.0, 4.0, 0.02)
    alpha = (1 / 8) / (0.02 + 1 / 8)
    f.reset(np.zeros(14))
    out = f.apply(np.ones(14))
    assert np.allclose(out, min((1 - alpha), 4.0 * 0.02))
    g = DeployFilter(0.0, 4.0, 0.02)
    g.reset(np.zeros(14))
    assert np.allclose(g.apply(np.ones(14)), 0.08)
