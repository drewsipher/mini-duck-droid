"""MuJoCo integration: the real walk brain through the engine, idle 2 s then walk 2 s, no fall."""
import os
import shutil

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
mujoco = pytest.importorskip("mujoco")

from graph_runtime.backends.mujoco import MujocoBackend                    # noqa: E402
from graph_runtime.engine import Engine, State                             # noqa: E402
from graph_runtime.graph import build_graph                                # noqa: E402
from graph_runtime.packages import HOME, load_package                      # noqa: E402
from graph_runtime.tests.conftest import PROJECT, WALK_OBS, graph_cfg, pose_clip, skill, write_package   # noqa: E402

WALK_ONNX = PROJECT / "sim" / "BEST_WALK_ONNX_2.onnx"


@pytest.fixture
def walk_root(skill_root):
    """Replace the null walk with the deployed brain (battery marked as a test fixture)."""
    if not WALK_ONNX.exists():
        pytest.skip("BEST_WALK_ONNX_2.onnx missing")
    d = skill_root / "walk" / "v1"
    shutil.rmtree(d)
    write_package(d, skill("walk", "locomotion",
                           commands={"vx": [-0.15, 0.15], "vy": [-0.2, 0.2], "wz": [-1, 1], "neck_pitch": [-0.34, 1.1],
                                     "head_pitch": [-0.78, 0.78], "head_yaw": [-1.5, 1.5], "head_roll": [-0.5, 0.5]},
                           entry={"from": ["bow", "sit"], "phase": [0, 1], "window_s": 0.4}, exit={"to": ["bow", "sit", "nod"]},
                           obs_spec={"names": WALK_OBS, "size": 101},
                           policy={"kind": "onnx", "action_scale": 0.25, "head_mode": "add", "lpf_hz": 0.0,
                                   "max_vel_rad_s": 4.0, "gait_period_steps": 27}),
                  pose_clip([HOME], 0.0), {"skill": "walk", "passed": True, "note": "test fixture"}, policy_onnx=WALK_ONNX)
    return skill_root


def test_backend_matches_home_and_reads_sensors():
    be = MujocoBackend()
    f = be.read()
    assert np.allclose(f.q, HOME, atol=1e-3)
    assert f.height_m == pytest.approx(0.15, abs=0.01) and f.tilt_deg < 1.0
    assert f.acc[2] > 8.0                                              # gravity on the body z axis
    be.write_targets(HOME)
    assert be.read().t == pytest.approx(0.02)
    be.close()


def test_idle_then_walk_with_the_walk_brain_no_fall(walk_root):
    g = build_graph(graph_cfg(walk_root))
    be = MujocoBackend(seed=0)
    eng = Engine(g, be)
    infos = eng.run(2.0)
    assert all(i.label == "idle" for i in infos)
    p0 = be.base_pose()[0]
    eng.set_velocity(0.12)
    infos += eng.run(2.0)
    p1 = be.base_pose()[0]
    tilts = [i.tilt_deg for i in infos]
    heights = [i.height_m for i in infos]
    assert max(tilts) < 20.0, f"max tilt {max(tilts):.1f}"
    assert min(heights) > 0.12
    assert not eng.recovering and eng.state == State.RUNNING and eng.active == "walk"
    assert p1[0] - p0[0] > 0.05, "the duck should have moved forward"
    assert infos[-1].label == "walk" and all(np.isfinite(infos[-1].targets))
    be.close()


def test_sit_abort_from_standing_stays_upright(walk_root):
    g = build_graph(graph_cfg(walk_root))
    be = MujocoBackend(seed=1)
    eng = Engine(g, be)
    eng.run(1.0)
    eng.abort("test")
    eng.run(5.0, until=lambda i: i.state == State.RELAXED)
    assert eng.relaxed and be.stiffness == pytest.approx(0.3)
    after = eng.run(2.0)
    assert max(i.tilt_deg for i in after) < 20.0
    assert min(i.height_m for i in after) > 0.10
    be.close()
