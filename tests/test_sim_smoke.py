"""sim/sim_test.py Sim with the deployed walk brain (BEST_WALK_ONNX_2.onnx) on the backlash scene
at the real-servo torque (1.86 N.m): stands, walks forward, is deterministic, obeys the clamp.
These are smoke tests, not the battery: one seed, no randomisation, short horizons.
"""
import contextlib
import os
import sys

import numpy as np
import pytest

from conftest import SIM, WALK_ONNX, XML

TORQUE = 1.86
FALL_TILT_DEG = 45.0          # proposed G1 fall rule, see tests/GATES.md
MIN_BASE_Z = 0.10             # proposed secondary fall rule (standing is 0.15-0.165 m)


@contextlib.contextmanager
def in_sim_dir():
    """sim_test resolves Open_Duck_Playground/... relative to the cwd."""
    old = os.getcwd()
    os.chdir(SIM)
    try:
        yield
    finally:
        os.chdir(old)


def make_sim(torque=TORQUE, extra=None):
    for p in (WALK_ONNX, XML):
        if not os.path.exists(p):
            pytest.skip(f"missing {p}")
    if SIM not in sys.path:
        sys.path.insert(0, SIM)
    with in_sim_dir():
        from sim_test import Sim
        s = Sim(WALK_ONNX, XML, 0.0, extra={"head_add_cmd": True, **(extra or {})})
    s.m.actuator_forcerange[:, :] = [-torque, torque]
    return s


def rollout(s, cmd, seconds):
    """Return dict of max tilt, min base z, start/end positions, peak |actuator force|."""
    cmd = np.asarray(list(cmd) + [0] * (7 - len(cmd)), float)
    p0 = s.base_pose()[0].copy()
    tilt_max, zmin, fmax = 0.0, 9.0, 0.0
    for _ in range(int(seconds * 50)):
        s.control(cmd)
        p, yaw, tilt = s.base_pose()
        tilt_max = max(tilt_max, tilt)
        zmin = min(zmin, p[2])
        fmax = max(fmax, float(np.abs(s.d.actuator_force).max()))
    p1 = s.base_pose()[0].copy()
    return dict(tilt_max=tilt_max, zmin=zmin, p0=p0, p1=p1, fmax=fmax, nan=bool(np.isnan(s.d.qpos).any()))


def test_obs_matches_policy_input():
    s = make_sim()
    inp = s.policy.get_inputs()[0]
    o = s.obs(np.zeros(7))
    assert inp.shape[-1] == o.shape[0], f"policy expects {inp.shape}, sim builds {o.shape}"
    assert np.isfinite(o).all()


def test_stands_3s_at_real_torque():
    s = make_sim()
    r = rollout(s, [0, 0, 0], 3.0)
    assert not r["nan"]
    assert r["tilt_max"] < FALL_TILT_DEG, f"fell while standing: max tilt {r['tilt_max']:.1f} deg"
    assert r["zmin"] > MIN_BASE_Z, f"base dropped to {r['zmin']:.3f} m while standing"
    assert np.linalg.norm(r["p1"][:2] - r["p0"][:2]) < 0.10, "standing duck drifted more than 10 cm in 3 s"


def test_walks_forward_4s():
    s = make_sim()
    rollout(s, [0, 0, 0], 2.0)                               # settle like sim_test.run()
    r = rollout(s, [0.15, 0, 0], 4.0)
    dx = r["p1"][0] - r["p0"][0]
    assert not r["nan"]
    assert r["tilt_max"] < FALL_TILT_DEG, f"fell walking: max tilt {r['tilt_max']:.1f} deg"
    assert r["zmin"] > MIN_BASE_Z
    assert dx > 0.10, f"base x moved only {dx:+.3f} m in 4 s at 0.15 m/s command"
    assert abs(r["p1"][1] - r["p0"][1]) < 0.5 * max(dx, 1e-6), "walked more sideways than forward"


def test_torque_clamp_is_live():
    s = make_sim()
    rollout(s, [0, 0, 0], 1.0)
    r = rollout(s, [0.15, 0, 0], 2.0)
    assert r["fmax"] <= TORQUE + 1e-6, f"actuator force {r['fmax']:.3f} exceeds the {TORQUE} N.m clamp"
    assert r["fmax"] > 0.2 * TORQUE, "no actuator ever worked: clamp or control is not reaching the model"


def test_rollout_is_deterministic():
    """Same ONNX, same model, same commands: bit-identical states. This is the baseline the
    sim-to-sim gate's 'matches' threshold is measured against (tests/GATES.md G2)."""
    outs = []
    for _ in range(2):
        s = make_sim()
        rollout(s, [0, 0, 0], 1.0)
        rollout(s, [0.15, 0, 0.3], 1.5)
        outs.append((s.d.qpos.copy(), s.targets.copy()))
    assert np.array_equal(outs[0][0], outs[1][0])
    assert np.array_equal(outs[0][1], outs[1][1])


def test_zero_torque_falls_and_rule_catches_it():
    """The fall rule must fire on an actual collapse (actuators off), so a passing battery means
    something. Also documents that in this scene the trunk has no floor collision: the fallen
    base ends below z = 0 instead of resting on the mat."""
    s = make_sim(torque=0.0)
    r = rollout(s, [0, 0, 0], 3.0)
    assert r["tilt_max"] > FALL_TILT_DEG or r["zmin"] < MIN_BASE_Z, r
