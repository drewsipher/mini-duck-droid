"""Shared fixtures: temporary skill packages with null policies and a MuJoCo-free fake backend."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from graph_runtime.backends.base import Backend, SensorFrame          # noqa: E402
from graph_runtime.packages import HOME, JOINTS, write_package         # noqa: E402

WALK_OBS = ["gyro", "accelerometer", "commands", "joint_pos_rel", "joint_vel_scaled", "last_action",
            "last_last_action", "last_last_last_action", "motor_targets", "feet_contacts", "gait_phase"]
ENVELOPE = {"torque_nm": 1.86, "max_vel_rad_s": 4.0, "thermal_c": 55.0, "fall_tilt_deg": 45.0, "min_height_m": 0.10}


def pose_clip(poses, seconds):
    """Linear clip through `poses` (list of 14-vectors) spread over `seconds`."""
    n = len(poses)
    frames = []
    for k, p in enumerate(poses):
        t = 0.0 if n == 1 else seconds * k / (n - 1)
        frames.append([round(t, 4), {j: float(v) for j, v in zip(JOINTS, p)}])
    return {"frames": frames, "interp": "linear"}


def skill(name, stype, **over):
    d = {"name": name, "type": stype, "version": 1, "commands": {}, "entry": {"from": ["*"], "phase": [0, 1], "window_s": 0.2},
         "exit": {"to": ["*"]}, "timeout_s": None, "envelope": dict(ENVELOPE), "obs_spec": {"names": [], "size": 1},
         "policy": {"kind": "null"}, "status": "test"}
    d.update(over)
    return d


def passed(name):
    return {"skill": name, "trials": 10, "passes": 10, "threshold": 0.9, "passed": True}


@pytest.fixture
def skill_root(tmp_path):
    """walk (locomotion, null policy standing at home), bow (episodic 1 s clip), sit (recovery 0.5 s ramp),
    plus a get-up `up` recovery-return node so reset can be tested."""
    root = tmp_path / "skills"
    crouch = HOME.copy()
    crouch[[3, 12]] = 1.5
    bow_pose = HOME.copy()
    bow_pose[5] = 0.4
    write_package(root / "walk" / "v1", skill("walk", "locomotion",
                  commands={"vx": [-0.15, 0.15], "vy": [-0.2, 0.2], "wz": [-1, 1], "neck_pitch": [-0.34, 1.1],
                            "head_pitch": [-0.78, 0.78], "head_yaw": [-1.5, 1.5], "head_roll": [-0.5, 0.5]},
                  entry={"from": ["bow", "sit"], "phase": [0, 1], "window_s": 0.2}, exit={"to": ["bow", "sit", "nod"]}),
                  pose_clip([HOME], 0.0), passed("walk"))
    nod_pose = HOME.copy()
    nod_pose[6] = 0.5                                   # head_pitch
    write_package(root / "nod" / "v1", skill("nod", "overlay",
                  entry={"from": ["walk"], "phase": [0, 1], "window_s": 0.2}, exit={"to": ["walk"]},
                  overlay={"mask": ["neck_pitch", "head_pitch"], "window_s": 0.2}),
                  pose_clip([HOME, nod_pose, HOME], 1.0), passed("nod"))
    write_package(root / "bow" / "v1", skill("bow", "episodic", timeout_s=2.0,
                  entry={"from": ["walk"], "phase": [0.0, 1.0], "window_s": 0.2, "contact": "double", "speed": [0.0, 0.03]},
                  exit={"to": ["walk", "sit"], "phase": [0.95, 1.0]}),
                  pose_clip([HOME, bow_pose, HOME], 1.0), passed("bow"))
    write_package(root / "sit" / "v1", skill("sit", "recovery",
                  entry={"from": ["*"], "phase": [0, 1], "window_s": 0.1}, exit={"to": ["walk"]},
                  recovery={"relax_stiffness": 0.3, "hold_s": 0.2},
                  envelope={"torque_nm": 1.86, "max_vel_rad_s": 3.0, "thermal_c": 55.0, "fall_tilt_deg": 90.0}),
                  pose_clip([HOME, crouch], 0.5), passed("sit"))
    return root


def graph_cfg(root, with_return=False, extra_edges=()):
    edges = [
        {"from": "walk", "to": "bow", "phase": [0.0, 1.0], "contact": "double", "speed": [0.0, 0.03], "window_s": 0.2, "wait_timeout_s": 1.0},
        {"from": "bow", "to": "walk", "phase": [0.95, 1.0], "window_s": 0.2, "wait_timeout_s": 0.5},
        {"from": "walk", "to": "sit", "window_s": 0.1},
        {"from": "bow", "to": "sit", "window_s": 0.1},
    ]
    if with_return:
        edges.append({"from": "sit", "to": "walk", "window_s": 0.2})
    edges.append({"from": "walk", "to": "nod", "phase": [0, 1], "speed": [0.0, 0.1], "window_s": 0.2, "wait_timeout_s": 0.5})
    edges += list(extra_edges)
    return {"name": "test", "root": str(root), "nodes": {"walk": "walk/v1", "bow": "bow/v1", "sit": "sit/v1", "nod": "nod/v1"},
            "start": "walk", "idle": {"node": "walk", "command": [0, 0, 0]}, "recovery": {"node": "sit", "window_s": 0.1},
            "edges": edges}


class FakeBackend(Backend):
    """Scriptable sensors: the test sets contacts / tilt / height; targets and stiffness calls are recorded."""
    name = "fake"

    def __init__(self):
        self.t = 0.0
        self.contacts = np.array([1.0, 1.0])
        self.tilt = 0.0
        self.height = 0.15
        self.q = HOME.copy()
        self.targets = []
        self.stiffness = [1.0]
        self.shown = []

    def read(self):
        return SensorFrame(t=self.t, gyro=np.zeros(3), acc=np.array([0.0, 0.0, 9.81]), q=self.q.copy(),
                           dq=np.zeros(14), contacts=self.contacts.copy(), tilt_deg=self.tilt, height_m=self.height)

    def write_targets(self, targets):
        t = self.check_targets(targets)
        self.targets.append(t.copy())
        self.q = t.copy()                      # perfect servos
        self.t += self.dt

    def set_stiffness(self, scale):
        self.stiffness.append(float(scale))

    def show(self, channel, value):
        self.shown.append((channel, value))


@pytest.fixture
def fake_backend():
    return FakeBackend()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj))
