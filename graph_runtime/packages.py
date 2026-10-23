"""Skill packages: load and validate `skills/<name>/v<N>/` directories.

A package is the hand-off format between the four teams (framework section 2.5, plan section 4):

    skill.json          name, type, version, commands, entry, exit, neighbours, timeout_s,
                        envelope, obs_spec, status (+ runtime `policy` block, optional `show_cues`)
    clip.json           reference motion at 50 Hz: {"frames": [[t, {joint: rad}], ...], "interp", "base"?}
    policy.onnx         50 Hz policy (a real ONNX file even for null policies, see onnx_minimal.py)
    report/battery.json sim battery ratios; the graph refuses a package whose battery did not pass

The loader refuses a package with a missing part. It never guesses: an unknown skill type, an
observation layout that does not add up, or a clip with the wrong joints is an error.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
          "neck_pitch", "head_pitch", "head_yaw", "head_roll",
          "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
NUM_JOINTS = len(JOINTS)
HEAD = slice(5, 9)
LEGS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
# The standing pose: the MJCF `home` keyframe and HWI.init_pos on the robot are these numbers.
HOME = np.array([0.002, 0.053, -0.63, 1.368, -0.784, 0.0, 0.0, 0.0, 0.0,
                 -0.003, -0.065, 0.635, 1.379, -0.796])
CONTROL_HZ = 50
DT = 1.0 / CONTROL_HZ

REQUIRED_FILES = ["skill.json", "clip.json", "policy.onnx", "report/battery.json"]
SKILL_TYPES = ("locomotion", "overlay", "episodic", "recovery")
POLICY_KINDS = ("onnx", "null")
HEAD_MODES = ("add", "from_command", "policy")
ACTION_MODES = ("home", "residual_on_reference")


class PackageError(Exception):
    """A package that must not be loaded."""


@dataclass
class Clip:
    """A joint-space reference motion at policy rate, in JOINTS order."""
    times: np.ndarray            # (N,)
    poses: np.ndarray            # (N, 14) rad
    interp: str = "linear"
    base: Any = None             # optional root track, kept verbatim

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0]) if len(self.times) else 0.0

    def sample(self, t: float) -> np.ndarray:
        """Pose at time t (clamped to the clip's ends), linear between frames."""
        if len(self.times) == 1:
            return self.poses[0].copy()
        t = float(np.clip(t, self.times[0], self.times[-1]))
        k = int(np.searchsorted(self.times, t, side="right") - 1)
        k = min(max(k, 0), len(self.times) - 2)
        t0, t1 = self.times[k], self.times[k + 1]
        a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        return (1 - a) * self.poses[k] + a * self.poses[k + 1]

    def phase(self, t: float) -> float:
        """Progress through the clip in [0, 1]."""
        return 1.0 if self.duration <= 0 else float(np.clip((t - self.times[0]) / self.duration, 0.0, 1.0))

    def velocity(self, t: float) -> np.ndarray:
        """Joint velocity at t: np.gradient over the frames (central differences, one-sided at the
        ends), the tracking recipe's definition (playground/common/clip_tracking_reference.py)."""
        if len(self.times) < 2 or self.duration <= 0:
            return np.zeros(self.poses.shape[1])
        if not hasattr(self, "_vel"):
            self._vel = np.gradient(self.poses, self.times, axis=0)
        t = float(np.clip(t, self.times[0], self.times[-1]))
        k = int(np.searchsorted(self.times, t, side="right") - 1)
        k = min(max(k, 0), len(self.times) - 2)
        t0, t1 = self.times[k], self.times[k + 1]
        a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        return (1 - a) * self._vel[k] + a * self._vel[k + 1]

    @classmethod
    def from_dict(cls, data: dict, where: str = "clip") -> "Clip":
        if isinstance(data, list):                    # bare puppet format [[t, {joint: rad}], ...]
            data = {"frames": data, "interp": "linear"}
        frames = data.get("frames")
        if not isinstance(frames, list) or not frames:
            raise PackageError(f"{where}: no frames")
        times, poses = [], []
        for i, fr in enumerate(frames):
            if not (isinstance(fr, (list, tuple)) and len(fr) == 2 and isinstance(fr[1], dict)):
                raise PackageError(f"{where}: frame {i} is not [t, {{joint: rad}}]")
            t, pose = fr
            missing = [j for j in JOINTS if j not in pose]
            extra = [j for j in pose if j not in JOINTS]
            if missing or extra:
                raise PackageError(f"{where}: frame {i} joints wrong (missing {missing}, unknown {extra})")
            times.append(float(t))
            poses.append([float(pose[j]) for j in JOINTS])
        times = np.asarray(times)
        if np.any(np.diff(times) < 0):
            raise PackageError(f"{where}: frame times must not decrease")
        return cls(times=times, poses=np.asarray(poses), interp=data.get("interp", "linear"), base=data.get("base"))

    def to_dict(self) -> dict:
        d = {"frames": [[round(float(t), 4), {j: round(float(v), 5) for j, v in zip(JOINTS, p)}]
                        for t, p in zip(self.times, self.poses)], "interp": self.interp}
        if self.base is not None:
            d["base"] = self.base
        return d


@dataclass
class PolicyConfig:
    """How the runtime turns the package's policy output into joint targets. Defaults are the
    deployed walk brain's settings; the filters are the deploy filters of sim_test.py."""
    kind: str = "onnx"
    action_scale: float = 0.25
    head_mode: str = "add"           # add: targets[5:9] += head command (duck runtime); from_command: head = home + command; policy: untouched
    lpf_hz: float = 0.0              # first-order low-pass on targets (0 = off); runtime LowPassActionFilter formula
    max_vel_rad_s: float = 4.0       # per-tick rate limit on targets (0 = off)
    accel_bias: float = 0.0          # added to accelerometer x (sim_test default was 1.3; eval/render use 0)
    gait_period_steps: int = 27      # locomotion phase clock (polynomial reference: 0.54 s at 50 Hz)
    input_name: str = "obs"
    open_loop: bool = False          # null policy playing its clip as targets, no feedback
    # home: target = HOME + scale * a (the walk brain). residual_on_reference: target = clip(t + dt) + scale * a
    # (the tracking recipe, sim/train_pg/recipe_v1.py: action 0 plays the clip open-loop).
    action_mode: str = "home"

    @classmethod
    def from_dict(cls, d: dict | None, where: str) -> "PolicyConfig":
        d = dict(d or {})
        cfg = cls()
        for k, v in d.items():
            if not hasattr(cfg, k):
                raise PackageError(f"{where}: unknown policy option {k!r}")
            setattr(cfg, k, v)
        if cfg.kind not in POLICY_KINDS:
            raise PackageError(f"{where}: policy.kind must be one of {POLICY_KINDS}, got {cfg.kind!r}")
        if cfg.head_mode not in HEAD_MODES:
            raise PackageError(f"{where}: policy.head_mode must be one of {HEAD_MODES}, got {cfg.head_mode!r}")
        if cfg.action_mode not in ACTION_MODES:
            raise PackageError(f"{where}: policy.action_mode must be one of {ACTION_MODES}, got {cfg.action_mode!r}")
        return cfg


@dataclass
class SkillPackage:
    path: Path
    name: str
    type: str
    version: int
    commands: dict                 # {input: [min, max]}
    entry: dict                    # {from: [...], phase: [lo, hi], window_s, contact?, speed?}
    exit: dict                     # {to: [...], phase?: [lo, hi]}
    neighbours: list
    timeout_s: float | None
    envelope: dict                 # torque_nm, max_vel_rad_s, thermal_c, fall_tilt_deg, min_height_m?
    obs_spec: dict                 # {names: [...], size}
    status: str
    policy: PolicyConfig
    clip: Clip
    battery: dict
    show_cues: list = field(default_factory=list)
    recovery: dict = field(default_factory=dict)   # type == recovery: {relax_stiffness, hold_s}
    overlay: dict = field(default_factory=dict)    # type == overlay: {mask: [joints], window_s}
    raw: dict = field(default_factory=dict)

    @property
    def policy_path(self) -> Path:
        return self.path / "policy.onnx"

    @property
    def battery_passed(self) -> bool:
        return battery_passed(self.battery)

    def allows_entry_from(self, other: str) -> bool:
        allowed = self.entry.get("from", [])
        return "*" in allowed or other in allowed

    def allows_exit_to(self, other: str) -> bool:
        allowed = self.exit.get("to", [])
        return "*" in allowed or other in allowed

    def __repr__(self):
        return f"SkillPackage({self.name} v{self.version} {self.type} @ {self.path})"


def battery_passed(battery: dict) -> bool:
    """A battery passes when it says so, or when its ratio clears its threshold."""
    if not isinstance(battery, dict):
        return False
    if "passed" in battery:
        return bool(battery["passed"])
    trials, passes = battery.get("trials"), battery.get("passes")
    if isinstance(trials, (int, float)) and isinstance(passes, (int, float)) and trials > 0:
        return passes / trials >= float(battery.get("threshold", 1.0))
    return False


def _need(d: dict, key: str, where: str, types=None):
    if key not in d:
        raise PackageError(f"{where}: missing field {key!r}")
    if types is not None and not isinstance(d[key], types):
        raise PackageError(f"{where}: field {key!r} has the wrong type ({type(d[key]).__name__})")
    return d[key]


def _window(v, where):
    if not (isinstance(v, (list, tuple)) and len(v) == 2) or not (0.0 <= float(v[0]) <= 1.0 and 0.0 <= float(v[1]) <= 1.0):
        raise PackageError(f"{where}: must be [lo, hi] inside [0, 1], got {v!r}")
    return [float(v[0]), float(v[1])]


def load_package(path: str | os.PathLike) -> SkillPackage:
    """Load one package directory, refusing anything incomplete or inconsistent."""
    path = Path(path)
    if not path.is_dir():
        raise PackageError(f"{path}: not a directory")
    missing = [f for f in REQUIRED_FILES if not (path / f).is_file()]
    if missing:
        raise PackageError(f"{path}: package incomplete, missing {missing}")
    empty = [f for f in REQUIRED_FILES if (path / f).stat().st_size == 0]
    if empty:
        raise PackageError(f"{path}: empty files {empty}")
    where = f"{path}/skill.json"
    try:
        skill = json.loads((path / "skill.json").read_text())
    except json.JSONDecodeError as e:
        raise PackageError(f"{where}: invalid JSON ({e})")
    name = _need(skill, "name", where, str)
    stype = _need(skill, "type", where, str)
    if stype not in SKILL_TYPES:
        raise PackageError(f"{where}: type must be one of {SKILL_TYPES}, got {stype!r}")
    version = int(_need(skill, "version", where, (int, float)))
    commands = _need(skill, "commands", where, dict)
    for k, rng in commands.items():
        if not (isinstance(rng, (list, tuple)) and len(rng) == 2 and float(rng[0]) <= float(rng[1])):
            raise PackageError(f"{where}: commands[{k!r}] must be [min, max]")
    entry = dict(_need(skill, "entry", where, dict))
    _need(entry, "from", f"{where}:entry", list)
    entry["phase"] = _window(entry.get("phase", [0.0, 1.0]), f"{where}:entry.phase")
    entry["window_s"] = float(entry.get("window_s", 0.3))
    if entry["window_s"] < 0:
        raise PackageError(f"{where}: entry.window_s must be >= 0")
    exit_ = dict(_need(skill, "exit", where, dict))
    _need(exit_, "to", f"{where}:exit", list)
    if "phase" in exit_:
        exit_["phase"] = _window(exit_["phase"], f"{where}:exit.phase")
    neighbours = list(skill.get("neighbours", sorted(set(entry["from"]) | set(exit_["to"]))))
    timeout_s = skill.get("timeout_s")
    if timeout_s is not None:
        timeout_s = float(timeout_s)
        if timeout_s <= 0:
            raise PackageError(f"{where}: timeout_s must be > 0 or null")
    envelope = dict(_need(skill, "envelope", where, dict))
    for k in ("torque_nm", "max_vel_rad_s", "thermal_c", "fall_tilt_deg"):
        if not isinstance(envelope.get(k), (int, float)):
            raise PackageError(f"{where}: envelope.{k} missing or not a number")
    obs_spec = dict(_need(skill, "obs_spec", where, dict))
    if "names" not in obs_spec and isinstance(obs_spec.get("layout"), list):      # the training runner's obs_spec.json shape
        obs_spec["names"] = [str(b["name"]) for b in obs_spec["layout"]]
        obs_spec.setdefault("blocks", {str(b["name"]): int(b["size"]) for b in obs_spec["layout"]})
    if "size" not in obs_spec and "obs_size" in obs_spec:
        obs_spec["size"] = obs_spec["obs_size"]
    _need(obs_spec, "names", f"{where}:obs_spec", list)
    _need(obs_spec, "size", f"{where}:obs_spec", (int, float))
    status = str(skill.get("status", ""))
    policy = PolicyConfig.from_dict(skill.get("policy"), where)
    if stype == "episodic" and timeout_s is None:
        raise PackageError(f"{where}: an episodic skill needs timeout_s")
    try:
        clip = Clip.from_dict(json.loads((path / "clip.json").read_text()), f"{path}/clip.json")
    except json.JSONDecodeError as e:
        raise PackageError(f"{path}/clip.json: invalid JSON ({e})")
    try:
        battery = json.loads((path / "report" / "battery.json").read_text())
    except json.JSONDecodeError as e:
        raise PackageError(f"{path}/report/battery.json: invalid JSON ({e})")
    if not isinstance(battery, dict):
        raise PackageError(f"{path}/report/battery.json: must be an object")
    recovery = dict(skill.get("recovery", {}))
    if stype == "recovery":
        recovery.setdefault("relax_stiffness", 0.3)
        recovery.setdefault("hold_s", 0.5)
    overlay = dict(skill.get("overlay", {}))
    if stype == "overlay":
        mask = overlay.get("mask")
        if not isinstance(mask, list) or not mask:
            raise PackageError(f"{where}: an overlay needs overlay.mask (the joints its clip drives)")
        bad = [j for j in mask if j not in JOINTS]
        if bad:
            raise PackageError(f"{where}: overlay.mask names unknown joints {bad}")
        overlay["window_s"] = float(overlay.get("window_s", entry["window_s"]))
    pkg = SkillPackage(path=path, name=name, type=stype, version=version, commands=commands, entry=entry,
                       exit=exit_, neighbours=neighbours, timeout_s=timeout_s, envelope=envelope,
                       obs_spec=obs_spec, status=status, policy=policy, clip=clip, battery=battery,
                       show_cues=list(skill.get("show_cues", [])), recovery=recovery, overlay=overlay, raw=skill)
    # The observation layout must add up before anything is run (policies.py knows the block sizes).
    from graph_runtime.policies import check_obs_spec
    check_obs_spec(pkg)
    return pkg


def write_package(path: str | os.PathLike, skill: dict, clip: Clip | dict, battery: dict,
                  policy_onnx: str | os.PathLike | None = None) -> Path:
    """Write a package directory. With policy_onnx=None a null policy file is written."""
    from graph_runtime.onnx_minimal import write_zero_policy
    path = Path(path)
    (path / "report").mkdir(parents=True, exist_ok=True)
    (path / "skill.json").write_text(json.dumps(skill, indent=1))
    clip_d = clip.to_dict() if isinstance(clip, Clip) else clip
    (path / "clip.json").write_text(json.dumps(clip_d))
    (path / "report" / "battery.json").write_text(json.dumps(battery, indent=1))
    if policy_onnx is None:
        size = int(skill.get("obs_spec", {}).get("size", 1))
        write_zero_policy(path / "policy.onnx", size, NUM_JOINTS)
    else:
        (path / "policy.onnx").write_bytes(Path(policy_onnx).read_bytes())
    return path
