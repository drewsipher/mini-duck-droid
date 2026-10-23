"""Policies: turn a SensorFrame plus the engine's command and action history into joint targets.

Two kinds live here:
  OnnxPolicy  an onnxruntime session whose observation is assembled from the package's
              obs_spec names (block registry below; unknown names are a hard error), then the
              deploy filters of sim_test.py / the duck runtime are applied in the same order:
              scale -> head command -> first-order low-pass -> per-tick rate limit.
  ClipPolicy  the null policy: plays the package clip open-loop as targets (recovery ramps and
              placeholder episodic clips). Same filters, no feedback.

Every policy reports its own phase in [0, 1) (gait clock for locomotion, clip progress for
episodic and recovery skills) so the engine can judge entry windows, and takes a HandOff at
reset: the previous three actions and the last commanded targets (transition rule 1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from graph_runtime.backends.base import SensorFrame
from graph_runtime.packages import DT, HEAD, HOME, NUM_JOINTS, PackageError, SkillPackage

COMMAND_SIZE = 7      # vx, vy, wz, neck_pitch, head_pitch, head_yaw, head_roll


class PolicyError(Exception):
    pass


@dataclass
class HandOff:
    """What the incoming policy is given at a hand-off."""
    actions: tuple                  # (a1, a2, a3): newest first, policy units
    targets: np.ndarray             # last joint targets actually commanded (rad)
    command: np.ndarray             # (7,) current operator command
    t: float                        # engine time


@dataclass
class PolicyContext:
    frame: SensorFrame
    command: np.ndarray             # (7,) clamped to the skill's trained ranges
    actions: tuple                  # (a1, a2, a3) newest first
    motor_targets: np.ndarray       # (14,) last targets commanded to the backend (post blend)
    t_skill: float                  # seconds since this skill became active
    dt: float = DT


# ---------------------------------------------------------------- observation blocks
# name -> (size, fn(ctx, policy) -> array). This is the deployed walk brain's layout, in the order
# sim_test.Sim.obs builds it: gyro, acc, cmd, q - home, qvel * 0.05, a1, a2, a3, targets, contacts, phase.
OBS_BLOCKS: dict[str, tuple[int, Callable]] = {
    "gyro": (3, lambda c, p: c.frame.gyro),
    "accelerometer": (3, lambda c, p: c.frame.acc + np.array([p.cfg.accel_bias, 0.0, 0.0])),
    "commands": (COMMAND_SIZE, lambda c, p: c.command),
    "joint_pos_rel": (NUM_JOINTS, lambda c, p: c.frame.q - HOME),
    "joint_vel_scaled": (NUM_JOINTS, lambda c, p: c.frame.dq * 0.05),
    "last_action": (NUM_JOINTS, lambda c, p: c.actions[0]),
    "last_last_action": (NUM_JOINTS, lambda c, p: c.actions[1]),
    "last_last_last_action": (NUM_JOINTS, lambda c, p: c.actions[2]),
    "motor_targets": (NUM_JOINTS, lambda c, p: c.motor_targets),
    "feet_contacts": (2, lambda c, p: c.frame.contacts),
    "gait_phase": (2, lambda c, p: p.clock.vec()),
}


def _need_gravity(c, p):
    if c.frame.gravity is None:
        raise PolicyError(f"{p.pkg.name}: obs block 'gravity' needs the backend to supply an orientation (frame.gravity)")
    return c.frame.gravity


def _clip_t(c, p):
    if not isinstance(p.clock, ClipClock):
        raise PolicyError(f"{p.pkg.name}: clip-tracking obs blocks need a clip clock (episodic/recovery/overlay package)")
    return min(p.clock.t, p.clock.duration), max(p.clock.duration, 1e-6)


def _clip_pose(p, t):
    return p.pkg.clip.sample(p.pkg.clip.times[0] + t)


# The tracking recipe's layout (sim/train_pg/recipe_v1.py OBS_LAYOUT, tracking.py _get_obs, 107 values):
# gyro 3, accelerometer 3, joint_pos_rel 14, joint_vel_scaled 14, prev_action_1 14, prev_action_2 14,
# motor_targets_rel 14, phase 3, ref_joint_pos_next_rel 14, ref_joint_vel_next_scaled 14.
# Clock convention (the trainer's): the observation at step k uses t = k * dt (k = 0 at entry) and the
# clip frame at t + dt; the action is a residual on that next frame (PolicyConfig.action_mode).
OBS_BLOCKS.update({
    "prev_action_1": (NUM_JOINTS, lambda c, p: c.actions[0]),
    "prev_action_2": (NUM_JOINTS, lambda c, p: c.actions[1]),
    "motor_targets_rel": (NUM_JOINTS, lambda c, p: c.motor_targets - HOME),
    "phase": (3, lambda c, p: (lambda t, T: np.array([t / T, np.cos(2 * np.pi * t / T), np.sin(2 * np.pi * t / T)]))(*_clip_t(c, p))),
    "ref_joint_pos_next_rel": (NUM_JOINTS, lambda c, p: _clip_pose(p, _clip_t(c, p)[0] + c.dt) - HOME),
    "ref_joint_vel_next_scaled": (NUM_JOINTS, lambda c, p: p.pkg.clip.velocity(p.pkg.clip.times[0] + _clip_t(c, p)[0] + c.dt) * 0.05),
})
# Clip-tracking blocks PROVISIONALLY guessed by tools/skillkit.PolicyRunner._block (the battery tool),
# kept so a package scored there runs here with the same numbers. Same clock convention as above.
OBS_BLOCKS.update({
    "gravity": (3, _need_gravity),
    "commands3": (3, lambda c, p: c.command[:3]),
    "clip_phase": (2, lambda c, p: (lambda t, T: np.array([np.cos(2 * np.pi * t / T), np.sin(2 * np.pi * t / T)]))(*_clip_t(c, p))),
    "clip_time": (1, lambda c, p: (lambda t, T: np.array([t / T]))(*_clip_t(c, p))),
    "ref_pose": (NUM_JOINTS, lambda c, p: _clip_pose(p, _clip_t(c, p)[0]) - HOME),
    "ref_pose_next": (NUM_JOINTS, lambda c, p: _clip_pose(p, _clip_t(c, p)[0] + c.dt) - HOME),
    "ref_vel": (NUM_JOINTS, lambda c, p: (_clip_pose(p, _clip_t(c, p)[0] + c.dt) - _clip_pose(p, _clip_t(c, p)[0])) / c.dt),
    "ref_contacts": (2, lambda c, p: np.ones(2)),
})


def check_obs_spec(pkg: SkillPackage) -> int:
    """Validate obs_spec against the block registry (and the ONNX input) without running anything."""
    names = pkg.obs_spec["names"]
    unknown = [n for n in names if n not in OBS_BLOCKS]
    if unknown:
        raise PackageError(f"{pkg.path}: obs_spec names unknown to the runtime: {unknown} "
                           f"(known: {sorted(OBS_BLOCKS)})")
    declared = pkg.obs_spec.get("blocks") or {}
    wrong = {n: (declared[n], OBS_BLOCKS[n][0]) for n in names if n in declared and int(declared[n]) != OBS_BLOCKS[n][0]}
    if wrong:
        raise PackageError(f"{pkg.path}: obs_spec.blocks sizes differ from the runtime's {wrong} (declared, runtime)")
    total = sum(OBS_BLOCKS[n][0] for n in names)
    size = int(pkg.obs_spec["size"])
    if pkg.policy.kind == "onnx":
        if total != size:
            raise PackageError(f"{pkg.path}: obs_spec.size {size} but the named blocks add up to {total}")
        shape = onnx_input_shape(pkg.policy_path, pkg.policy.input_name)
        dims = [d for d in shape if isinstance(d, int)]
        if dims and dims[-1] != size:
            raise PackageError(f"{pkg.path}: policy.onnx input {shape} does not take {size} values")
    elif size < 1:
        raise PackageError(f"{pkg.path}: obs_spec.size must be >= 1")
    return total


def onnx_input_shape(path, input_name="obs"):
    import onnxruntime as ort
    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as e:                      # noqa: BLE001 - any loader failure means an unusable package
        raise PackageError(f"{path}: not a loadable ONNX model ({e})")
    for i in sess.get_inputs():
        if i.name == input_name:
            return list(i.shape)
    raise PackageError(f"{path}: no input named {input_name!r} (has {[i.name for i in sess.get_inputs()]})")


# ---------------------------------------------------------------- filters and clocks
class DeployFilter:
    """sim_test.control's smoothing, in its order: first-order low-pass then per-tick rate limit,
    both against the previous commanded target. Reset seeds the state with the hand-off targets so
    the first output after a transition is continuous."""

    def __init__(self, lpf_hz: float, max_vel_rad_s: float, dt: float = DT):
        self.alpha = (1.0 / lpf_hz) / (dt + 1.0 / lpf_hz) if lpf_hz and lpf_hz > 0 else 0.0
        self.step = max_vel_rad_s * dt if max_vel_rad_s and max_vel_rad_s > 0 else None
        self.prev = HOME.copy()

    def reset(self, targets: np.ndarray):
        self.prev = np.asarray(targets, dtype=float).copy()

    def apply(self, targets: np.ndarray) -> np.ndarray:
        t = np.asarray(targets, dtype=float)
        if self.alpha:
            t = self.alpha * self.prev + (1 - self.alpha) * t
        if self.step is not None:
            t = np.clip(t, self.prev - self.step, self.prev + self.step)
        self.prev = t.copy()
        return t


class GaitClock:
    """The locomotion policy's free-running phase (sim_test: imitation_i over nb_steps)."""

    def __init__(self, period_steps: int, factor: float = 1.0):
        self.n = int(period_steps)
        self.factor = factor
        self.i = 0.0

    def advance(self):
        self.i = (self.i + self.factor) % self.n

    @property
    def phase(self) -> float:
        return self.i / self.n

    def vec(self) -> np.ndarray:
        ang = self.phase * 2 * np.pi
        return np.array([np.cos(ang), np.sin(ang)])


class ClipClock:
    """Progress through a clip; phase saturates at 1 when the clip is over."""

    def __init__(self, duration: float):
        self.duration = float(duration)
        self.t = 0.0

    def advance(self, dt: float):
        self.t += dt

    @property
    def phase(self) -> float:
        return 1.0 if self.duration <= 0 else min(self.t / self.duration, 1.0)

    @property
    def done(self) -> bool:
        return self.t >= self.duration

    def vec(self) -> np.ndarray:
        ang = self.phase * 2 * np.pi
        return np.array([np.cos(ang), np.sin(ang)])


# ---------------------------------------------------------------- policies
class BasePolicy:
    kind = "base"

    def __init__(self, pkg: SkillPackage):
        self.pkg = pkg
        self.cfg = pkg.policy
        self.filter = DeployFilter(self.cfg.lpf_hz, self.cfg.max_vel_rad_s)
        self.clock = GaitClock(self.cfg.gait_period_steps) if pkg.type == "locomotion" else ClipClock(pkg.clip.duration)
        self.handoff: HandOff | None = None

    def reset(self, handoff: HandOff):
        """Rule 1: the incoming policy starts from the outgoing one's last actions and targets."""
        self.handoff = handoff
        self.filter.reset(handoff.targets)
        if isinstance(self.clock, ClipClock):
            self.clock.t = 0.0

    @property
    def phase(self) -> float:
        return self.clock.phase

    @property
    def done(self) -> bool:
        return isinstance(self.clock, ClipClock) and self.clock.done

    def step(self, ctx: PolicyContext) -> tuple[np.ndarray, np.ndarray]:
        """-> (joint targets rad, action in policy units for the shared action history)."""
        raise NotImplementedError

    def _head(self, targets: np.ndarray, command: np.ndarray) -> np.ndarray:
        if self.cfg.head_mode == "add":                 # the duck runtime: policy head + head command
            targets[HEAD] = targets[HEAD] + command[3:7]
        elif self.cfg.head_mode == "from_command":      # brains trained with HEAD_FROM_CMD
            targets[HEAD] = HOME[HEAD] + command[3:7]
        return targets


class OnnxPolicy(BasePolicy):
    kind = "onnx"

    def __init__(self, pkg: SkillPackage):
        super().__init__(pkg)
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(pkg.policy_path), providers=["CPUExecutionProvider"])
        self.names = list(pkg.obs_spec["names"])
        self.input_name = self.cfg.input_name

    def observe(self, ctx: PolicyContext) -> np.ndarray:
        parts = []
        for n in self.names:
            size, fn = OBS_BLOCKS[n]
            v = np.asarray(fn(ctx, self), dtype=np.float32).reshape(-1)
            if v.shape[0] != size:
                raise PolicyError(f"{self.pkg.name}: obs block {n!r} gave {v.shape[0]} values, expected {size}")
            parts.append(v)
        return np.concatenate(parts).astype(np.float32)

    def step(self, ctx: PolicyContext):
        if isinstance(self.clock, GaitClock):
            self.clock.advance()                     # the walk brain: sim_test advances the clock, then observes
        obs = self.observe(ctx)                      # a clip clock observes at t = k * dt, then advances (tracking.py)
        a = self.sess.run(None, {self.input_name: obs[None, :]})[0][0].astype(float)
        if a.shape != (NUM_JOINTS,):
            raise PolicyError(f"{self.pkg.name}: policy output {a.shape}, expected ({NUM_JOINTS},)")
        if self.cfg.action_mode == "residual_on_reference":
            if not isinstance(self.clock, ClipClock):
                raise PolicyError(f"{self.pkg.name}: residual_on_reference needs a clip clock (not a locomotion package)")
            base = self.pkg.clip.sample(self.pkg.clip.times[0] + self.clock.t + ctx.dt)      # the coming frame
        else:
            base = HOME
        if isinstance(self.clock, ClipClock):
            self.clock.advance(ctx.dt)
        t = base + a * self.cfg.action_scale
        t = self._head(t, ctx.command)
        # the filters act relative to what the servos were last told (the engine's commanded targets,
        # blended or not): that is what the trainer's `motor_targets` state is, and what the policy observed
        self.filter.prev = np.asarray(ctx.motor_targets, dtype=float).copy()
        return self.filter.apply(t), a


class ClipPolicy(BasePolicy):
    """Null policy: the clip is the target. Open loop, clearly labelled as such in the package.
    For a locomotion package (tests, placeholders) the clip loops and never reports done."""
    kind = "null"

    def __init__(self, pkg: SkillPackage):
        super().__init__(pkg)
        self.loop = pkg.type == "locomotion"
        self.clock = ClipClock(pkg.clip.duration)

    @property
    def phase(self) -> float:
        if self.loop and self.clock.duration > 0:
            return (self.clock.t % self.clock.duration) / self.clock.duration
        return self.clock.phase

    @property
    def done(self) -> bool:
        return (not self.loop) and self.clock.done

    def step(self, ctx: PolicyContext):
        self.clock.advance(ctx.dt)
        t_clip = self.clock.t
        if self.loop and self.clock.duration > 0:
            t_clip = t_clip % self.clock.duration
        pose = self.pkg.clip.sample(self.pkg.clip.times[0] + t_clip)
        t = self._head(pose.copy(), ctx.command)
        a = (pose - HOME) / self.cfg.action_scale        # what an onnx policy would have output
        return self.filter.apply(t), a


def make_policy(pkg: SkillPackage) -> BasePolicy:
    if pkg.policy.kind == "onnx":
        return OnnxPolicy(pkg)
    if pkg.policy.kind == "null":
        return ClipPolicy(pkg)
    raise PolicyError(f"{pkg.name}: unknown policy kind {pkg.policy.kind!r}")
