"""The 50 Hz motion graph engine: one active skill, operator commands, transitions, abort.

Transition rules (framework section 2.4) as implemented here:

  1. Previous two actions. The engine keeps one shared action history (a1, a2, a3, newest first,
     in policy units) and the last commanded targets. At a hand-off the incoming policy is reset
     with exactly those (HandOff), and its deploy filters are seeded with the last targets, so its
     first observation and its first filtered output are continuous with what the robot just did.
  2. Authored windows. A requested edge waits (state WAITING) until the OUTGOING skill's phase is in
     the edge's window, the feet match the edge's contact mode and the commanded planar speed is in
     the edge's speed range; while waiting the velocity command is pulled inside that range. A
     window that never opens cancels the request (or aborts, for an episodic skill's return edge).
  3. Blend the reference, never the policy outputs. WHAT IS BLENDED: the commanded joint targets,
     which are the reference the PD loop tracks (for a clip skill they ARE the reference clip; for
     the walk brain they are its filtered motor targets, which it also observes as `motor_targets`).
     At hand-off the outgoing skill's last targets are frozen as `hold`; for window_s the command is
     (1 - s) * hold + s * incoming(t) with s = smoothstep(t / window_s). Only the incoming policy runs
     during the blend: the outgoing policy is not stepped, nothing is averaged between two live
     controllers, and when s reaches 1 the hold is dropped for good. WHY: two closed loops fighting
     over the same joints is not a cross-fade, and a policy stepped on a state it no longer controls
     produces garbage; freezing the last reference and ramping the new one in is what an animator's
     cross-fade does and is what the tracking policies are trained to start from (blended starts).
  4. (training: neighbours' end states) not the runtime's job.
  5. Only listed edges exist: graph.py refuses anything else; the engine can only follow Edge objects.

Safety every tick: tilt above the active skill's envelope (fall_tilt_deg) or trunk height below
min_height_m (when the backend knows it) enters the recovery skill at once, as does an operator
abort or an episodic timeout. Recovery = sit: the recovery package's clip ramps to a crouch, is
held hold_s, then the servos are softened to relax_stiffness (state RELAXED, terminal until reset).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from graph_runtime.backends.base import Backend, SensorFrame
from graph_runtime.graph import Edge, MotionGraph
from graph_runtime.packages import DT, HOME, JOINTS, NUM_JOINTS
from graph_runtime.policies import BasePolicy, DeployFilter, HandOff, PolicyContext, make_policy

COMMAND_NAMES = ["vx", "vy", "wz", "neck_pitch", "head_pitch", "head_yaw", "head_roll"]
JOINT_INDEX = {j: i for i, j in enumerate(JOINTS)}
COMMAND_SLEW = np.array([0.3, 0.3, 2.0, 1.5, 1.5, 1.5, 1.5])     # per second: m/s^2, rad/s^2, rad/s
IDLE_EPS = 1e-3


class State(str, Enum):
    RUNNING = "running"
    WAITING = "waiting"
    BLENDING = "blending"
    RELAXED = "relaxed"


def smoothstep(x: float) -> float:
    """3x^2 - 2x^3 on [0, 1]: starts and ends with zero slope, monotone in between."""
    x = min(max(float(x), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


@dataclass
class Event:
    t: float
    kind: str
    text: str

    def __str__(self):
        return f"{self.t:7.2f}s  {self.kind:10s} {self.text}"


@dataclass
class TickInfo:
    t: float
    tick: int
    state: State
    active: str
    label: str                   # skill name, or "idle" for locomotion at zero command
    status: str                  # one line for overlays
    blend: float                 # 0..1 weight of the incoming skill (1 when not blending)
    phase: float
    tilt_deg: float
    height_m: float | None
    speed: float
    command: np.ndarray
    targets: np.ndarray
    recovering: bool
    recovery_reason: str | None
    events: list = field(default_factory=list)
    overlay: str | None = None   # overlay skill layered on the active locomotion skill, if any
    overlay_w: float = 0.0       # its ramp weight


class ShowScheduler:
    """Show functions (head, antennas, eyes, sound) are a side channel scheduled alongside skills.
    Stub: a package may list `show_cues` [{"t": 0.5, "channel": "antenna_left", "value": 1.0}, ...];
    they fire at t seconds into the skill through backend.show(channel, value). No training involved."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.cues: list = []
        self.fired: list = []

    def start(self, cues: list):
        self.cues = sorted((dict(c) for c in cues), key=lambda c: float(c.get("t", 0.0)))
        self.fired = []

    def tick(self, t_skill: float) -> list:
        out = []
        while self.cues and float(self.cues[0].get("t", 0.0)) <= t_skill:
            c = self.cues.pop(0)
            self.backend.show(str(c.get("channel", "")), c.get("value"))
            self.fired.append(c)
            out.append(c)
        return out

    def cancel(self):
        self.cues = []


class Engine:
    def __init__(self, graph: MotionGraph, backend: Backend, policy_factory: Callable = make_policy,
                 safety: bool = True, log: Callable | None = None):
        self.graph = graph
        self.backend = backend
        self.dt = float(getattr(backend, "dt", DT))
        self.safety = safety
        self.log = log
        self.policies: dict[str, BasePolicy] = {n: policy_factory(p) for n, p in graph.nodes.items()}
        self.show = ShowScheduler(backend)
        self.t = 0.0
        self.tick_i = 0
        self.events: list[Event] = []
        self.cmd_target = np.zeros(7)        # what the operator asked for
        self.cmd_base = np.zeros(7)          # slewed and clamped operator command
        self.cmd = np.zeros(7)               # what the policies see: cmd_base plus overlay deltas
        self.actions = (np.zeros(NUM_JOINTS), np.zeros(NUM_JOINTS), np.zeros(NUM_JOINTS))
        self.targets = HOME.copy()
        self.frame: SensorFrame | None = None
        self.tilt_filt = 0.0
        self.state = State.RUNNING
        self.active = graph.start
        self.t_skill = 0.0
        self.pending: Edge | None = None
        self.pending_auto = False
        self.wait_t = 0.0
        self.hold: np.ndarray | None = None
        self.blend_edge: Edge | None = None
        self.blend_t = 0.0
        self.blend_w = 1.0
        self.recovering = False
        self.recovery_reason: str | None = None
        self.recovery_done_t: float | None = None
        self.relaxed = False
        self.overlay: dict | None = None     # {"node", "pol", "edge", "t", "w"} while an overlay plays
        self.overlay_req: list | None = None  # [edge, waited_s] while an overlay waits for its window
        self._unmet_text = ""
        self.segments: list = []             # (skill, t_enter)
        self.limiter = DeployFilter(0.0, float(self.pkg.envelope.get("max_vel_rad_s", 0.0)), self.dt)
        self.limiter.reset(self.targets)
        self.policies[self.active].reset(HandOff(self.actions, self.targets.copy(), self.cmd.copy(), 0.0))
        self.show.start(self.pkg.show_cues)
        self.segments.append((self.active, 0.0))
        self._event("start", f"{self.active} (graph {graph.name})")

    # ------------------------------------------------------------ operator API
    @property
    def pkg(self):
        return self.graph.nodes[self.active]

    def set_velocity(self, vx: float, vy: float = 0.0, wz: float = 0.0):
        self.cmd_target[:3] = [vx, vy, wz]

    def set_head(self, neck_pitch: float = 0.0, head_pitch: float = 0.0, head_yaw: float = 0.0, head_roll: float = 0.0):
        self.cmd_target[3:7] = [neck_pitch, head_pitch, head_yaw, head_roll]

    def trigger(self, name: str) -> bool:
        """Request a transition along the edge active -> name. Returns False if there is no such edge
        or the engine is busy (waiting, blending, recovering)."""
        if self.recovering or self.relaxed:
            return self._refuse(f"trigger {name}: recovering")
        if name not in self.graph.nodes:
            return self._refuse(f"trigger {name}: no such node")
        if self.graph.is_overlay(name):
            if self.overlay is not None or self.overlay_req is not None:
                return self._refuse(f"trigger {name}: an overlay is already playing or waiting")
            if self.state != State.RUNNING:
                return self._refuse(f"trigger {name}: busy ({self.state.value})")
            edge = self.graph.edge(self.active, name)
            if edge is None:
                return self._refuse(f"trigger {name}: overlay not layered on {self.active}")
            self.overlay_req = [edge, 0.0]
            self._event("request", f"overlay {name} on {self.active}")
            return True
        if self.state != State.RUNNING:
            return self._refuse(f"trigger {name}: busy ({self.state.value})")
        edge = self.graph.edge(self.active, name)
        if edge is None:
            return self._refuse(f"trigger {name}: no edge {self.active} -> {name}")
        if edge.kind == "recovery":
            return self._refuse(f"trigger {name}: recovery is entered with abort(), not a trigger")
        self._request(edge, auto=False)
        return True

    def abort(self, reason: str = "operator abort") -> bool:
        return self._enter_recovery(reason)

    def reset(self) -> bool:
        """After a recovery: stiffen again and take the recovery node's return edge, if the graph has one."""
        if not self.relaxed:
            return self._refuse("reset: not relaxed")
        edge = self.graph.return_edge(self.active)
        if edge is None:
            return self._refuse(f"reset: no return edge from {self.active} (needs a get-up skill); staying relaxed")
        self.backend.set_stiffness(1.0)
        self.relaxed = False
        self.recovering = False
        self.recovery_reason = None
        self.recovery_done_t = None
        self.state = State.RUNNING
        self._event("reset", f"stiffness 1.0, taking {edge.src} -> {edge.dst}")
        self._request(edge, auto=True)
        return True

    # ------------------------------------------------------------ the tick
    def tick(self) -> TickInfo:
        n0 = len(self.events)
        frame = self.backend.read()
        self.frame = frame
        tilt = self._tilt(frame)
        pkg = self.pkg
        if self.safety and not self.recovering:
            env = pkg.envelope
            if tilt > float(env["fall_tilt_deg"]):
                self._enter_recovery(f"tilt {tilt:.0f} deg > {env['fall_tilt_deg']} ({pkg.name} envelope)")
            elif frame.height_m is not None and env.get("min_height_m") is not None and frame.height_m < float(env["min_height_m"]):
                self._enter_recovery(f"height {frame.height_m:.3f} m < {env['min_height_m']} ({pkg.name} envelope)")
        self._shape_command()
        speed = float(np.hypot(self.cmd[0], self.cmd[1]))
        if self.state == State.WAITING:
            self._check_window(frame, speed)
        self._overlay_tick(frame, speed)
        pol = self.policies[self.active]
        ctx = PolicyContext(frame, self.cmd.copy(), self.actions, self.targets.copy(), self.t_skill, self.dt)
        if self.relaxed:
            targets, action = self.targets.copy(), self.actions[0]
        else:
            targets, action = pol.step(ctx)
        w = 1.0
        if self.state == State.BLENDING:
            self.blend_t += self.dt
            w = smoothstep(self.blend_t / self.blend_edge.window_s) if self.blend_edge.window_s > 0 else 1.0
            targets = (1.0 - w) * self.hold + w * targets
            if self.blend_t >= self.blend_edge.window_s - 1e-9:
                self.state = State.RUNNING
                self.hold = None
                w = 1.0
                self._event("handoff", f"{self.active} in charge (blend {self.blend_edge.window_s:.2f}s done)")
        self.blend_w = w
        targets = self.limiter.apply(targets)
        self.backend.write_targets(targets)
        self.targets = targets
        self.actions = (np.asarray(action, dtype=float).copy(), self.actions[0], self.actions[1])
        self.t += self.dt
        self.t_skill += self.dt
        self.tick_i += 1
        self._after_step(pol)
        for c in self.show.tick(self.t_skill):
            self._event("show", f"{c.get('channel')} = {c.get('value')}")
        return TickInfo(self.t, self.tick_i, self.state, self.active, self.label, self.status, self.blend_w,
                        pol.phase, tilt, frame.height_m, speed, self.cmd.copy(), self.targets.copy(),
                        self.recovering, self.recovery_reason, [str(e) for e in self.events[n0:]],
                        self.overlay["node"] if self.overlay else None, self.overlay["w"] if self.overlay else 0.0)

    def run(self, seconds: float, until: Callable | None = None) -> list:
        """Tick for `seconds` (or until `until(info)` is true). Returns the TickInfos."""
        out = []
        for _ in range(int(round(seconds / self.dt))):
            info = self.tick()
            out.append(info)
            if until is not None and until(info):
                break
        return out

    # ------------------------------------------------------------ labels
    @property
    def label(self) -> str:
        base = self.active
        if self.active == self.graph.idle_node and np.all(np.abs(self.cmd[:3]) < IDLE_EPS) and not self.recovering:
            base = "idle"
        if self.overlay is not None:
            base += f" + {self.overlay['node']}"
        return base

    @property
    def status(self) -> str:
        if self.state == State.RELAXED:
            return f"relaxed in {self.active}: {self.recovery_reason}"
        if self.recovering:
            base = f"RECOVERY ({self.recovery_reason})"
        else:
            base = "running"
        if self.state == State.WAITING and self.pending is not None:
            return f"waiting for {self.pending.dst}: {self._unmet_text}"
        if self.state == State.BLENDING and self.blend_edge is not None:
            return f"{base}: blending {self.blend_edge.src} -> {self.blend_edge.dst} {100 * self.blend_w:3.0f}%"
        if self.overlay_req is not None:
            return f"{base}, overlay {self.overlay_req[0].dst} waiting: {self._unmet_text}"
        if self.overlay is not None:
            return f"{base}, overlay {self.overlay['node']} {100 * self.overlay['w']:3.0f}%"
        return base

    # ------------------------------------------------------------ internals
    def _event(self, kind: str, text: str):
        e = Event(self.t, kind, text)
        self.events.append(e)
        if self.log:
            self.log(str(e))

    def _refuse(self, text: str) -> bool:
        self._event("refused", text)
        return False

    def _tilt(self, frame: SensorFrame) -> float:
        if frame.tilt_deg is not None:
            return float(frame.tilt_deg)
        raw = SensorFrame.tilt_from_accel(frame.acc)
        self.tilt_filt = 0.8 * self.tilt_filt + 0.2 * raw
        return self.tilt_filt

    def _shape_command(self):
        """Slew the operator command, clamp it to the active skill's trained ranges, and while waiting
        for an edge with a speed range pull the planar speed inside it. Recovery forces zero."""
        target = self.cmd_target.copy()
        if self.recovering or self.relaxed:
            target[:] = 0.0
        ranges = self.pkg.commands
        for i, name in enumerate(COMMAND_NAMES):
            rng = ranges.get(name)
            if rng is None:
                target[i] = 0.0
            else:
                target[i] = float(np.clip(target[i], rng[0], rng[1]))
        if self.state == State.WAITING and self.pending is not None and self.pending.speed is not None:
            hi = self.pending.speed[1]
            sp = float(np.hypot(target[0], target[1]))
            if sp > hi:
                target[:2] *= (hi / sp) if sp > 0 else 0.0
        step = COMMAND_SLEW * self.dt
        self.cmd_base = np.clip(target, self.cmd_base - step, self.cmd_base + step)
        self.cmd = self.cmd_base.copy()            # overlays add their deltas on top, fresh every tick

    def _request(self, edge: Edge, auto: bool):
        self.pending = edge
        self.pending_auto = auto
        self.wait_t = 0.0
        self.state = State.WAITING
        self._unmet_text = ""
        self._event("request", f"{edge.src} -> {edge.dst}" + (" (auto)" if auto else ""))

    def _check_window(self, frame: SensorFrame, speed: float):
        edge = self.pending
        self.wait_t += self.dt
        unmet = edge.unmet(self.policies[self.active].phase, frame.contacts, speed)
        if not unmet:
            self._event("window", f"{edge.src} -> {edge.dst} open after {self.wait_t:.2f}s "
                                  f"(phase {self.policies[self.active].phase:.2f}, contacts {frame.contacts.tolist()}, speed {speed:.2f})")
            self._start_blend(edge)
        else:
            self._unmet_text = ", ".join(unmet)
            if self.wait_t > edge.wait_timeout_s:
                self.pending = None
                self.state = State.RUNNING
                if edge.kind == "return" or self.pending_auto:
                    self._event("timeout", f"return window {edge.src} -> {edge.dst} never opened ({self._unmet_text})")
                    self._enter_recovery(f"{edge.src}: return window never opened")
                else:
                    self._event("timeout", f"gave up waiting for {edge.dst} after {edge.wait_timeout_s}s ({self._unmet_text})")

    def _overlay_tick(self, frame: SensorFrame, speed: float):
        """Overlays: a clip layered on the active locomotion skill through its command inputs (masked
        joints -> head commands), ramped in and out with smoothstep over the overlay's window. The
        locomotion policy keeps balancing; this is the framework's 'masked upper-body targets' layer."""
        if self.overlay_req is not None:
            edge, waited = self.overlay_req
            waited += self.dt
            unmet = edge.unmet(self.policies[self.active].phase, frame.contacts, speed)
            if not unmet:
                pol = self.policies[edge.dst]
                pol.reset(HandOff(tuple(a.copy() for a in self.actions), HOME.copy(), self.cmd.copy(), self.t))
                self.overlay = {"node": edge.dst, "pol": pol, "edge": edge, "t": 0.0, "w": 0.0}
                self.overlay_req = None
                self._event("overlay", f"{edge.dst} starts on {self.active} (waited {waited:.2f}s)")
            elif waited > edge.wait_timeout_s:
                self.overlay_req = None
                self._event("timeout", f"overlay {edge.dst}: window never opened ({', '.join(unmet)})")
            else:
                self.overlay_req[1] = waited
                self._unmet_text = ", ".join(unmet)
        if self.overlay is None:
            return
        ov = self.overlay
        pkg = self.graph.nodes[ov["node"]]
        ctx = PolicyContext(frame, self.cmd.copy(), self.actions, self.targets.copy(), ov["t"], self.dt)
        pose, _ = ov["pol"].step(ctx)
        ov["t"] += self.dt
        win = float(pkg.overlay.get("window_s", 0.3))
        if win > 0:
            w = min(smoothstep(ov["t"] / win), smoothstep((pkg.clip.duration - ov["t"]) / win))
        else:
            w = 1.0
        ov["w"] = max(w, 0.0)
        delta = pose - HOME
        ranges = self.pkg.commands
        for j in pkg.overlay["mask"]:
            ji = JOINT_INDEX[j]
            ci = 3 + (ji - 5)                          # head joints map onto commands 3..6
            rng = ranges.get(COMMAND_NAMES[ci], [-np.inf, np.inf])
            self.cmd[ci] = float(np.clip(self.cmd[ci] + ov["w"] * delta[ji], rng[0], rng[1]))
        if ov["pol"].done:
            self.overlay = None
            self._event("overlay", f"{pkg.name} done")

    def _cancel_overlay(self, why: str):
        if self.overlay is not None or self.overlay_req is not None:
            name = self.overlay["node"] if self.overlay else self.overlay_req[0].dst
            self.overlay = None
            self.overlay_req = None
            self._event("overlay", f"{name} cancelled: {why}")

    def _start_blend(self, edge: Edge):
        """The hand-off: freeze the outgoing reference, seed the incoming policy (rule 1), start the ramp (rule 3)."""
        self._cancel_overlay(f"hand-off to {edge.dst}")
        self.hold = self.targets.copy()
        self.blend_edge = edge
        self.blend_t = 0.0
        self.pending = None
        self.active = edge.dst
        self.t_skill = 0.0
        self.limiter = DeployFilter(0.0, float(self.pkg.envelope.get("max_vel_rad_s", 0.0)), self.dt)
        self.limiter.reset(self.hold)
        self.policies[self.active].reset(HandOff(tuple(a.copy() for a in self.actions), self.hold.copy(), self.cmd.copy(), self.t))
        self.show.cancel()
        self.show.start(self.pkg.show_cues)
        self.segments.append((self.active, self.t))
        self.state = State.BLENDING if edge.window_s > 0 else State.RUNNING
        self._event("handoff", f"{edge.src} -> {edge.dst}: blending reference over {edge.window_s:.2f}s")

    def _enter_recovery(self, reason: str) -> bool:
        if self.recovering or self.relaxed:
            return False
        edge = self.graph.abort_edge(self.active)
        if edge is None:                   # graph validation makes this impossible; keep the robot safe anyway
            self._event("abort", f"no abort edge from {self.active}: relaxing in place")
            self.recovering = True
            self.recovery_reason = reason
            self._relax()
            return True
        self.recovering = True
        self.recovery_reason = reason
        self.pending = None
        self._event("abort", f"{reason}: {self.active} -> {edge.dst}")
        self._start_blend(edge)
        return True

    def _relax(self):
        soft = float(self.pkg.recovery.get("relax_stiffness", 0.3)) if self.pkg.type == "recovery" else 0.3
        self.backend.set_stiffness(soft)
        self.relaxed = True
        self.state = State.RELAXED
        self._event("relax", f"servos at stiffness {soft:g}, holding the pose")

    def _after_step(self, pol: BasePolicy):
        pkg = self.pkg
        if self.relaxed:
            return
        if pkg.type == "recovery" and pol.done and self.state == State.RUNNING:
            if self.recovery_done_t is None:
                self.recovery_done_t = self.t
                self._event("recovery", f"{pkg.name} clip done, holding {pkg.recovery.get('hold_s', 0.5)}s")
            elif self.t - self.recovery_done_t >= float(pkg.recovery.get("hold_s", 0.5)) - 1e-9:
                self._relax()
            return
        if pkg.type == "episodic" and pol.done and self.state == State.RUNNING and not self.recovering:
            edge = self.graph.return_edge(self.active)
            if edge is None:
                self._enter_recovery(f"{pkg.name} finished with no return edge")
            else:
                self._event("done", f"{pkg.name} clip finished")
                self._request(edge, auto=True)
        if pkg.timeout_s is not None and self.t_skill > pkg.timeout_s and not self.recovering:
            self._enter_recovery(f"{pkg.name} timed out after {pkg.timeout_s}s")
