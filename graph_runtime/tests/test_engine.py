"""State machine logic without MuJoCo: windows, timeouts, abort, blend ramp, action continuity."""
import numpy as np
import pytest

from graph_runtime.engine import Engine, State, smoothstep
from graph_runtime.graph import build_graph
from graph_runtime.packages import HOME
from graph_runtime.policies import BasePolicy, HandOff, make_policy
from graph_runtime.tests.conftest import FakeBackend, graph_cfg


def make(skill_root, backend, with_return=False, **kw):
    g = build_graph(graph_cfg(skill_root, with_return=with_return))
    return Engine(g, backend, **kw)


# ---------------------------------------------------------------- blend ramp shape
def test_smoothstep_shape():
    xs = np.linspace(0, 1, 101)
    ys = np.array([smoothstep(x) for x in xs])
    assert ys[0] == 0.0 and ys[-1] == 1.0
    assert np.all(np.diff(ys) >= 0)                                   # monotone
    assert smoothstep(0.5) == pytest.approx(0.5)
    assert smoothstep(0.01) < 0.001 and smoothstep(0.99) > 0.999        # zero slope at both ends
    assert smoothstep(-1) == 0.0 and smoothstep(2) == 1.0
    d = np.diff(ys) / np.diff(xs)
    assert d[0] < 0.05 and d[-1] < 0.05 and d.max() == pytest.approx(1.5, abs=0.01)


# ---------------------------------------------------------------- idle and commands
def test_idle_is_locomotion_at_zero_command(skill_root, fake_backend):
    eng = make(skill_root, fake_backend)
    info = eng.tick()
    assert info.active == "walk" and info.label == "idle" and info.state == State.RUNNING
    eng.set_velocity(0.12)
    infos = eng.run(1.0)
    assert infos[-1].label == "walk"
    assert infos[-1].command[0] == pytest.approx(0.12)
    assert infos[5].command[0] < 0.12                                  # slewed, not stepped
    eng.set_velocity(5.0)                                              # clamped to the trained range
    assert eng.run(2.0)[-1].command[0] == pytest.approx(0.15)


# ---------------------------------------------------------------- entry windows
def test_transition_waits_for_contact_and_speed_window(skill_root, fake_backend):
    eng = make(skill_root, fake_backend)
    eng.set_velocity(0.12)
    eng.run(1.0)
    fake_backend.contacts[:] = [1.0, 0.0]                              # single support: window shut
    assert eng.trigger("bow")
    infos = eng.run(0.4)
    assert all(i.state == State.WAITING for i in infos)
    assert "contact double" in infos[-1].status
    # speed is pulled inside the edge's range while waiting (0.12 -> 0.03 at 0.3 m/s^2)
    assert infos[-1].speed <= 0.03 + 1e-6
    fake_backend.contacts[:] = [1.0, 1.0]
    infos = eng.run(0.1)
    assert infos[0].active == "bow" and infos[0].state == State.BLENDING
    assert any("window" in e for e in infos[0].events)


def test_phase_window_wraps(skill_root):
    from graph_runtime.graph import Edge
    e = Edge("a", "b", [0.9, 0.1])
    assert e.phase_open(0.95) and e.phase_open(0.05) and not e.phase_open(0.5)
    e2 = Edge("a", "b", [0.2, 0.4])
    assert e2.phase_open(0.3) and not e2.phase_open(0.1)
    assert e.contact_open([1, 1]) and Edge("a", "b", [0, 1], "double").contact_open([1, 0]) is False
    assert Edge("a", "b", [0, 1], "flight").contact_open([0, 0])


def test_window_wait_timeout_cancels_request(skill_root, fake_backend):
    eng = make(skill_root, fake_backend)
    eng.run(0.2)
    fake_backend.contacts[:] = [0.0, 0.0]
    assert eng.trigger("bow")
    infos = eng.run(1.5)                                               # wait_timeout_s is 1.0
    assert infos[-1].state == State.RUNNING and infos[-1].active == "walk"
    assert any(e.kind == "timeout" for e in eng.events)
    assert not eng.recovering


def test_trigger_refused_when_busy_or_no_edge(skill_root, fake_backend):
    eng = make(skill_root, fake_backend)
    eng.run(0.1)
    assert not eng.trigger("sit_down")                                 # no such node
    assert not eng.trigger("walk")                                     # no self edge
    assert eng.trigger("bow")
    assert not eng.trigger("bow")                                      # already waiting
    eng.run(0.3)                                                       # now blending / running bow
    assert eng.active == "bow"
    assert not eng.trigger("sit")                                      # recovery is abort(), not a trigger
    assert eng.trigger("walk")                                         # the return edge may be asked for early...
    infos = eng.run(0.3)
    assert all(i.state == State.WAITING and i.active == "bow" for i in infos)   # ...but waits for the exit window


# ---------------------------------------------------------------- episodic flow, timeouts
def test_episodic_plays_and_returns_to_locomotion(skill_root, fake_backend):
    eng = make(skill_root, fake_backend)
    infos = eng.run(0.2)
    eng.trigger("bow")
    infos += eng.run(3.0)
    seq = []
    for i in infos:
        if not seq or seq[-1] != i.active:
            seq.append(i.active)
    assert seq == ["walk", "bow", "walk"]
    kinds = [e.kind for e in eng.events]
    assert kinds.count("handoff") >= 4                                 # 2 blends x (start, done)
    assert "done" in kinds and not eng.recovering
    bow_ticks = [i for i in infos if i.active == "bow"]
    assert 0.95 <= len(bow_ticks) * eng.dt < 2.0                       # the 1 s clip, then straight back


def test_episodic_timeout_aborts(skill_root, fake_backend):
    eng = make(skill_root, fake_backend)
    eng.run(0.2)
    eng.trigger("bow")
    eng.run(0.3)
    assert eng.active == "bow"
    fake_backend.contacts[:] = [0.0, 0.0]                              # irrelevant for the return edge (contact any)
    # freeze the clip clock so the skill never finishes -> timeout_s (2 s) -> recovery
    eng.policies["bow"].clock.duration = 1e9
    infos = eng.run(3.0)
    assert eng.recovering and "timed out" in eng.recovery_reason
    assert infos[-1].active == "sit"


def test_return_window_never_opening_aborts(skill_root, fake_backend):
    eng = make(skill_root, fake_backend)
    eng.run(0.2)
    eng.trigger("bow")
    eng.run(0.3)
    # make the return edge impossible: require flight while the fake feet stay planted
    e = eng.graph.edge("bow", "walk")
    e.contact = "flight"
    eng.run(2.5)
    assert eng.recovering and "return window" in eng.recovery_reason


# ---------------------------------------------------------------- abort and safety
@pytest.mark.parametrize("node", ["walk", "bow"])
def test_abort_reaches_recovery_from_every_node(skill_root, fake_backend, node):
    eng = make(skill_root, fake_backend)
    eng.run(0.2)
    if node == "bow":
        eng.trigger("bow")
        eng.run(0.3)
        assert eng.active == "bow"
    assert eng.abort("test")
    infos = eng.run(2.0)
    assert infos[0].active == "sit" and infos[0].recovering
    assert infos[-1].state == State.RELAXED
    assert fake_backend.stiffness[-1] == pytest.approx(0.3)            # relax = soft hold
    assert not eng.abort("again")                                      # already recovering
    assert not eng.trigger("bow")


def test_tilt_trips_safety_into_recovery(skill_root, fake_backend):
    eng = make(skill_root, fake_backend)
    eng.run(0.2)
    fake_backend.tilt = 60.0
    info = eng.tick()
    assert info.active == "sit" and info.recovering and "tilt 60" in eng.recovery_reason
    assert all(i.command[0] == 0 for i in eng.run(0.2))                # commands forced to zero


def test_height_trips_safety(skill_root, fake_backend):
    eng = make(skill_root, fake_backend)
    eng.run(0.2)
    fake_backend.height = 0.05
    eng.tick()
    assert eng.recovering and "height" in eng.recovery_reason


def test_safety_off_does_not_abort(skill_root, fake_backend):
    eng = make(skill_root, fake_backend, safety=False)
    fake_backend.tilt = 80.0
    eng.run(0.2)
    assert not eng.recovering


def test_reset_after_recovery_needs_return_edge(skill_root, fake_backend):
    eng = make(skill_root, fake_backend)
    eng.abort("x")
    eng.run(2.0)
    assert eng.relaxed
    assert not eng.reset()                                             # demo-style graph: no get-up edge
    eng2 = make(skill_root, FakeBackend(), with_return=True)
    eng2.abort("x")
    eng2.run(2.0)
    assert eng2.reset()
    infos = eng2.run(1.0)
    assert infos[-1].active == "walk" and not eng2.relaxed
    assert eng2.backend.stiffness[-1] == 1.0


# ---------------------------------------------------------------- blending and continuity
def test_blend_is_a_reference_crossfade_from_held_targets(skill_root, fake_backend):
    eng = make(skill_root, fake_backend)
    # run walk with a non-home pose so the hand-off has something to blend from
    walk = eng.policies["walk"]
    off = HOME.copy()
    off[0] += 0.3
    walk.pkg.clip.poses[:] = off
    eng.run(1.0)
    hold = eng.targets.copy()
    assert hold[0] == pytest.approx(off[0], abs=1e-6)
    eng.trigger("bow")
    infos = eng.run(0.2)                                               # window 0.2 s = 10 ticks
    assert infos[0].state == State.BLENDING and infos[-1].state == State.RUNNING
    traj = np.array([i.targets[0] for i in infos])
    ws = np.array([i.blend for i in infos])
    # first blended tick stays within one smoothstep increment of the held pose, the last tick is the incoming clip (home)
    assert abs(traj[0] - hold[0]) < 0.01
    assert traj[-1] == pytest.approx(HOME[0], abs=1e-6)
    assert np.all(np.diff(traj) <= 1e-9)                               # monotone ramp from hold to incoming
    assert ws[0] == pytest.approx(smoothstep(0.1)) and ws[-1] == 1.0
    # after the window the hold is gone and the incoming policy alone sets the targets
    after = eng.run(0.1)
    assert all(i.blend == 1.0 for i in after) and eng.hold is None


def test_incoming_policy_receives_previous_actions_and_targets(skill_root, fake_backend):
    seen = {}

    class Spy(BasePolicy):
        def reset(self, handoff):
            super().reset(handoff)
            seen[self.pkg.name] = handoff

        def step(self, ctx):
            pose = self.pkg.clip.sample(self.clock.t if hasattr(self.clock, "t") else 0.0)
            if hasattr(self.clock, "advance"):
                try:
                    self.clock.advance(ctx.dt)
                except TypeError:
                    self.clock.advance()
            a = np.full(14, 0.1 * (len(fake_backend.targets) % 7))    # distinctive action pattern
            return self.filter.apply(pose), a

    eng = make(skill_root, fake_backend, policy_factory=Spy)
    eng.run(0.3)
    a1, a2, a3 = (a.copy() for a in eng.actions)
    last_targets = eng.targets.copy()
    eng.trigger("bow")
    eng.tick()                                                         # window open at once: hand-off this tick
    h = seen["bow"]
    assert np.allclose(h.actions[0], a1) and np.allclose(h.actions[1], a2) and np.allclose(h.actions[2], a3)
    assert np.allclose(h.targets, last_targets)
    # and the incoming policy's filter state was seeded with them: no jump in the first command
    assert np.max(np.abs(fake_backend.targets[-1] - last_targets)) < 0.05


def test_final_rate_limit_uses_active_envelope(skill_root, fake_backend):
    eng = make(skill_root, fake_backend)
    eng.run(0.1)
    # ask the walk clip to jump 1 rad at once: the envelope's 4 rad/s limits it to 0.08 rad per tick
    eng.policies["walk"].pkg.clip.poses[:, 3] += 1.0
    before = eng.targets[3]
    eng.tick()
    assert eng.targets[3] - before == pytest.approx(4.0 * eng.dt, abs=1e-6)


# ---------------------------------------------------------------- overlays
def test_overlay_plays_through_head_commands_without_a_transition(skill_root, fake_backend):
    eng = make(skill_root, fake_backend)
    eng.run(0.2)
    assert eng.trigger("nod")
    infos = eng.run(1.4)
    assert all(i.active == "walk" and i.state == State.RUNNING for i in infos)      # no hand-off
    played = [i for i in infos if i.overlay == "nod"]
    assert 0.9 < len(played) * eng.dt < 1.1
    assert played[0].label == "idle + nod"
    hp = np.array([i.command[4] for i in played])                                   # head_pitch command
    assert hp[0] < 0.01 and hp.max() == pytest.approx(0.5, abs=0.03) and hp[-1] < 0.01
    assert np.all(np.diff(hp[: len(hp) // 2]) >= -1e-9)                             # ramps in smoothly
    assert infos[-1].overlay is None and infos[-1].command[4] == 0.0
    assert any(e.kind == "overlay" and "done" in e.text for e in eng.events)


def test_overlay_waits_for_window_and_is_cancelled_by_handoff(skill_root, fake_backend):
    eng = make(skill_root, fake_backend)
    eng.set_velocity(0.15)
    eng.run(1.0)
    assert eng.trigger("nod")                                                       # speed window is [0, 0.1]
    infos = eng.run(0.2)
    assert infos[-1].overlay is None and "overlay nod waiting" in infos[-1].status
    eng.set_velocity(0.0)
    eng.run(0.6)
    assert eng.overlay is not None
    assert not eng.trigger("nod")                                                   # one overlay at a time
    eng.abort("x")
    eng.tick()
    assert eng.overlay is None and any("cancelled" in e.text for e in eng.events)
    assert eng.active == "sit"


def test_overlay_window_timeout(skill_root, fake_backend):
    eng = make(skill_root, fake_backend)
    eng.set_velocity(0.15)
    eng.run(1.0)
    eng.trigger("nod")
    eng.run(1.0)                                                                    # wait_timeout_s 0.5
    assert eng.overlay is None and eng.overlay_req is None
    assert any(e.kind == "timeout" and "overlay" in e.text for e in eng.events)


def test_show_cues_fire_on_schedule(skill_root, fake_backend):
    eng = make(skill_root, fake_backend)
    eng.graph.nodes["bow"].show_cues[:] = [{"t": 0.0, "channel": "antenna_left", "value": 1.0},
                                           {"t": 0.5, "channel": "sound", "value": "beep1"}]
    eng.run(0.2)
    eng.trigger("bow")
    eng.run(0.3)
    assert fake_backend.shown == [("antenna_left", 1.0)]
    eng.run(0.4)
    assert fake_backend.shown[-1] == ("sound", "beep1")
