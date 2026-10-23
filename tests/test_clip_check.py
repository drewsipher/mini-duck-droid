"""anim/clip_check.py: the stage 2 gate (floor penetration, self-collision, joint limits, servo
speed, start/end pose). Exit 0 = pass, 1 = fail; the --json report names each failure.
"""
import json
import os

import pytest

from conftest import ANIM, CLIPS, PY, SIM, run

CLIP_CHECK = os.path.join(ANIM, "clip_check.py")
FPS = 50
REPORT_FIELDS = ["pass", "fails", "floor_penetration_m", "self_collisions_m", "limit_breach_rad",
                 "over_speed", "start_vs_home_rad", "end_vs_home_rad"]


def check(clip, *args, tmp_path):
    """Run the checker; return (exit code, report dict, CompletedProcess)."""
    out = tmp_path / (os.path.basename(clip) + ".report.json")
    r = run([PY, CLIP_CHECK, clip, "--json", str(out), *args], cwd=SIM, timeout=180)
    assert out.exists(), f"no report written. stdout:\n{r.stdout[-1500:]}\nstderr:\n{r.stderr[-1500:]}"
    return r.returncode, json.load(open(out)), r


def synth(home, seconds, fn):
    """Frames at 50 Hz: home pose plus fn(t) offsets, like anim/make_test_clips.py."""
    frames = []
    for k in range(int(round(seconds * FPS)) + 1):
        t = k / FPS
        pose = dict(home)
        for j, v in fn(t).items():
            pose[j] = home[j] + v
        frames.append([round(t, 3), {j: round(v, 5) for j, v in pose.items()}])
    return frames


def ramp(t, t0, t1):
    """0 -> 1 between t0 and t1, smoothstep."""
    x = min(max((t - t0) / (t1 - t0), 0.0), 1.0)
    return x * x * (3 - 2 * x)


def bump(t, t0, t1, hold):
    rise = (t1 - t0 - hold) / 2
    return ramp(t, t0, t0 + rise) * (1 - ramp(t, t1 - rise, t1))


# ---------------------------------------------------------------- real clips

def test_turn_left_passes_with_loop(tmp_path):
    clip = os.path.join(CLIPS, "s2a_turn_left.json")
    rc, rep, r = check(clip, "--loop", tmp_path=tmp_path)
    assert rc == 0, f"exit {rc}; fails {rep.get('fails')}\n{r.stdout[-800:]}"
    assert rep["pass"] is True and rep["fails"] == []
    assert rep["floor_penetration_m"] <= 0.003
    assert rep["self_collisions_m"] == {}
    assert rep["limit_breach_rad"] == {} and rep["over_speed"] == {}
    assert rep["frames"] == 28


def test_gait_clip_needs_loop_flag(tmp_path):
    """A recorded gait clip does not start at home (measured 0.20 rad > 0.15 tol), so without
    --loop the checker must fail it on the pose rule, not on anything physical."""
    clip = os.path.join(CLIPS, "s2a_turn_left.json")
    rc, rep, _ = check(clip, tmp_path=tmp_path)
    assert rc == 1
    assert "does not start at home" in rep["fails"]
    assert set(rep["fails"]) <= {"does not start at home", "does not end at home"}


def test_test_bow_passes(tmp_path):
    clip = os.path.join(CLIPS, "test_bow.json")
    if not os.path.exists(clip):
        pytest.skip("tracker/data/clips/test_bow.json not generated (anim/make_test_clips.py)")
    rc, rep, r = check(clip, tmp_path=tmp_path)
    assert rc == 0, f"exit {rc}; fails {rep.get('fails')}\n{r.stdout[-800:]}"
    assert rep["pass"] is True
    assert rep["start_vs_home_rad"] < 0.15 and rep["end_vs_home_rad"] < 0.15


def test_report_has_gate_fields(tmp_path):
    clip = os.path.join(CLIPS, "test_bow.json")
    if not os.path.exists(clip):
        pytest.skip("test_bow.json not generated")
    _, rep, _ = check(clip, tmp_path=tmp_path)
    for f in REPORT_FIELDS:
        assert f in rep, f"report field {f} missing"
    assert isinstance(rep["fails"], list) and isinstance(rep["pass"], bool)
    assert isinstance(rep["limit_breach_rad"], dict) and isinstance(rep["over_speed"], dict)
    assert isinstance(rep["self_collisions_m"], dict)
    assert rep["joints"] and len(rep["joints"]) == 14
    assert abs(rep["seconds"] - (rep["frames"] - 1) / FPS) < 1e-6


# ---------------------------------------------------------------- synthetic failures

def test_home_hold_passes(home, write_clip, tmp_path):
    clip = write_clip("hold", synth(home, 1.0, lambda t: {}))
    rc, rep, _ = check(clip, tmp_path=tmp_path)
    assert rc == 0 and rep["pass"], rep["fails"]


def test_knee_beyond_range_fails_on_joint_limits(home, joint_ranges, write_clip, tmp_path):
    hi = joint_ranges["left_knee"][1]                       # 1.5708 in the MJCF
    over = hi + 0.3 - home["left_knee"]                     # peak 0.3 rad past the limit
    clip = write_clip("knee_over", synth(home, 2.0, lambda t: {"left_knee": over * bump(t, 0.2, 1.8, 0.4)}))
    rc, rep, _ = check(clip, tmp_path=tmp_path)
    assert rc == 1
    assert "joint limits" in rep["fails"], rep["fails"]
    assert "left_knee" in rep["limit_breach_rad"]
    assert abs(rep["limit_breach_rad"]["left_knee"] - 0.3) < 0.01
    assert rep["over_speed"] == {}, "the ramp is slow; only the limit should trip"


def test_one_frame_jump_fails_on_speed(home, write_clip, tmp_path):
    def fn(t):
        if t < 0.2:
            return {}
        if t < 0.4:
            return {"head_yaw": 1.0}                        # 1 rad step in one 20 ms frame = 50 rad/s
        return {"head_yaw": 1.0 * (1 - ramp(t, 0.4, 1.0))}  # ~2.6 rad/s peak coming back
    clip = write_clip("jump", synth(home, 1.2, fn))
    rc, rep, _ = check(clip, tmp_path=tmp_path)
    assert rc == 1
    assert any(f.startswith("servo speed") for f in rep["fails"]), rep["fails"]
    assert "head_yaw" in rep["over_speed"] and rep["over_speed"]["head_yaw"] > 40
    assert "joint limits" not in rep["fails"]


def test_max_vel_flag_is_wired(home, write_clip, tmp_path):
    """The same jump passes when the speed envelope is raised above it, so the flag reaches the gate."""
    clip = write_clip("jump2", synth(home, 1.2, lambda t: {"head_yaw": 1.0 if 0.2 <= t < 0.4 else 1.0 * (1 - ramp(t, 0.4, 1.0)) if t >= 0.4 else 0.0}))
    rc, rep, _ = check(clip, "--max-vel", "60", tmp_path=tmp_path)
    assert not any(f.startswith("servo speed") for f in rep["fails"]), rep["fails"]
    assert rep["over_speed"] == {}


def test_not_starting_at_home_fails_without_loop(home, write_clip, tmp_path):
    off = 0.4                                                # left_hip_pitch home -0.63 -> -0.23, inside its range
    clip = write_clip("offhome", synth(home, 1.0, lambda t: {"left_hip_pitch": off * (1 - ramp(t, 0.0, 0.6))}))
    rc, rep, _ = check(clip, tmp_path=tmp_path)
    assert rc == 1
    assert "does not start at home" in rep["fails"], rep["fails"]
    assert "does not end at home" not in rep["fails"]
    assert abs(rep["start_vs_home_rad"] - off) < 0.01


def test_not_ending_at_home_fails(home, write_clip, tmp_path):
    clip = write_clip("endoff", synth(home, 1.0, lambda t: {"neck_pitch": 0.5 * ramp(t, 0.2, 0.8)}))
    rc, rep, _ = check(clip, tmp_path=tmp_path)
    assert rc == 1
    assert "does not end at home" in rep["fails"], rep["fails"]


def test_loop_that_does_not_close_fails(home, write_clip, tmp_path):
    clip = write_clip("openloop", synth(home, 1.0, lambda t: {"neck_pitch": 0.5 * ramp(t, 0.2, 0.8)}))
    rc, rep, _ = check(clip, "--loop", tmp_path=tmp_path)
    assert rc == 1
    assert "loop does not close" in rep["fails"], rep["fails"]


def test_feet_not_level_reports_penetration(home, write_clip, tmp_path):
    """Without a base track the lowest foot is dropped to the floor, so lifting one knee a lot
    while the other leg stays put must not report penetration (the other foot is on the floor)."""
    clip = write_clip("lift", synth(home, 1.5, lambda t: {"left_knee": 0.15 * bump(t, 0.2, 1.3, 0.3),
                                                           "left_hip_pitch": -0.15 * bump(t, 0.2, 1.3, 0.3)}))
    rc, rep, _ = check(clip, tmp_path=tmp_path)
    assert rep["floor_penetration_m"] <= 0.003, rep
    assert "floor penetration" not in rep["fails"]
