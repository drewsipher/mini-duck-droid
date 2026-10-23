#!/usr/bin/env python3
"""Build the demo skill packages under graph_runtime/demo_skills/ and run their sim batteries.

    cd <project> && MUJOCO_GL=egl sim/.venv/bin/python graph_runtime/make_demo_skills.py [--trials 10] [--out DIR]

  walk/v1   locomotion: sim/BEST_WALK_ONNX_2.onnx as deployed (action scale 0.25, head = policy + command,
            no low-pass, 4 rad/s rate limit). Clip = one recorded gait cycle. Battery: N trials of
            3 s standing then 3 s walking at 0.10-0.14 m/s with +-0.02 rad start-pose noise; counts falls.
  sit/v1    recovery: null policy, clip ramps home -> crouch in 1.5 s, held 0.5 s, then servos at 0.3 stiffness.
            Battery: N trials of abort from standing; pass = still upright and not collapsed 3 s after relaxing.
  bow/v1    episodic, OPEN-LOOP PLACEHOLDER: null policy plays tracker/data/clips/test_bow.json as targets.
            Battery: N trials of idle -> bow -> back to walk through the engine; pass = no recovery, no fall.

The batteries run through the Engine on the MuJoCo backend (the runtime's own code path), so they are
also the first sim-to-sim check of the state machine. Packages are first written with a pending
battery, then the batteries are run with graphs loaded without the battery gate, then the reports are
written for real.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from graph_runtime.backends.mujoco import DEFAULT_XML, MujocoBackend          # noqa: E402
from graph_runtime.engine import Engine, State                                 # noqa: E402
from graph_runtime.graph import build_graph                                    # noqa: E402
from graph_runtime.packages import HOME, JOINTS, Clip, write_package           # noqa: E402

WALK_ONNX = PROJECT / "sim" / "BEST_WALK_ONNX_2.onnx"
BOW_CLIP = PROJECT / "tracker" / "data" / "clips" / "test_bow.json"
WALK_OBS = ["gyro", "accelerometer", "commands", "joint_pos_rel", "joint_vel_scaled", "last_action",
            "last_last_action", "last_last_last_action", "motor_targets", "feet_contacts", "gait_phase"]
ENVELOPE = {"torque_nm": 1.86, "max_vel_rad_s": 4.0, "thermal_c": 55.0, "fall_tilt_deg": 45.0, "min_height_m": 0.10}
FALL_TILT, FALL_HEIGHT = 45.0, 0.08
# Crouch found by graph_runtime's MuJoCo scan (2026-10-23): stable held and at 0.3 stiffness, inside the
# measured real joint limits (joint_limits/real_limits_20261014.json). hip_pitch -0.4/+0.4, knee 1.5,
# ankle -1.1, neck -0.3, head_pitch 0.3; the duck's legs are short so the trunk only drops ~1 cm.
CROUCH = HOME.copy()
CROUCH[[2, 3, 4]] = [-0.4, 1.5, -1.1]
CROUCH[[11, 12, 13]] = [0.4, 1.5, -1.1]
CROUCH[[5, 6]] = [-0.3, 0.3]


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def pending_battery(name):
    return {"skill": name, "passed": False, "pending": True, "note": "battery not run yet"}


# ---------------------------------------------------------------- skill.json contents
def walk_skill():
    return {
        "name": "walk", "type": "locomotion", "version": 1,
        "description": "Deployed walk brain BEST_WALK_ONNX_2 packaged as locomotion v1. Idle is this skill at zero command.",
        "commands": {"vx": [-0.15, 0.15], "vy": [-0.2, 0.2], "wz": [-1.0, 1.0],
                     "neck_pitch": [-0.34, 1.1], "head_pitch": [-0.78, 0.78], "head_yaw": [-1.5, 1.5], "head_roll": [-0.5, 0.5]},
        "entry": {"from": ["bow"], "phase": [0.0, 1.0], "window_s": 0.4},
        "exit": {"to": ["bow", "bow_overlay", "sit"]},
        "neighbours": ["bow", "bow_overlay", "sit"],
        "timeout_s": None,
        "envelope": dict(ENVELOPE),
        "obs_spec": {"names": WALK_OBS, "size": 101},
        "policy": {"kind": "onnx", "action_scale": 0.25, "head_mode": "add", "lpf_hz": 0.0, "max_vel_rad_s": 4.0,
                   "accel_bias": 0.0, "gait_period_steps": 27, "input_name": "obs"},
        "status": "deployed v1 (BEST_WALK_ONNX_2); lpf off because an 8 Hz low-pass stops this brain walking in sim",
        "source": {"onnx": "sim/BEST_WALK_ONNX_2.onnx", "obs_layout": "sim/sim_test.py Sim.obs", "packaged_by": "graph_runtime/make_demo_skills.py"},
        "robot_model_hash": None, "actuator_model_version": None, "recipe_version": None, "seed": None,
    }


def sit_skill():
    return {
        "name": "sit", "type": "recovery", "version": 1,
        "description": "Abort / recovery: canned ramp from the standing pose to a crouch, hold, then soften the servos.",
        "commands": {},
        "entry": {"from": ["*"], "phase": [0.0, 1.0], "window_s": 0.3},
        "exit": {"to": []},
        "neighbours": ["*"],
        "timeout_s": None,
        "envelope": {"torque_nm": 1.86, "max_vel_rad_s": 3.0, "thermal_c": 55.0, "fall_tilt_deg": 90.0, "min_height_m": None},
        "obs_spec": {"names": [], "size": 1},
        "policy": {"kind": "null", "action_scale": 0.25, "head_mode": "policy", "lpf_hz": 0.0, "max_vel_rad_s": 3.0, "open_loop": True},
        "recovery": {"relax_stiffness": 0.3, "hold_s": 0.5},
        "status": "canned sit-down; relax = 0.3 stiffness hold (full torque-off collapses the duck; HWI.low_torque_kps on the robot)",
        "source": {"crouch": CROUCH.tolist(), "packaged_by": "graph_runtime/make_demo_skills.py"},
    }


def bow_skill():
    return {
        "name": "bow", "type": "episodic", "version": 1,
        "description": "OPEN-LOOP PLACEHOLDER: the test_bow clip played as joint targets by a null policy, no feedback.",
        "commands": {},
        "entry": {"from": ["walk"], "phase": [0.0, 1.0], "window_s": 0.4, "contact": "double", "speed": [0.0, 0.03]},
        "exit": {"to": ["walk", "sit"], "phase": [0.95, 1.0]},
        "neighbours": ["walk", "sit"],
        "timeout_s": 4.0,
        "envelope": dict(ENVELOPE),
        "obs_spec": {"names": [], "size": 1},
        "policy": {"kind": "null", "action_scale": 0.25, "head_mode": "policy", "lpf_hz": 0.0, "max_vel_rad_s": 4.0, "open_loop": True},
        "status": "open-loop placeholder until the Controls agent's tracking policy lands (sim/train_pg/runs_skills/*/package_draft)",
        "source": {"clip": "tracker/data/clips/test_bow.json", "packaged_by": "graph_runtime/make_demo_skills.py"},
    }


def bow_overlay_skill():
    return {
        "name": "bow_overlay", "type": "overlay", "version": 1,
        "description": "Head-only bow layered on the walk brain: the clip's head joints are fed to the locomotion policy "
                       "as its (trained) head command inputs, so the brain keeps balancing. No new policy.",
        "commands": {},
        "entry": {"from": ["walk"], "phase": [0.0, 1.0], "window_s": 0.3, "speed": [0.0, 0.15]},
        "exit": {"to": ["walk"]},
        "neighbours": ["walk"],
        "timeout_s": None,
        "envelope": dict(ENVELOPE),
        "obs_spec": {"names": [], "size": 1},
        "policy": {"kind": "null", "action_scale": 0.25, "head_mode": "policy", "lpf_hz": 0.0, "max_vel_rad_s": 4.0, "open_loop": True},
        "overlay": {"mask": ["neck_pitch", "head_pitch", "head_yaw", "head_roll"], "window_s": 0.3},
        "status": "overlay v1: same test_bow clip as bow/v1, played through the walk brain's head commands",
        "source": {"clip": "tracker/data/clips/test_bow.json", "packaged_by": "graph_runtime/make_demo_skills.py"},
    }


def sit_clip(seconds=1.5):
    frames = []
    n = int(seconds * 50)
    for k in range(n + 1):
        a = k / n
        s = 0.5 - 0.5 * np.cos(np.pi * a)
        frames.append([round(k / 50, 4), {j: float(v) for j, v in zip(JOINTS, HOME + s * (CROUCH - HOME))}])
    return {"frames": frames, "interp": "linear", "authored_in": {"tool": "graph_runtime/make_demo_skills.py", "note": "home -> crouch, cosine ease"}}


# ---------------------------------------------------------------- batteries through the engine
def graph_cfg(root, nodes):
    edges = [{"from": "walk", "to": "sit", "window_s": 0.3}]
    if "bow" in nodes:
        edges += [{"from": "walk", "to": "bow", "phase": [0, 1], "contact": "double", "speed": [0, 0.03], "window_s": 0.4, "wait_timeout_s": 3.0},
                  {"from": "bow", "to": "walk", "phase": [0.95, 1.0], "window_s": 0.4, "wait_timeout_s": 1.0},
                  {"from": "bow", "to": "sit", "window_s": 0.3}]
    if "bow_overlay" in nodes:
        edges += [{"from": "walk", "to": "bow_overlay", "phase": [0, 1], "speed": [0, 0.15], "window_s": 0.3, "wait_timeout_s": 2.0}]
    return {"name": "battery", "root": str(root), "nodes": {n: f"{n}/v1" for n in nodes}, "start": "walk",
            "idle": {"node": "walk", "command": [0, 0, 0]}, "recovery": {"node": "sit", "window_s": 0.3}, "edges": edges}


def overlay_battery(root, trials, xml, torque):
    """Standing and walking: the overlay must play to the end without a recovery or a fall."""
    results = []
    for k in range(trials):
        walking = k % 2 == 1
        eng, be = make_engine(root, ["walk", "bow_overlay", "sit"], xml, torque, seed=300 + k, pose_noise=0.02)
        eng.run(1.0)
        if walking:
            eng.set_velocity(0.12)
            eng.run(1.5)
        ok_trigger = eng.trigger("bow_overlay")
        seen = {"v": False}

        def until(i):
            seen["v"] = seen["v"] or i.overlay == "bow_overlay"
            return i.recovering or (seen["v"] and i.overlay is None)
        infos = eng.run(8.0, until=until)
        infos += eng.run(1.0)
        tilt, hmin, fall = track(infos)
        ok = ok_trigger and seen["v"] and not eng.recovering and fall is None
        results.append({"seed": 300 + k, "walking": walking, "passed": bool(ok), "max_tilt_deg": round(tilt, 1),
                        "min_height_m": round(hmin, 3), "recovery": eng.recovery_reason})
        be.close()
        print(f"  overlay trial {k} ({'walking' if walking else 'standing'}): pass {ok}  max tilt {tilt:.1f}  recovery {eng.recovery_reason}", flush=True)
    return results


def make_engine(root, nodes, xml, torque, seed, pose_noise):
    g = build_graph(graph_cfg(root, nodes), project_root=PROJECT, require_battery=False)
    be = MujocoBackend(xml=xml, torque_nm=torque, seed=seed, pose_noise=pose_noise)
    return Engine(g, be), be


def track(infos):
    tilt = max(i.tilt_deg for i in infos)
    hmin = min(i.height_m for i in infos)
    fall = next((i.t for i in infos if i.tilt_deg > FALL_TILT or i.height_m < FALL_HEIGHT), None)
    return tilt, hmin, fall


def walk_battery(root, trials, xml, torque):
    results = []
    for k in range(trials):
        vx = 0.10 + 0.04 * (k / max(trials - 1, 1))
        eng, be = make_engine(root, ["walk", "sit"], xml, torque, seed=k, pose_noise=0.02)
        infos = eng.run(3.0)
        p0 = be.base_pose()[0]
        eng.set_velocity(vx)
        infos += eng.run(3.0)
        p1 = be.base_pose()[0]
        tilt, hmin, fall = track(infos)
        fell = fall is not None or eng.recovering
        results.append({"seed": k, "vx": round(vx, 3), "fell": bool(fell), "fall_t": fall, "max_tilt_deg": round(tilt, 1),
                        "min_height_m": round(hmin, 3), "fwd_m": round(float(p1[0] - p0[0]), 3),
                        "recovery": eng.recovery_reason})
        be.close()
        print(f"  walk trial {k}: vx {vx:.3f}  fell {fell}  max tilt {tilt:.1f}  fwd {p1[0] - p0[0]:+.3f} m", flush=True)
    return results


def sit_battery(root, trials, xml, torque):
    results = []
    for k in range(trials):
        eng, be = make_engine(root, ["walk", "sit"], xml, torque, seed=100 + k, pose_noise=0.02)
        eng.run(1.0)
        eng.abort("battery")
        infos = eng.run(8.0, until=lambda i: i.state == State.RELAXED)
        t_relax = eng.t
        after = eng.run(3.0)
        tilt_after = max(i.tilt_deg for i in after)
        h_after = min(i.height_m for i in after)
        ok = eng.relaxed and tilt_after < 20.0 and h_after > 0.10
        results.append({"seed": 100 + k, "passed": bool(ok), "relaxed_at_s": round(t_relax, 2), "max_tilt_after_relax_deg": round(tilt_after, 1),
                        "min_height_after_relax_m": round(h_after, 3), "stiffness": be.stiffness})
        be.close()
        print(f"  sit trial {k}: relaxed {eng.relaxed} at {t_relax:.2f}s  tilt after {tilt_after:.1f}  height {h_after:.3f}  pass {ok}", flush=True)
    return results


def bow_battery(root, trials, xml, torque):
    results = []
    for k in range(trials):
        eng, be = make_engine(root, ["walk", "bow", "sit"], xml, torque, seed=200 + k, pose_noise=0.02)
        eng.run(1.0)
        ok_trigger = eng.trigger("bow")
        infos = eng.run(10.0, until=lambda i: (i.active == "walk" and i.state == State.RUNNING and i.t > 2.0) or i.recovering)
        infos += eng.run(1.0)
        tilt, hmin, fall = track(infos)
        ok = ok_trigger and not eng.recovering and fall is None and any(i.active == "bow" for i in infos)
        results.append({"seed": 200 + k, "passed": bool(ok), "triggered": ok_trigger, "max_tilt_deg": round(tilt, 1),
                        "min_height_m": round(hmin, 3), "recovery": eng.recovery_reason,
                        "events": [str(e) for e in eng.events if e.kind in ("window", "handoff", "done", "abort", "timeout")]})
        be.close()
        print(f"  bow trial {k}: pass {ok}  max tilt {tilt:.1f}  recovery {eng.recovery_reason}", flush=True)
    return results


def record_walk_clip(root, xml, torque, vx=0.12):
    """One gait period (27 control steps) of measured joint angles while walking, as the reference motion."""
    eng, be = make_engine(root, ["walk", "sit"], xml, torque, seed=0, pose_noise=0.0)
    eng.run(2.0)
    eng.set_velocity(vx)
    eng.run(3.0)
    pol = eng.policies["walk"]
    while pol.clock.i > 1e-6:          # start the clip at phase 0
        eng.tick()
    frames = []
    for k in range(28):
        q = be.read().q
        frames.append([round(k / 50, 4), {j: round(float(v), 5) for j, v in zip(JOINTS, q)}])
        eng.tick()
    be.close()
    return {"frames": frames, "interp": "linear", "authored_in": {"tool": "graph_runtime/make_demo_skills.py",
            "note": f"measured joint angles of BEST_WALK_ONNX_2 walking at vx {vx} m/s in MuJoCo, one gait period from phase 0"}}


def report(name, protocol, results, xml, torque, pose_noise, threshold=0.9):
    trials = len(results)
    passes = sum(1 for r in results if r.get("passed", not r.get("fell", True)))
    return {"skill": name, "version": 1, "date": now(), "generated_by": "graph_runtime/make_demo_skills.py",
            "simulator": {"engine": "graph_runtime.engine on MujocoBackend", "xml": str(xml), "torque_nm": torque,
                          "timestep_s": 0.002, "control_hz": 50, "start_pose_noise_rad": pose_noise},
            "protocol": protocol, "trials": trials, "passes": passes, "fails": trials - passes,
            "ratio": f"{passes}/{trials}", "threshold": threshold, "passed": trials > 0 and passes / trials >= threshold,
            "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT / "graph_runtime" / "demo_skills"))
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--xml", default=str(DEFAULT_XML))
    ap.add_argument("--torque", type=float, default=1.86)
    ap.add_argument("--skip-battery", action="store_true")
    a = ap.parse_args()
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)

    print("writing packages (pending batteries)")
    write_package(root / "sit" / "v1", sit_skill(), sit_clip(), pending_battery("sit"))
    write_package(root / "bow" / "v1", bow_skill(), json.load(open(BOW_CLIP)), pending_battery("bow"))
    write_package(root / "bow_overlay" / "v1", bow_overlay_skill(), json.load(open(BOW_CLIP)), pending_battery("bow_overlay"))
    walk_dir = write_package(root / "walk" / "v1", walk_skill(),
                             {"frames": [[0.0, {j: float(v) for j, v in zip(JOINTS, HOME)}]], "interp": "linear"},
                             pending_battery("walk"), policy_onnx=WALK_ONNX)
    print("recording the walk reference clip")
    (walk_dir / "clip.json").write_text(json.dumps(record_walk_clip(root, a.xml, a.torque)))
    if a.skip_battery:
        return
    print(f"walk battery ({a.trials} trials)")
    r = walk_battery(root, a.trials, a.xml, a.torque)
    rep = report("walk", f"{a.trials} trials: 3 s standing then 3 s walking at vx 0.10-0.14 m/s; fall = tilt > {FALL_TILT} deg or trunk < {FALL_HEIGHT} m",
                 r, a.xml, a.torque, 0.02)
    rep["falls"] = sum(1 for x in r if x["fell"])
    (walk_dir / "report" / "battery.json").write_text(json.dumps(rep, indent=1))
    print(f"  walk: {rep['ratio']} passed, falls {rep['falls']}")
    n_small = max(3, a.trials // 2)
    print(f"sit battery ({n_small} trials)")
    r = sit_battery(root, n_small, a.xml, a.torque)
    rep = report("sit", f"{n_small} trials: abort from standing; pass = relaxed, tilt < 20 deg and trunk > 0.10 m for 3 s after relaxing", r, a.xml, a.torque, 0.02)
    (root / "sit" / "v1" / "report" / "battery.json").write_text(json.dumps(rep, indent=1))
    print(f"  sit: {rep['ratio']} passed")
    print(f"bow battery ({n_small} trials)")
    r = bow_battery(root, n_small, a.xml, a.torque)
    rep = report("bow", f"{n_small} trials through the engine: idle 1 s, trigger bow, return to walk; pass = no recovery and no fall (open-loop clip)", r, a.xml, a.torque, 0.02)
    (root / "bow" / "v1" / "report" / "battery.json").write_text(json.dumps(rep, indent=1))
    print(f"  bow: {rep['ratio']} passed")
    print(f"bow_overlay battery ({n_small} trials)")
    r = overlay_battery(root, n_small, a.xml, a.torque)
    rep = report("bow_overlay", f"{n_small} trials through the engine, alternating standing and walking at 0.12 m/s: "
                 "overlay bow on the walk brain; pass = played to the end, no recovery, no fall", r, a.xml, a.torque, 0.02)
    (root / "bow_overlay" / "v1" / "report" / "battery.json").write_text(json.dumps(rep, indent=1))
    print(f"  bow_overlay: {rep['ratio']} passed")


if __name__ == "__main__":
    main()
