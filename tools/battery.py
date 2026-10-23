#!/usr/bin/env python3
"""Gate G1, the sim battery: multi-seed rollouts of a skill package with the training recipe's
randomisation, scored with the same termination the trainer uses, reported as ratios.

    cd sim && MUJOCO_GL=egl .venv/bin/python ../tools/battery.py ../skills/walk/v1 [--seeds 5 --episodes 10]
        [--no-video] [--tests forward,turn_left] [--lpf 8 --max-vel 4]   (overrides, recorded in the report)

Per seed the floor friction, link masses and the servo torque ceiling are drawn once
(tools/skillkit.randomise); per episode the start pose is jittered, the settle time before the
command varies, and pushes come from a random direction. A fall is trunk tilt beyond
envelope.fall_tilt_deg or base height below thresholds.min_height_m (skill.json).

locomotion: standing 3 s, forward, backward, turn left/right, side-step left/right (6 s each,
            commands at the trained range limits), push recovery (15 N for 0.1 s while standing).
            A velocity test also has to move: at least thresholds.min_tracking_fraction of the
            command, in the right direction, or standing still would score as walking.
episodic (also overlay, recovery): the clip is played from the home pose with the policy;
            success = no fall and mean joint error below thresholds.joint_err_rad (default 0.15).

Writes report/battery.json, report/battery.png and one short mp4 per test (first episode of
seed 0). Exit 0 when the battery passed (overall >= thresholds.battery_pass_ratio and every
test >= battery_min_test_ratio), 2 when it ran but failed, 1 on a tool error.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import skillkit as sk  # noqa: E402

PUSH_N, PUSH_STEPS, PUSH_AFTER_S = 15.0, 5, 3.0
ZERO7 = np.zeros(7)


def gyro_jitter(gyro_xy):
    """Fast (>8 Hz) roll/pitch gyro RMS in deg/s, as sim/smooth_cmp_lib.bands."""
    g = np.asarray(gyro_xy, float)
    if len(g) < 16:
        return float("nan")
    g = g - g.mean(0)
    f = np.fft.rfftfreq(len(g), 1 / 50)
    G = np.fft.rfft(g, axis=0)
    hi = np.fft.irfft(G * (f > 8)[:, None], len(g), axis=0)
    return float(np.degrees(np.sqrt((hi ** 2).mean(0))).mean())


def locomotion_tests(skill):
    c = skill.get("commands", {})
    tests = [("standing", 3.0, [0, 0, 0], None)]
    if "vx" in c:
        tests += [("forward", 6.0, [c["vx"][1], 0, 0], "fwd"), ("backward", 6.0, [c["vx"][0], 0, 0], "fwd")]
    if "wz" in c:
        tests += [("turn_left", 6.0, [0, 0, c["wz"][1]], "yaw"), ("turn_right", 6.0, [0, 0, c["wz"][0]], "yaw")]
    if "vy" in c:
        tests += [("side_left", 6.0, [0, c["vy"][1], 0], "left"), ("side_right", 6.0, [0, c["vy"][0], 0], "left")]
    tests.append(("push_recovery", PUSH_AFTER_S, [0, 0, 0], "push"))
    return tests


def run_locomotion_episode(sim, runner, thr, cmd3, seconds, kind, rng, rec=None, label=""):
    """settle (1-2 s at zero command) then the command; returns the episode's metrics."""
    settle = float(rng.uniform(1.0, 2.0))
    fell = False
    for _ in range(int(settle * 50)):
        runner.step(ZERO7)
        if rec:
            rec.frame(f"{label} settle")
        if sk.fallen(sim, thr):
            fell = True
            break
    p0, yaw0, _ = sim.base_pose()
    yawt = sk.yaw_rate_tracker()
    yawt.add(yaw0)
    cmd = np.array(list(cmd3) + [0, 0, 0, 0], float)
    tilt_max, gyro, push_dir = 0.0, [], None
    if kind == "push":
        ang = float(rng.uniform(0, 2 * np.pi))
        push_dir = round(float(np.degrees(ang)), 1)
    n = int(seconds * 50)
    for k in range(n):
        if fell:
            break
        if kind == "push":
            sim.d.xfrc_applied[sim.base, :3] = [PUSH_N * np.cos(ang), PUSH_N * np.sin(ang), 0] if k < PUSH_STEPS else 0
        o, _ = runner.step(cmd)
        gyro.append(o[0:2])
        _, yaw, tilt = sim.base_pose()
        yawt.add(yaw)
        tilt_max = max(tilt_max, tilt)
        if rec:
            rec.frame(f"{label} {'FALLEN' if sk.fallen(sim, thr) else ''}")
        if sk.fallen(sim, thr):
            fell = True
    sim.d.xfrc_applied[sim.base, :] = 0
    fwd, left = sk.heading_velocity(sim, p0, yaw0, seconds)
    m = {"fell": fell, "fwd_m_s": round(float(fwd), 4), "left_m_s": round(float(left), 4),
         "yaw_deg_s": round(float(np.degrees(yawt.total) / seconds), 2), "tilt_max_deg": round(float(tilt_max), 1),
         "settle_s": round(settle, 2), "jitter_deg_s": round(gyro_jitter(gyro), 2)}
    if push_dir is not None:
        m["push_dir_deg"] = push_dir
    return m


def locomotion_success(m, cmd3, kind, thr):
    if m["fell"]:
        return False
    if kind is None or kind == "push":
        return True
    frac = thr["min_tracking_fraction"]
    if kind == "fwd":
        return m["fwd_m_s"] * np.sign(cmd3[0]) >= frac * abs(cmd3[0])
    if kind == "left":
        return m["left_m_s"] * np.sign(cmd3[1]) >= frac * abs(cmd3[1])
    if kind == "yaw":
        return np.radians(m["yaw_deg_s"]) * np.sign(cmd3[2]) >= frac * abs(cmd3[2])
    return True


def run_clip_episode(sim, runner, thr, clip, rt, rng, rec=None, label=""):
    """Play the package clip from the home pose with the policy driving the servos."""
    times, q = clip
    T = sk.clip_duration(times)
    cmd = np.array(list(rt.get("episodic_cmd", [0, 0, 0])) + [0, 0, 0, 0], float)
    fell, errs, tilt_max = False, [], 0.0
    for _ in range(50):                                   # 1 s at home before the clip starts
        runner.step(ZERO7, clip_t=0.0)
        if rec:
            rec.frame(f"{label} home")
        if sk.fallen(sim, thr):
            fell = True
            break
    n = int(round(T * 50)) + 25                           # the clip plus 0.5 s to settle at its end
    for k in range(n):
        if fell:
            break
        t = min(k / 50.0, T)
        runner.step(cmd, clip_t=t)
        ref = sk.clip_pose(times, q, t)
        errs.append(np.abs(sim.d.qpos[sim.qadr] - ref).mean())
        _, _, tilt = sim.base_pose()
        tilt_max = max(tilt_max, tilt)
        if rec:
            rec.frame(f"{label} t={t:.2f}s")
        if sk.fallen(sim, thr):
            fell = True
    err = float(np.mean(errs)) if errs else float("nan")
    return {"fell": fell, "mean_joint_err_rad": round(err, 4), "max_joint_err_rad": round(float(np.max(errs)) if errs else float("nan"), 4),
            "tilt_max_deg": round(float(tilt_max), 1)}


def summarise(episodes, success_fn):
    ok = [bool(success_fn(e)) for e in episodes]
    out = {"success": int(sum(ok)), "total": len(ok), "ratio": round(sum(ok) / len(ok), 3) if ok else None,
           "no_fall_ratio": round(sum(not e["fell"] for e in episodes) / len(episodes), 3) if episodes else None}
    keys = [k for k in episodes[0] if isinstance(episodes[0][k], (int, float)) and not isinstance(episodes[0][k], bool)] if episodes else []
    out["mean"] = {k: round(float(np.nanmean([e[k] for e in episodes])), 4) for k in keys}
    return out


def plot(report, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = [n for n, t in report["tests"].items() if t.get("ratio") is not None]
    vals = [report["tests"][n]["ratio"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=120)
    cols = ["#2a9d8f" if v >= report["thresholds"]["battery_min_test_ratio"] else "#e76f51" for v in vals]
    ax.bar(range(len(names)), vals, color=cols)
    ax.axhline(report["thresholds"]["battery_min_test_ratio"], color="#888", lw=0.8, ls="--")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success ratio")
    ax.set_title(f"{report['skill']} v{report['version']}: battery {report['overall_ratio']:.2f} "
                 f"({report['seeds']} seeds x {report['episodes']} episodes)", fontsize=10)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{report['tests'][names[i]]['success']}/{report['tests'][names[i]]['total']}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("skill_dir")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=10, help="episodes per seed and test")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--tests", help="comma list to run a subset")
    ap.add_argument("--out", help="report path (default <skill_dir>/report/battery.json)")
    ap.add_argument("--lpf", type=float, help="override runtime.lpf_hz (experiment; recorded in the report)")
    ap.add_argument("--max-vel", type=float, help="override envelope.max_vel_rad_s (experiment)")
    ap.add_argument("--torque", type=float, help="override envelope.torque_nm (experiment)")
    ap.add_argument("--no-status", action="store_true", help="do not promote skill.json status on a pass")
    a = ap.parse_args(argv)

    skill_dir = os.path.abspath(a.skill_dir)
    skill = sk.load_skill(skill_dir)
    onnx = os.path.join(skill_dir, "policy.onnx")
    clip_path = os.path.join(skill_dir, "clip.json")
    out = os.path.abspath(a.out) if a.out else os.path.join(skill_dir, "report", "battery.json")
    report_dir = os.path.dirname(out)
    os.makedirs(report_dir, exist_ok=True)
    rt, thr = sk.runtime_settings(skill), sk.thresholds(skill)
    overrides = {}
    if a.lpf is not None:
        rt["lpf_hz"] = a.lpf; overrides["lpf_hz"] = a.lpf
    if a.max_vel is not None:
        rt["max_vel"] = a.max_vel; overrides["max_vel_rad_s"] = a.max_vel
    if a.torque is not None:
        rt["torque_nm"] = a.torque; overrides["torque_nm"] = a.torque
    kind = "locomotion" if skill["type"] == "locomotion" else "clip"
    clip = None
    if kind == "clip" or sk.WALK_OBS_NAMES != (skill.get("obs_spec", {}).get("names") or sk.WALK_OBS_NAMES):
        times, q, _ = sk.load_clip(clip_path)
        clip = (times, q)
    tests = locomotion_tests(skill) if kind == "locomotion" else [("clip_tracking", None, None, "clip")]
    if a.tests:
        want = set(a.tests.split(","))
        tests = [t for t in tests if t[0] in want]
    print(f"battery {skill['name']} v{skill['version']} ({skill['type']}): {a.seeds} seeds x {a.episodes} episodes, "
          f"torque {rt['torque_nm']} N.m lpf {rt['lpf_hz']} Hz max_vel {rt['max_vel']} rad/s, "
          f"fall > {thr['fall_tilt_deg']} deg or z < {thr['min_height_m']} m")
    t0 = time.time()
    results = {name: [] for name, *_ in tests}
    rand = []
    video_done = set()
    for seed in range(a.seeds):
        rng = np.random.default_rng(1000 + seed)
        sim = sk.make_sim(onnx, rt, clip_path=clip_path)
        rand.append({"seed": seed, **sk.randomise(sim, rng, rt["torque_nm"])})
        runner = sk.PolicyRunner(sim, skill, clip=clip)
        for name, seconds, cmd3, tkind in tests:
            for ep in range(a.episodes):
                sk.reset_episode(sim, rng)
                rec = None
                if not a.no_video and name not in video_done:
                    rec = sk.Recorder(sim, f"{skill['name']} v{skill['version']} {name}")
                    video_done.add(name)
                if kind == "locomotion":
                    m = run_locomotion_episode(sim, runner, thr, cmd3, seconds, tkind, rng, rec, name)
                else:
                    m = run_clip_episode(sim, runner, thr, clip, rt, rng, rec, name)
                m.update({"seed": seed, "episode": ep})
                results[name].append(m)
                if rec:
                    rec.save(os.path.join(report_dir, f"battery_{name}.mp4"))
                    rec.close()
        done = sum(len(v) for v in results.values())
        print(f"  seed {seed}: {rand[-1]}  ({done} episodes, {time.time() - t0:.0f} s)")
    tests_out = {}
    for name, seconds, cmd3, tkind in tests:
        eps = results[name]
        if kind == "locomotion":
            s = summarise(eps, lambda e, c=cmd3, k=tkind: locomotion_success(e, c, k, thr))
            s["command"] = {"vx": cmd3[0], "vy": cmd3[1], "wz": cmd3[2]}
            s["seconds"] = seconds
        else:
            s = summarise(eps, lambda e: (not e["fell"]) and e["mean_joint_err_rad"] < thr["joint_err_rad"])
        s["episodes"] = eps
        tests_out[name] = s
    total = sum(t["total"] for t in tests_out.values())
    succ = sum(t["success"] for t in tests_out.values())
    overall = round(succ / total, 3) if total else None
    passed = bool(total and overall >= thr["battery_pass_ratio"]
                  and all(t["ratio"] >= thr["battery_min_test_ratio"] for t in tests_out.values()))
    mean_err = None
    if kind == "clip":
        mean_err = round(float(np.nanmean([e["mean_joint_err_rad"] for e in results["clip_tracking"]])), 4)
    report = {
        "skill": skill["name"], "version": skill["version"], "type": skill["type"],
        "seeds": a.seeds, "episodes": a.episodes,
        "tests": tests_out, "overall_ratio": overall, "overall": f"{succ}/{total}", "passed": passed,
        "thresholds": {**thr, "torque_nm": rt["torque_nm"], "max_vel_rad_s": rt["max_vel"], "lpf_hz": rt["lpf_hz"],
                       "push_n": PUSH_N, "push_s": PUSH_STEPS / 50},
        "overrides": overrides,
        "randomisation": {"per_seed": rand, "per_episode": "start pose +-0.03 rad, settle 1-2 s, push direction"},
        "mean_joint_err_rad": mean_err,
        "robot_model_hash": sk.model_hash(), "runtime_s": round(time.time() - t0, 1),
        "created": sk.now_iso(), "tool": "tools/battery.py",
    }
    sk.dump_json(report, out)
    plot(report, os.path.join(report_dir, os.path.basename(out).replace(".json", ".png")))
    print(f"\n{'test':<16}{'ratio':<8}{'ok/n':<8}no-fall  mean")
    for name, t in tests_out.items():
        print(f"{name:<16}{t['ratio']:<8.2f}{t['success']}/{t['total']:<6}{t['no_fall_ratio']:<8.2f}"
              + " ".join(f"{k}={v}" for k, v in t["mean"].items() if k in ("fwd_m_s", "left_m_s", "yaw_deg_s", "tilt_max_deg", "mean_joint_err_rad", "jitter_deg_s")))
    print(f"overall {succ}/{total} = {overall}  {'PASSED' if passed else 'FAILED'}  ({report['runtime_s']} s) -> {out}")
    if passed and not a.no_status and not overrides and out.endswith(os.path.join("report", "battery.json")):
        if skill.get("status", "draft") == "draft":
            skill["status"] = "battery_passed"
            skill.setdefault("status_history", []).append({"at": sk.now_iso(), "from": "draft", "to": "battery_passed", "by": "tools/battery.py"})
            sk.dump_json(skill, os.path.join(skill_dir, "skill.json"))
            print("status: draft -> battery_passed")
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
