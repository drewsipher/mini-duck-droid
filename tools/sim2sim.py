#!/usr/bin/env python3
"""Gate G2, sim-to-sim: the package's ONNX driven by the REAL runtime code
(sim/upstream_runtime: scripts/v2_rl_walk_mujoco.py RLWalk.run + mini_bdx_runtime HWI / OnnxInfer /
LowPassActionFilter / PolyReferenceMotion), with the hardware mocked by MuJoCo, compared with
the scorer sim_test.Sim on the same commands and seed.

    cd sim && MUJOCO_GL=egl .venv/bin/python ../tools/sim2sim.py ../skills_archive/upstream_walk_v1 [--cmd 0.15,0,0] [--seconds 5] [--seed 0]

Only the hardware is replaced:
  rustypot.feetech  -> FakeFeetech: present positions / velocities from MuJoCo qpos / qvel of the
                       14 servo joints (HWI then subtracts offsets and rounds to 0.001 rad as on the duck);
                       write_goal_position sets the position actuators and advances physics 20 ms.
  raw_imu.Imu       -> the model's gyro and accelerometer sensors at the imu site.
  FeetContacts      -> foot/floor contacts from the contact list.
  time              -> a clock that advances 20 ms per control loop (the filter warm-up and the
                       loop timing in run() see the same time the duck would).
Everything else (observation assembly and order, phase clock, inference, action scale, low-pass,
head command handling, joint order) is the upstream code, unmodified.

Two comparisons are made and both are in report/sim2sim.json:
  lockstep  one physics driven by the runtime; sim_test.Sim shadows it (its state is copied
            every step) so each stack computes its targets from the identical state. The RMS of
            the target difference over the commanded window is the pipeline-level number and
            decides `match`.
  free_run  sim_test.Sim run on its own from the same start: joint trajectories, falls, and the
            velocity each stack achieves (tracking).

match = lockstep RMS < thresholds.sim2sim_rms_rad (default 0.02 rad) and neither stack falls.
Why 0.02 rad: the servo reports 0.0015 rad ticks and the model has +-0.0087 rad of backlash, so
differences below ~0.01 rad never reach the output shaft; a one-step lag of the 27-step gait
clock or a missing filter shifts targets by 0.03-0.1 rad RMS while walking. 0.02 sits between
the two: float/rounding noise passes, any pipeline difference that would change the gait fails.
"""
import argparse
import importlib.util
import os
import sys
import time as _time
import types
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import skillkit as sk  # noqa: E402

PKG_DIR = os.path.join(sk.UPSTREAM, "mini_bdx_runtime")
SCRIPT = os.path.join(sk.UPSTREAM, "scripts", "v2_rl_walk_mujoco.py")
CONFIG = os.path.join(sk.UPSTREAM, "example_config.json")


# ----------------------------------------------------------------------------- hardware mocks
class FakeClock:
    """time.time()/time.sleep() for the runtime: advances only when the loop sleeps."""

    def __init__(self):
        self.t = 1000.0

    def time(self):
        return self.t

    def sleep(self, dt):
        self.t += max(0.0, float(dt))

    def module(self):
        m = types.ModuleType("time")
        m.time, m.sleep, m.perf_counter, m.monotonic = self.time, self.sleep, self.time, self.time
        return m


class FakeFeetech:
    """Stands in for rustypot.feetech(port, baud); the bus of 14 STS3215 backed by MuJoCo."""

    def __init__(self, phys, on_write):
        self.phys, self.on_write = phys, on_write
        self.ids = None

    def set_kps(self, ids, kps):
        pass

    def set_kds(self, ids, kds):
        pass

    def disable_torque(self, ids):
        pass

    def read_present_position(self, ids):
        return [float(x) for x in self.phys.d.qpos[self.phys.qadr]]

    def read_present_velocity(self, ids):
        return [float(x) for x in self.phys.d.qvel[self.phys.vadr]]

    def write_goal_position(self, ids, positions):
        self.on_write(np.asarray(positions, dtype=float))


class FakeImu:
    def __init__(self, phys):
        self.phys = phys

    def get_data(self):
        d, p = self.phys.d, self.phys
        return {"gyro": d.sensordata[p.gyro_a:p.gyro_a + 3].copy(),
                "accelero": d.sensordata[p.acc_a:p.acc_a + 3].copy()}


class FakeFeet:
    def __init__(self, phys):
        self.phys = phys

    def get(self):
        left, right = self.phys.contacts()
        return [bool(left), bool(right)]

    def stop(self):
        pass


class StopRun(KeyboardInterrupt):
    """RLWalk.run() ends its loop on KeyboardInterrupt, like Ctrl-C on the duck."""


def load_runtime(fake_io, clock):
    """Import the upstream walk script with the Pi-only modules stubbed out."""
    if PKG_DIR not in sys.path:
        sys.path.insert(0, PKG_DIR)
    for name in ("board", "busio", "digitalio", "pwmio", "adafruit_bno055", "pygame"):
        sys.modules.setdefault(name, mock.MagicMock(name=name))
    rusty = types.ModuleType("rustypot")
    rusty.feetech = lambda port, baud: fake_io
    sys.modules["rustypot"] = rusty
    spec = importlib.util.spec_from_file_location("v2_rl_walk_mujoco", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.time = clock.module()
    sys.modules["mini_bdx_runtime.rustypot_position_hwi"].time = clock.module()
    return mod


# ----------------------------------------------------------------------------- the comparison
def run_comparison(skill_dir, cmd3, seconds, settle, seed, video=True, runtime_cutoff=None):
    skill = sk.load_skill(skill_dir)
    onnx = os.path.join(skill_dir, "policy.onnx")
    rt, thr = sk.runtime_settings(skill), sk.thresholds(skill)
    names = skill.get("obs_spec", {}).get("names") or sk.WALK_OBS_NAMES
    if list(names) != sk.WALK_OBS_NAMES:
        return None, ("skipped: the upstream runtime only builds the 101-wide walk observation; this package's "
                      "obs_spec needs the motion-graph runtime (graph_runtime/) before sim-to-sim can run")
    n_settle, n_cmd = int(settle * 50), int(seconds * 50)
    n_total = n_settle + n_cmd

    def cmd_at(k):
        c = np.zeros(7)
        if k >= n_settle:
            c[:3] = cmd3
        return c

    # identical simulators: physics for the runtime, its lockstep shadow, a second shadow whose
    # gait clock is delayed one step to match the runtime's (diagnostic only), and the free run
    phys = sk.make_sim(onnx, rt)
    shadow = sk.make_sim(onnx, rt)
    aligned = sk.make_sim(onnx, rt)
    free = sk.make_sim(onnx, rt)
    for s in (phys, shadow, aligned, free):
        sk.reset_episode(s, np.random.default_rng(seed))
    aligned.imitation_i = -1.0                      # its first step lands on phase(0), like the runtime's second
    import mujoco

    log = {"rt_targets": [], "sh_targets": [], "al_targets": [], "rt_obs": [], "sh_obs": [], "rt_q": [], "rt_fell": False,
           "rt_tilt": [], "rt_pos": [], "rt_yaw": []}
    state = {"k": 0, "live": False, "rec": None}
    clock = FakeClock()

    def on_write(targets):
        if not state["live"]:                     # HWI.turn_on() writes the init pose before the loop
            phys.d.ctrl[:] = targets
            return
        k = state["k"]
        cmd = cmd_at(k)
        # the shadow computes its targets from the very same state the runtime just observed
        mujoco.mj_copyData(shadow.d, shadow.m, phys.d)
        sh_obs, _ = shadow.control(cmd)
        log["sh_targets"].append(shadow.targets.copy())
        log["sh_obs"].append(sh_obs.astype(np.float64))
        mujoco.mj_copyData(aligned.d, aligned.m, phys.d)
        aligned.control(cmd)
        log["al_targets"].append(aligned.targets.copy())
        log["rt_targets"].append(targets.copy())
        # physics follows the runtime
        phys.d.ctrl[:] = targets
        for _ in range(phys.decim):
            mujoco.mj_step(phys.m, phys.d)
        pos, yaw, tilt = phys.base_pose()
        log["rt_q"].append(phys.d.qpos[phys.qadr].copy())
        log["rt_tilt"].append(tilt)
        log["rt_pos"].append(pos)
        log["rt_yaw"].append(yaw)
        if state["rec"]:
            state["rec"].frame(f"runtime  t={k / 50:.1f}s cmd={cmd[:3]}")
        if sk.fallen(phys, thr):
            log["rt_fell"] = True
        state["k"] = k + 1
        rl.last_commands = list(cmd_at(k + 1))      # the loop reads it at the next get_obs()
        if state["k"] >= n_total or log["rt_fell"]:
            raise StopRun()

    fake_io = FakeFeetech(phys, on_write)
    mod = load_runtime(fake_io, clock)
    mod.Imu = lambda *a, **kw: FakeImu(phys)
    mod.FeetContacts = lambda *a, **kw: FakeFeet(phys)
    cwd = os.getcwd()
    os.chdir(os.path.dirname(SCRIPT))               # PolyReferenceMotion("./polynomial_coefficients.pkl")
    try:
        cutoff = (rt["lpf_hz"] or None) if runtime_cutoff is None else (runtime_cutoff or None)
        rl = mod.RLWalk(onnx, duck_config_path=CONFIG, commands=False, action_scale=rt["action_scale"],
                        cutoff_frequency=cutoff)
    finally:
        os.chdir(cwd)
    infer0 = rl.policy.infer

    def infer_logged(obs):
        log["rt_obs"].append(np.asarray(obs, dtype=np.float64).copy())
        return infer0(obs)
    rl.policy.infer = infer_logged
    rl.last_commands = list(cmd_at(0))
    if video:
        state["rec"] = sk.Recorder(phys, f"{skill['name']} v{skill['version']} sim2sim")
    state["live"] = True
    t0 = _time.time()
    rl.run()                                        # the real loop; StopRun ends it
    rt_wall = _time.time() - t0

    # ---- free run of the scorer on its own
    fr = {"q": [], "fell": False, "targets": [], "pos": [], "yaw": []}
    for k in range(n_total):
        cmd = cmd_at(k)
        free.control(cmd)
        fr["q"].append(free.d.qpos[free.qadr].copy())
        fr["targets"].append(free.targets.copy())
        pos, yaw, _ = free.base_pose()
        fr["pos"].append(pos)
        fr["yaw"].append(yaw)
        if sk.fallen(free, thr):
            fr["fell"] = True
            break
    if state["rec"]:
        state["rec"].save(os.path.join(skill_dir, "report", "sim2sim.mp4"))
        state["rec"].close()

    # ---- numbers
    rt_t, sh_t = np.array(log["rt_targets"]), np.array(log["sh_targets"])
    n = min(len(rt_t), len(sh_t))
    rt_t, sh_t = rt_t[:n], sh_t[:n]
    w = slice(n_settle, n)                          # the commanded window
    diff = rt_t - sh_t
    rms = float(np.sqrt(np.mean(diff[w] ** 2))) if n > n_settle else float("nan")
    rms_settle = float(np.sqrt(np.mean(diff[:n_settle] ** 2))) if n_settle else 0.0
    per_joint = {j: round(float(np.sqrt(np.mean(diff[w, i] ** 2))), 4) for i, j in enumerate(sk.JOINTS)} if n > n_settle else {}
    max_diff = float(np.abs(diff[w]).max()) if n > n_settle else float("nan")
    al_t = np.array(log["al_targets"])[:n]
    rms_aligned = float(np.sqrt(np.mean((rt_t[w] - al_t[w]) ** 2))) if n > n_settle else float("nan")
    # observation blocks: where do the two stacks disagree?
    rt_o, sh_o = np.array(log["rt_obs"][:n]), np.array(log["sh_obs"][:n])
    obs_diff, i0 = {}, 0
    if len(rt_o) and rt_o.shape == sh_o.shape:
        for name, size in sk.WALK_OBS_BLOCKS:
            obs_diff[name] = round(float(np.sqrt(np.mean((rt_o[w, i0:i0 + size] - sh_o[w, i0:i0 + size]) ** 2))), 5)
            i0 += size
    # free run vs runtime trajectories
    rq, fq = np.array(log["rt_q"]), np.array(fr["q"])
    m = min(len(rq), len(fq))
    q_rms = float(np.sqrt(np.mean((rq[n_settle:m] - fq[n_settle:m]) ** 2))) if m > n_settle else float("nan")

    # velocities over the commanded window, in the heading at its start (yaw accumulated step by step)
    def vel(pos, yaw, fell):
        pos, yaw = np.array(pos), np.unwrap(np.array(yaw))
        if len(pos) <= n_settle + 1:
            return {"fell": fell}
        a, b = n_settle - 1 if n_settle else 0, len(pos) - 1
        secs = (b - a) / 50.0
        dp = pos[b] - pos[a]
        c, s_ = np.cos(-yaw[a]), np.sin(-yaw[a])
        return {"fwd_m_s": round(float((c * dp[0] - s_ * dp[1]) / secs), 4), "left_m_s": round(float((s_ * dp[0] + c * dp[1]) / secs), 4),
                "yaw_deg_s": round(float(np.degrees(yaw[b] - yaw[a]) / secs), 2), "fell": fell}
    tracking = {"command": {"vx": cmd3[0], "vy": cmd3[1], "wz": cmd3[2]},
                "runtime": vel(log["rt_pos"], log["rt_yaw"], log["rt_fell"]),
                "sim_test": vel(fr["pos"], fr["yaw"], fr["fell"])}

    match = bool(np.isfinite(rms) and rms < thr["sim2sim_rms_rad"] and not log["rt_fell"] and not fr["fell"])
    match_aligned = bool(np.isfinite(rms_aligned) and rms_aligned < thr["sim2sim_rms_rad"] and not log["rt_fell"] and not fr["fell"])
    notes = []
    if runtime_cutoff is not None:
        notes.append(f"DIAGNOSTIC RUN: runtime launched with cutoff_frequency={cutoff} instead of the package's {rt['lpf_hz'] or None}")
    notes.append(f"runtime path: {os.path.relpath(SCRIPT, sk.ROOT)} RLWalk.run with cutoff_frequency={cutoff}, "
                 f"action_scale={rt['action_scale']}; HWI rounds positions/velocities to 0.001; no motor speed clip "
                 f"(commented out upstream) whereas sim_test clips at {rt['max_vel']} rad/s")
    notes.append("phase clock: the runtime builds the observation before advancing imitation_i (first step sees [0,0]), "
                 "sim_test advances first; both run 27 steps per period. With sim_test's clock delayed one step "
                 f"to match, the lockstep RMS is {rms_aligned:.4f} rad (diagnostic, not used for match)")
    if obs_diff:
        worst = sorted(obs_diff.items(), key=lambda kv: -kv[1])[:3]
        notes.append("largest observation-block RMS differences (runtime - sim_test, same state): " + ", ".join(f"{k} {v}" for k, v in worst))
    if log["rt_fell"]:
        notes.append(f"the runtime stack fell at {len(rt_t) / 50:.2f} s")
    if fr["fell"]:
        notes.append(f"sim_test fell at {len(fq) / 50:.2f} s in the free run")
    if n < n_total:
        notes.append(f"compared {n} of {n_total} steps")
    report = {
        "skill": skill["name"], "version": skill["version"], "match": match, "match_clock_aligned": match_aligned,
        "rms_target_diff_rad": round(rms, 5), "max_diff_rad": round(max_diff, 5), "settle_rms_rad": round(rms_settle, 5),
        "rms_clock_aligned_rad": round(rms_aligned, 5),
        "threshold_rad": thr["sim2sim_rms_rad"], "per_joint_rms_rad": per_joint, "obs_block_rms": obs_diff,
        "free_run": {"joint_pos_rms_rad": round(q_rms, 5), "runtime_fell": log["rt_fell"], "sim_test_fell": fr["fell"], "steps_compared": m},
        "tracking": tracking,
        "command": list(map(float, cmd3)), "seconds": seconds, "settle_s": settle, "seed": seed, "steps": n,
        "runtime": {"script": os.path.relpath(SCRIPT, sk.ROOT), "cutoff_frequency": cutoff,
                    "action_scale": rt["action_scale"], "config": os.path.relpath(CONFIG, sk.ROOT),
                    "wall_s": round(rt_wall, 2)},
        "sim_test": {"lpf_hz": rt["lpf_hz"], "max_vel_rad_s": rt["max_vel"], "torque_nm": rt["torque_nm"], "head_mode": rt["head_mode"]},
        "notes": notes, "created": sk.now_iso(), "tool": "tools/sim2sim.py",
    }
    plot_path = os.path.join(skill_dir, "report", "sim2sim.png")
    try:
        plot(report, rt_t, sh_t, diff, n_settle, plot_path)
    except Exception as e:                           # the numbers matter, the picture is a bonus
        notes.append(f"plot failed: {e}")
    return report, None


def plot(report, rt_t, sh_t, diff, n_settle, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(rt_t)) / 50.0
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), dpi=120, sharex=True)
    ax = axes[0]
    for i, j in ((2, "left_hip_pitch"), (3, "left_knee"), (4, "left_ankle")):
        ax.plot(t, np.degrees(rt_t[:, i]), lw=1.0, label=f"{j} runtime")
        ax.plot(t, np.degrees(sh_t[:, i]), lw=1.0, ls="--", label=f"{j} sim_test")
    ax.axvline(n_settle / 50.0, color="#888", lw=0.8)
    ax.set_ylabel("target (deg)")
    ax.legend(fontsize=6, ncol=3)
    ax.set_title(f"{report['skill']} v{report['version']} sim-to-sim: lockstep target RMS {report['rms_target_diff_rad']:.4f} rad "
                 f"({'match' if report['match'] else 'MISMATCH'}, threshold {report['threshold_rad']})", fontsize=10)
    ax = axes[1]
    ax.plot(t, np.degrees(np.sqrt((diff ** 2).mean(1))), color="#e76f51", lw=1.0)
    ax.axhline(np.degrees(report["threshold_rad"]), color="#888", lw=0.8, ls="--")
    ax.set_ylabel("|target diff| RMS over joints (deg)")
    ax.set_xlabel("s  (command starts at the vertical line)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("skill_dir")
    ap.add_argument("--cmd", default=None, help="vx,vy,wz (default: forward at the trained vx max, or 0,0,0)")
    ap.add_argument("--seconds", type=float, default=5.0, help="commanded window compared")
    ap.add_argument("--settle", type=float, default=2.0, help="seconds standing before the command")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--no-status", action="store_true")
    ap.add_argument("--runtime-cutoff", type=float, help="diagnostic: launch the runtime with this low-pass cutoff (Hz) instead of the package's")
    a = ap.parse_args(argv)
    skill_dir = os.path.abspath(a.skill_dir)
    skill = sk.load_skill(skill_dir)
    if a.cmd:
        cmd3 = [float(x) for x in a.cmd.split(",")]
    else:
        vx = skill.get("commands", {}).get("vx")
        cmd3 = [vx[1], 0.0, 0.0] if vx else [0.0, 0.0, 0.0]
    out = os.path.abspath(a.out) if a.out else os.path.join(skill_dir, "report", "sim2sim.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    report, skipped = run_comparison(skill_dir, cmd3, a.seconds, a.settle, a.seed, video=not a.no_video,
                                     runtime_cutoff=a.runtime_cutoff)
    if a.runtime_cutoff is not None and not a.out:
        out = os.path.join(skill_dir, "report", f"sim2sim_runtime_cutoff{a.runtime_cutoff:g}.json")
    if skipped:
        sk.dump_json({"skill": skill["name"], "version": skill["version"], "match": None, "rms_target_diff_rad": None,
                      "max_diff_rad": None, "notes": [skipped], "created": sk.now_iso(), "tool": "tools/sim2sim.py"}, out)
        print(skipped)
        return 0
    sk.dump_json(report, out)
    print(f"\nsim2sim {report['skill']} v{report['version']} cmd {cmd3} over {a.seconds} s (seed {a.seed})")
    print(f"  lockstep target RMS {report['rms_target_diff_rad']:.4f} rad (max {report['max_diff_rad']:.4f}, settle window {report['settle_rms_rad']:.4f}, "
          f"clock-aligned {report['rms_clock_aligned_rad']:.4f}), threshold {report['threshold_rad']}")
    print(f"  per joint: " + " ".join(f"{k.replace('left_', 'L').replace('right_', 'R')}={v:.3f}" for k, v in report["per_joint_rms_rad"].items()))
    print(f"  obs blocks: " + " ".join(f"{k}={v}" for k, v in report["obs_block_rms"].items()))
    print(f"  free run: joint RMS {report['free_run']['joint_pos_rms_rad']:.4f} rad, runtime fell {report['free_run']['runtime_fell']}, "
          f"sim_test fell {report['free_run']['sim_test_fell']}; tracking {report['tracking']}")
    print(f"  {'MATCH' if report['match'] else 'MISMATCH'} -> {out}")
    if report["match"] and not a.no_status and out.endswith(os.path.join("report", "sim2sim.json")):
        if skill.get("status") == "battery_passed":
            skill["status"] = "sim2sim_passed"
            skill.setdefault("status_history", []).append({"at": sk.now_iso(), "from": "battery_passed", "to": "sim2sim_passed", "by": "tools/sim2sim.py"})
            sk.dump_json(skill, os.path.join(skill_dir, "skill.json"))
            print("status: battery_passed -> sim2sim_passed")
    return 0 if report["match"] else 2


if __name__ == "__main__":
    sys.exit(main())
