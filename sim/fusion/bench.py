"""Stand <-> walk fusion experiments on frozen experts (CPU, strict-battery physics randomisation).

    cd sim && MUJOCO_GL=egl train_pg/.venv/bin/python fusion/bench.py <experiment> --stand S.onnx --walk W.onnx --out DIR

    phase-in    stand 1.5 s, then walk with its clock started at each phase: success per start phase
    phase-out   walk 2 s, then a hard switch to stand at each walk phase: success per switch phase
    scenario    stand -> walk -> stand by command: hard switch / 0.3 s cross-fade (--walk-start, --ds-only)
    blend       hold a fixed mix w of the two for 6 s: output blend vs weight blend, success and speed per w
    stress      12 s, the command flips every 0.5-2 s, pushes on: hard switch / cross-fade / learned gate (--gate)

Each writes DIR/<experiment>.json and .png (and scenario a video of one episode).
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CLIPS = os.path.join(HERE, "..", "..", "tracker", "data", "clips")
A = None          # parsed args, set in each worker
SIM = None


def setup(a):
    global A, SIM
    import mujoco
    import fusion_sim as F
    A = a
    m0 = mujoco.MjModel.from_xml_path(F.ET.SCENE_XML)
    m0.opt.timestep = F.R.SIM_DT
    stand = F.Expert("stand", a.stand, os.path.join(CLIPS, "Stand.json"), m0)
    walk = F.Expert("walk", a.walk, os.path.join(CLIPS, "Walk.json"), m0)
    SIM = F.FusionSim(m0, [stand, walk])


def episodes(seed):
    return np.random.default_rng(seed)


def summary(r, t_from=None, t_to=None):
    tr = r["trace"]
    rows = [x for x in tr if (t_from is None or x["t"] >= t_from) and (t_to is None or x["t"] <= t_to)]
    return {"success": r["success"], "seconds": r["seconds"],
            "vx": float(np.mean([x["vx"] for x in rows])) if rows else None,
            "x_end": tr[-1]["x"], "y_end": tr[-1]["y"], "jump_max": max(x["jump"] for x in tr),
            "tilt_max": max(x["tilt"] for x in tr)}


# ---------------------------------------------------------------- one job per (setting, seed)
def job_phase_in(args):
    import fusion_sim as F
    p, seed = args
    c = F.Commanded([(0, "stand"), (1.5, "walk")], walk_start=p)
    r = SIM.run(episodes(1000 * seed + int(p * 1000)), c, 4.5, start=("stand", 0))
    return p, seed, summary(r, 2.5)


def job_phase_out(args):
    import fusion_sim as F
    p, seed = args
    ew = SIM.experts[1]
    t_sw = 2.0 + p * ew.cycle * F.R.CTRL_DT                           # walk from frame 0: phase p is reached at 2 s + p * T
    c = F.Commanded([(0, "walk"), (t_sw, "stand")], ds_only=False)
    r = SIM.run(episodes(1000 * seed + int(p * 1000)), c, t_sw + 3.0, start=("walk", 0))
    return p, seed, summary(r, t_sw + 1.0)


SCHED = [(0, "stand"), (1.5, "walk"), (5.5, "stand")]


def controller(name, ramp, ds):
    import fusion_sim as F
    if name == "learned gate":
        return F.Gate(A.gate, SCHED)
    return F.Commanded(SCHED, ramp_s=ramp, walk_start=A.walk_start, ds_only=ds)


def job_scenario(args):
    name, ramp, ds, seed = args
    c = controller(name, ramp, ds)
    r = SIM.run(episodes(seed), c, 8.5, start=("stand", 0), pushes=A.pushes)
    s = summary(r, 2.5, 5.5)
    s["switch_times"] = getattr(c, "switch_times", None)
    s["settle_vx"] = float(np.max(np.abs([x["vx"] for x in r["trace"] if x["t"] >= 7.0]))) if r["success"] else None
    return name, seed, s


def stress_schedule(seed):
    rng = np.random.default_rng(10_000 + seed)
    t, cmd, sched = 0.0, "stand", []
    while t < 12.0:
        sched.append((t, cmd))
        t += rng.uniform(0.5, 2.0)
        cmd = "walk" if cmd == "stand" else "stand"
    return sched


def job_stress(args):
    import fusion_sim as F
    name, seed = args
    sched = stress_schedule(seed)
    c = F.Gate(A.gate, sched) if name == "learned gate" else F.Commanded(sched, ramp_s=0.3 if "fade" in name else 0.0, ds_only=True)
    r = SIM.run(episodes(seed), c, 12.0, start=("stand", 0), pushes=True)
    tr = r["trace"]
    walk_rows = [x for x in tr if c.command(x["t"]) == "walk"]
    stop_rows = [x for x in tr if c.command(x["t"]) == "stand"]
    return name, seed, {"success": r["success"], "seconds": r["seconds"],
                        "vx_walk": float(np.mean([x["vx"] for x in walk_rows])) if walk_rows else None,
                        "speed_stop": float(np.mean([abs(x["vx"]) for x in stop_rows])) if stop_rows else None,
                        "tilt_max": max(x["tilt"] for x in tr)}


def stress(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import fusion_sim as F
    setup(a)
    names = ["hard switch", "0.3 s cross-fade"] + (["learned gate"] if a.gate else [])
    res = pool_map(job_stress, [(n, s) for n in names for s in range(a.episodes)], a)
    rows = []
    for n in names:
        rs = [x for v, _, x in res if v == n]
        ok = [x for x in rs if x["success"]]
        rows.append({"variant": n, "success": len(ok), "total": len(rs),
                     "mean_seconds": float(np.mean([x["seconds"] for x in rs])),
                     "vx_walk": float(np.mean([x["vx_walk"] for x in rs if x["vx_walk"] is not None])),
                     "speed_stop": float(np.mean([x["speed_stop"] for x in rs if x["speed_stop"] is not None])),
                     "tilt_max_deg": float(np.degrees(np.mean([x["tilt_max"] for x in rs])))})
        print(rows[-1])
    out = {"experiment": "stress", "rows": rows, "stand": a.stand, "walk": a.walk, "gate": a.gate,
           "note": "12 s from standing; command flips every 0.5-2 s (schedule per seed); recipe pushes on"}
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    cols = ["#3a7bd5", "#e08a2e", "#2a9d5c"][:len(rows)]
    axs[0].bar(range(len(rows)), [r["success"] / r["total"] for r in rows], color=cols); axs[0].set_ylabel("survived 12 s"); axs[0].set_ylim(0, 1.05)
    axs[1].bar(range(len(rows)), [r["vx_walk"] for r in rows], color=cols); axs[1].set_ylabel("mean vx while 'walk' (m/s)")
    axs[2].bar(range(len(rows)), [r["speed_stop"] for r in rows], color=cols); axs[2].set_ylabel("mean |vx| while 'stop' (m/s)")
    for ax in axs:
        ax.set_xticks(range(len(rows))); ax.set_xticklabels([r["variant"] for r in rows], rotation=15, fontsize=8)
    fig.tight_layout()
    if a.gate:                                      # film the gate on seed 0
        c = F.Gate(a.gate, stress_schedule(0))
        vid = F.Video(SIM.m0, "learned gate, stress: command flips, pushes")
        SIM.run(episodes(0), c, 12.0, start=("stand", 0), pushes=True,
                render=lambda d, row, fell: vid(d, row, fell, cmd="command: " + c.command(row["t"])))
        vid.save(os.path.join(a.out, "stress_gate.mp4"))
    return out, fig


def job_blend(args):
    import fusion_sim as F
    from experts import MLP
    kind, w, seed = args
    W = [1 - w, w]
    if kind == "output":
        r = SIM.run(episodes(seed), F.Fixed(W), 6.0, start=("stand", 0))
    else:
        net = SIM.experts[0].net.blend(SIM.experts[1].net, w)
        r = SIM.run(episodes(seed), F.Fixed(W), 6.0, start=("stand", 0), blend_net=(net, None))
    return kind, w, seed, summary(r, 2.0)


def pool_map(fn, jobs, a):
    with Pool(a.procs, initializer=setup, initargs=(a,)) as pool:
        return pool.map(fn, jobs, chunksize=1)


# ---------------------------------------------------------------- experiments
def phase_sweep(a, which):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup(a)
    cyc = SIM.experts[1].cycle
    phases = [i / cyc for i in range(0, cyc, a.phase_step)]
    jobs = [(p, s) for p in phases for s in range(a.episodes)]
    res = pool_map(job_phase_in if which == "in" else job_phase_out, jobs, a)
    per = {p: [r for q, _, r in res if q == p] for p in phases}
    rows = [{"phase": p, "success": sum(r["success"] for r in rs), "total": len(rs),
             "vx": float(np.mean([r["vx"] for r in rs if r["success"]])) if any(r["success"] for r in rs) else None}
            for p, rs in per.items()]
    ew = SIM.experts[1]
    out = {"experiment": f"phase-{which}", "stand": a.stand, "walk": a.walk, "episodes_per_phase": a.episodes, "rows": rows,
           "double_support_phases": [i / cyc for i in range(cyc) if ew.double_support(i)]}
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for p in out["double_support_phases"]:
        ax.axvspan(p - 0.5 / cyc, p + 0.5 / cyc, color="#9ad0a0", alpha=0.35, lw=0)
    ax.bar([r["phase"] for r in rows], [r["success"] / r["total"] for r in rows], width=0.9 / cyc * a.phase_step, color="#3a7bd5")
    ax.set_ylim(0, 1.05); ax.set_xlim(-0.02, 1.0)
    ax.set_xlabel("walk phase (0 = left foot lifts; green = both feet down)")
    ax.set_ylabel("success")
    ax.set_title("stand -> walk: walk clock started at this phase" if which == "in" else "walk -> stand: hard switch at this phase")
    fig.tight_layout()
    return out, fig


def scenario(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import fusion_sim as F
    setup(a)
    variants = [("hard switch, double support", 0.0, True), ("0.3 s cross-fade, double support", 0.3, True),
                ("hard switch, any phase", 0.0, False)] + ([("learned gate", None, None)] if a.gate else [])
    jobs = [(n, r, ds, s) for n, r, ds in variants for s in range(a.episodes)]
    res = pool_map(job_scenario, jobs, a)
    rows = []
    for n, r, ds in variants:
        rs = [x for v, _, x in res if v == n]
        ok = [x for x in rs if x["success"]]
        rows.append({"variant": n, "ramp_s": r, "ds_only": ds, "success": len(ok), "total": len(rs),
                     "walk_vx": float(np.mean([x["vx"] for x in ok])) if ok else None,
                     "jump_max_rad": float(np.mean([x["jump_max"] for x in rs])),
                     "settle_vx_max": float(np.mean([x["settle_vx"] for x in ok])) if ok else None,
                     "fail_seeds": [s for v, s, x in res if v == n and not x["success"]]})
        print(rows[-1])
    out = {"experiment": "scenario", "schedule": SCHED, "walk_start": a.walk_start, "pushes": a.pushes, "rows": rows,
           "stand": a.stand, "walk": a.walk}
    # one episode of the first variant, traced and filmed
    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    shown = [variants[0], variants[1]] + ([variants[-1]] if a.gate else [])
    for (n, r, ds), col in zip(shown, ["#3a7bd5", "#e08a2e", "#2a9d5c"]):
        c = controller(n, r, ds)
        vid = F.Video(SIM.m0, f"{n}  (command: stand / walk / stand)") if n == shown[-1][0] else None
        rr = SIM.run(episodes(0), c, 8.5, start=("stand", 0), pushes=a.pushes,
                     render=(lambda d, row, fell, c=c: vid(d, row, fell, cmd="command: " + c.command(row["t"]))) if vid else None)
        if vid:
            vid.save(os.path.join(a.out, "scenario.mp4"))
        tr = rr["trace"]
        t = [x["t"] for x in tr]
        axs[0].plot(t, [x["w"][1] for x in tr], color=col, label=n)
        axs[1].plot(t, [x["vx"] for x in tr], color=col, lw=0.8)
        axs[2].plot(t, [np.degrees(x["tilt"]) for x in tr], color=col, lw=0.8)
    for ax, lab in zip(axs, ["walk weight", "forward speed m/s", "tilt deg"]):
        ax.set_ylabel(lab)
        for ts, _ in SCHED[1:]:
            ax.axvline(ts, color="#888", ls=":", lw=0.8)
    axs[0].legend(fontsize=8); axs[2].set_xlabel("t (s); dotted = command changes")
    fig.tight_layout()
    return out, fig


def blend(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup(a)
    ws = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
    jobs = [(k, w, s) for k in ("output", "weights") for w in ws for s in range(a.episodes)]
    res = pool_map(job_blend, jobs, a)
    rows = []
    for k in ("output", "weights"):
        for w in ws:
            rs = [x for kk, ww, _, x in res if kk == k and ww == w]
            ok = [x for x in rs if x["success"]]
            rows.append({"kind": k, "w_walk": w, "success": len(ok), "total": len(rs),
                         "vx": float(np.mean([x["vx"] for x in ok])) if ok else None})
    out = {"experiment": "blend", "rows": rows, "stand": a.stand, "walk": a.walk,
           "note": "fixed mix for 6 s from the stand pose; weights = one network with interpolated weights tracking the blended reference"}
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for k, col in (("output", "#3a7bd5"), ("weights", "#d5503a")):
        rr = [r for r in rows if r["kind"] == k]
        axs[0].plot(ws, [r["success"] / r["total"] for r in rr], "o-", color=col, label=f"{k} blend")
        axs[1].plot(ws, [r["vx"] if r["vx"] is not None else np.nan for r in rr], "o-", color=col)
    axs[0].set_ylabel("success (6 s upright)"); axs[1].set_ylabel("mean forward speed m/s")
    for ax in axs:
        ax.set_xlabel("walk weight w (0 = stand, 1 = walk)")
    axs[0].set_ylim(-0.05, 1.05); axs[0].legend(fontsize=8)
    fig.tight_layout()
    return out, fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["phase-in", "phase-out", "scenario", "blend", "stress"])
    ap.add_argument("--stand", required=True)
    ap.add_argument("--walk", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--phase-step", type=int, default=2, help="sweep every Nth walk frame")
    ap.add_argument("--walk-start", type=float, default=0.0)
    ap.add_argument("--pushes", action="store_true")
    ap.add_argument("--gate", default=None, help="the learned gate's policy.onnx (scenario: adds it as a variant)")
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    a.stand, a.walk, a.out = map(os.path.abspath, (a.stand, a.walk, a.out))
    a.gate = a.gate and os.path.abspath(a.gate)
    os.makedirs(a.out, exist_ok=True)
    out, fig = {"phase-in": lambda: phase_sweep(a, "in"), "phase-out": lambda: phase_sweep(a, "out"),
                "scenario": lambda: scenario(a), "blend": lambda: blend(a), "stress": lambda: stress(a)}[a.experiment]()
    json.dump(out, open(os.path.join(a.out, f"{a.experiment}.json"), "w"), indent=1)
    fig.savefig(os.path.join(a.out, f"{a.experiment}.png"), dpi=110)
    print("wrote", os.path.join(a.out, f"{a.experiment}.json"))


if __name__ == "__main__":
    main()
