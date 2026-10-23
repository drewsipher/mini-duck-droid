#!/usr/bin/env python3
"""Graphs for report 1 (today's training runs). Writes PNGs into report1/."""
import glob
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import eval_policy as ev
from autohead import AutoHead
from sim_test import PG, Sim, run

OUT = "report1"
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C_OLD, C_NEW, C_3B, C_REAL = "#7a7a7a", "#e07b00", "#3a7bd5", "#2a9d5c"
STAGE2 = "train_pg/runs/stage2_head/2026_10_13_181321_100761600.onnx"
STAGE3B = sorted(glob.glob("train_pg/runs/stage3b_y02/*.onnx"), key=lambda p: int(p.split("_")[-1][:-5]))[-1]
data = {}

# 1. mass split
fig, ax = plt.subplots(figsize=(6, 2.6))
parts = ["Head + neck", "Body", "Left leg", "Right leg"]
model = [0.582, 0.699, 0.414, 0.414]
yours = [0.5, 0.82, 0.21, 0.21]          # the user's estimate: head 1/4, body 1/2, legs 1/8 of ~1.65 kg
x = np.arange(4)
ax.bar(x - 0.2, model, 0.4, label="Simulation model (2.1 kg)", color=C_NEW)
ax.bar(x + 0.2, yours, 0.4, label="Your estimate (~1.65 kg)", color=C_OLD)
ax.set_xticks(x, parts); ax.set_ylabel("kg"); ax.legend(frameon=False)
ax.set_title("Where his weight is")
fig.tight_layout(); fig.savefig(f"{OUT}/mass.png"); plt.close(fig)

# 2. training curves
from tensorflow.python.summary.summary_iterator import summary_iterator  # noqa: E402


def curves(run):
    f = sorted(glob.glob(f"train_pg/runs/{run}/events.out.tfevents*"))[-1]
    d = {}
    for e in summary_iterator(f):
        for v in e.summary.value:
            val = v.simple_value if v.HasField("simple_value") else (v.tensor.float_val[0] if v.tensor.float_val else None)
            d.setdefault(v.tag, {})[e.step] = val
    return d


runs = [("realservo_v1", "1. Weak servos from scratch (stopped)", "#b23b3b", 0),
        ("stage1_walk", "2. Stage 1: full strength", "#2a9d5c", 0),
        ("stage2_head", "3. Stage 2: real strength + head follows commands", C_NEW, 150.7),
        ("stage3_sides", "4. Stage 3: side-step fix (collapsed)", "#8e44ad", 250.7),
        ("stage3b_y02", "5. Stage 3b: wider side-step range", C_3B, 250.7)]
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
for run_name, label, col, offset in runs:
    try:
        d = curves(run_name)
    except IndexError:
        continue
    st = sorted(d["eval/episode_reward"])
    xs = [offset + s / 1e6 for s in st]
    axs[0].plot(xs, [d["eval/episode_reward"][s] for s in st], "-o", ms=3, color=col, label=label)
    axs[1].plot(xs, [d["eval/avg_episode_length"][s] / 10 for s in st], "-o", ms=3, color=col)
    axs[2].plot(xs, [d["eval/episode_reward/imitation"][s] for s in st], "-o", ms=3, color=col)
axs[0].set_title("Training score"); axs[1].set_title("Survives (% of a 20 s attempt)")
axs[2].set_title("Copying the reference walk")
for a in axs:
    a.set_xlabel("million steps (stages continue from the one before)")
axs[0].legend(fontsize=7.5, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/training.png"); plt.close(fig)

# 3. scorecard at 2.0 N.m
brains = [("Current brain", "BEST_WALK_ONNX_2.onnx", False, C_OLD), ("Stage 2", STAGE2, True, C_NEW),
          ("Stage 3b", STAGE3B, True, C_3B)]
moves = [("fwd", [0.15, 0, 0], "forward"), ("back", [-0.15, 0, 0], "backward"),
         ("left", [0, 0.2, 0], "side-step L"), ("right", [0, -0.2, 0], "side-step R"),
         ("turnL", [0, 0, 1.0], "turn L"), ("turnR", [0, 0, -1.0], "turn R")]
score = {}
for name, path, head, _ in brains:
    ev.HEAD_FROM_CMD = head
    row = {}
    for key, cmd, _ in moves:
        r = run(ev.make(path, 2.0), cmd + [0, 0, 0, 0], 6)
        row[key] = r["yaw"] if "turn" in key else (r["left"] if key in ("left", "right") else r["fwd"])
    row["eval"] = ev.evaluate(path, 2.0)
    score[name] = row
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
x = np.arange(4)
for i, (name, _, _, col) in enumerate(brains):
    axs[0].bar(x + (i - 1) * 0.27, [abs(score[name][k]) for k, _, _ in moves[:4]], 0.27, color=col, label=name)
    axs[1].bar(np.arange(2) + (i - 1) * 0.27, [abs(score[name][k]) for k, _, _ in moves[4:]], 0.27, color=col)
axs[0].set_xticks(x, [m[2] for m in moves[:4]]); axs[0].set_ylabel("speed reached, m/s")
axs[0].axhline(0.15, ls=":", c="k", lw=0.8); axs[0].text(3.45, 0.152, "asked 0.15", fontsize=7, ha="right")
axs[1].set_xticks(np.arange(2), ["turn left", "turn right"]); axs[1].set_ylabel("deg/s")
axs[1].axhline(57, ls=":", c="k", lw=0.8); axs[1].text(1.45, 58, "asked 57", fontsize=7, ha="right")
axs[0].legend(frameon=False); axs[0].set_title("Speed from a standstill (sim, servos at 2.0 N·m)")
axs[1].set_title("Turning")
fig.tight_layout(); fig.savefig(f"{OUT}/speeds.png"); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))
names = [b[0] for b in brains]; cols = [b[3] for b in brains]
head = np.array([score[n]["eval"]["head"] for n in names])
for i, (n, c) in enumerate(zip(names, cols)):
    axs[0].bar(np.arange(4) + (i - 1) * 0.27, head[i], 0.27, color=c, label=n)
axs[0].set_xticks(np.arange(4), ["neck", "head pitch", "head yaw", "head roll"]); axs[0].axhline(0, c="k", lw=0.6)
axs[0].set_ylabel("deg from straight"); axs[0].set_title("Head when standing, untouched"); axs[0].legend(frameon=False, fontsize=8)
axs[1].bar(names, [score[n]["eval"]["fwd_sway"][1] for n in names], color=cols)
axs[1].set_ylabel("deg/s"); axs[1].set_title("Front-back wobble walking")
axs[2].bar(names, [score[n]["eval"]["push"] for n in names], color=cols)
axs[2].set_ylim(0, 8); axs[2].set_title("Shoves survived (of 8)")
fig.tight_layout(); fig.savefig(f"{OUT}/scorecard.png"); plt.close(fig)
data["score"] = {n: {k: (float(v) if not isinstance(v, dict) else None) for k, v in score[n].items()} for n in names}

# 4. servo strength vs behaviour (current brain)
lims = [3.35, 2.5, 2.0, 1.86, 1.5]
res = {k: [] for k in ("fwd", "back", "pitch")}
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("sc", "smooth_cmp_lib.py"); sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
ev.HEAD_FROM_CMD = False
for L in lims:
    res["fwd"].append(run(ev.make("BEST_WALK_ONNX_2.onnx", L), [0.15, 0, 0, 0, 0, 0, 0], 6)["fwd"])
    res["back"].append(-run(ev.make("BEST_WALK_ONNX_2.onnx", L), [-0.15, 0, 0, 0, 0, 0, 0], 6)["fwd"])
    s = ev.make("BEST_WALK_ONNX_2.onnx", L); o = []
    for _ in range(100):
        s.control(np.zeros(7))
    for _ in range(400):
        oo, _ = s.control(np.array([0.15, 0, 0, 0, 0, 0, 0], float)); o.append(oo)
    res["pitch"].append(sc.bands(np.array(o[100:], dtype=np.float64)[:, 0:2])[0][1])
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
axs[0].plot(lims, res["fwd"], "-o", color=C_NEW, label="forward"); axs[0].plot(lims, res["back"], "-o", color=C_3B, label="backward")
axs[0].set_xlabel("servo strength limit in sim, N·m"); axs[0].set_ylabel("speed, m/s"); axs[0].invert_xaxis()
axs[0].axvspan(1.8, 2.6, color="#ffe9c7", zorder=0); axs[0].text(2.2, max(res["fwd"]) * 0.98, "his real servos\n(to be measured)", ha="center", fontsize=8)
axs[0].legend(frameon=False); axs[0].set_title("Current brain: weaker servos, slower walk")
axs[1].plot(lims, res["pitch"], "-o", color="#b23b3b", label="sim")
axs[1].axhline(20.3, color=C_REAL, ls="--", label="your duck (walk 7)")
axs[1].set_xlabel("servo strength limit in sim, N·m"); axs[1].set_ylabel("front-back wobble, deg/s"); axs[1].invert_xaxis()
axs[1].legend(frameon=False); axs[1].set_title("…and more wobble, toward what your duck shows")
fig.tight_layout(); fig.savefig(f"{OUT}/torque.png"); plt.close(fig)
data["torque"] = {"lims": lims, **{k: [float(x) for x in v] for k, v in res.items()}}

# 5. head + body over the routine (current vs stage 2 with auto head)
ROUT = [("stand", 2, [0, 0, 0]), ("fwd", 4, [0.15, 0, 0]), ("back", 4, [-0.15, 0, 0]), ("turn L", 3.5, [0, 0, 1.0]),
        ("turn R", 3.5, [0, 0, -1.0]), ("step L", 3.5, [0, 0.2, 0]), ("step R", 3.5, [0, -0.2, 0])]
fig, axs = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True)
for ax, (name, path, mode, col) in zip(axs, [("Current brain", "BEST_WALK_ONNX_2.onnx", "plain", C_OLD),
                                              ("Stage 2 + auto head", STAGE2, "auto", C_NEW)]):
    s = Sim(path, f"{PG}/xmls/scene_flat_terrain_backlash.xml", 0.0,
            extra={"head_from_cmd": mode == "auto", "head_add_cmd": mode == "plain"})
    s.m.actuator_forcerange[:, :] = [-2.0, 2.0]
    ah = AutoHead(); hd, t, edges = [], [], [0]
    for lab, sec, cmd in ROUT:
        for _ in range(int(sec * 50)):
            head = list(ah(*cmd)) if mode == "auto" else [0, 0, 0, 0]
            o, _ = s.control(np.array(cmd + head, float))
            hd.append(np.degrees(s.d.qpos[s.qadr[5:9]])); t.append(len(t) / 50)
        edges.append(t[-1])
    hd = np.array(hd)
    for j, (lab, c) in enumerate(zip(["neck pitch", "head pitch", "head yaw", "head roll"],
                                     ["#1f77b4", "#9467bd", "#e07b00", "#2a9d5c"])):
        ax.plot(t, hd[:, j], color=c, lw=1.4, label=lab)
    for e, (lab, _, _) in zip(edges[:-1], ROUT):
        ax.axvline(e, color="#ccc", lw=0.8); ax.text(e + 0.1, ax.get_ylim()[1] if False else 22, lab, fontsize=8, color="#555")
    ax.set_ylim(-25, 26); ax.set_ylabel("head joint, deg"); ax.set_title(name, loc="left")
axs[0].legend(ncol=4, frameon=False, fontsize=8, loc="lower left"); axs[1].set_xlabel("seconds")
fig.tight_layout(); fig.savefig(f"{OUT}/head_trace.png"); plt.close(fig)

json.dump(data, open(f"{OUT}/data.json", "w"), indent=1, default=float)
print("plots done")
