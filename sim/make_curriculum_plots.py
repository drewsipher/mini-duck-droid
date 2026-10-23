#!/usr/bin/env python3
"""Training curves of the scaffold curriculum, stage after stage on one time axis.

    cd sim && train_pg/.venv/bin/python make_curriculum_plots.py [out_dir] [runs_dir ...]
    STAGES="s0@train_pg/runs s1@train_pg/runs s2a s2b s2c s2d s3" ... make_curriculum_plots.py report2 train_pg/runs3

Reads the tfevents of cur_<stage> in the first runs dir that has it (no tensorboard package
needed) and the earlier unscaffolded walking run as a reference. Every eval metric is an *episode sum*, so
it is divided by the episode length to give a per-step value; otherwise a duck that falls
early looks worse on everything.
"""
import glob
import os
import struct
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX.proto import event_pb2

OUT = sys.argv[1] if len(sys.argv) > 1 else "report2"
RUNS_DIRS = sys.argv[2:] or ["train_pg/runs"]
LABELS = {"s0": "s0: 0.7 g, held 50-80 %", "s1": "s1: 0.85 g, held 20-50 %", "s2": "s2: 1 g, held 0-20 %",
          "s2a": "s2a: 1 g, held 20-40 %, spring 3-6", "s2b": "s2b: held 5-20 %, spring 2-4",
          "s2c": "s2c: held 0-10 %, spring 1-2", "s2d": "s2d: not held, spring 0-1",
          "s3": "s3: real servos, no help", "h1": "h1: hand 3-10 %, spring 3-6, real servos, drop at 30 cm", "h2": "h2: hand, lead cost 5/m, model servos", "a2": "a2: hand, lead cost, real servos, ankles 20 Hz / 8 rad/s", "p1": "p1: puppet stride as reference, spring 3-6, real servos",
          "p2": "p2: puppet stride + s2a's recorded turn as reference, spring 3-6, real servos"}
COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#d62728", "#17becf"]
# A stage token may carry its own runs dir: "s0@train_pg/runs"; otherwise the first dir that
# has it is used. Pin the dir when the same stage name exists in several attempts.
STAGES = [(k.split("@")[0], LABELS.get(k.split("@")[0], k), COLORS[i % len(COLORS)], k.partition("@")[2] or None)
          for i, k in enumerate(os.environ.get("STAGES", "s0 s1 s2 s3").split())]
REFERENCE = ("stage1_walk", "earlier brain, trained unheld")


def find_run(name, only=None):
    for d in ([only] if only else RUNS_DIRS):
        if glob.glob(os.path.join(d, name, "events*")):
            return os.path.join(d, name)
    return None


def scalars(run_dir):
    out = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(run_dir, "events*"))):
        with open(path, "rb") as f:
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                (n,) = struct.unpack("<Q", hdr)
                f.read(4)
                ev = event_pb2.Event()
                ev.ParseFromString(f.read(n))
                f.read(4)
                for v in ev.summary.value:
                    if v.HasField("simple_value"):
                        out[v.tag].append((ev.step, v.simple_value))
    return {k: np.array(v) for k, v in out.items()}


PANELS = [  # (title, tag, per-step?, unit)
    ("Eval reward per step", "eval/episode_reward", True, ""),
    ("Episode length (1000 = never fell)", "eval/avg_episode_length", False, "steps"),
    ("Foot lift while swinging", "eval/episode_swing_peak", True, "m, mean over both feet"),
    ("Imitation of the reference gait", "eval/episode_reward/imitation", True, ""),
    ("Speed tracking (max 2.5)", "eval/episode_reward/tracking_lin_vel", True, ""),
    ("Turn tracking (max 6)", "eval/episode_reward/tracking_ang_vel", True, ""),
    ("Action change per step (smoothness cost)", "eval/episode_cost/action_rate", True, ""),
    ("Both feet off the floor", "eval/episode_scaffold/airborne", True, "fraction of steps"),
]

fig, axes = plt.subplots(2, 4, figsize=(18, 7.5))
offset = 0.0
bounds = []
have_any = False
for key, label, color, where in STAGES:
    run = find_run(f"cur_{key}", where)
    if run is None:
        continue
    sc = scalars(run)
    if "eval/episode_reward" not in sc:
        continue
    have_any = True
    steps = sc["eval/episode_reward"][:, 0] / 1e6
    length = sc["eval/avg_episode_length"][:, 1]
    for ax, (title, tag, per_step, unit) in zip(axes.flat, PANELS):
        if tag not in sc:
            continue
        y = sc[tag][:, 1] / (length if per_step else 1.0)
        ax.plot(offset + steps, y, "o-", color=color, label=label, ms=3)
    offset += steps[-1]
    bounds.append((offset, key))

if not have_any:
    sys.exit("no curriculum runs found under " + " ".join(RUNS_DIRS))

ref = scalars(find_run(REFERENCE[0]) or find_run(REFERENCE[0], "train_pg/runs") or "")
if "eval/episode_reward" in ref:
    length = ref["eval/avg_episode_length"][:, 1]
    for ax, (title, tag, per_step, unit) in zip(axes.flat, PANELS):
        if tag in ref:
            y = ref[tag][:, 1] / (length if per_step else 1.0)
            ax.axhline(np.mean(y[-3:]), color="gray", ls="--", lw=1, label=REFERENCE[1] + " (final)")

for ax, (title, tag, per_step, unit) in zip(axes.flat, PANELS):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("million steps" + (", stages end to end" if len(bounds) > 1 else ""))
    if unit:
        ax.set_ylabel(unit)
    for x, key in bounds[:-1]:
        ax.axvline(x, color="k", lw=0.6, alpha=0.4)
    ax.grid(alpha=0.3)
axes.flat[0].legend(fontsize=7, loc="lower right")
fig.suptitle(os.environ.get("PLOT_TITLE", "Scaffold curriculum, 2026-10-21: what the duck learned while being held, then let go step by step"), fontsize=13)
fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
name = os.environ.get("PLOT_NAME", "curriculum_curves.png")
fig.savefig(f"{OUT}/{name}", dpi=110)
print("saved", f"{OUT}/{name}")
