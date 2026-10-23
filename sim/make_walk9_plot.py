#!/usr/bin/env python3
"""Walk 9 (after tightening screws, phase offset 0) vs walk 4 (the good run): body tilt over time."""
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
def medf(x, k=25):
    p = np.pad(x, (k // 2, k - 1 - k // 2), mode="edge"); return np.median(sliding_window_view(p, k), axis=1)
def load(w):
    o = np.array(pickle.load(open(f"/home/hermes/src/mini-duck-droid/walk_recordings/obs_walk{w}.pkl", "rb")))
    acc, cmd = o[:, 3:6], o[:, 6:13]
    return medf(np.degrees(np.arctan2(acc[:, 0], acc[:, 2]))), cmd
COL = {"forward": "#e07b00", "back": "#3a7bd5", "turn": "#8e5bd5", "side-step": "#2a8a52"}
def kind(c):
    if abs(c[2]) > 0.05: return "turn"
    if abs(c[1]) > 0.02: return "side-step"
    if c[0] > 0.02: return "forward"
    if c[0] < -0.02: return "back"
    return None
fig, axs = plt.subplots(2, 1, figsize=(10, 5.4), sharey=True)
for ax, w, title in [(axs[0], 4, "Walk 4 (Oct 13, the good run)"), (axs[1], 9, "Walk 9 (today: screws tightened, gait timing 0)")]:
    tilt, cmd = load(w)
    t = np.arange(len(tilt)) / 50
    k = [kind(c) for c in cmd]
    for name, col in COL.items():
        m = np.array([x == name for x in k])
        ax.fill_between(t, -40, 40, where=m, color=col, alpha=0.13, lw=0, label=name)
    ax.plot(t, np.clip(tilt, -40, 40), color="#222", lw=1)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_ylim(-40, 40); ax.set_title(title, loc="left"); ax.set_ylabel("leaning back (°)")
    ax.set_xlim(0, min(t[-1], 120))
    if w == 9:
        ax.set_xlabel("seconds of recording (pauses cut out)")
axs[0].legend(frameon=False, ncol=4, fontsize=9, loc="lower right")
axs[1].text(0.5, 33, "falls: line hits the top", fontsize=9, color="#b23b3b")
fig.tight_layout(); fig.savefig("report1/walk9.png"); print("saved")
