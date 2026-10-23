#!/usr/bin/env python3
"""Measured joint stops (joint_limits_probe.py, 2026-10-14) against the simulator's joint ranges."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
real = json.load(open("/home/hermes/src/mini-duck-droid/joint_limits/real_limits_20261014.json"))
m = mujoco.MjModel.from_xml_path("train_pg/playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash_collision.xml")
home = dict(zip([m.actuator(i).name for i in range(m.nu)], m.keyframe("home").ctrl))
names = list(real)
fig, ax = plt.subplots(figsize=(8.5, 5.2))
for i, n in enumerate(names):
    y = len(names) - i
    s0, s1 = np.degrees(m.jnt_range[m.joint(n).id])
    r0, r1 = np.degrees(real[n])
    ax.plot([s0, s1], [y + 0.15] * 2, color="#9aa4b2", lw=6, solid_capstyle="butt", label="simulator" if i == 0 else None)
    ax.plot([r0, r1], [y - 0.15] * 2, color="#e07b00", lw=6, solid_capstyle="butt", label="his real stops" if i == 0 else None)
    ax.plot(np.degrees(home[n]), y, "k|", ms=14, mew=2, label="standing pose" if i == 0 else None)
ax.set_yticks(range(len(names), 0, -1)); ax.set_yticklabels([n.replace("_", " ") for n in names])
ax.set_xlabel("joint angle (degrees)"); ax.set_xlim(-170, 170); ax.axvline(0, color="#ddd", lw=1, zorder=0)
ax.legend(frameon=False, loc="lower right", fontsize=9)
ax.set_title("How far each joint bends: simulator vs his real stops")
fig.tight_layout(); fig.savefig("report1/limits.png"); print("saved")
