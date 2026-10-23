#!/usr/bin/env python3
"""Strength test graphs for the report.
    make_strength_plot.py out.png "label=file.csv" ...   (last one is the valid, clamped test)"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
KP = 17.11 * np.pi / 180        # sim kp, N.m per degree of error
STYLE = [("o", "#b8b8b8"), ("s", "#8fb3e0"), ("D", "#e07b00")]
tests = []
for spec in sys.argv[2:]:
    label, path = spec.split("=", 1)
    r = list(csv.DictReader(open(path)))
    tests.append((label, *(np.array([float(x[k]) for x in r]) for k in ("press_deg", "torque_nm", "short_deg", "load_pct"))))

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.7))
a.axhspan(1.6, 2.6, color="#999999", alpha=0.13)
a.text(0.4, 2.35, "datasheet / sim range for the limit (not reached)", color="#555", fontsize=8.5)
for (label, press, tq, short, load), (m, c) in zip(tests, STYLE):
    a.plot(press, tq, m + "-", color=c, lw=2 if c == "#e07b00" else 1.3, label=label)
a.set_xlabel("how far the ankle was told to press (degrees)"); a.set_ylabel("torque at the ankle (N·m)")
a.set_ylim(0, 2.7); a.set_xlim(0, 16); a.set_title("Right ankle pressing the scale"); a.legend(frameon=False, loc="center left", fontsize=8.5)

label, press, tq, short, load = tests[-1]
fit = (tq @ short) / (short @ short)
e = np.linspace(0, 3, 10)
b.plot(e, KP * e, "--", color="#7a7a7a", label=f"simulator servo ({KP:.2f} N·m per °)")
b.plot(e, fit * e, "-", color="#e07b00", alpha=0.5, label=f"real ankle fit ({fit:.2f} N·m per °)")
for (lab, p_, t_, s_, l_), (m, c) in zip(tests, STYLE):
    b.plot(s_, t_, m, color=c, ms=6)
b.set_xlabel("how far the servo was short of its target (degrees)"); b.set_ylabel("torque (N·m)")
b.set_title("Stiffness: real vs simulator"); b.legend(frameon=False, loc="upper left", fontsize=8.5)
b.set_xlim(0, 3); b.set_ylim(0, 1.3)
fig.tight_layout(); fig.savefig(sys.argv[1]); print("saved", sys.argv[1])
print(f"real stiffness {fit:.3f} N.m/deg = {fit*180/np.pi:.1f} N.m/rad = {fit/KP:.2f}x sim; load% at peak {load[-1]}")
