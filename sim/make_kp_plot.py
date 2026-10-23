#!/usr/bin/env python3
"""Current brain in sim with servos stiffened to match strength test 3 (numbers from kp_test.py, torque 2.3)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
k = [1.0, 1.3, 1.5, 1.7]
fwd, back = [0.118, 0.136, 0.138, 0.134], [0.051, 0.085, 0.090, 0.093]
turnL, turnR = [46, 55, 55, 56], [51, 62, 68, 70]
jit = [4.04, 4.87, 5.86, 6.47]          # front-back shake above 8 Hz, deg/s
fig, ax = plt.subplots(1, 3, figsize=(10, 3.1))
ax[0].plot(k, fwd, "o-", color="#e07b00", label="forward"); ax[0].plot(k, back, "s-", color="#3a7bd5", label="backward")
ax[0].set_ylabel("m/s"); ax[0].set_title("Walking speed"); ax[0].legend(frameon=False); ax[0].set_ylim(0, 0.16)
ax[1].plot(k, turnL, "o-", color="#e07b00", label="left"); ax[1].plot(k, turnR, "s-", color="#3a7bd5", label="right")
ax[1].set_ylabel("°/s"); ax[1].set_title("Turning speed"); ax[1].legend(frameon=False); ax[1].set_ylim(0, 80)
ax[2].plot(k, jit, "o-", color="#b23b3b"); ax[2].set_ylabel("°/s"); ax[2].set_title("Fast shake (jitter)"); ax[2].set_ylim(0, 7)
for a in ax:
    a.axvline(1.48, color="#2a8a52", ls=":", lw=1.5); a.set_xlabel("servo stiffness vs simulator (×)")
ax[2].text(1.5, 1.0, "your\nservos", color="#2a8a52", fontsize=9)
fig.tight_layout(); fig.savefig("report1/kp.png"); print("saved")
