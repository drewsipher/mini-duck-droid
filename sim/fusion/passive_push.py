"""Passive push limits: the duck holding its home pose with NO policy, pushed once, until it falls.

    cd sim && MUJOCO_GL=egl PYTHONPATH=train_pg train_pg/.venv/bin/python fusion/passive_push.py --out fusion/runs/passive_push

The servos hold the home pose (position targets fixed). One push at t = 1 s: a change of base velocity of
<size> m/s thrown in <direction> (duck frame), delivered either as an instant velocity kick (as training
does) or as a constant force over <duration> s giving the same total change. Fall = base below
MIN_BASE_HEIGHT_M or tilted more than MAX_ORI_ERR_RAD within 4 s after the push. Nominal model.

This is the floor a push-recovery expert must beat and where its training range should start
(memory: test-passive-limits-first). Writes passive_push.json and .png (one panel per duration).
"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fusion_sim as F  # noqa: E402

R = F.R
DIRS = [("forward", 0), ("left-fwd", 45), ("left", 90), ("left-back", 135), ("backward", 180), ("right-back", 225), ("right", 270), ("right-fwd", 315)]
SIZES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
DURATIONS = [0.0, 0.1, 0.3, 0.6]        # 0 = instant kick


def episode(m, size, deg, duration, settle_s=1.0, after_s=4.0):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.keyframe("home").id)
    d.ctrl[:] = m.keyframe("home").ctrl
    mujoco.mj_forward(m, d)
    base = 1
    mass = float(np.sum(m.body_mass))
    steps_settle = int(settle_s / m.opt.timestep)
    steps_after = int(after_s / m.opt.timestep)
    th = np.radians(deg)                                           # heading is identity at the home pose
    u = np.array([np.cos(th), np.sin(th), 0.0])
    force_steps = max(1, int(round(duration / m.opt.timestep))) if duration > 0 else 0
    tilt_max = 0.0
    for t in range(steps_settle + steps_after):
        if t == steps_settle:
            if force_steps == 0:
                d.qvel[0:2] += size * u[:2]
            else:
                d.xfrc_applied[base, :3] = mass * size / duration * u
        if force_steps and t == steps_settle + force_steps:
            d.xfrc_applied[base, :3] = 0.0
        mujoco.mj_step(m, d)
        tilt = float(np.arccos(np.clip(d.xmat[base][8], -1, 1)))
        tilt_max = max(tilt_max, tilt)
        if d.qpos[2] < R.MIN_BASE_HEIGHT_M or tilt > R.MAX_ORI_ERR_RAD:
            return False, tilt_max, (t - steps_settle) * m.opt.timestep
    return True, tilt_max, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(F.ET.SCENE_XML)
    m.opt.timestep = R.SIM_DT
    rows = []
    for dur in DURATIONS:
        for label, deg in DIRS:
            limit = None
            for size in SIZES:
                ok, tilt_max, t_fall = episode(m, size, deg, dur)
                rows.append({"duration_s": dur, "direction": label, "deg": deg, "size_ms": size, "survived": ok,
                             "tilt_max_deg": round(float(np.degrees(tilt_max)), 1), "fell_after_s": t_fall})
                if not ok and limit is None:
                    limit = size
            print(f"{'kick' if dur == 0 else f'{dur:.1f} s push':>10}  {label:10s} falls from {limit if limit else '> %.1f' % SIZES[-1]} m/s")
    json.dump({"experiment": "passive_push", "note": __doc__.strip().splitlines()[0], "rows": rows,
               "sizes": SIZES, "durations": DURATIONS}, open(os.path.join(a.out, "passive_push.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(DURATIONS), figsize=(3.6 * len(DURATIONS), 3.6), sharey=True)
    for ax, dur in zip(axs, DURATIONS):
        grid = np.array([[1.0 if r["survived"] else 0.0 for r in rows if r["duration_s"] == dur and r["deg"] == deg] for _, deg in DIRS])
        ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto", origin="lower")
        ax.set_xticks(range(len(SIZES))); ax.set_xticklabels([f"{s:g}" for s in SIZES], fontsize=7)
        ax.set_yticks(range(len(DIRS))); ax.set_yticklabels([l for l, _ in DIRS], fontsize=7)
        ax.set_xlabel("velocity change m/s"); ax.set_title("instant kick" if dur == 0 else f"force over {dur:.1f} s", fontsize=9)
    axs[0].set_ylabel("thrown toward")
    fig.suptitle("Passive duck (servos hold the home pose, no policy): green = stayed up", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "passive_push.png"), dpi=110)
    print("wrote", os.path.join(a.out, "passive_push.png"))


if __name__ == "__main__":
    main()
