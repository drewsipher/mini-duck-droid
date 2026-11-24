"""How much an expert fidgets while standing, and what it costs to calm it at deploy time.

    cd sim && MUJOCO_GL=egl PYTHONPATH=train_pg train_pg/.venv/bin/python fusion/jitter.py \
        --expert stand=../skills/stand/v1/policy.onnx --expert recovery=../skills/recovery/v1/policy.onnx \
        [--variants] [--push 0.3,180,0.2] [--seeds 10] [--push-seeds 20]

Per expert (and, with --variants, per deploy-time knob on it: lower low-pass cutoff, smaller action scale,
tighter rate clamp), on the Stand clip:
  flutter   mean |target_t - target_{t-1}| over ticks and joints, deg per 20 ms tick: what the servos are asked to do
  command   mean over ticks of the largest per-joint change of the policy's own command (before the filter), deg
  hf        fraction of the target-motion power above 5 Hz (quiet standing)
  up        pushed runs that stayed up (--push dv,deg,force_s: 0.3 m/s thrown backward as a 0.2 s shove, every 2 s)
The action scale of each expert comes from obs_spec.json beside its ONNX (0.25 if none).
"""
import argparse
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fusion_sim as F  # noqa: E402

CLIP = os.path.join(HERE, "..", "..", "tracker", "data", "clips", "Stand.json")


def alpha(hz):
    return (1.0 / hz) / (F.R.CTRL_DT + 1.0 / hz)


class Probe:
    def __init__(self):
        self.T, self.cmd = [], []

    def __call__(self, d, row, fell):
        self.T.append(d.ctrl.copy())
        self.cmd.append(row["jump"])


def measure(sim, name, seeds, push, push_seeds, lpf_hz=None, scale=None, max_vel=None):
    e = sim.experts[sim.names.index(name)]
    sim.alpha = alpha(lpf_hz or F.R.ACT_LPF_HZ)
    sim.max_step = (max_vel or F.R.MAX_MOTOR_VEL) * F.R.CTRL_DT
    scale0 = e.scale
    if scale is not None:
        e.scale = scale
    W = F.Fixed([1.0 if n == name else 0.0 for n in sim.names])
    fl, cm, hf = [], [], []
    for s in range(seeds):
        p = Probe()
        sim.run(np.random.default_rng(1000 + s), W, 6.0, start=(name, 0), render=p)
        dT = np.diff(np.array(p.T), axis=0)
        fl.append(np.degrees(np.abs(dT)).mean())
        cm.append(np.degrees(np.mean(p.cmd)))
        f = np.fft.rfftfreq(len(dT), F.R.CTRL_DT)
        P = (np.abs(np.fft.rfft(dT - dT.mean(0), axis=0)) ** 2).sum(1)
        hf.append(P[f > 5].sum() / max(P[1:].sum(), 1e-12))
    up, flp = 0, []
    if push:
        dv, deg, fs = push
        for s in range(push_seeds):
            p = Probe()
            r = sim.run(np.random.default_rng(2000 + s), W, 6.0, start=(name, 0), pushes=True, push_mag=(dv, dv),
                        push_dir=np.radians(deg), push_every_s=2.0, push_force_s=fs, render=p)
            up += r["success"]
            if len(p.T) > 2:
                flp.append(np.degrees(np.abs(np.diff(np.array(p.T), axis=0))).mean())
    e.scale = scale0
    return dict(flutter=np.mean(fl), command=np.mean(cm), hf=np.mean(hf), up=up,
                flutter_pushed=np.mean(flp) if flp else float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expert", action="append", required=True, help="name=policy.onnx (repeatable)")
    ap.add_argument("--clip", default=CLIP)
    ap.add_argument("--variants", action="store_true", help="also try LPF 4 Hz, scale x0.7 and a 2 rad/s clamp on each expert")
    ap.add_argument("--push", default="0.3,180,0.2", help="dv,deg,force_s; '' for none")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--push-seeds", type=int, default=20)
    a = ap.parse_args()
    push = tuple(float(x) for x in a.push.split(",")) if a.push else None

    m0 = mujoco.MjModel.from_xml_path(F.ET.SCENE_XML)
    m0.opt.timestep = F.R.SIM_DT
    experts = []
    for spec in a.expert:
        name, onnx = spec.split("=", 1)
        experts.append(F.Expert(name, os.path.abspath(onnx), os.path.abspath(a.clip), m0))
    sim = F.FusionSim(m0, experts)
    print(f"{'variant':40s} {'flutter deg/tick':>16s} {'command deg':>11s} {'hf>5Hz':>7s} {'pushed flutter':>14s} {'up/%d' % a.push_seeds:>6s}", flush=True)
    for e in experts:
        rows = [(f"{e.name} (scale {e.scale:g}, {F.R.ACT_LPF_HZ:g} Hz)", {})]
        if a.variants:
            rows += [(f"{e.name}, LPF 4 Hz", dict(lpf_hz=4.0)), (f"{e.name}, scale {0.7 * e.scale:.2f}", dict(scale=0.7 * e.scale)),
                     (f"{e.name}, clamp 2 rad/s", dict(max_vel=2.0))]
        for label, kw in rows:
            r = measure(sim, e.name, a.seeds, push, a.push_seeds, **kw)
            print(f"{label:40s} {r['flutter']:16.3f} {r['command']:11.2f} {r['hf']:7.2f} {r['flutter_pushed']:14.3f} {r['up']:6d}", flush=True)


if __name__ == "__main__":
    main()
