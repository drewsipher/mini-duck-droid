#!/usr/bin/env python3
"""Step and chirp response of ONE STS3215 joint in MuJoCo with the training model's actuator
parameters (BAM fit: damping, frictionloss, armature, kp read from the training XML), at the
deploy torque limit, with and without a 0.1 kg load on the link.

    cd sim && MUJOCO_GL=egl .venv/bin/python ../mechatronics/servo_response_sim.py

Writes mechatronics/response/step.png, chirp.png, response.json and prints a summary table.

The bench rig this mirrors (mechatronics/bench_procedure.md): a servo bolted down, horn axis
horizontal, a shank-sized arm hanging from the horn (shank + foot of the duck, about 0.15 kg,
79 mm), optionally a 0.1 kg weight at the end; targets sent at 50 Hz over the bus; position
read back at 50 Hz. The step is 0.5 rad from hanging. The sweep is sines of fixed amplitude
from 0.5 to 15 Hz; the chirp picture is the same thing in one run.

Variants: the model servo at the deploy torque (1.86 N.m) and at the training XML's 3.23 N.m,
the "measured stiffness" servo (kp x1.48 from the ankle scale test, sim/report1), and the
deploy command chain (8 Hz first-order low-pass + 4 rad/s rate limit) in front of the model.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
XML = ROOT / "sim/train_pg/playground/open_duck_mini_v2/xmls/open_duck_mini_v2_backlash.xml"
OUT = HERE / "response"

DT = 0.001                 # physics step
CTRL_HZ = 50               # policy / bus command rate
SUB = int(round(1 / CTRL_HZ / DT))
DEPLOY_TORQUE = 1.86
DEPLOY_LPF_HZ = 8.0
DEPLOY_MAX_VEL = 4.0
KP_MEASURED_FACTOR = 1.48  # report1: real ankle 0.44 N.m/deg vs sim 0.30
# shank + foot of the duck, from the MJCF (knee_and_ankle_assembly_2 0.0726 kg, foot_assembly 0.0752 kg, 78.65 mm)
SHANK_KG, FOOT_KG, SHANK_M = 0.07259, 0.07524, 0.07865

# validated categorical palette (dataviz skill, fixed order) and neutral ink
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def actuator_params():
    """damping, frictionloss, armature, kp, forcerange of class sts3215 in the training XML."""
    txt = XML.read_text()
    blk = re.search(r'<default class="sts3215">(.*?)</default>', txt, re.S).group(1)
    blk = re.sub(r"<!--.*?-->", "", blk, flags=re.S)       # drop the commented-out older fit
    j = re.search(r'<joint[^>]*damping="([\d.]+)"[^>]*frictionloss="([\d.]+)"[^>]*armature="([\d.]+)"', blk)
    p = re.search(r'<position[^>]*kp="([\d.]+)"[^>]*forcerange="-?([\d.]+) ([\d.]+)"', blk)
    return dict(damping=float(j[1]), frictionloss=float(j[2]), armature=float(j[3]), kp=float(p[1]),
                forcerange=float(p[3]))


def model_xml(prm, kp, fr, payload_kg):
    pay = (f'<body name="payload" pos="0 0 {-(SHANK_M + 0.03):.4f}"><geom type="sphere" size="0.012" '
           f'mass="{payload_kg}" rgba="0.8 0.3 0.2 1"/></body>') if payload_kg else ""
    return f"""
<mujoco model="sts3215_bench">
  <compiler angle="radian"/>
  <option timestep="{DT}" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="servo" pos="0 0 0.3">
      <geom type="box" size="0.012 0.02 0.0175" mass="0.06" rgba="0.3 0.3 0.3 1"/>
      <body name="link">
        <joint name="j" type="hinge" axis="0 1 0" range="-3 3" damping="{prm['damping']}"
               frictionloss="{prm['frictionloss']}" armature="{prm['armature']}"/>
        <geom type="capsule" fromto="0 0 0 0 0 {-SHANK_M:.4f}" size="0.008" mass="{SHANK_KG}"/>
        <geom type="box" pos="0 0 {-SHANK_M - 0.012:.4f}" size="0.02 0.015 0.012" mass="{FOOT_KG}"/>
        {pay}
      </body>
    </body>
  </worldbody>
  <actuator><position name="a" joint="j" kp="{kp}" kv="0" forcerange="-{fr} {fr}"/></actuator>
</mujoco>"""


class Rig:
    def __init__(self, prm, kp, fr, payload_kg, chain):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_string(model_xml(prm, kp, fr, payload_kg))
        self.d = mujoco.MjData(self.m)
        self.chain = chain
        self.alpha = (1 / DEPLOY_LPF_HZ) / (1 / CTRL_HZ + 1 / DEPLOY_LPF_HZ)
        self.max_step = DEPLOY_MAX_VEL / CTRL_HZ
        self.reset()

    def reset(self):
        self.mj.mj_resetData(self.m, self.d)
        self.filt = 0.0
        self.prev = 0.0

    def command(self, target):
        """One 50 Hz tick: optional deploy chain, then SUB physics steps. Returns per-step arrays."""
        if self.chain == "deploy":
            self.filt = self.alpha * self.filt + (1 - self.alpha) * target
            t = float(np.clip(self.filt, self.prev - self.max_step, self.prev + self.max_step))
        else:
            t = target
        self.prev = t
        self.d.ctrl[0] = t
        q, v, f = np.empty(SUB), np.empty(SUB), np.empty(SUB)
        for k in range(SUB):
            self.mj.mj_step(self.m, self.d)
            q[k], v[k], f[k] = self.d.qpos[0], self.d.qvel[0], self.d.actuator_force[0]
        return t, q, v, f

    def run(self, targets):
        """targets: sequence at 50 Hz. Returns dict of arrays at DT resolution + the applied ctrl."""
        Q, V, F, T = [], [], [], []
        for tg in targets:
            t, q, v, f = self.command(float(tg))
            Q.append(q); V.append(v); F.append(f); T.append(np.full(SUB, t))
        return dict(q=np.concatenate(Q), v=np.concatenate(V), f=np.concatenate(F), ctrl=np.concatenate(T))


def step_test(rig, amp=0.5, settle=1.0, dur=1.5):
    rig.reset()
    n0, n1 = int(settle * CTRL_HZ), int(dur * CTRL_HZ)
    r = rig.run(np.concatenate([np.zeros(n0), np.full(n1, amp)]))
    q = r["q"][n0 * SUB:] ; v = r["v"][n0 * SUB:]; f = r["f"][n0 * SUB:]
    t = np.arange(len(q)) * DT
    final = q[-int(0.2 / DT):].mean()
    def first(cond):
        i = np.argmax(cond)
        return float(t[i]) if cond.any() else None
    t10, t90 = first(q >= 0.1 * amp), first(q >= 0.9 * amp)
    rise = None if t10 is None or t90 is None else round(t90 - t10, 4)
    peak = q.max()
    overshoot = round(100 * max(0.0, peak - amp) / amp, 2)
    band = 0.02 * amp
    out = np.abs(q - amp) > band
    settling = None if out[-1] else round(float(t[len(t) - 1 - np.argmax(out[::-1])] + DT), 4) if out.any() else 0.0
    return dict(rise_time_s=rise, t_90_s=t90, overshoot_pct=overshoot, settling_2pct_s=settling,
                steady_state_error_rad=round(float(amp - final), 4), peak_velocity_rad_s=round(float(np.abs(v).max()), 3),
                peak_torque_nm=round(float(np.abs(f).max()), 3), time_at_torque_limit_s=round(float((np.abs(f) >= 0.999 * rig.m.actuator_forcerange[0, 1]).sum() * DT), 3),
                trace=dict(t=t[::10].round(4).tolist(), q=q[::10].round(5).tolist(), v=v[::10].round(4).tolist(), f=f[::10].round(4).tolist()))


def sine_fit(t, y, f):
    """Least-squares amplitude and phase (rad) of a sine at frequency f in y(t)."""
    w = 2 * np.pi * f
    A = np.stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)], 1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.hypot(c[0], c[1])), float(np.arctan2(c[1], c[0]))


def sweep(rig, amp, freqs):
    gains, phases = [], []
    for f in freqs:
        rig.reset()
        cycles = max(4, int(np.ceil(2.0 * f)))
        n = int(np.ceil(cycles / f * CTRL_HZ)) + CTRL_HZ   # + 1 s settle
        tk = np.arange(n) / CTRL_HZ
        r = rig.run(amp * np.sin(2 * np.pi * f * tk))
        t = np.arange(len(r["q"])) * DT
        keep = t >= t[-1] - 2 / f                           # last two cycles
        a, ph = sine_fit(t[keep], r["q"][keep], f)
        gains.append(a / amp)
        phases.append(np.degrees(ph))
    gains, phases = np.array(gains), np.unwrap(np.radians(phases)) * 180 / np.pi
    return gains, phases


def bandwidth(freqs, gains, level=1 / np.sqrt(2)):
    for i in range(1, len(freqs)):
        if gains[i] < level <= gains[i - 1]:
            # interpolate in log f
            x0, x1 = np.log(freqs[i - 1]), np.log(freqs[i])
            y0, y1 = gains[i - 1], gains[i]
            return float(np.exp(x0 + (level - y0) * (x1 - x0) / (y1 - y0)))
    return None if gains[-1] >= level else float(freqs[0])


def chirp_run(rig, amp=0.1, f0=0.5, f1=15.0, dur=20.0):
    rig.reset()
    n = int(dur * CTRL_HZ) + CTRL_HZ
    tk = np.arange(n) / CTRL_HZ - 1.0
    k = np.log(f1 / f0) / dur
    phase = 2 * np.pi * f0 * (np.exp(k * np.clip(tk, 0, None)) - 1) / k
    cmd = np.where(tk >= 0, amp * np.sin(phase), 0.0)
    r = rig.run(cmd)
    t = np.arange(len(r["q"])) * DT - 1.0
    return t, r["ctrl"], r["q"], lambda tt: f0 * np.exp(k * tt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--quick", action="store_true", help="fewer sweep points")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a.out.mkdir(exist_ok=True)
    prm = actuator_params()
    kp = prm["kp"]
    print(f"actuator from {XML.name}: {prm}")
    variants = [
        ("model, 1.86 N·m", dict(kp=kp, fr=DEPLOY_TORQUE, load=0.0, chain="raw"), C[0], "-"),
        ("model, 1.86 N·m, +0.1 kg", dict(kp=kp, fr=DEPLOY_TORQUE, load=0.1, chain="raw"), C[0], "--"),
        ("model, 3.23 N·m (training XML)", dict(kp=kp, fr=prm["forcerange"], load=0.0, chain="raw"), C[1], "-"),
        ("measured stiffness ×1.48, 1.86 N·m", dict(kp=kp * KP_MEASURED_FACTOR, fr=DEPLOY_TORQUE, load=0.0, chain="raw"), C[2], "-"),
        ("measured ×1.48, +0.1 kg", dict(kp=kp * KP_MEASURED_FACTOR, fr=DEPLOY_TORQUE, load=0.1, chain="raw"), C[2], "--"),
        ("model, 1.86 N·m, deploy chain 8 Hz + 4 rad/s", dict(kp=kp, fr=DEPLOY_TORQUE, load=0.0, chain="deploy"), C[3], "-"),
    ]
    freqs = np.geomspace(0.5, 15, 12 if a.quick else 25)
    amps = [0.05, 0.2]
    results = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "xml": str(XML.relative_to(ROOT)), "actuator": prm,
               "physics_dt": DT, "command_hz": CTRL_HZ, "link": dict(shank_kg=SHANK_KG, foot_kg=FOOT_KG, length_m=SHANK_M),
               "step_amplitude_rad": 0.5, "sweep_hz": freqs.round(3).tolist(), "sweep_amplitudes_rad": amps,
               "velocity_ceiling_rad_s": {}, "variants": {}}
    t0 = time.time()
    for name, cfg, col, ls in variants:
        rig = Rig(prm, cfg["kp"], cfg["fr"], cfg["load"], cfg["chain"])
        st = step_test(rig)
        sw = {}
        for amp in amps:
            g, ph = sweep(rig, amp, freqs)
            sw[str(amp)] = dict(gain=g.round(4).tolist(), phase_deg=ph.round(1).tolist(), bandwidth_3db_hz=bandwidth(freqs, g),
                                gain_at_4hz=float(np.interp(4.0, freqs, g)), phase_at_4hz_deg=float(np.interp(4.0, freqs, ph)))
        results["variants"][name] = dict(config=cfg, step=st, sweep=sw)
        results["velocity_ceiling_rad_s"][name] = round((cfg["fr"] - prm["frictionloss"]) / prm["damping"], 3)
        print(f"{name:46s} rise {st['rise_time_s']} s  overshoot {st['overshoot_pct']} %  settle {st['settling_2pct_s']} s  "
              f"ss err {st['steady_state_error_rad']} rad  vmax {st['peak_velocity_rad_s']} rad/s  "
              f"f-3dB {sw['0.05']['bandwidth_3db_hz'] and round(sw['0.05']['bandwidth_3db_hz'],2)} Hz @0.05 rad / "
              f"{sw['0.2']['bandwidth_3db_hz'] and round(sw['0.2']['bandwidth_3db_hz'],2)} Hz @0.2 rad", flush=True)
    results["notes"] = [
        "Velocity ceiling = (forcerange - frictionloss) / damping: the model servo cannot move faster than this in steady state, whatever the command.",
        "Armature 0.027 kg.m^2 (reflected rotor inertia) is ~40x the shank+foot inertia (~7e-4) and ~4x a 0.1 kg end load (~1.2e-3), so the load barely changes the response in the model; if the real servo slows visibly with 0.1 kg, the armature in the model is too high.",
        "Bandwidth at 0.2 rad is slew-rate limited (torque ceiling), at 0.05 rad it is close to the linear response.",
    ]
    (a.out / "response.json").write_text(json.dumps(results, indent=1))

    # ---- step figure: position, velocity, torque
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "grid.color": GRID, "axes.grid": True,
                         "grid.linewidth": 0.6, "legend.frameon": False})
    fig, ax = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
    ax[0].axhline(0.5, color=INK2, lw=1, ls=":", label="target 0.5 rad")
    for name, cfg, col, ls in variants:
        tr = results["variants"][name]["step"]["trace"]
        ax[0].plot(tr["t"], tr["q"], color=col, ls=ls, lw=2, label=name)
        ax[1].plot(tr["t"], tr["v"], color=col, ls=ls, lw=2)
        ax[2].plot(tr["t"], tr["f"], color=col, ls=ls, lw=2)
    ax[0].set_ylabel("joint angle (rad)"); ax[1].set_ylabel("velocity (rad/s)"); ax[2].set_ylabel("actuator torque (N·m)")
    ax[2].set_xlabel("time after the step (s)")
    ax[2].axhline(DEPLOY_TORQUE, color=INK2, lw=0.8, ls=":"); ax[2].text(0.98, DEPLOY_TORQUE + 0.05, "1.86 N·m deploy limit", color=INK2, fontsize=8, ha="right")
    ax[1].axhline(DEPLOY_MAX_VEL, color=INK2, lw=0.8, ls=":"); ax[1].text(0.98, DEPLOY_MAX_VEL + 0.1, "4 rad/s deploy limit", color=INK2, fontsize=8, ha="right")
    ax[0].set_title("One STS3215 joint in MuJoCo: 0.5 rad step, commands at 50 Hz", loc="left", fontsize=12)
    ax[0].legend(fontsize=8.5, loc="lower right", ncol=1)
    ax[2].set_xlim(0, 1.0)
    fig.tight_layout(); fig.savefig(a.out / "step.png", dpi=130); plt.close(fig)

    # ---- chirp figure: Bode (gain, phase) from the sweep + one time-domain chirp
    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1, 1])
    axg, axp, axc = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    for name, cfg, col, ls in variants:
        sw = results["variants"][name]["sweep"]["0.05"]
        axg.plot(freqs, sw["gain"], color=col, ls=ls, lw=2, marker="o", ms=3, label=name)
        axp.plot(freqs, sw["phase_deg"], color=col, ls=ls, lw=2, marker="o", ms=3)
    big = results["variants"][variants[0][0]]["sweep"]["0.2"]
    axg.plot(freqs, big["gain"], color=C[0], lw=1.2, ls=":", marker="s", ms=3, label="model, 1.86 N·m at 0.2 rad (slew-limited)")
    axg.axhline(1 / np.sqrt(2), color=INK2, lw=0.8, ls=":"); axg.text(0.52, 0.72, "−3 dB", color=INK2, fontsize=8)
    axg.set_xscale("log"); axp.set_xscale("log")
    axg.set_ylabel("amplitude ratio (out / in)"); axp.set_ylabel("phase lag (deg)"); axp.set_xlabel("frequency (Hz)")
    axg.set_title("Frequency response, sine amplitude 0.05 rad (solid/dashed) and 0.2 rad (dotted)", loc="left", fontsize=12)
    axg.legend(fontsize=8, loc="lower left")
    for axx in (axg, axp):
        axx.set_xticks([0.5, 1, 2, 4, 8, 15]); axx.set_xticklabels(["0.5", "1", "2", "4", "8", "15"])
    rig = Rig(prm, kp, DEPLOY_TORQUE, 0.0, "raw")
    t, cmd, q, finst = chirp_run(rig)
    axc.plot(t, cmd, color=INK2, lw=1, label="command, 0.1 rad chirp 0.5→15 Hz")
    axc.plot(t, q, color=C[0], lw=1.6, label="model, 1.86 N·m")
    axc.set_xlim(0, 20); axc.set_xlabel("time (s)   [instantaneous frequency: " + ", ".join(f"{finst(x):.1f} Hz @ {x:.0f} s" for x in (0, 5, 10, 15, 20)) + "]")
    axc.set_ylabel("angle (rad)"); axc.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout(); fig.savefig(a.out / "chirp.png", dpi=130); plt.close(fig)
    print(f"wrote {a.out.relative_to(ROOT)}/step.png, chirp.png, response.json  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
