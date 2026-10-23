"""Sensor noise like the real duck's, then the runtime's --cutoff_frequency filter: does it
calm the servo commands without hurting walking?"""
import sys, pickle, numpy as np, importlib.util, mujoco
from sim_test import Sim, PG
from sim_routine_lib import ROUTINE
spec = importlib.util.spec_from_file_location("sc", "smooth_cmp_lib.py"); sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
XML = f"{PG}/xmls/scene_flat_terrain_backlash.xml"

class NoisySim(Sim):
    def __init__(self, onnx, gyro_sd, acc_sd, cutoff, seed=0):
        super().__init__(onnx, XML, 0.0)
        self.rng = np.random.default_rng(seed); self.gyro_sd = gyro_sd; self.acc_sd = acc_sd
        self.alpha = None if cutoff is None else (1 / cutoff) / (1 / 50 + 1 / cutoff)
        self.filt = self.default.copy(); self.ticks = 0
    def obs(self, cmd):
        o = super().obs(cmd)
        o[0:3] += self.rng.normal(0, self.gyro_sd, 3); o[3:6] += self.rng.normal(0, self.acc_sd, 3)
        return o
    def control(self, cmd):
        self.imitation_i = (self.imitation_i + 1.0) % self.nb_steps
        ang = self.imitation_i / self.nb_steps * 2 * np.pi
        self.phase = np.array([np.cos(ang), np.sin(ang)])
        o = self.obs(cmd)
        a = self.policy.run(None, {"obs": [o]})[0][0]
        self.a3, self.a2, self.a1 = self.a2, self.a1, a.copy()
        t = self.default + a * 0.25
        t = np.clip(t, self.prev_targets - 0.1048, self.prev_targets + 0.1048)
        if self.alpha is not None:
            self.filt = self.alpha * self.filt + (1 - self.alpha) * t
            if self.ticks > 50: t = self.filt.copy()
        self.ticks += 1
        self.prev_targets = t.copy(); self.targets = t; self.d.ctrl[:] = t
        for _ in range(self.decim): mujoco.mj_step(self.m, self.d)
        return o, a

def hf(x):
    x = x - x.mean(0); f = np.fft.rfftfreq(len(x), 1/50); X = np.fft.rfft(x, axis=0)
    return np.degrees(np.sqrt((np.fft.irfft(X * (f > 8)[:, None], len(x), axis=0)**2).mean(0))).mean()

def trial(onnx, gyro_sd, acc_sd, cutoff, seed=0):
    s = NoisySim(onnx, gyro_sd, acc_sd, cutoff, seed)
    obs, fell, pos = [], 0, []
    for sec, cmd in ROUTINE:
        p0 = s.d.qpos[:2].copy(); y0 = s.base_pose()[1]
        for _ in range(int(sec * 50)):
            o, _ = s.control(np.array(cmd + [0, 0, 0, 0], float)); obs.append(o)
            if s.base_pose()[2] > 45: fell = 1
        dp = s.d.qpos[:2] - p0; c, sn = np.cos(-y0), np.sin(-y0)
        pos.append((c * dp[0] - sn * dp[1]) / sec)
    o = np.array(obs, dtype=np.float64); r = sc.summary(o)
    fwd = [i for i in range(len(o)) if sc.label(o[i, 6:9]) == "fwd"]
    inp = [i for i in range(500)]
    return dict(fell=fell, fwd_speed=pos[1], back_speed=pos[2],
                tgt_jit_stand=hf(o[100:500, 83:97]), tgt_jit_fwd=hf(o[fwd[50]:fwd[-1], 83:97]),
                body_jit_fwd=r.get("fwd", (None, np.full(2, np.nan)))[1].mean(),
                body_jit_stand=r.get("in-place", (None, np.full(2, np.nan)))[1].mean())

onnx = sys.argv[1] if len(sys.argv) > 1 else "BEST_WALK_ONNX_2.onnx"
print(onnx, "   (real walk 7: servo-command jitter standing 0.53, walking 0.75; body jitter standing 4.7, walking 10.3)")
print(f"{'':34s}{'falls':>6s}{'fwd m/s':>9s}{'back m/s':>9s}{'cmd jit stand':>14s}{'cmd jit walk':>13s}{'body jit stand':>15s}{'body jit walk':>14s}")
for label, g, a in [("no noise", 0, 0), ("noise like training", 0.058, 0.03), ("heavier noise", 0.15, 0.3)]:
    for cutoff in ([None] if label != "heavier noise" else [None, 50, 37.5, 25, 15]):
        rs = [trial(onnx, g, a, cutoff, seed) for seed in range(3)]
        m = {k: np.mean([r[k] for r in rs]) for k in rs[0]}
        name = f"{label}, filter {cutoff if cutoff else 'off'}"
        print(f"{name:34s}{int(sum(r['fell'] for r in rs)):>4d}/3{m['fwd_speed']:9.3f}{m['back_speed']:9.3f}{m['tgt_jit_stand']:14.2f}{m['tgt_jit_fwd']:13.2f}{m['body_jit_stand']:15.1f}{m['body_jit_fwd']:14.1f}")
