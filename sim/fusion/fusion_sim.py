"""Run several frozen tracking experts in one sim and mix them: the bench for gating and fusion.

Every expert runs every step ("shadowing"): it sees the shared sensors and motor targets plus
its own clip clock, reference and previous actions, exactly as in its training, and proposes
raw_i = clip_i(k_i + 1) + action_scale * a_i. A controller picks weights w_i each step and the
duck gets sum_i w_i raw_i, then the deploy low-pass and rate clamp (output blending). With
weight blending there is one network, made by interpolating the experts' weights, tracking the
same blend of the references.

The physics randomisation is the strict battery's (eval_tracking.TrackingSim.randomised_model).
A fall is the base below MIN_BASE_HEIGHT_M or tilted more than MAX_ORI_ERR_RAD from upright
(no reference to compare against once experts are mixed).
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
sys.path.insert(0, SIM)
sys.path.insert(0, os.path.join(SIM, "train_pg"))
import eval_tracking as ET                                           # noqa: E402
import recipe_v1 as R                                                # noqa: E402
from playground.common.clip_tracking_reference import ClipTrackingReference, quat_rotate  # noqa: E402
from experts import MLP                                              # noqa: E402

HORIZON_S = 8.0


class Expert:
    """One frozen tracking skill: its network, its clip, its clock and its own action history."""

    def __init__(self, name, onnx, clip, m0, loop=True, net=None):
        self.name, self.loop = name, loop
        self.net = net or MLP.from_onnx(onnx)
        self.ref = ClipTrackingReference(clip, m0, loop=loop, horizon_s=HORIZON_S, quiet=True)
        self.cycle = self.ref.cycle_steps
        # per cycle frame: both feet on the floor in the reference (within 1 mm of the lowest foot height)
        z = self.ref.foot_pos[:, :, 2]
        self.down = (z < z.min() + 0.001).all(axis=1)
        self.reset()

    def reset(self, k=0):
        self.k = k
        self.a1 = np.zeros(14); self.a2 = np.zeros(14)
        self.hist = [np.zeros(14) for _ in range(3)]

    def i(self, k):
        """Row of the unwrapped reference for clock k (loops: kept off the ends, joints repeat per cycle)."""
        return self.cycle + k % self.cycle if self.loop else int(np.clip(k, 0, self.ref.n - 1))

    def phase(self, k=None):
        k = self.k if k is None else k
        return (k % self.cycle) / self.cycle if self.loop else min(k, self.ref.n - 1) / max(self.ref.n - 1, 1)

    def double_support(self, k=None):
        """Both feet down in the reference at clock k."""
        return bool(self.down[self.i(self.k if k is None else k)])


class FusionSim(ET.TrackingSim):
    def __init__(self, m0, experts):
        first = experts[0]
        super().__init__(m0, first.ref, None, True)
        self.experts = experts
        self.names = [e.name for e in experts]

    def expert_obs(self, e, shared, targets):
        kn = e.i(e.k + 1)
        ph = e.phase()
        return np.concatenate([shared, e.a1, e.a2, targets - self.home,
                               [ph, np.cos(2 * np.pi * ph), np.sin(2 * np.pi * ph)],
                               e.ref.q[kn] - self.home, e.ref.qd[kn] * R.DOF_VEL_SCALE])

    def run(self, rng, controller, episode_s, start=("stand", 0), pushes=False, render=None, blend_net=None):
        """One episode. controller(sim, t, w_prev) -> weights (len experts). start = (expert, clock).
        blend_net: (MLP, fn t -> w) runs that one network instead, tracking the blended reference."""
        m = self.randomised_model(rng)
        d = mujoco.MjData(m)
        act_delay = int(rng.integers(R.ACTION_DELAY_STEPS[0], R.ACTION_DELAY_STEPS[1] + 1))
        imu_delay = int(rng.integers(R.IMU_DELAY_STEPS[0], R.IMU_DELAY_STEPS[1] + 1))
        for e in self.experts:
            e.reset()
        e0 = self.experts[self.names.index(start[0])]
        e0.k = start[1]
        r, i0 = e0.ref, e0.i(e0.k)
        d.qpos[:] = r.qpos_full[i0]
        d.qpos[0:2] -= r.qpos_full[i0][0:2]                              # start at the origin
        d.qvel[:] = 0.0
        d.qvel[self.vadr] = r.qd[i0]
        d.qvel[0:3] = r.base_linvel[i0]
        qc = r.base_quat[i0]
        d.qvel[3:6] = quat_rotate(np.array([qc[0], -qc[1], -qc[2], -qc[3]]), r.base_angvel[i0])
        targets = r.q[i0].copy()
        d.ctrl[:] = targets
        mujoco.mj_forward(m, d)
        imu_hist, trace = [], []
        w = np.array([1.0 if n == start[0] else 0.0 for n in self.names])
        raw_prev, fell = targets.copy(), False
        steps = int(round(episode_s / R.CTRL_DT))
        push_every = int(round(rng.uniform(*R.PUSH_INTERVAL_S) / R.CTRL_DT)) if pushes else 0
        bn_a = [np.zeros(14), np.zeros(14)]; bn_hist = [np.zeros(14) for _ in range(3)]
        for t in range(steps):
            gyro = d.sensordata[self.gyro_a:self.gyro_a + 3].copy()
            acc = d.sensordata[self.acc_a:self.acc_a + 3].copy()
            imu_hist.insert(0, np.concatenate([gyro, acc])); del imu_hist[4:]
            imu = imu_hist[min(imu_delay, len(imu_hist) - 1)]
            q = d.qpos[self.qadr].copy()
            q += np.where(self.badr >= 0, d.qpos[np.maximum(self.badr, 0)], 0.0)
            shared = np.concatenate([imu, q - self.home, d.qvel[self.vadr] * R.DOF_VEL_SCALE])
            w = np.asarray(controller(self, t * R.CTRL_DT, w, shared=shared, targets=targets), float)   # may also reset expert clocks
            refs = np.array([e.ref.q[e.i(e.k + 1)] for e in self.experts])
            if blend_net is None:
                raws = []
                for e in self.experts:
                    a = e.net(self.expert_obs(e, shared, targets))
                    e.hist.insert(0, a); del e.hist[3:]
                    raws.append(e.ref.q[e.i(e.k + 1)] + R.ACTION_SCALE * e.hist[min(act_delay, 2)])
                    e.a2, e.a1 = e.a1, a
                raw = w @ np.array(raws)
            else:
                # one network, weights blended at w; it tracks the same blend of the references, on the last expert's clock
                net, ew = blend_net[0], self.experts[-1]
                ph = ew.phase()
                bref = w @ refs
                bref_d = w @ np.array([e.ref.qd[e.i(e.k + 1)] for e in self.experts])
                o = np.concatenate([shared, bn_a[0], bn_a[1], targets - self.home, [ph, np.cos(2 * np.pi * ph), np.sin(2 * np.pi * ph)],
                                    bref - self.home, bref_d * R.DOF_VEL_SCALE])
                a = net(o)
                bn_hist.insert(0, a); del bn_hist[3:]
                raw = bref + R.ACTION_SCALE * bn_hist[min(act_delay, 2)]
                bn_a = [a, bn_a[0]]
            new = self.alpha * targets + (1 - self.alpha) * raw
            targets = np.clip(new, targets - self.max_step, targets + self.max_step)
            d.ctrl[:] = targets
            if push_every and (t + 1) % push_every == 0:
                th, mag = rng.uniform(0, 2 * np.pi), rng.uniform(*R.PUSH_MAGNITUDE_MS)
                d.qvel[0] += mag * np.cos(th); d.qvel[1] += mag * np.sin(th)
            for _ in range(self.decim):
                mujoco.mj_step(m, d)
            for e in self.experts:
                e.k += 1
            b = self.base_body
            tilt = float(np.arccos(np.clip(d.xmat[b][8], -1, 1)))
            h = float(d.qpos[self.ref.base_qpos_addr + 2])
            trace.append({"t": (t + 1) * R.CTRL_DT, "w": w.tolist(), "x": float(d.qpos[0]), "y": float(d.qpos[1]),
                          "vx": float(d.qvel[0]), "h": h, "tilt": tilt, "jump": float(np.abs(raw - raw_prev).max()),
                          "phase": {e.name: e.phase() for e in self.experts}})
            raw_prev = raw
            fell = h < R.MIN_BASE_HEIGHT_M or tilt > R.MAX_ORI_ERR_RAD or not np.isfinite(d.qpos).all()
            if render:
                render(d, trace[-1], fell)
            if fell:
                break
        return {"success": not fell, "seconds": len(trace) * R.CTRL_DT, "trace": trace}


# ----------------------------------------------------------------------------- controllers
class Commanded:
    """The baseline: a command schedule [(t, 'stand'|'walk'), ...] mapped to one expert at a time.
    ramp_s = 0 is a hard switch, > 0 a linear output cross-fade over that long. Into walk the walk
    clock restarts at walk_start (phase); out of walk it waits for double support when ds_only."""

    def __init__(self, schedule, ramp_s=0.0, walk_start=0.0, ds_only=True, walk="walk", stand="stand"):
        self.schedule, self.ramp_s, self.walk_start, self.ds_only = schedule, ramp_s, walk_start, ds_only
        self.walk, self.stand = walk, stand
        self.target = None
        self.switch_times = []

    def command(self, t):
        c = self.schedule[0][1]
        for ts, name in self.schedule:
            if t >= ts:
                c = name
        return c

    def __call__(self, sim, t, w, **_):
        want = self.command(t)
        if self.target is None:
            self.target = want if want in sim.names else sim.names[int(np.argmax(w))]
        if want != self.target:
            ew = sim.experts[sim.names.index(self.walk)]
            if want == self.walk:
                ew.reset(int(round(self.walk_start * ew.cycle)))
                self.target = want; self.switch_times.append(t)
            elif not self.ds_only or ew.double_support():
                self.target = want; self.switch_times.append(t)
        goal = np.array([1.0 if n == self.target else 0.0 for n in sim.names])
        if self.ramp_s <= 0:
            return goal
        step = R.CTRL_DT / self.ramp_s
        return w + np.clip(goal - w, -step, step)


class Fixed:
    def __init__(self, w):
        self.w = np.asarray(w, float)

    def __call__(self, sim, t, w, **_):
        return self.w


class Gate:
    """The learned gate (fusion_gate.py): moves w_walk from the IMU, joints, motor targets, command,
    walk phase and its previous w. The walk clock free-runs from the episode start, as in training.
    Reads gate_spec.json next to the ONNX: action_mode "target" (v5+: w follows (a + 1) / 2 at most
    w_rate per step), else w_rate (v2-v4: w += w_rate * a), else neither (v1: w = (a + 1) / 2)."""

    def __init__(self, onnx, schedule, w_rate=None):
        import json
        self.net, self.schedule = MLP.from_onnx(onnx), schedule
        spec = os.path.join(os.path.dirname(onnx), "gate_spec.json")
        sp = json.load(open(spec)) if os.path.exists(spec) else {}
        self.w_rate = w_rate if w_rate is not None else sp.get("w_rate")
        self.target = sp.get("action_mode") == "target"
        self.prior = sp.get("prior", "none")
        self.command = Commanded.command.__get__(self)

    def __call__(self, sim, t, w, shared, targets):
        ew = sim.experts[sim.names.index("walk")]
        ph = ew.phase()
        cmd = 1.0 if self.command(t) == "walk" else 0.0
        o = np.concatenate([shared, targets - sim.home, [cmd, np.cos(2 * np.pi * ph), np.sin(2 * np.pi * ph)], [w[sim.names.index("walk")]]])
        a = self.net(o)[0]
        wp = w[sim.names.index("walk")]
        if self.target:
            goal = np.clip(cmd + a, 0.0, 1.0) if self.prior == "command" else 0.5 * (a + 1.0)
            ww = float(wp + np.clip(goal - wp, -self.w_rate, self.w_rate))
        else:
            ww = float(np.clip(wp + self.w_rate * a if self.w_rate else 0.5 * (a + 1.0), 0.0, 1.0))
        return np.array([1.0 - ww if n == "stand" else ww for n in sim.names])


# ----------------------------------------------------------------------------- video
class Video:
    W, H, FPS = 640, 420, 25

    def __init__(self, m, title):
        from PIL import ImageFont
        self.r = mujoco.Renderer(m, self.H, self.W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = 1
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.9, 135, -14
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 19)
            self.small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        except OSError:
            self.font = self.small = ImageFont.load_default()
        self.title, self.frames, self.n = title, [], 0

    def __call__(self, d, row, fell, names=("stand", "walk"), cmd=""):
        self.n += 1
        if self.n % 2:
            return
        from PIL import Image, ImageDraw
        self.r.update_scene(d, self.cam)
        img = Image.fromarray(self.r.render())
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, self.W, 30], fill=(20, 20, 20))
        dr.text((8, 4), self.title, font=self.font, fill=(255, 255, 255))
        dr.rectangle([0, self.H - 50, self.W, self.H], fill=(20, 20, 20))
        x0 = 8
        for n, wi in zip(names, row["w"]):
            dr.rectangle([x0, self.H - 44, x0 + 120, self.H - 30], outline=(120, 120, 120))
            dr.rectangle([x0, self.H - 44, x0 + int(120 * wi), self.H - 30], fill=(90, 170, 255) if n == "walk" else (255, 190, 80))
            dr.text((x0 + 126, self.H - 47), f"{n} {wi:.2f}", font=self.small, fill=(230, 230, 230))
            x0 += 250
        status = f"t {row['t']:.2f} s   x {100 * row['x']:.1f} cm   vx {row['vx']:.2f} m/s   tilt {np.degrees(row['tilt']):.0f} deg  {cmd}"
        dr.text((8, self.H - 24), status + ("   FELL" if fell else ""), font=self.small, fill=(255, 90, 90) if fell else (255, 210, 90))
        self.frames.append(np.asarray(img))

    def save(self, path, hold=25):
        import imageio.v2 as imageio
        if self.frames:
            self.frames.extend([self.frames[-1]] * hold)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with imageio.get_writer(path, fps=self.FPS, codec="libx264", quality=7, macro_block_size=16) as w:
            for f in self.frames:
                w.append_data(f)
        self.r.close()
