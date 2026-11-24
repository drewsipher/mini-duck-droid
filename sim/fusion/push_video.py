"""Film one expert being pushed: one section per push direction, three views per frame, the push drawn as an arrow.

    cd sim && MUJOCO_GL=egl PYTHONPATH=train_pg train_pg/.venv/bin/python fusion/push_video.py \
        --onnx <policy.onnx> --clip ../tracker/data/clips/Stand.json --out pushes.mp4 [--mag 0.5] [--seconds 7] [--every 2]

Each section: the expert alone from its clip start, a kick of --mag m/s every --every s thrown in the section's
direction (forward, backward, left, right, then random at --mag + 0.1). Views: side, front, three-quarter.
The arrow is the kick: it points the way the duck is thrown, its length is the kick size, it fades over 0.6 s.
"""
import argparse
import os
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fusion_sim as F  # noqa: E402

VW, VH, FPS = 480, 360, 25
HEAD, FOOT = 34, 54
VIEWS = [("side", 90), ("front", 0), ("three-quarter", 45)]        # camera azimuth: 0 puts the camera in front of
#   the duck (it faces +x; mjv puts the camera at azimuth 0 on the +x side looking back at it), 90 on its left side
ARROW_S = 0.6                                                     # arrow fade time


class ThreeViews:
    def __init__(self, m):
        self.r = mujoco.Renderer(m, VH, VW)
        self.cams = []
        for _, az in VIEWS:
            c = mujoco.MjvCamera()
            c.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            c.trackbodyid = 1
            c.distance, c.azimuth, c.elevation = 0.95, az, -12
            self.cams.append(c)
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 19)
            self.small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        except OSError:
            self.font = self.small = ImageFont.load_default()
        self.frames = []

    def arrow(self, d, base_body, push, age):
        """Add the kick arrow to the current scene: from behind the duck, pointing the way it is thrown."""
        scn = self.r.scene
        if scn.ngeom >= scn.maxgeom or push is None:
            return
        v = np.array(push, float)
        mag = np.linalg.norm(v)
        if mag < 1e-6:
            return
        u = np.array([v[0] / mag, v[1] / mag, 0.0])
        base = d.xpos[base_body].copy()
        base[2] += 0.02
        length = 0.12 + 0.36 * mag                                   # 0.5 m/s -> 0.30 m
        tip = base - u * 0.05
        tail = tip - u * length
        fade = max(0.0, 1.0 - age / ARROW_S)
        rgba = np.array([1.0, 0.25 + 0.5 * (1 - min(mag / 0.6, 1.0)), 0.1, 0.25 + 0.75 * fade], np.float32)
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).flatten(), rgba)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.012 + 0.01 * fade, tail, tip)
        scn.ngeom += 1

    def frame(self, d, base_body, title, status, push, age, fell):
        W = VW * len(VIEWS)
        img = Image.new("RGB", (W, HEAD + VH + FOOT), (20, 20, 20))
        for i, (name, _) in enumerate(VIEWS):
            self.r.update_scene(d, self.cams[i])
            if age is not None and age < ARROW_S:
                self.arrow(d, base_body, push, age)
            view = Image.fromarray(self.r.render())
            img.paste(view, (i * VW, HEAD))
            dr = ImageDraw.Draw(img)
            tw = dr.textlength(name, font=self.small)
            dr.rectangle([i * VW + 4, HEAD + 4, i * VW + 14 + tw, HEAD + 24], fill=(20, 20, 20))
            dr.text((i * VW + 9, HEAD + 6), name, font=self.small, fill=(235, 235, 235))
        dr = ImageDraw.Draw(img)
        dr.text((8, 6), title, font=self.font, fill=(255, 255, 255))
        dr.text((8, HEAD + VH + 8), status + ("   FELL" if fell else ""), font=self.small, fill=(255, 90, 90) if fell else (255, 210, 90))
        return np.asarray(img)


def film_section(sim, views, name, title, rng, seconds, every, mag, push_dir, force_s=0.0):
    base_body = sim.base_body
    state = {"push": None, "t_push": None, "n": 0}
    W = [1.0 if n == name else 0.0 for n in sim.names]
    frames = []

    def render(d, row, fell):
        if row["push"] is not None:
            state["push"], state["t_push"], state["n"] = row["push"], row["t"], state["n"] + 1
        age = None if state["t_push"] is None else row["t"] - state["t_push"]
        if int(round(row["t"] / F.R.CTRL_DT)) % 2:
            return
        if state["push"] is None:
            kick = "waiting for the first push"
        else:
            p = state["push"]
            kick = f"push {state['n']}: {np.hypot(*p):.2f} m/s"
        status = f"t {row['t']:.2f} s    {kick}    tilt {np.degrees(row['tilt']):.0f} deg    moved {100 * np.hypot(row['x'], row['y']):.0f} cm"
        frames.append(views.frame(d, base_body, title, status, state["push"], age, fell))

    r = sim.run(rng, F.Fixed(W), seconds, start=(name, 0), pushes=True, push_mag=(mag, mag), push_dir=push_dir,
                push_every_s=every, push_force_s=force_s, render=render)
    frames.extend([frames[-1]] * int(FPS * 0.8))                      # hold the last frame
    return frames, r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--clip", default=os.path.join(HERE, "..", "..", "tracker", "data", "clips", "Stand.json"))
    ap.add_argument("--name", default="recovery")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mag", type=float, default=0.5)
    ap.add_argument("--seconds", type=float, default=7.0)
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force-s", type=float, default=0.0, help="push as a force over this long instead of an instant kick")
    a = ap.parse_args()

    m0 = mujoco.MjModel.from_xml_path(F.ET.SCENE_XML)
    m0.opt.timestep = F.R.SIM_DT
    expert = F.Expert(a.name, os.path.abspath(a.onnx), os.path.abspath(a.clip), m0)
    sim = F.FusionSim(m0, [expert])
    views = ThreeViews(m0)
    sections = [("thrown forward", 0.0, a.mag), ("thrown backward", 180.0, a.mag), ("thrown left", 90.0, a.mag),
                ("thrown right", 270.0, a.mag), (f"random direction, {a.mag + 0.1:.1f} m/s", None, a.mag + 0.1)]
    frames = []
    for i, (label, deg, mag) in enumerate(sections):
        kind = f"shove over {a.force_s:g} s" if a.force_s else "kick"
        title = f"{a.name}: {label}, {mag:.1f} m/s {kind} every {a.every:.0f} s   ({i + 1}/{len(sections)})"
        fr, r = film_section(sim, views, a.name, title, np.random.default_rng(a.seed * 100 + i), a.seconds, a.every, mag,
                             None if deg is None else np.radians(deg), a.force_s)
        print(f"{label}: {'stayed up' if r['success'] else 'fell at %.2f s' % r['seconds']}", flush=True)
        frames += fr
    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with imageio.get_writer(a.out, fps=FPS, codec="libx264", quality=7, macro_block_size=16) as w:
        for f in frames:
            w.append_data(f)
    print("wrote", a.out, f"{len(frames) / FPS:.0f} s")


if __name__ == "__main__":
    main()
