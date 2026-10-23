#!/usr/bin/env python3
"""Shutdown poses with the duck's measured joint limits (joint_limits/real_limits_20261014.json),
rendered side by side: report1/tuck.mp4."""
import numpy as np, mujoco, imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import tuck_sim as s

W, H = 420, 380
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
PANELS = [
    ("Motors off while standing", dict(hip=-0.63, knee=1.37, ankle=-0.78, neck=0.0, head_pitch=0.0), 90),
    ("Current crouch (lean back)", dict(hip=-0.18, knee=1.55, ankle=-1.18, neck=0.0, head_pitch=0.0), 90),
    ("Option 1: tuck, head down", dict(hip=-0.4, knee=1.68, ankle=-1.13, neck=-0.69, head_pitch=-0.6), 90),
    ("Option 2: tuck, chin up", dict(hip=0.0, knee=1.68, ankle=-1.13, neck=-0.3, head_pitch=1.08), 90),
]
cols, stats = [], []
for label, p, az in PANELS:
    rs = {}
    def render(m, d, t, t2, t3, label=label, az=az):
        r = rs.get(id(m)) or rs.setdefault(id(m), mujoco.Renderer(m, H, W))
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.04, 0, 0.17]; cam.distance, cam.azimuth, cam.elevation = 0.95, az, -8
        r.update_scene(d, cam)
        img = Image.fromarray(r.render()); dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, W, 26], fill=(20, 20, 20)); dr.text((8, 4), label, font=F, fill=(255, 255, 255))
        st = "standing" if t < 1 else "moving into pose" if t < t2 else "holding" if t < t3 else "MOTORS OFF"
        dr.rectangle([0, H - 24, W, H], fill=(20, 20, 20)); dr.text((8, H - 21), f"{t:4.1f}s  {st}", font=S, fill=(255, 210, 90))
        return np.asarray(img)
    frames = []
    r = s.run(p, 1.5, 1.9, frames, render)
    cols.append(frames); stats.append((label, r))
    print(f"{label:34s} head peak {r['peak_head_v']:.2f} m/s, ends pitch {r['pitch_end']:+.0f}, face {r['face_deg']:+.0f}, head {r['head_cm']:.1f} cm")
n = max(len(c) for c in cols); cols = [c + [c[-1]] * (n - len(c)) for c in cols]
with imageio.get_writer("report1/tuck.mp4", fps=25, codec="libx264", quality=7, macro_block_size=4,
                        ffmpeg_params=["-pix_fmt", "yuv420p"]) as w:
    for i in range(n):
        w.append_data(np.concatenate([np.concatenate([cols[0][i], cols[1][i]], 1),
                                      np.concatenate([cols[2][i], cols[3][i]], 1)], 0))
Image.fromarray(np.concatenate([np.concatenate([cols[0][-1], cols[1][-1]], 1),
                                np.concatenate([cols[2][-1], cols[3][-1]], 1)], 0)).save("report1/tuck_end.png")
print("saved report1/tuck.mp4")
