#!/usr/bin/env python3
"""Render policies running the test routine side by side, camera following the duck.

    MUJOCO_GL=egl .venv/bin/python render_video.py out.mp4 \
        "Current brain=BEST_WALK_ONNX_2.onnx" "Stage 2=path.onnx:head" [--torque 2.0]

":head" after a path = policy trained with HEAD_FROM_CMD (head follows commands);
":auto" = HEAD_FROM_CMD plus the automatic head pose from autohead.py.
"""
import argparse

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from autohead import AutoHead
from sim_test import PG, Sim

XML = f"{PG}/xmls/scene_flat_terrain_backlash.xml"
ROUTINE = [("standing", 2.5, [0, 0, 0]), ("forward", 4, [0.15, 0, 0]), ("backward", 4, [-0.15, 0, 0]),
           ("turn left", 3.5, [0, 0, 1.0]), ("turn right", 3.5, [0, 0, -1.0]),
           ("side-step left", 3.5, [0, 0.2, 0]), ("side-step right", 3.5, [0, -0.2, 0]),
           ("head moves (standing)", 5, "head"), ("standing", 1.5, [0, 0, 0])]
HEAD_SWING = [[0.6, 0, 0, 0], [0, 0, 0.35, 0], [0, 0, -0.35, 0], [0, 0, 0, 0.5], [0, 0, 0, -0.5]]
W, H, FPS = 480, 400, 25
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
except OSError:
    FONT = SMALL = ImageFont.load_default()


def rollout_frames(label, onnx, mode, torque, opts=()):
    extra = {"head_from_cmd": mode in ("head", "auto"), "head_add_cmd": mode == "plain"}
    for o in opts:                                  # ":lpf=8" action low-pass, ":vel=4" motor speed limit
        k, v = o.split("=")
        extra[{"lpf": "lpf_hz", "vel": "max_motor_vel", "alpf": "ankle_lpf_hz", "avel": "ankle_max_vel", "las": "leg_action_scale",
               "ref": "use_ref"}[k]] = float(v)       # ":ref=0" = ignore the REF_CLIP env (old gait clock) for this column
    s = Sim(onnx, XML, 0.0, extra=extra)
    s.m.actuator_forcerange[:, :] = [-torque, torque]
    r = mujoco.Renderer(s.m, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = s.base
    cam.distance, cam.azimuth, cam.elevation = 0.95, 140, -12
    ah = AutoHead()
    frames, step = [], 0
    for name, sec, cmd in ROUTINE:
        n = int(sec * 50)
        for k in range(n):
            if cmd == "head":
                drive, head = [0, 0, 0], HEAD_SWING[min(k * len(HEAD_SWING) // n, len(HEAD_SWING) - 1)]
            else:
                drive, head = cmd, [0, 0, 0, 0]
            if mode == "auto":
                head = list(ah(*drive, extra=np.array(head, float)))
            s.control(np.array(list(drive) + list(head), float))
            if step % 2 == 0:                       # 50 Hz control -> 25 fps video
                r.update_scene(s.d, cam)
                img = Image.fromarray(r.render())
                d = ImageDraw.Draw(img)
                d.rectangle([0, 0, W, 30], fill=(20, 20, 20))
                d.text((8, 4), label, font=FONT, fill=(255, 255, 255))
                d.rectangle([0, H - 26, W, H], fill=(20, 20, 20))
                fell = " - FALLEN" if s.base_pose()[2] > 45 else ""
                d.text((8, H - 23), f"{name}{fell}", font=SMALL, fill=(255, 210, 90) if not fell else (255, 90, 90))
                frames.append(np.asarray(img))
            step += 1
    r.close()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("policies", nargs="+", help='"Label=path.onnx[:head|:auto][:lpf=8][:vel=4]"')
    ap.add_argument("--torque", type=float, default=2.0)
    a = ap.parse_args()
    columns = []
    for spec in a.policies:
        label, rest = spec.split("=", 1)
        path, *tags = rest.split(":")
        mode = next((t for t in tags if "=" not in t), "plain")
        columns.append(rollout_frames(label, path, mode, a.torque, [t for t in tags if "=" in t]))
        print("rendered", label, len(columns[-1]), "frames")
    n = min(len(c) for c in columns)
    with imageio.get_writer(a.out, fps=FPS, codec="libx264", quality=7, macro_block_size=16,
                            ffmpeg_params=["-pix_fmt", "yuv420p"]) as w:
        for i in range(n):
            w.append_data(np.concatenate([c[i] for c in columns], axis=1))
    print("saved", a.out)


if __name__ == "__main__":
    main()
