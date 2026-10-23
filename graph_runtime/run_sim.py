#!/usr/bin/env python3
"""Run a scripted operator session through the motion graph engine in MuJoCo and render it.

    cd <project> && MUJOCO_GL=egl sim/.venv/bin/python graph_runtime/run_sim.py \
        --graph graphs/demo.json --script "idle 2s; walk 0.12 3s; bow; walk 0.12 2s; abort" --out demo.mp4

Script steps, separated by ";":
    idle <dur>                     velocity 0 (locomotion at zero command) for <dur>
    walk <vx> [<vy> <wz>] <dur>    velocity command for <dur>
    turn <wz> <dur>                yaw-rate command for <dur>
    head <neck> <pitch> <yaw> <roll> <dur>
    wait <dur>                     keep the current command
    <skill>                        trigger that skill, then wait until the engine is back in a locomotion node
    abort                          operator abort (recovery = sit), then wait until relaxed
    reset                          re-arm after a recovery (needs a return edge in the graph)
Durations are seconds, "2s" or "2". The video shows the active skill, the transition state, the
command, tilt and phase; the console gets the event log and a summary (falls, timings).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from graph_runtime.backends.mujoco import COLLISION_XML, DEFAULT_XML, MujocoBackend   # noqa: E402
from graph_runtime.engine import Engine, State                                        # noqa: E402
from graph_runtime.graph import describe, load_graph                                  # noqa: E402

FALL_TILT = 45.0


def parse_duration(s: str) -> float:
    return float(s[:-1]) if s.endswith("s") else float(s)


def parse_script(text: str) -> list:
    steps = []
    for raw in text.split(";"):
        parts = raw.strip().split()
        if not parts:
            continue
        op = parts[0].lower()
        if op == "idle":
            steps.append(("idle", parse_duration(parts[1])))
        elif op == "walk":
            nums = [float(p.rstrip("s")) for p in parts[1:]]
            if len(nums) == 2:
                steps.append(("walk", (nums[0], 0.0, 0.0), parse_duration(parts[-1])))
            elif len(nums) == 4:
                steps.append(("walk", tuple(nums[:3]), parse_duration(parts[-1])))
            else:
                raise SystemExit(f"walk needs 'walk vx dur' or 'walk vx vy wz dur': {raw!r}")
        elif op == "turn":
            steps.append(("walk", (0.0, 0.0, float(parts[1])), parse_duration(parts[2])))
        elif op == "head":
            steps.append(("head", tuple(float(p) for p in parts[1:5]), parse_duration(parts[5])))
        elif op == "wait":
            steps.append(("wait", parse_duration(parts[1])))
        elif op in ("abort", "reset"):
            steps.append((op,))
        else:
            steps.append(("trigger", parts[0]))
    return steps


class Recorder:
    def __init__(self, backend, out, fps=25, control_hz=50):
        self.backend, self.out, self.fps = backend, out, fps
        self.every = max(1, int(round(control_hz / fps)))
        self.frames = []
        from PIL import ImageFont
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            self.small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except OSError:
            self.font = self.small = ImageFont.load_default()

    def capture(self, info, step_text):
        if (info.tick - 1) % self.every:
            return
        from PIL import Image, ImageDraw
        img = Image.fromarray(self.backend.render())
        W, H = img.size
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W, 32], fill=(20, 20, 20))
        colour = (255, 90, 90) if info.recovering else ((140, 220, 255) if info.overlay else (255, 255, 255))
        d.text((8, 5), info.label, font=self.font, fill=colour)
        d.text((W - 92, 8), f"t={info.t:5.2f}s", font=self.small, fill=(200, 200, 200))
        fell = info.tilt_deg > FALL_TILT
        d.rectangle([0, H - 44, W, H], fill=(20, 20, 20))
        status = info.status if len(info.status) < 58 else info.status[:55] + "..."
        d.text((8, H - 42), status, font=self.small, fill=(255, 210, 90) if not fell else (255, 90, 90))
        cmd = info.command
        line = (f"cmd {cmd[0]:+.2f} {cmd[1]:+.2f} {cmd[2]:+.2f}  tilt {info.tilt_deg:4.1f}  phase {info.phase:.2f}"
                f"  blend {100 * info.blend:3.0f}%" + ("  FALLEN" if fell else "") + f"   [{step_text}]")
        d.text((8, H - 22), line, font=self.small, fill=(200, 200, 200) if not fell else (255, 90, 90))
        self.frames.append(np.asarray(img))

    def save(self):
        import imageio.v2 as imageio
        with imageio.get_writer(self.out, fps=self.fps, codec="libx264", quality=7, macro_block_size=16,
                                ffmpeg_params=["-pix_fmt", "yuv420p"]) as w:
            for f in self.frames:
                w.append_data(f)


def run_script(engine: Engine, steps: list, on_tick=None, max_wait=15.0) -> list:
    infos = []
    step_text = ""

    def go(seconds, until=None):
        for _ in range(int(round(seconds / engine.dt))):
            info = engine.tick()
            infos.append(info)
            if on_tick:
                on_tick(info, step_text)
            if until is not None and until(info):
                return True
        return False

    for st in steps:
        op = st[0]
        step_text = " ".join(str(x) for x in st)
        if op == "idle":
            engine.set_velocity(0.0, 0.0, 0.0)
            go(st[1])
        elif op == "walk":
            engine.set_velocity(*st[1])
            go(st[2])
        elif op == "head":
            engine.set_head(*st[1])
            go(st[2])
        elif op == "wait":
            go(st[1])
        elif op == "trigger":
            name = st[1]
            t0 = engine.t
            while not engine.trigger(name):          # busy: retry for a while
                if engine.relaxed or engine.t - t0 > 3.0:
                    break
                go(engine.dt)
            entered = {"v": False}
            is_overlay = engine.graph.is_overlay(name)

            def back(info, name=name):
                if info.active == name or info.overlay == name:
                    entered["v"] = True
                if info.state == State.RELAXED:
                    return True
                if is_overlay:
                    return entered["v"] and info.overlay is None and engine.overlay_req is None
                loco = engine.graph.nodes[info.active].type == "locomotion"
                return entered["v"] and loco and info.state == State.RUNNING
            go(max_wait, until=back)
        elif op == "abort":
            engine.abort("script abort")
            go(max_wait, until=lambda i: i.state == State.RELAXED)
            go(1.5)
        elif op == "reset":
            engine.reset()
            go(2.0)
    return infos


def summarize(engine: Engine, infos: list) -> dict:
    fell = [i for i in infos if i.tilt_deg > FALL_TILT]
    segs = []
    for k, (skill, t0) in enumerate(engine.segments):
        t1 = engine.segments[k + 1][1] if k + 1 < len(engine.segments) else engine.t
        segs.append({"skill": skill, "enter_s": round(t0, 2), "seconds": round(t1 - t0, 2)})
    return {"seconds": round(engine.t, 2), "final_state": engine.state.value, "final_skill": engine.active,
            "recovery_reason": engine.recovery_reason, "fell": bool(fell), "first_fall_s": round(fell[0].t, 2) if fell else None,
            "max_tilt_deg": round(max(i.tilt_deg for i in infos), 1) if infos else None,
            "min_height_m": round(min(i.height_m for i in infos if i.height_m is not None), 3) if infos else None,
            "segments": segs, "events": [str(e) for e in engine.events]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--script", default="idle 2s; walk 0.12 3s; bow; walk 0.12 2s; abort")
    ap.add_argument("--out", default=None, help="mp4 path (omit for no video)")
    ap.add_argument("--xml", default=str(DEFAULT_XML), help="MJCF scene, or 'collision' for the body-collision variant")
    ap.add_argument("--torque", type=float, default=1.86)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--pose-noise", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--log", default=None, help="write per-tick log and summary JSON here")
    ap.add_argument("--no-safety", action="store_true")
    ap.add_argument("--allow-failed-battery", action="store_true",
                    help="SIM ONLY: load packages whose battery failed, to watch what goes wrong")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    xml = COLLISION_XML if a.xml == "collision" else a.xml
    if a.allow_failed_battery:
        print("WARNING: --allow-failed-battery: loading packages that did not pass their battery (simulation only)")
    g = load_graph(a.graph, project_root=PROJECT, require_battery=not a.allow_failed_battery)
    print(describe(g))
    be = MujocoBackend(xml=xml, torque_nm=a.torque, seed=a.seed, pose_noise=a.pose_noise)
    eng = Engine(g, be, safety=not a.no_safety, log=None if a.quiet else print)
    rec = Recorder(be, a.out, fps=a.fps) if a.out else None
    steps = parse_script(a.script)
    print("script:", steps)
    t0 = time.time()
    infos = run_script(eng, steps, on_tick=rec.capture if rec else None)
    wall = time.time() - t0
    s = summarize(eng, infos)
    print(f"\nsummary: {s['seconds']} s simulated in {wall:.1f} s wall; final {s['final_skill']} ({s['final_state']}); "
          f"fell {s['fell']}" + (f" at {s['first_fall_s']} s" if s['fell'] else "") +
          f"; max tilt {s['max_tilt_deg']} deg; min height {s['min_height_m']} m")
    for seg in s["segments"]:
        print(f"  {seg['enter_s']:6.2f}s  {seg['skill']:8s} {seg['seconds']:.2f}s")
    if rec:
        rec.save()
        print("video", a.out)
    if a.log:
        ticks = [{"t": round(i.t, 3), "label": i.label, "state": i.state.value, "blend": round(i.blend, 3), "tilt": round(i.tilt_deg, 2),
                  "height": None if i.height_m is None else round(i.height_m, 4), "speed": round(i.speed, 3), "phase": round(i.phase, 3)}
                 for i in infos]
        Path(a.log).write_text(json.dumps({"summary": s, "script": a.script, "graph": a.graph, "ticks": ticks}, indent=1))
        print("log", a.log)
    be.close()
    return 0 if not s["fell"] else 1


if __name__ == "__main__":
    sys.exit(main())
