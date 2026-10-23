#!/usr/bin/env python3
"""Run the real-duck test routine in MuJoCo and save obs in the runtime's format, so
../analyse2.py can compare sim with real runs.
    .venv/bin/python sim_routine.py BEST_WALK_ONNX_2.onnx out.pkl [--xml ...] [--accel-bias 0]"""
import argparse, pickle, numpy as np
from sim_test import Sim, PG

ROUTINE = [(10, [0, 0, 0]), (10, [0.15, 0, 0]), (10, [-0.15, 0, 0]), (10, [0, 0, 1.0]),
           (10, [0, 0, -1.0]), (5, [0, 0.2, 0]), (5, [0, -0.2, 0]), (5, [0, 0, 0])]
ap = argparse.ArgumentParser()
ap.add_argument("onnx"); ap.add_argument("out")
ap.add_argument("--xml", default=f"{PG}/xmls/scene_flat_terrain.xml")
ap.add_argument("--accel-bias", type=float, default=0.0)
ap.add_argument("--action-scale", type=float, default=0.25)
ap.add_argument("--phase-factor", type=float, default=1.0)
a = ap.parse_args()
s = Sim(a.onnx, a.xml, a.accel_bias, a.action_scale, a.phase_factor)
obs = []
for sec, cmd in ROUTINE:
    for _ in range(int(sec * 50)):
        o, _ = s.control(np.array(cmd + [0, 0, 0, 0], float))
        o = o.copy(); o[3] -= a.accel_bias      # save the physical reading
        obs.append(o)
pickle.dump([np.array(x, dtype=np.float64) for x in obs], open(a.out, "wb"))
print("saved", a.out, len(obs))
