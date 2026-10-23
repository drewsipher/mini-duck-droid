"""From a standstill, which commands make each policy start stepping?"""
import sys, numpy as np
from sim_test import Sim, PG
xml=f"{PG}/xmls/scene_flat_terrain.xml"
for onnx in ["BEST_WALK_ONNX_2.onnx","BEST_WALK_ONNX.onnx"]:
    print(onnx)
    for label, cmds in [("vy", [[0,v,0] for v in (-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.2)]),
                        ("vx", [[v,0,0] for v in (-0.15,-0.1,-0.05,0.05,0.1,0.15)]),
                        ("yaw",[[0,0,v] for v in (-1,-0.5,-0.2,0.2,0.5,1)]),
                        ("vy- with nudge",[[0.03,-0.2,0],[-0.03,-0.2,0],[0,-0.2,0.1],[0,-0.2,-0.1]])]:
        row=[]
        for c in cmds:
            s=Sim(onnx,xml,0.0)
            for _ in range(100): s.control(np.zeros(7))
            p0=s.d.qpos[:3].copy(); f=[]
            for _ in range(200):
                o,_=s.control(np.array(c+[0,0,0,0],float)); f.append(o[97:99])
            moved=np.linalg.norm(s.d.qpos[:2]-p0[:2])
            row.append(f"{c}: {'STEPS' if np.mean(f)<0.95 else 'stands'} ({moved*100:.0f}cm)")
        print(f"  {label:15s}", " | ".join(row))
