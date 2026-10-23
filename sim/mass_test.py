import numpy as np, importlib.util, mujoco
from sim_test import Sim, PG
from sim_routine_lib import ROUTINE
spec = importlib.util.spec_from_file_location("sc", "smooth_cmp_lib.py"); sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
XML = f"{PG}/xmls/scene_flat_terrain_backlash.xml"
print(f"{'':22s}{'mass':>6s} {'fwd sway r/p':>13s} {'fwd jit r/p':>12s} {'turn sway p':>11s} {'turn jit p':>10s}  falls")
for onnx in ["BEST_WALK_ONNX_2.onnx", "BEST_WALK_ONNX.onnx"]:
    for scale in (1.0, 0.85, 0.75):
        s = Sim(onnx, XML, 0.0)
        s.m.body_mass[:] *= scale; s.m.body_inertia[:] *= scale
        obs, fell = [], False
        for sec, cmd in ROUTINE:
            for _ in range(int(sec * 50)):
                o, _ = s.control(np.array(cmd + [0, 0, 0, 0], float)); obs.append(o)
                if not fell and s.base_pose()[2] > 45: fell = len(obs) / 50
        r = sc.summary(np.array(obs, dtype=np.float64))
        g = lambda k, i, j: r[k][i][j] if k in r else np.nan
        print(f"{onnx[:20]:22s}{2.107*scale:6.2f} {g('fwd',0,0):6.1f}/{g('fwd',0,1):5.1f} {g('fwd',1,0):6.1f}/{g('fwd',1,1):4.1f} "
              f"{(g('turnL',0,1)+g('turnR',0,1))/2:10.1f} {(g('turnL',1,1)+g('turnR',1,1))/2:10.1f}  {('FELL at %.0fs' % fell) if fell else '-'}")
print("REAL walk 7 (1.5-1.8 kg)        28.9/ 20.3   10.2/10.4       19.4       14.1")
