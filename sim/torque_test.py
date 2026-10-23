import numpy as np, importlib.util
from sim_test import Sim, PG, run
spec = importlib.util.spec_from_file_location("sc", "smooth_cmp_lib.py"); sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
from sim_routine_lib import ROUTINE
XML = f"{PG}/xmls/scene_flat_terrain_backlash.xml"
TESTS=[("forward",[0.15,0,0]),("backward",[-0.15,0,0]),("step L",[0,0.2,0]),("turn L",[0,0,1.0]),("turn R",[0,0,-1.0])]
print(f"{'torque limit':14s}" + "".join(f"{n:>17s}" for n,_ in TESTS) + "   fwd sway r/p  fwd jit p  routine")
for lim in (3.35, 2.5, 1.86, 1.5):
    cells=[]
    for n,cmd in TESTS:
        s=Sim("BEST_WALK_ONNX_2.onnx", XML, 0.0)
        s.m.actuator_forcerange[:, :] = [-lim, lim]
        r=run(s, cmd+[0,0,0,0], 8)
        v = r['yaw'] if 'turn' in n else (r['left'] if 'step' in n else r['fwd'])
        cells.append(f"{v:+8.3f}{' FELL' if r['fell'] else '     '}    ")
    s=Sim("BEST_WALK_ONNX_2.onnx", XML, 0.0); s.m.actuator_forcerange[:, :] = [-lim, lim]
    obs=[]; fell=False
    for sec,cmd in ROUTINE:
        for _ in range(int(sec*50)):
            o,_=s.control(np.array(cmd+[0,0,0,0],float)); obs.append(o); fell |= s.base_pose()[2]>45
    rr=sc.summary(np.array(obs,dtype=np.float64)); g=lambda k,i,j: rr[k][i][j] if k in rr else np.nan
    print(f"{lim:5.2f} N·m     " + "".join(cells) + f"  {g('fwd',0,0):5.1f}/{g('fwd',0,1):5.1f}  {g('fwd',1,1):6.1f}   {'FELL' if fell else 'ok'}")
print("real walk 7                                                                         28.9/ 20.3    10.4")
