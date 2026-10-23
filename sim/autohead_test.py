import numpy as np, eval_policy as e
from autohead import AutoHead
e.HEAD_FROM_CMD = True
F = "train_pg/runs/stage2_head/2026_10_13_181321_100761600.onnx"
import importlib.util
spec = importlib.util.spec_from_file_location("sc", "smooth_cmp_lib.py"); sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
ROUT = [(3, [0, 0, 0]), (6, [0.15, 0, 0]), (6, [0, 0, 1.0]), (6, [0, 0, -1.0]), (6, [0.1, 0, 0.6]), (4, [0, 0.2, 0]), (4, [-0.15, 0, 0]), (3, [0, 0, 0])]
for torque in (1.86, 2.3):
    for mode in ("still head", "auto head"):
        s = e.make(F, torque); ah = AutoHead(); obs = []; fell = False; seg = []
        for sec, c in ROUT:
            p0 = s.d.qpos[:2].copy(); y0 = s.base_pose()[1]; yaw = 0; yp = y0
            for _ in range(int(sec * 50)):
                head = ah(*c) if mode == "auto head" else np.zeros(4)
                o, _ = s.control(np.array(c + list(head), float)); obs.append(o)
                fell |= s.base_pose()[2] > 45
                y = s.base_pose()[1]; yaw += (y - yp + np.pi) % (2 * np.pi) - np.pi; yp = y
            dp = s.d.qpos[:2] - p0; cc, sn = np.cos(-y0), np.sin(-y0)
            seg.append(((cc * dp[0] - sn * dp[1]) / sec, np.degrees(yaw) / sec))
        o = np.array(obs, dtype=np.float64)
        lo, hi = sc.bands(o[150:450, 0:2])
        print(f"{torque} N.m {mode:10s}: fwd {seg[1][0]:+.3f} m/s (pitch sway {lo[1]:.1f}, roll {lo[0]:.1f})  turnL {seg[2][1]:+.0f}  turnR {seg[3][1]:+.0f}  arc fwd {seg[4][0]:+.3f} turn {seg[4][1]:+.0f}  {'FELL' if fell else ''}")
