"""A joint sitting a couple of degrees off from where the policy thinks it is: what drift does it cause?"""
import numpy as np
from sim_test import Sim, PG, run
XML = f"{PG}/xmls/scene_flat_terrain_backlash.xml"
J = ["left_hip_yaw","left_hip_roll","left_hip_pitch","left_knee","left_ankle","neck","head_pitch","head_yaw","head_roll",
     "right_hip_yaw","right_hip_roll","right_hip_pitch","right_knee","right_ankle"]
cases = [("none", {})] + [(f"{j} {d:+d} deg", {j: d}) for j in ["left_hip_yaw","left_hip_roll","left_hip_pitch","left_ankle","right_hip_roll"] for d in (2, -2)]
print(f"{'':26s}  forward: fwd  side  turn |  turn L: fwd  side | stand still?")
for name, b in cases:
    bias = np.zeros(14)
    for j, d in b.items(): bias[J.index(j)] = np.radians(d)
    out = []
    for cmd in ([0.15, 0, 0], [0, 0, 1.0], [0, 0, 0]):
        s = Sim("BEST_WALK_ONNX_2.onnx", XML, 0.0, extra={"joint_bias": bias})
        out.append(run(s, cmd + [0, 0, 0, 0], 8))
    f, t, z = out
    print(f"{name:26s}  {f['fwd']:+.3f} {f['left']:+.3f} {f['yaw']:+5.1f} | {t['fwd']:+.3f} {t['left']:+.3f} | "
          f"{'yes' if z['feet'].mean() > 0.97 else 'shuffles'} tilt {z['tilt_max']:.1f}{' FELL' if any(r['fell'] for r in out) else ''}")
