#!/usr/bin/env python3
"""What a stiffer servo (strength test 3: real ~1.5x sim kp) does to the current brain in sim."""
import sys

import eval_policy as ev

orig = ev.make
for scale in [1.0, 1.3, 1.5, 1.7]:
    def make(onnx, torque, scale=scale):
        s = orig(onnx, torque)
        s.m.actuator_gainprm[:, 0] *= scale
        s.m.actuator_biasprm[:, 1] *= scale
        return s
    ev.make = make
    o = ev.evaluate("BEST_WALK_ONNX_2.onnx", float(sys.argv[1]) if len(sys.argv) > 1 else 2.3)
    sp = o["speeds"]
    print(f"kp x{scale}: fwd {sp['fwd']:.3f} back {sp['back']:.3f} L {sp['left']:.3f} R {sp['right']:.3f} "
          f"turnL {sp['turnL']:.0f} turnR {sp['turnR']:.0f} | sway {o['fwd_sway']} jit {o['fwd_jit']} "
          f"stand_jit {o['stand_jit']:.1f} push {o['push']}/8 fell {o['fell']} starts {o['starts']}", flush=True)
