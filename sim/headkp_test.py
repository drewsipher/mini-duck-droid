#!/usr/bin/env python3
"""The real runtime sets the 4 head servos to P=8 (legs P=30); the sim gives all 14 the same kp.
Strength test: real P=30 ~ 1.48x sim kp, so real head ~ 1.48*8/30 = 0.39x sim kp. Effect on the current brain."""
import eval_policy as ev

orig = ev.make
for label, legs, head in [("sim as trained (all 1.0x)", 1.0, 1.0), ("real legs 1.48x, head same", 1.48, 1.48),
                          ("real legs 1.48x, head soft 0.39x", 1.48, 0.39)]:
    def make(onnx, torque, legs=legs, head=head):
        s = orig(onnx, torque)
        sc = [legs] * 14
        sc[5:9] = [head] * 4
        for i, k in enumerate(sc):
            s.m.actuator_gainprm[i, 0] *= k
            s.m.actuator_biasprm[i, 1] *= k
        return s
    ev.make = make
    o = ev.evaluate("BEST_WALK_ONNX_2.onnx", 2.3)
    sp = o["speeds"]
    print(f"{label:34s} fwd {sp['fwd']:.3f} back {sp['back']:.3f} turnL {sp['turnL']:.0f} turnR {sp['turnR']:.0f} | "
          f"sway roll/pitch {o['fwd_sway'][0]:.1f}/{o['fwd_sway'][1]:.1f} jit {o['fwd_jit'][0]:.1f}/{o['fwd_jit'][1]:.1f} "
          f"push {o['push']}/8 head-swing steps {o['head_lift']:.2f} fell {o['fell'] or o['head_fell']} head@rest {o['head'].round(1)}", flush=True)
