import sys, numpy as np, eval_policy as e
from sim_test import run
e.HEAD_FROM_CMD = "--head-from-cmd" in sys.argv
torque = 2.0
e.START = e.START + [("left.2", [0, 0.2, 0]), ("right.2", [0, -0.2, 0])]
for f in [a for a in sys.argv[1:] if a.endswith(".onnx")]:
    row = []
    for name, cmd in e.START:
        r = run(e.make(f, torque), cmd + [0, 0, 0, 0], 6)
        v = r["yaw"] if "turn" in name else (r["left"] if name in ("left", "right") else r["fwd"])
        row.append(f"{name} {v:+.3f}" if "turn" not in name else f"{name} {v:+.0f}")
    print(f"{f.split('/')[-1][:32]:32s} @{torque}: " + "  ".join(row))
