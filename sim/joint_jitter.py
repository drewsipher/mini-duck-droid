import pickle, sys, numpy as np
sys.path.insert(0, '.')
import importlib.util
spec = importlib.util.spec_from_file_location("sc", "smooth_cmp_lib.py"); sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
J = ["L hyaw","L hrol","L hpit","L knee","L ank","neck","hpitch","hyaw","hroll","R hyaw","R hrol","R hpit","R knee","R ank"]
def hf(x):  # >8 Hz RMS of a signal, deg
    x = x - x.mean(0); f = np.fft.rfftfreq(len(x), 1/50); X = np.fft.rfft(x, axis=0)
    return np.degrees(np.sqrt((np.fft.irfft(X * (f > 8)[:, None], len(x), axis=0)**2).mean(0)))
def segs(o, name):
    lab = [sc.label(c) for c in o[:, 6:9]]
    idx = np.array([i for i, l in enumerate(lab) if l == name])
    return [r for r in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1) if len(r) >= 150]
for name in ["fwd", "in-place"]:
    print(f"\n{name}: fast (>8 Hz) joint motion, deg RMS   [position | target the policy sent]")
    print(f"{'':8s}" + "".join(f"{j:>7s}" for j in J))
    for path in sys.argv[1:]:
        o = np.array(pickle.load(open(path, "rb")))
        rs = segs(o, name)
        if not rs: continue
        pos = np.mean([hf(o[r, 13:27]) for r in rs], 0); tgt = np.mean([hf(o[r, 83:97]) for r in rs], 0)
        print(f"{path.split('/')[-1][:8]:8s}" + "".join(f"{v:7.2f}" for v in pos) + "  pos")
        print(f"{'':8s}" + "".join(f"{v:7.2f}" for v in tgt) + "  target")
