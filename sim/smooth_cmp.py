"""Body shake per command, from the gyro: slow sway (<4 Hz) and fast jitter (>8 Hz), deg/s RMS."""
import pickle, sys, numpy as np
def bands(g):
    g = g - g.mean(0)
    f = np.fft.rfftfreq(len(g), 1/50); G = np.fft.rfft(g, axis=0)
    lo = np.fft.irfft(G * (f < 4)[:, None], len(g), axis=0)
    hi = np.fft.irfft(G * (f > 8)[:, None], len(g), axis=0)
    return np.degrees(np.sqrt((lo**2).mean(0))), np.degrees(np.sqrt((hi**2).mean(0)))
def label(c):
    vx, vy, w = c
    if abs(vx) < .02 and abs(vy) < .02 and abs(w) < .05: return "in-place"
    if abs(vy) < .02 and abs(w) < .05: return "fwd" if vx > 0 else "back"
    if abs(vx) < .02 and abs(w) < .05: return "left" if vy > 0 else "right"
    if abs(vx) < .02 and abs(vy) < .02: return "turnL" if w > 0 else "turnR"
    return None
print(f"{'':28s}{'slow sway roll/pitch':>22s}{'fast jitter roll/pitch':>26s}")
for path in sys.argv[1:]:
    o = np.array(pickle.load(open(path, "rb")))
    from numpy.lib.stride_tricks import sliding_window_view as sw
    a = np.pad(o[:, 3:6], ((12, 12), (0, 0)), mode="edge")
    a = np.median(sw(a, 25, axis=0), axis=-1)          # smoothed accel, to skip falls only
    tilt = np.degrees(np.arctan2(np.hypot(a[:, 0], a[:, 1]), a[:, 2]))
    lab = [label(c) for c in o[:, 6:9]]
    for name in ["in-place", "fwd", "back", "turnL", "turnR", "left", "right"]:
        idx = np.array([i for i, l in enumerate(lab) if l == name and tilt[i] < 20])
        if len(idx) < 100: continue
        # longest contiguous chunks only, so we don't splice segments
        runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        runs = [r for r in runs if len(r) >= 100]
        if not runs: continue
        lo = np.mean([bands(o[r, 0:2])[0] for r in runs], 0); hi = np.mean([bands(o[r, 0:2])[1] for r in runs], 0)
        print(f"  {path.split('/')[-1][:16]:16s} {name:9s}  {lo[0]:6.1f} / {lo[1]:5.1f}            {hi[0]:6.1f} / {hi[1]:5.1f}")
    print()
