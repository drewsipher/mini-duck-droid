import pickle, sys, numpy as np
o = np.array(pickle.load(open(sys.argv[1], 'rb')))
J = ["L_hyaw","L_hroll","L_hpitch","L_knee","L_ankle","neck","hpitch","hyaw","hroll","R_hyaw","R_hroll","R_hpitch","R_knee","R_ankle"]
gyro, acc, cmd = o[:, 0:3], o[:, 3:6], o[:, 6:13]
pos = o[:, 13:27]; vel = o[:, 27:41] / 0.05; act = o[:, 41:55]; tgt = o[:, 83:97]; feet = o[:, 97:99]; ph = o[:, 99:101]
init = None
# tgt is absolute; pos is relative to init. Recover init from the first step's target ~ init when action ~0
N = len(o); print(f"{N} steps = {N/50:.0f} s at 50 Hz")
back = np.degrees(np.arctan2(acc[:, 0], acc[:, 2])); right = np.degrees(np.arctan2(acc[:, 1], acc[:, 2]))
# phase jumps = pauses/resumes
ang = np.unwrap(np.arctan2(ph[:, 1], ph[:, 0])); d = np.diff(ang)
med = np.median(d); gaps = np.where(np.abs(d - med) > 0.3)[0]
print("phase step median", round(med, 4), "-> gait period", round(2*np.pi/med/50, 2), "s; resumes/gaps at", list(gaps[:40]))
# segment by command
def label(c):
    vx, vy, w = c[:3]
    s = []
    if abs(vx) > 0.02: s.append("fwd" if vx > 0 else "back")
    if abs(vy) > 0.02: s.append("left" if vy > 0 else "right")
    if abs(w) > 0.05: s.append("turnL" if w > 0 else "turnR")
    return "+".join(s) or "stand"
labs = [label(c) for c in cmd]
segs = []; start = 0
for i in range(1, N + 1):
    if i == N or labs[i] != labs[start] or (i - 1) in gaps:
        segs.append((start, i, labs[start])); start = i
print("\ncommand ranges seen: vx", cmd[:,0].min().round(3), cmd[:,0].max().round(3), " vy", cmd[:,1].min().round(3), cmd[:,1].max().round(3), " yaw", cmd[:,2].min().round(3), cmd[:,2].max().round(3))
print("head cmds max abs", np.abs(cmd[:,3:]).max(0).round(2))
print("\nSEGMENTS (>=1 s):")
for a, b, l in segs:
    if b - a < 50: continue
    sl = slice(a, b)
    print(f"  {a/50:6.1f}-{b/50:6.1f}s {l:12s} vx {cmd[sl,0].mean():+.2f} vy {cmd[sl,1].mean():+.2f} w {cmd[sl,2].mean():+.2f} | "
          f"back {back[sl].mean():+5.1f} (min {back[sl].min():+5.1f} max {back[sl].max():+5.1f}) right {right[sl].mean():+5.1f} (±{right[sl].std():.1f}) | "
          f"gyro rms {np.sqrt((gyro[sl]**2).mean(0)).round(2)} | feet both-off {np.mean((feet[sl]<0.5).all(1))*100:3.0f}% L {feet[sl,0].mean()*100:3.0f}% R {feet[sl,1].mean()*100:3.0f}%")
# summary per command type
print("\nBY COMMAND TYPE:")
from collections import defaultdict
agg = defaultdict(list)
for i, l in enumerate(labs): agg[l].append(i)
for l, idx in sorted(agg.items(), key=lambda x: -len(x[1])):
    idx = np.array(idx)
    print(f"  {l:12s} {len(idx)/50:6.1f}s  back {back[idx].mean():+5.1f}±{back[idx].std():.1f}  right {right[idx].mean():+5.1f}±{right[idx].std():.1f}  "
          f"gyro rms(r,p,y) {np.sqrt((gyro[idx]**2).mean(0)).round(2)}  |tilt|>15°: {np.mean((np.abs(back[idx])>15)|(np.abs(right[idx])>15))*100:.1f}%")
# tracking: compare pos (relative) with target relative. Need init: init = tgt - act*scale ... get it from pos+init == ~tgt at rest. Estimate init as median(tgt - pos) over low-velocity steps
still = np.abs(vel).max(1) < 0.2
init_est = np.median((tgt - pos)[still] if still.sum() > 50 else (tgt - pos), axis=0)
rel_t = tgt - init_est
err = np.degrees(pos - rel_t)
print("\nJOINT TRACKING (target vs measured, deg)  [target in obs is the one sent the step before]")
print(f"  {'joint':9s} {'rms err':>8s} {'p95':>6s} {'max':>6s} {'mean':>6s}  {'target range':>14s} {'vel p99 rad/s':>13s}")
for k, n in enumerate(J):
    e = err[:, k]
    print(f"  {n:9s} {np.sqrt((e**2).mean()):8.2f} {np.percentile(np.abs(e),95):6.2f} {np.abs(e).max():6.1f} {e.mean():+6.2f}  "
          f"{np.degrees(rel_t[:,k].min()):+6.1f}..{np.degrees(rel_t[:,k].max()):+5.1f} {np.percentile(np.abs(vel[:,k]),99):13.2f}")
# lag: cross-correlate target and measured for leg joints
print("\nLAG (steps of 20 ms, measured behind target) for legs:")
for k in [0,1,2,3,4,9,10,11,12,13]:
    t = rel_t[:, k] - rel_t[:, k].mean(); p = pos[:, k] - pos[:, k].mean()
    best = max(range(0, 8), key=lambda L: np.dot(t[:N-L], p[L:]) / (np.linalg.norm(t[:N-L]) * np.linalg.norm(p[L:]) + 1e-9))
    amp = np.std(p) / (np.std(t) + 1e-9)
    print(f"  {J[k]:9s} lag {best}  amplitude ratio measured/target {amp:.2f}")
# action saturation
print("\nACTIONS: |action| p99", np.percentile(np.abs(act), 99, axis=0).round(2))
# falls: large tilt events
big = np.where((np.abs(back) > 25) | (np.abs(right) > 25))[0]
if len(big):
    ev = np.split(big, np.where(np.diff(big) > 25)[0] + 1)
    print(f"\nBIG TILT EVENTS (>25°): {len(ev)}")
    for e in ev:
        a = e[0]; pre = slice(max(0, a - 75), a)
        print(f"  at {a/50:6.1f}s cmd {labs[a]:10s} tilt back {back[a]:+.0f} right {right[a]:+.0f}; 1.5 s before: back {back[pre].mean():+.1f} right {right[pre].mean():+.1f}, cmd {labs[max(0,a-50)]}")
np.save("init_est.npy", init_est)
