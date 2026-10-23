import pickle, numpy as np
from numpy.lib.stride_tricks import sliding_window_view
def median_filter(x,k):
    p=np.pad(x,(k//2,k-1-k//2),mode='edge'); return np.median(sliding_window_view(p,k),axis=1)
o = np.array(pickle.load(open(''+__import__('sys').argv[1]+'','rb'))); N=len(o)
gyro, acc, cmd = o[:,0:3], o[:,3:6], o[:,6:13]; pos=o[:,13:27]; feet=o[:,97:99]; act=o[:,41:55]
g = np.linalg.norm(acc,axis=1)
print("accel norm median", np.median(g).round(2), " p5/p95", np.percentile(g,[5,95]).round(1))
back = np.degrees(np.arctan2(acc[:,0],acc[:,2])); right=np.degrees(np.arctan2(acc[:,1],acc[:,2]))
sb = median_filter(back, 25); sr = median_filter(right, 25)   # 0.5 s median: removes step impacts
# pauses: jumps between consecutive frames in joint pos or last_action discontinuity
jump = np.abs(np.diff(pos,axis=0)).max(1)
djump = np.where(jump > np.radians(8))[0]
# last_action shift consistency: obs[t].last_last_action should equal obs[t-1].last_action unless a pause/restart happened? (actions persist across pause) -> use time gap via shifted equality anyway
la, lla = o[:,41:55], o[:,55:69]
brk = np.where(np.abs(lla[1:] - la[:-1]).max(1) > 1e-6)[0]
print("frames where action history doesn't chain:", len(brk), brk[:20])
print("big joint jumps between frames (>8deg):", len(djump), (djump/50).round(1)[:40])
# upright vs not
up = (np.abs(sb) < 20) & (np.abs(sr) < 20)
print(f"upright (smoothed tilt <20°): {up.mean()*100:.0f}% of recorded time")
# fall-ish: smoothed tilt crosses 30°
f = np.where(~((np.abs(sb) < 30) & (np.abs(sr) < 30)))[0]
if len(f):
    ev = np.split(f, np.where(np.diff(f) > 10)[0]+1)
    print(f"\nSUSTAINED TILT >30° episodes: {len(ev)}")
    for e in ev:
        a=e[0]; pre=slice(max(0,a-50),a)
        c = cmd[max(0,a-50):a,:3].mean(0)
        print(f"  {a/50:6.1f}s for {len(e)/50:4.1f}s: goes {'BACK' if sb[e[len(e)//2]]>0 else 'FWD'} {abs(sb[e[len(e)//2]]):.0f}° / {'RIGHT' if sr[e[len(e)//2]]>0 else 'LEFT'} {abs(sr[e[len(e)//2]]):.0f}°   "
              f"prior 1 s: tilt back {sb[pre].mean():+.1f} right {sr[pre].mean():+.1f}, cmd vx {c[0]:+.2f} vy {c[1]:+.2f} w {c[2]:+.2f}")
# upright stats by command type
def lab(c):
    vx,vy,w=c[:3]; s=[]
    if abs(vx)>.02: s.append("fwd" if vx>0 else "back")
    if abs(vy)>.02: s.append("left" if vy>0 else "right")
    if abs(w)>.05: s.append("turnL" if w>0 else "turnR")
    return "+".join(s) or "in-place"
labs=np.array([lab(c) for c in cmd])
print("\nWHILE UPRIGHT, by command (smoothed tilt; sway = std of roll over 0.6 s gait):")
for l in np.unique(labs):
    idx = np.where((labs==l)&up)[0]
    if len(idx)<25: continue
    sway = np.std(right[idx])
    print(f"  {l:12s} {len(idx)/50:6.1f}s  lean {'BACK' if sb[idx].mean()>0 else 'FWD '} {abs(sb[idx].mean()):4.1f}°  {'RIGHT' if sr[idx].mean()>0 else 'LEFT '} {abs(sr[idx].mean()):4.1f}°   "
          f"pitch wobble ±{np.std(back[idx]):.1f}° roll wobble ±{sway:.1f}°  feet down L {feet[idx,0].mean()*100:.0f}% R {feet[idx,1].mean()*100:.0f}% both {((feet[idx]>.5).all(1)).mean()*100:.0f}%")
# drift in place: yaw rate while commanded zero
idx=np.where((labs=="in-place")&up)[0]
print(f"\nIn place: mean yaw rate {np.degrees(gyro[idx,2].mean()):+.1f} deg/s (+ = turning left)")
for l in ["turnL","turnR"]:
    idx=np.where((labs==l)&up)[0]
    if len(idx): print(f"  {l}: commanded {np.degrees(cmd[idx,2].mean()):+.0f} deg/s, actual {np.degrees(gyro[idx,2].mean()):+.0f} deg/s")
