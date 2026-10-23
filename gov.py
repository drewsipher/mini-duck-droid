import pickle,sys,numpy as np,math
sys.path.insert(0,'duck_live'); from stability_governor import StabilityGovernor, tilt_angle_deg, tilt_rate
o=np.array(pickle.load(open(sys.argv[1],'rb'))); acc=o[:,3:6]; gy=o[:,0:3]; cmd=o[:,6:9]
g=StabilityGovernor(True,10,25,2.5,7.0,0.35,0.3); sc=[]
for i in range(len(o)):
    a=acc[i-1] if i else acc[0]; w=gy[i-1] if i else gy[0]
    pitch=math.degrees(math.atan2(-a[0],math.hypot(a[1],a[2]))); roll=math.degrees(math.atan2(a[1],a[2]))
    sc.append(g.update(tilt_angle_deg(pitch,roll),tilt_rate(w)))
sc=np.array(sc); mv=np.abs(cmd).max(1)>0.02
print(f"governor scale (simulated): all {sc.mean():.2f}, while moving {sc[mv].mean():.2f}, p10 {np.percentile(sc[mv],10):.2f}; <0.7 for {np.mean(sc[mv]<0.7)*100:.0f}% of moving time")
bk=cmd[:,0]<-0.02
if bk.any(): print(f"backward: cmd vx mean {cmd[bk,0].mean():+.3f} (stick max -0.15), governor scale {sc[bk].mean():.2f}, implied stick {np.mean(cmd[bk,0]/sc[bk]):+.3f}")
fw=cmd[:,0]>0.02
if fw.any(): print(f"forward:  cmd vx mean {cmd[fw,0].mean():+.3f}, governor scale {sc[fw].mean():.2f}")
# tilt-rate vs tilt angle: which term dominates
pitch=np.degrees(np.arctan2(-acc[:,0],np.hypot(acc[:,1],acc[:,2]))); roll=np.degrees(np.arctan2(acc[:,1],acc[:,2]))
t=np.hypot(pitch,roll); r=np.hypot(gy[:,0],gy[:,1])
print(f"raw per-tick tilt >10° {np.mean(t>10)*100:.0f}% of time; gyro rate >2.5 {np.mean(r>2.5)*100:.0f}%")
