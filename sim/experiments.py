"""Which real-duck differences reproduce his extra front-back sway and jitter?"""
import sys, pickle, collections, numpy as np
from sim_test import Sim, PG
from sim_routine_lib import ROUTINE
sys.argv = ["x"]
import importlib.util
spec = importlib.util.spec_from_file_location("sc", "smooth_cmp_lib.py"); sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)

class Sim2(Sim):
    def __init__(self, *a, head_kp_scale=1.0, kp_scale=1.0, act_delay=0, imu_delay=0, **k):
        super().__init__(*a, **k)
        for i in range(self.m.nu):
            s = kp_scale * (head_kp_scale if 5 <= i <= 8 else 1.0)
            kp = self.m.actuator_gainprm[i, 0] * s
            self.m.actuator_gainprm[i, 0] = kp; self.m.actuator_biasprm[i, 1] = -kp
        self.act_q = collections.deque([self.default.copy()] * act_delay)
        self.imu_q = collections.deque(maxlen=imu_delay + 1)
    def obs(self, cmd):
        o = super().obs(cmd)
        self.imu_q.append(o[0:6].copy()); o[0:6] = self.imu_q[0]
        return o
    def control(self, cmd):
        if not self.act_q:
            return super().control(cmd)
        real_step = mujoco_step = None
        # run the policy, but apply the target from act_delay ticks ago
        o, a = self._policy_only(cmd)
        self.act_q.append(self.targets.copy()); self.d.ctrl[:] = self.act_q.popleft()
        import mujoco
        for _ in range(self.decim): mujoco.mj_step(self.m, self.d)
        return o, a
    def _policy_only(self, cmd):
        self.imitation_i = (self.imitation_i + self.phase_factor) % self.nb_steps
        ang = self.imitation_i / self.nb_steps * 2 * np.pi
        self.phase = np.array([np.cos(ang), np.sin(ang)])
        o = self.obs(cmd)
        a = self.policy.run(None, {"obs": [o]})[0][0]
        self.a3, self.a2, self.a1 = self.a2, self.a1, a.copy()
        t = np.clip(self.default + a * self.action_scale, self.prev_targets - 0.1048, self.prev_targets + 0.1048)
        self.prev_targets = t.copy(); self.targets = t
        return o, a

def routine(**kw):
    xml = kw.pop("xml", f"{PG}/xmls/scene_flat_terrain_backlash.xml")
    pf = kw.pop("phase_factor", 1.0); onnx = kw.pop("onnx", "BEST_WALK_ONNX_2.onnx")
    s = Sim2(onnx, xml, 0.0, 0.25, pf, **kw)
    obs, fell = [], False
    for sec, cmd in ROUTINE:
        for _ in range(int(sec * 50)):
            o, _ = s.control(np.array(cmd + [0, 0, 0, 0], float)); obs.append(o)
            if s.base_pose()[2] > 45: fell = True
    return np.array(obs, dtype=np.float64), fell

EXPS = [("baseline (backlash model)", {}), ("step timing 0.9 (your -0.1)", {"phase_factor": 0.9}),
        ("head servos soft (8 vs 30)", {"head_kp_scale": 8 / 30}), ("legs 30% weaker", {"kp_scale": 0.7}),
        ("output 1 tick late", {"act_delay": 1}), ("IMU 2 ticks late", {"imu_delay": 2}),
        ("all of the above", {"phase_factor": 0.9, "head_kp_scale": 8 / 30, "kp_scale": 0.7, "act_delay": 1, "imu_delay": 2})]
real = np.array(pickle.load(open("../walk_recordings/obs_walk7.pkl", "rb")))
print(f"{'':30s} {'fwd sway r/p':>13s} {'fwd jit r/p':>12s} {'turn sway p':>12s} {'turn jit p':>11s} {'inplace jit':>12s}")
def row(name, o, fell=False):
    r = sc.summary(o)
    g = lambda k, i, j: r.get(k, (np.full(2, np.nan), np.full(2, np.nan)))[i][j]
    print(f"{name:30s} {g('fwd',0,0):5.1f}/{g('fwd',0,1):5.1f}  {g('fwd',1,0):5.1f}/{g('fwd',1,1):4.1f}  "
          f"{(g('turnL',0,1)+g('turnR',0,1))/2:9.1f}  {(g('turnL',1,1)+g('turnR',1,1))/2:9.1f}  "
          f"{g('in-place',1,0):5.1f}/{g('in-place',1,1):4.1f} {'FELL' if fell else ''}")
row("REAL walk 7", real)
for name, kw in EXPS:
    o, fell = routine(**kw); row(name, o, fell)
