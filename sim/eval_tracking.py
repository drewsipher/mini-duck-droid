#!/usr/bin/env python3
"""Strict evaluation of a clip-tracking skill: the same thresholds as training termination,
N seeds x M episodes with randomised physics, started at t = 0 (the deploy case), sensors
noise-free, deterministic. Reports a success ratio (e.g. 47/50), the mean joint tracking error
(rad) and the mean base tracking error (m), renders reference ghost vs policy side by side, and
can write the skill package draft (framework §2.5).

    cd sim && MUJOCO_GL=egl PYTHONPATH=train_pg train_pg/.venv/bin/python eval_tracking.py \
        train_pg/runs_skills/bow/preview/policy.onnx ../tracker/data/clips/test_bow.json \
        --name bow --out-dir train_pg/runs_skills/bow --package [--loop] [--seeds 5] [--episodes 10]

Outputs: <out-dir>/eval/battery.json, <out-dir>/eval/eval.mp4 and, with --package,
<out-dir>/package_draft/{skill.json, clip.json, policy.onnx, report/battery.json, report/eval.mp4}.
The observation is built exactly as tracking.py builds it (obs_spec.json next to the ONNX is
checked against the ONNX input size); the action is a residual on the clip's next frame, then
the deploy low-pass (8 Hz) and target-rate clamp (4 rad/s), as in training and the runtime.
"""
import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import mujoco
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PG = os.path.join(HERE, "train_pg")
sys.path.insert(0, TRAIN_PG)
import recipe_v1 as R                                                   # noqa: E402
from playground.common.clip_tracking_reference import ClipTrackingReference, quat_angle, quat_rotate  # noqa: E402

SCENE_XML = os.path.join(TRAIN_PG, R.SCENE_XML)
ROBOT_XML = os.path.join(TRAIN_PG, R.ROBOT_XML)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


class Policy:
    def __init__(self, onnx_path):
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name, self.obs_size = inp.name, int(inp.shape[-1])

    def __call__(self, obs):
        return self.sess.run(None, {self.input_name: obs[None].astype(np.float32)})[0][0]


class TrackingSim:
    """CPU mirror of tracking.ClipTracking for one episode (no sensor noise)."""

    def __init__(self, base_model, ref, policy, loop, pushes=False):
        self.m0, self.ref, self.policy, self.loop, self.pushes = base_model, ref, policy, loop, pushes
        m = base_model
        self.act_names = [m.actuator(i).name for i in range(m.nu)]
        self.qadr = np.array([m.joint(n).qposadr[0] for n in self.act_names])
        self.vadr = np.array([m.joint(n).dofadr[0] for n in self.act_names])
        self.badr = np.array([m.joint(n + "_backlash").qposadr[0] if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n + "_backlash") >= 0 else -1
                              for n in self.act_names])
        s = lambda n: m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)]
        self.gyro_a, self.acc_a = s("gyro"), s("accelerometer")
        self.home = np.array(m.keyframe("home").ctrl)
        self.base_body = ref.base_body
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.alpha = (1.0 / R.ACT_LPF_HZ) / (R.CTRL_DT + 1.0 / R.ACT_LPF_HZ) if R.ACT_LPF_HZ > 0 else 0.0
        self.max_step = R.MAX_MOTOR_VEL * R.CTRL_DT
        self.decim = int(round(R.CTRL_DT / R.SIM_DT))
        self.episode_steps = int(round(R.LOOP_EPISODE_S / R.CTRL_DT)) if loop else ref.episode_steps

    # ---- physics randomisation, the recipe's set (randomize.py), one draw per episode
    def randomised_model(self, rng):
        m = copy.copy(self.m0)
        m.geom_friction[self.floor, 0] = rng.uniform(0.5, 1.0)
        m.dof_frictionloss[self.vadr] = self.m0.dof_frictionloss[self.vadr] * rng.uniform(0.9, 1.1, len(self.vadr))
        m.dof_armature[self.vadr] = self.m0.dof_armature[self.vadr] * rng.uniform(1.0, 1.05, len(self.vadr))
        m.body_ipos[1] = self.m0.body_ipos[1] + rng.uniform(-0.05, 0.05, 3)
        m.body_mass[:] = self.m0.body_mass * rng.uniform(0.9, 1.1, m.nbody)
        m.body_mass[1] += rng.uniform(-0.1, 0.1)
        m.qpos0[self.qadr] = self.m0.qpos0[self.qadr] + rng.uniform(-0.03, 0.03, len(self.qadr))
        f = rng.uniform(R.KP_RANGE[0], R.KP_RANGE[1], m.nu)
        kp = self.m0.actuator_gainprm[:, 0]
        m.actuator_gainprm[:, 0] = kp * f
        m.actuator_biasprm[:, 1] = -kp * f
        lim = rng.uniform(R.TORQUE_RANGE[0], R.TORQUE_RANGE[1], m.nu)
        m.actuator_forcerange[:, 0], m.actuator_forcerange[:, 1] = -lim, lim
        if R.MASS_SCALE:
            m.body_mass[:] *= rng.uniform(R.MASS_SCALE[0], R.MASS_SCALE[1])
        m.opt.timestep = R.SIM_DT
        return m

    def obs(self, d, k, a1, a2, targets, imu_hist, imu_delay):
        gyro = d.sensordata[self.gyro_a:self.gyro_a + 3].copy()
        acc = d.sensordata[self.acc_a:self.acc_a + 3].copy()
        imu_hist.insert(0, np.concatenate([gyro, acc]))
        del imu_hist[4:]
        imu = imu_hist[min(imu_delay, len(imu_hist) - 1)]
        q = d.qpos[self.qadr].copy()
        q += np.where(self.badr >= 0, d.qpos[np.maximum(self.badr, 0)], 0.0)
        qd = d.qvel[self.vadr]
        kn = self.ref.index(k + 1)
        return np.concatenate([
            imu[:3], imu[3:], q - self.home, qd * R.DOF_VEL_SCALE, a1, a2, targets - self.home,
            self.ref.phase(k), self.ref.q[kn] - self.home, self.ref.qd[kn] * R.DOF_VEL_SCALE,
        ]).astype(np.float32)

    def errors(self, d, k):
        i = self.ref.index(k)
        q = d.qpos[self.qadr]
        pos, quat = d.xpos[self.base_body], d.xquat[self.base_body]
        return {
            "joint_rms": float(np.sqrt(np.mean(np.square(q - self.ref.q[i])))),
            "joint_max": float(np.abs(q - self.ref.q[i]).max()),
            "base_pos": float(np.linalg.norm(pos - self.ref.base_pos[i])),
            "base_ori": float(quat_angle(quat, self.ref.base_quat[i])),
            "height": float(d.qpos[self.ref.base_qpos_addr + 2]),
        }

    def run(self, rng, render=None):
        """One episode from t = 0. Returns a dict of results; render(d, k, status) is called per step."""
        m = self.randomised_model(rng)
        d = mujoco.MjData(m)
        act_delay = int(rng.integers(R.ACTION_DELAY_STEPS[0], R.ACTION_DELAY_STEPS[1] + 1))
        imu_delay = int(rng.integers(R.IMU_DELAY_STEPS[0], R.IMU_DELAY_STEPS[1] + 1))
        k = 0
        d.qpos[:] = self.ref.qpos_full[0]
        d.qvel[:] = 0.0
        d.qvel[self.vadr] = self.ref.qd[0]
        d.qvel[0:3] = self.ref.base_linvel[0]
        d.qvel[3:6] = quat_rotate(np.array([self.ref.base_quat[0, 0], -self.ref.base_quat[0, 1], -self.ref.base_quat[0, 2], -self.ref.base_quat[0, 3]]), self.ref.base_angvel[0])
        d.ctrl[:] = self.ref.q[0]
        mujoco.mj_forward(m, d)
        targets = self.ref.q[0].copy()
        a1 = np.zeros(m.nu); a2 = np.zeros(m.nu)
        act_hist = [np.zeros(m.nu) for _ in range(3)]
        imu_hist = []
        errs, fell, t = [], False, 0
        # --pushes: the recipe's velocity kicks (training has them, the standard battery does not)
        push_every = int(round(rng.uniform(*R.PUSH_INTERVAL_S) / R.CTRL_DT)) if self.pushes else 0
        e0 = self.errors(d, k)
        if render:
            render(d, k, e0, False)
        while True:
            o = self.obs(d, k, a1, a2, targets, imu_hist, imu_delay)
            a = self.policy(o)
            act_hist.insert(0, a); del act_hist[3:]
            a_d = act_hist[min(act_delay, len(act_hist) - 1)]
            raw = self.ref.q[self.ref.index(k + 1)] + R.ACTION_SCALE * a_d
            new = self.alpha * targets + (1 - self.alpha) * raw
            targets = np.clip(new, targets - self.max_step, targets + self.max_step)
            d.ctrl[:] = targets
            if push_every and (t + 1) % push_every == 0:
                th, mag = rng.uniform(0, 2 * np.pi), rng.uniform(*R.PUSH_MAGNITUDE_MS)
                d.qvel[0] += mag * np.cos(th); d.qvel[1] += mag * np.sin(th)
            for _ in range(self.decim):
                mujoco.mj_step(m, d)
            k += 1; t += 1
            a2, a1 = a1, a
            e = self.errors(d, k)
            errs.append(e)
            fell = (e["height"] < R.MIN_BASE_HEIGHT_M) or (e["base_ori"] > R.MAX_ORI_ERR_RAD) or not np.isfinite(d.qpos).all()
            finished = (t >= self.episode_steps) if self.loop else (k >= self.ref.n - 1)
            if render:
                render(d, k, e, fell)
            if fell or finished:
                break
        return {
            "success": not fell, "steps": t, "seconds": t * R.CTRL_DT, "action_delay": act_delay, "imu_delay": imu_delay,
            "joint_err_rad": float(np.mean([e["joint_rms"] for e in errs])),
            "joint_err_max_rad": float(np.max([e["joint_max"] for e in errs])),
            "base_err_m": float(np.mean([e["base_pos"] for e in errs])),
            "base_ori_err_rad": float(np.mean([e["base_ori"] for e in errs])),
            "base_ori_err_max_rad": float(np.max([e["base_ori"] for e in errs])),
            "floor_friction": float(m.geom_friction[self.floor, 0]), "torque_mean": float(np.mean(m.actuator_forcerange[:, 1])),
            "kp_mean_factor": float(np.mean(m.actuator_gainprm[:, 0] / self.m0.actuator_gainprm[:, 0])),
        }


# ----------------------------------------------------------------------------- video
W, H, FPS = 480, 400, 25


class Renderer:
    """Side by side: reference ghost (kinematic playback of the clip) | the policy in the sim."""

    def __init__(self, m, ref, label_ref, label_pol):
        from PIL import ImageFont
        self.r = mujoco.Renderer(m, H, W)
        self.ghost = mujoco.MjData(m)
        self.m, self.ref = m, ref
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = ref.base_body
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.95, 140, -12
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            self.small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        except OSError:
            self.font = self.small = ImageFont.load_default()
        self.labels = (label_ref, label_pol)
        self.frames = []

    def _panel(self, d, label, foot, warn=False):
        from PIL import Image, ImageDraw
        self.r.update_scene(d, self.cam)
        img = Image.fromarray(self.r.render())
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, W, 30], fill=(20, 20, 20))
        dr.text((8, 4), label, font=self.font, fill=(255, 255, 255))
        dr.rectangle([0, H - 26, W, H], fill=(20, 20, 20))
        dr.text((8, H - 23), foot, font=self.small, fill=(255, 90, 90) if warn else (255, 210, 90))
        return np.asarray(img)

    def __call__(self, d, k, e, fell, tag=""):
        if k % 2:
            return
        i = self.ref.index(k)
        self.ghost.qpos[:] = self.ref.qpos_full[i]
        mujoco.mj_forward(self.m, self.ghost)       # the tracking camera follows subtree_com, which mj_kinematics alone does not fill
        t = k * R.CTRL_DT
        left = self._panel(self.ghost, self.labels[0], f"t = {t:.2f} s  clip frame {i}")
        status = f"t = {t:.2f} s  joint err {np.degrees(e['joint_rms']):.1f} deg  base {100 * e['base_pos']:.1f} cm  orient err {np.degrees(e['base_ori']):.0f} deg"
        right = self._panel(d, self.labels[1] + tag, status + ("  FELL" if fell else ""), warn=fell)
        self.frames.append(np.concatenate([left, right], axis=1))

    def hold(self, n):
        if self.frames:
            self.frames.extend([self.frames[-1]] * n)

    def save(self, path):
        import imageio.v2 as imageio
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with imageio.get_writer(path, fps=FPS, codec="libx264", quality=7, macro_block_size=16, ffmpeg_params=["-pix_fmt", "yuv420p"]) as w:
            for f in self.frames:
                w.append_data(f)
        self.r.close()


# ----------------------------------------------------------------------------- package draft
# Pass criteria written into every tracking package (skill.json "thresholds"), so the package
# carries its own gate and tools/battery.py, the dashboard and graph_runtime all read the same
# numbers. 0.94 = 47/50, the episodic G1 bar proposed in tests/GATES.md; joint error 0.15 rad is
# the tools/skillkit default. A termination (height / orientation, identical to training) is a fail.
PACKAGE_THRESHOLDS = {"battery_pass_ratio": 0.94, "battery_min_test_ratio": 0.94, "joint_err_rad": 0.15,
                      "min_height_m": R.MIN_BASE_HEIGHT_M, "max_ori_err_rad": R.MAX_ORI_ERR_RAD}
RECOVERY_SKILLS = ["sit"]         # the abort target(s) every skill must be able to fall back to
THERMAL_C_DEFAULT = 60.0          # mechatronics/envelope.json cool-down threshold (proposed)


def thermal_limit():
    """The servo temperature above which a skill must not start, from the Mechatronics envelope."""
    path = os.path.join(HERE, "..", "mechatronics", "envelope.json")
    try:
        rule = json.load(open(path))["thermal"]["proposed_rule"]          # {warn_c, cool_down_c, hard_stop_c, ...}
        return float(rule["cool_down_c"])
    except (OSError, ValueError, KeyError, TypeError):
        return THERMAL_C_DEFAULT


def pipeline_battery(raw, skill_name, version=1, thresholds=None):
    """This evaluator's report in the shape tools/battery.py writes and graph_runtime reads
    (tests / overall_ratio / overall / passed), with the raw report kept inside."""
    thr = dict(PACKAGE_THRESHOLDS, **(thresholds or {}))
    ratio = raw["success"] / raw["total"] if raw["total"] else 0.0
    err_ok = raw["joint_err_rad"] is not None and raw["joint_err_rad"] <= thr["joint_err_rad"]
    test = {"success": raw["success"], "total": raw["total"], "ratio": round(ratio, 3),
            "no_fall_ratio": round(ratio, 3),
            "mean": {"mean_joint_err_rad": round(raw["joint_err_rad"], 4), "base_err_m": round(raw["base_err_m"], 4),
                     "base_ori_err_rad": round(raw.get("base_ori_err_rad", float("nan")), 4)}}
    passed = bool(raw["total"] and ratio >= thr["battery_pass_ratio"] and ratio >= thr["battery_min_test_ratio"] and err_ok)
    return {"skill": skill_name, "version": version, "type": "episodic", "seeds": raw["seeds"], "episodes": raw["episodes"],
            "tests": {"clip_tracking": test}, "overall_ratio": round(ratio, 3), "overall": f"{raw['success']}/{raw['total']}",
            "passed": passed, "thresholds": {**thr, **raw.get("thresholds", {})},
            "mean_joint_err_rad": round(raw["joint_err_rad"], 4), "evaluator": "sim/eval_tracking.py",
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "raw": raw}


def write_package(out_dir, name, onnx, clip, battery, spec, run_meta, loop):
    """package_draft/ in the skills/schema.json layout, loadable by graph_runtime as it stands."""
    pk = os.path.join(out_dir, "package_draft")
    os.makedirs(os.path.join(pk, "report"), exist_ok=True)
    shutil.copy(onnx, os.path.join(pk, "policy.onnx"))
    shutil.copy(clip, os.path.join(pk, "clip.json"))
    bat = pipeline_battery(battery, name)
    json.dump(bat, open(os.path.join(pk, "report", "battery.json"), "w"), indent=1)
    video = os.path.join(out_dir, "eval", "eval.mp4")
    if os.path.exists(video):
        shutil.copy(video, os.path.join(pk, "report", "battery_clip_tracking.mp4"))
    duration = battery["clip"]["duration_s"]
    layout = spec.get("layout", [])
    obs_spec = {"names": [b["name"] for b in layout], "size": int(spec.get("obs_size", sum(b["size"] for b in layout))),
                "blocks": {b["name"]: int(b["size"]) for b in layout}, "layout": layout}
    if spec.get("notes"):
        obs_spec["notes"] = spec["notes"]
    sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    envelope = dict(R.ENVELOPE)
    if envelope.get("thermal_c") is None:
        envelope["thermal_c"] = thermal_limit()
    run_path = run_meta.get("_path") or os.path.join(os.path.dirname(os.path.abspath(onnx)), "run.json")
    skill = {
        "name": name, "type": "episodic", "loop": bool(loop), "version": 1,
        "commands": {},
        # idle is the walk skill at zero command (plan decision 5), not a package of its own; every
        # skill may exit into recovery (framework 2.3: recovery is reachable from every node)
        "entry": {"from": ["walk"], "phase": [0, 1], "window_s": 0.3},
        "exit": {"to": ["walk"] + RECOVERY_SKILLS},
        "neighbours": ["walk"] + RECOVERY_SKILLS,
        "timeout_s": round((R.LOOP_EPISODE_S if loop else duration) + 1.0, 3),
        "envelope": envelope,
        "robot_model_hash": sha256_file(ROBOT_XML), "robot_xml": os.path.relpath(ROBOT_XML, TRAIN_PG),
        "actuator_model_version": R.ACTUATOR_MODEL_VERSION,
        "recipe_version": f"tracking-v{R.RECIPE_VERSION}",
        "seed": run_meta.get("seed"),
        "obs_spec": obs_spec,
        # graph_runtime: target = clip(t + dt) + action_scale * action, then the deploy filters
        "policy": {"kind": "onnx", "action_mode": "residual_on_reference", "action_scale": R.ACTION_SCALE,
                   "lpf_hz": R.ACT_LPF_HZ, "max_vel_rad_s": R.MAX_MOTOR_VEL, "head_mode": "policy",
                   "input_name": sess.get_inputs()[0].name},
        # tools/skillkit: the same settings in its vocabulary
        "runtime": {"action_scale": R.ACTION_SCALE, "lpf_hz": R.ACT_LPF_HZ, "head_mode": "policy"},
        "thresholds": dict(PACKAGE_THRESHOLDS),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"clip": os.path.abspath(clip), "run": run_path, "num_timesteps": run_meta.get("num_timesteps"),
                   "wall_time_s": run_meta.get("wall_time_s"), "onnx": os.path.abspath(onnx)},
        "hashes": {"policy_onnx": sha256_file(onnx), "clip_json": sha256_file(clip)},
        "status": "battery_passed" if bat["passed"] else "draft",
        "status_history": [{"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "from": None,
                            "to": "battery_passed" if bat["passed"] else "draft", "by": "sim/eval_tracking.py"}],
    }
    json.dump(skill, open(os.path.join(pk, "skill.json"), "w"), indent=1)
    return pk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("clip")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--seeds", type=int, default=R.BATTERY_SEEDS)
    ap.add_argument("--episodes", type=int, default=R.BATTERY_EPISODES)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out-dir", default=None, help="default: the ONNX's directory")
    ap.add_argument("--package", action="store_true", help="also write <out-dir>/package_draft/")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--pushes", action="store_true", help="also apply the recipe's random velocity kicks (training has them)")
    a = ap.parse_args()
    onnx = os.path.abspath(a.onnx)
    out_dir = os.path.abspath(a.out_dir or os.path.dirname(onnx))
    name = a.name or os.path.basename(out_dir)
    os.makedirs(os.path.join(out_dir, "eval"), exist_ok=True)

    m0 = mujoco.MjModel.from_xml_path(SCENE_XML)
    m0.opt.timestep = R.SIM_DT
    ref = ClipTrackingReference(a.clip, m0, loop=a.loop, horizon_s=R.LOOP_EPISODE_S + 2.0)
    policy = Policy(onnx)
    spec_path = os.path.join(os.path.dirname(onnx), "obs_spec.json")
    spec = json.load(open(spec_path)) if os.path.exists(spec_path) else None
    if spec and spec["obs_size"] != policy.obs_size:
        raise SystemExit(f"obs_spec.json says {spec['obs_size']} inputs, the ONNX wants {policy.obs_size}")
    if policy.obs_size != R.OBS_SIZE:
        raise SystemExit(f"ONNX input size {policy.obs_size} != recipe OBS_SIZE {R.OBS_SIZE}")
    run_meta = {}
    run_path = os.path.join(os.path.dirname(onnx), "run.json")
    if os.path.exists(run_path):
        run_meta = json.load(open(run_path)); run_meta["_path"] = run_path

    sim = TrackingSim(m0, ref, policy, a.loop, pushes=a.pushes)
    results, per_seed = [], []
    first_fail = None
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        ok = 0
        for ep in range(a.episodes):
            r = sim.run(rng)
            r.update(seed=seed, episode=ep)
            results.append(r); ok += r["success"]
            if not r["success"] and first_fail is None:
                first_fail = (seed, ep)
        per_seed.append({"seed": seed, "success": ok, "total": a.episodes})
        print(f"seed {seed}: {ok}/{a.episodes}  joint err {np.mean([r['joint_err_rad'] for r in results if r['seed'] == seed]):.4f} rad", flush=True)
    succ = sum(r["success"] for r in results); total = len(results)
    battery = {
        "skill": name, "onnx": onnx, "clip": ref.summary(), "seeds": a.seeds, "episodes": a.episodes,
        "success": succ, "total": total, "ratio": f"{succ}/{total}", "success_rate": succ / total,
        "thresholds": {"min_base_height_m": R.MIN_BASE_HEIGHT_M, "max_ori_err_rad": R.MAX_ORI_ERR_RAD,
                       "episode_s": R.LOOP_EPISODE_S if a.loop else ref.duration, "start": "t = 0, reference state"},
        "joint_err_rad": float(np.mean([r["joint_err_rad"] for r in results])),
        "joint_err_max_rad": float(np.max([r["joint_err_max_rad"] for r in results])),
        "base_err_m": float(np.mean([r["base_err_m"] for r in results])),
        "base_ori_err_rad": float(np.mean([r["base_ori_err_rad"] for r in results])),
        "mean_episode_s": float(np.mean([r["seconds"] for r in results])),
        "per_seed": per_seed, "episodes_detail": results,
        "randomisation": {"floor_friction": [0.5, 1.0], "link_mass_scale": [0.9, 1.1], "trunk_mass_kg": [-0.1, 0.1], "trunk_com_m": [-0.05, 0.05],
                          "joint_zero_rad": [-0.03, 0.03], "frictionloss_scale": [0.9, 1.1], "armature_scale": [1.0, 1.05],
                          "kp_scale": list(R.KP_RANGE), "torque_nm": list(R.TORQUE_RANGE), "action_delay_steps": list(R.ACTION_DELAY_STEPS),
                          "imu_delay_steps": list(R.IMU_DELAY_STEPS), "sensor_noise": "off",
                          "pushes": f"{R.PUSH_MAGNITUDE_MS} m/s every {R.PUSH_INTERVAL_S} s" if a.pushes else "off"},
        "recipe_version": R.RECIPE_VERSION, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(f"\n{name}: {battery['ratio']} success ({100 * battery['success_rate']:.0f} %), joint err {battery['joint_err_rad']:.4f} rad "
          f"({np.degrees(battery['joint_err_rad']):.2f} deg), base err {100 * battery['base_err_m']:.2f} cm, "
          f"ori err {np.degrees(battery['base_ori_err_rad']):.1f} deg, mean episode {battery['mean_episode_s']:.2f} s")

    video = os.path.join(out_dir, "eval", "eval.mp4")
    if not a.no_video:
        rend = Renderer(m0, ref, f"reference: {os.path.basename(a.clip)}", f"policy: {name}")
        segments = [(0, 0, "")]
        if first_fail and first_fail != (0, 0):
            segments.append((first_fail[0], first_fail[1], f"  (seed {first_fail[0]} ep {first_fail[1]}: a failure)"))
        for seed, ep, tag in segments:
            rng = np.random.default_rng(seed)
            for _ in range(ep):                      # replay the rng draws so the physics is the same episode
                sim.run(rng)
            sim.run(rng, render=lambda d, k, e, fell, tag=tag: rend(d, k, e, fell, tag))
            rend.hold(FPS)
        rend.save(video)
        battery["video"] = video
        print("video", video)
    json.dump(battery, open(os.path.join(out_dir, "eval", "battery.json"), "w"), indent=1)
    print("battery", os.path.join(out_dir, "eval", "battery.json"))
    if a.package:
        pk = write_package(out_dir, name, onnx, a.clip, battery, spec, run_meta, a.loop)
        print("package draft", pk)


if __name__ == "__main__":
    main()
