"""Shared helpers for the skill-library tools (skill_package, battery, sim2sim, regression).

Everything here is import-only: paths, the package layout, the MJCF hash, clip loading,
and a policy runner that mirrors sim_test.Sim but builds its observation from the package's
obs_spec so tracking policies with a different layout can be scored by the same battery.

The simulator mirror (sim/sim_test.py) is imported, never modified. It resolves the gait
clock pickle relative to the current directory, so `chdir_sim()` must be called before `Sim`
is constructed; every tool does that at start-up after resolving its own arguments.
"""
import datetime as _dt
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(ROOT, "sim")
PG = os.path.join(SIM, "train_pg", "playground", "open_duck_mini_v2")
MODEL_XML = os.path.join(PG, "xmls", "open_duck_mini_v2_backlash.xml")
SCENE_XML = os.path.join(PG, "xmls", "scene_flat_terrain_backlash.xml")
SKILLS = os.path.join(ROOT, "skills")
SCHEMA = os.path.join(SKILLS, "schema.json")
UPSTREAM = os.path.join(SIM, "upstream_runtime")
PYTHON = os.path.join(SIM, ".venv", "bin", "python")

JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
          "neck_pitch", "head_pitch", "head_yaw", "head_roll",
          "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
LEGS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
HEAD = [5, 6, 7, 8]

# The deployed walk brain's observation (sim/sim_test.py, upstream v2_rl_walk_mujoco.get_obs):
WALK_OBS_BLOCKS = [("gyro", 3), ("accelerometer", 3), ("commands", 7), ("joint_pos_rel", 14),
                   ("joint_vel_scaled", 14), ("last_action", 14), ("last_last_action", 14),
                   ("last_last_last_action", 14), ("motor_targets", 14), ("feet_contacts", 2),
                   ("gait_phase", 2)]
WALK_OBS_NAMES = [n for n, _ in WALK_OBS_BLOCKS]
WALK_OBS_SIZE = sum(s for _, s in WALK_OBS_BLOCKS)          # 101
BLOCK_SIZES = dict(WALK_OBS_BLOCKS)
# Blocks a clip-tracking policy may ask for (see PolicyRunner._block). Sizes are fixed here;
# a package can override any of them with obs_spec.blocks {name: size}.
BLOCK_SIZES.update({"clip_phase": 2, "clip_time": 1, "ref_pose": 14, "ref_pose_next": 14,
                    "ref_vel": 14, "ref_contacts": 2, "commands3": 3, "gravity": 3})

REQUIRED_FILES = ["skill.json", "clip.json", "policy.onnx", os.path.join("report", "battery.json")]
STATUS_ORDER = ["draft", "battery_passed", "sim2sim_passed", "hardware_passed", "released"]

DEFAULT_RUNTIME = {"action_scale": 0.25, "lpf_hz": 8.0, "head_mode": "add", "accel_bias": 0.0,
                   "gait_clock": "poly", "episodic_cmd": [0.0, 0.0, 0.0]}
DEFAULT_THRESHOLDS = {"min_height_m": 0.09, "joint_err_rad": 0.15, "min_tracking_fraction": 0.2,
                      "battery_pass_ratio": 0.9, "battery_min_test_ratio": 0.8, "sim2sim_rms_rad": 0.02}


def now_iso():
    return _dt.datetime.now().astimezone().replace(microsecond=0).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_hash(path=MODEL_XML):
    """sha256 of the robot MJCF. The scene files include it, so a mass or actuator change in
    the robot shows here; a floor-friction change in the scene does not (by design)."""
    return sha256_file(path)


def chdir_sim():
    """sim_test.py loads its gait-clock pickle from a path relative to sim/."""
    os.chdir(SIM)
    if SIM not in sys.path:
        sys.path.insert(0, SIM)


def import_sim_test():
    chdir_sim()
    import sim_test   # noqa: E402
    return sim_test


def load_json(path):
    with open(path) as f:
        return json.load(f)


def dump_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def load_skill(skill_dir):
    return load_json(os.path.join(skill_dir, "skill.json"))


def runtime_settings(skill):
    """How the policy's output becomes servo targets: taken from skill.json (envelope + runtime),
    defaults are the deployed walk's (torque 1.86 N.m, 8 Hz low-pass, 4 rad/s, head = policy + cmd)."""
    env = skill.get("envelope", {})
    rt = dict(DEFAULT_RUNTIME)
    rt.update(skill.get("runtime", {}))
    rt["torque_nm"] = float(env.get("torque_nm", 1.86))
    rt["max_vel"] = float(env.get("max_vel_rad_s", 4.0))
    rt["fall_tilt_deg"] = float(env.get("fall_tilt_deg", 45.0))
    return rt


def thresholds(skill):
    t = dict(DEFAULT_THRESHOLDS)
    t.update(skill.get("thresholds", {}))
    t["fall_tilt_deg"] = float(skill.get("envelope", {}).get("fall_tilt_deg", 45.0))
    return t


# ----------------------------------------------------------------------------- clips
def load_clip(path):
    """A Duck Puppet clip: {"frames": [[t, {joint: rad}], ...], "interp": "linear", "base": optional}
    or the bare frames list. Returns (times[n], q[n, 14] in JOINTS order, meta dict)."""
    data = load_json(path)
    frames = data if isinstance(data, list) else data["frames"]
    if not frames:
        raise ValueError(f"{path}: clip has no frames")
    times = np.array([float(f[0]) for f in frames])
    q = np.array([[float(f[1].get(j, 0.0)) for j in JOINTS] for f in frames])
    meta = {} if isinstance(data, list) else {k: v for k, v in data.items() if k != "frames"}
    return times, q, meta


def clip_pose(times, q, t, loop=False):
    """Linear interpolation of the clip at time t (held at the ends, or wrapped when loop)."""
    T = times[-1]
    if loop and T > 0:
        t = t % T
    return np.array([np.interp(t, times, q[:, k]) for k in range(q.shape[1])])


def clip_duration(times):
    return float(times[-1] - times[0])


# ----------------------------------------------------------------------------- simulation
def make_sim(onnx, rt, xml=SCENE_XML, clip_path=None):
    """A sim_test.Sim configured from the package's runtime settings (the same knobs
    eval_policy.py and record_policy_clip.py expose on their command lines)."""
    chdir_sim()
    if rt.get("gait_clock", "poly") in ("clip", "hybrid") and clip_path:
        os.environ["REF_CLIP"] = clip_path
        os.environ["REF_HYBRID"] = "1" if rt["gait_clock"] == "hybrid" else "0"
        os.environ.setdefault("REF_QUIET", "1")
    else:
        os.environ.pop("REF_CLIP", None)
    from sim_test import Sim
    extra = {"head_from_cmd": rt["head_mode"] == "from_cmd", "head_add_cmd": rt["head_mode"] == "add",
             "lpf_hz": float(rt["lpf_hz"]), "max_motor_vel": float(rt["max_vel"])}
    if rt.get("leg_action_scale"):
        extra["leg_action_scale"] = float(rt["leg_action_scale"])
    s = Sim(onnx, xml, float(rt.get("accel_bias", 0.0)), action_scale=float(rt["action_scale"]), extra=extra)
    s.m.actuator_forcerange[:, :] = [-rt["torque_nm"], rt["torque_nm"]]
    s.nominal = {"mass": s.m.body_mass.copy(), "friction": s.m.geom_friction.copy(),
                 "forcerange": s.m.actuator_forcerange.copy(), "gain": s.m.actuator_gainprm[:, 0].copy()}
    return s


def randomise(sim, rng, torque_nm):
    """Per-seed domain randomisation in the training recipe's ranges
    (sim/train_pg/playground/common/randomize.py): floor friction U(0.5, 1.0), link masses
    x U(0.9, 1.1), trunk mass +U(-0.1, 0.1) kg, and the servo torque ceiling U(0.8, 1.0) x the
    package envelope, so the battery never asks more torque than the envelope allows."""
    import mujoco
    m, nom = sim.m, sim.nominal
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    friction = float(rng.uniform(0.5, 1.0))
    m.geom_friction[floor, 0] = friction
    mass_scale = rng.uniform(0.9, 1.1, size=m.nbody)
    m.body_mass[:] = nom["mass"] * mass_scale
    trunk = m.body("base").id if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base") >= 0 else 1
    dmass = float(rng.uniform(-0.1, 0.1))
    m.body_mass[trunk] = max(0.05, m.body_mass[trunk] + dmass)
    torque = float(torque_nm * rng.uniform(0.8, 1.0))
    m.actuator_forcerange[:, :] = [-torque, torque]
    mujoco.mj_setConst(m, sim.d)
    return {"floor_friction": round(friction, 3), "mass_scale_mean": round(float(mass_scale.mean()), 3),
            "trunk_dmass_kg": round(dmass, 3), "torque_nm": round(torque, 3)}


def reset_episode(sim, rng=None, pose_noise=0.03):
    """Back to the home keyframe with the training recipe's initial-pose jitter, and a fresh
    policy state (actions, filter, gait clock)."""
    import mujoco
    m, d = sim.m, sim.d
    mujoco.mj_resetDataKeyframe(m, d, m.keyframe("home").id)
    if rng is not None and pose_noise > 0:
        d.qpos[sim.qadr] += rng.uniform(-pose_noise, pose_noise, size=len(sim.qadr))
    d.ctrl[:] = sim.default
    mujoco.mj_forward(m, d)
    sim.reset_policy_state()
    sim.phase_f = 0.0


def fallen(sim, thr):
    """Training-style termination: the trunk tilted past the envelope or the base too low."""
    pos, _, tilt = sim.base_pose()
    return bool(tilt > thr["fall_tilt_deg"] or pos[2] < thr["min_height_m"])


def heading_velocity(sim, p0, yaw0, seconds):
    """Displacement since (p0, yaw0) expressed in the start heading: forward, left m/s."""
    p1, _, _ = sim.base_pose()
    dp = p1 - p0
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    return (c * dp[0] - s * dp[1]) / seconds, (s * dp[0] + c * dp[1]) / seconds


def yaw_rate_tracker():
    """Accumulates unwrapped yaw; call .add(yaw) each step and read .total (rad)."""
    class _T:
        def __init__(self):
            self.prev, self.total = None, 0.0

        def add(self, yaw):
            if self.prev is not None:
                self.total += (yaw - self.prev + np.pi) % (2 * np.pi) - np.pi
            self.prev = yaw
    return _T()


# ----------------------------------------------------------------------------- policy runner
class PolicyRunner:
    """Runs a package's ONNX on a sim_test.Sim.

    For the deployed walk layout (obs_spec.names == WALK_OBS_NAMES) this is exactly
    Sim.control(cmd). For any other layout the observation is assembled block by block from
    obs_spec.names (sizes from obs_spec.blocks or BLOCK_SIZES), and the action is turned into
    servo targets the same way (scale, low-pass, speed limit, head mode).

    HOOK (Controls agent): when sim/train_pg/playground/open_duck_mini_v2/tracking.py lands,
    mirror its observation blocks in _block() below under the same names it exports in the
    package's obs_spec. Until then only the blocks listed in BLOCK_SIZES exist and an unknown
    name raises, so a package can never be scored on a guessed observation.
    """

    def __init__(self, sim, skill, clip=None):
        self.sim = sim
        self.skill = skill
        self.rt = runtime_settings(skill)
        spec = skill.get("obs_spec", {}) or {}
        self.names = list(spec.get("names", WALK_OBS_NAMES))
        sizes = dict(BLOCK_SIZES)
        sizes.update(spec.get("blocks", {}))
        self.sizes = sizes
        self.walk_layout = self.names == WALK_OBS_NAMES
        unknown = [n for n in self.names if n not in sizes]
        if unknown:
            raise ValueError(f"obs_spec names not implemented in tools/skillkit.PolicyRunner: {unknown} "
                             f"(add them to _block(); see the HOOK note)")
        total = sum(sizes[n] for n in self.names)
        declared = int(spec.get("size", total))
        if declared != total:
            raise ValueError(f"obs_spec.size {declared} != sum of blocks {total}")
        onnx_dim = sim.policy.get_inputs()[0].shape[-1]
        if isinstance(onnx_dim, int) and onnx_dim != total:
            raise ValueError(f"policy.onnx expects {onnx_dim} inputs but obs_spec gives {total}")
        self.size = total
        self.clip = clip                     # (times, q) for the ref_* blocks
        self.clip_t = 0.0

    def step(self, cmd, clip_t=None):
        """One 50 Hz control step. cmd is the 7-vector (vx, vy, wz, neck, pitch, yaw, roll)."""
        cmd = np.asarray(cmd, dtype=float)
        if clip_t is not None:
            self.clip_t = clip_t
        if self.walk_layout:
            return self.sim.control(cmd)
        return self._generic_step(cmd)

    # ---- generic layout: the same mapping as Sim.control, observation from the blocks
    def _generic_step(self, cmd):
        s = self.sim
        s.imitation_i = (s.imitation_i + s.phase_factor) % s.nb_steps
        ang = s.imitation_i / s.nb_steps * 2 * np.pi
        s.phase = np.array([np.cos(ang), np.sin(ang)])
        o = np.concatenate([np.asarray(self._block(n, cmd), dtype=np.float32).ravel() for n in self.names]).astype(np.float32)
        a = s.policy.run(None, {"obs": [o]})[0][0]
        s.a3, s.a2, s.a1 = s.a2, s.a1, a.copy()
        t = s.default + a * s.action_scale
        if self.rt["head_mode"] == "from_cmd":
            t[5:9] = s.default[5:9] + cmd[3:7]
        elif self.rt["head_mode"] == "add":
            t[5:9] = t[5:9] + cmd[3:7]
        if self.rt["lpf_hz"]:
            alpha = (1 / self.rt["lpf_hz"]) / (0.02 + 1 / self.rt["lpf_hz"])
            t = alpha * s.prev_targets + (1 - alpha) * t
        step = self.rt["max_vel"] * 0.02
        t = np.clip(t, s.prev_targets - step, s.prev_targets + step)
        s.prev_targets = t.copy()
        s.targets = t
        s.d.ctrl[:] = t
        import mujoco
        for _ in range(s.decim):
            mujoco.mj_step(s.m, s.d)
        return o, a

    def _block(self, name, cmd):
        s, d = self.sim, self.sim.d
        if name == "gyro":
            return d.sensordata[s.gyro_a:s.gyro_a + 3]
        if name == "accelerometer":
            acc = d.sensordata[s.acc_a:s.acc_a + 3].copy()
            acc[0] += s.accel_bias
            return acc
        if name == "gravity":
            return d.site_xmat[0].reshape(3, 3).T @ np.array([0, 0, -1.0])
        if name == "commands":
            return cmd[:7]
        if name == "commands3":
            return cmd[:3]
        if name == "joint_pos_rel":
            return d.qpos[s.qadr] - s.default
        if name == "joint_vel_scaled":
            return d.qvel[s.vadr] * 0.05
        if name == "last_action":
            return s.a1
        if name == "last_last_action":
            return s.a2
        if name == "last_last_last_action":
            return s.a3
        if name == "motor_targets":
            return s.targets
        if name == "feet_contacts":
            return s.contacts()
        if name == "gait_phase":
            return s.phase
        # ---- clip-tracking blocks (HOOK: keep in step with tracking.py)
        if self.clip is None:
            raise ValueError(f"block {name} needs a clip")
        times, q = self.clip
        T = max(times[-1], 1e-6)
        if name == "clip_phase":
            ang = 2 * np.pi * min(self.clip_t / T, 1.0)
            return np.array([np.cos(ang), np.sin(ang)])
        if name == "clip_time":
            return np.array([min(self.clip_t / T, 1.0)])
        if name == "ref_pose":
            return clip_pose(times, q, self.clip_t) - s.default
        if name == "ref_pose_next":
            return clip_pose(times, q, self.clip_t + 0.02) - s.default
        if name == "ref_vel":
            return (clip_pose(times, q, self.clip_t + 0.02) - clip_pose(times, q, self.clip_t)) / 0.02
        if name == "ref_contacts":
            return np.ones(2)
        raise ValueError(f"unknown obs block {name}")


# ----------------------------------------------------------------------------- video
class Recorder:
    """Small tracking-camera mp4 of one episode (mujoco.Renderer, 25 fps, 320x240)."""

    def __init__(self, sim, label, w=320, h=240, fps=25):
        import mujoco
        self.ok = True
        try:
            self.r = mujoco.Renderer(sim.m, h, w)
        except Exception as e:                      # no EGL/OSMesa: the battery still scores
            print(f"[video] renderer unavailable ({e.__class__.__name__}: {e}); no mp4")
            self.ok = False
            return
        self.sim, self.label, self.fps, self.w, self.h = sim, label, fps, w, h
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = sim.base
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.95, 140, -12
        self.frames, self.k = [], 0
        try:
            from PIL import Image, ImageDraw, ImageFont
            self._pil = (Image, ImageDraw)
            try:
                self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except OSError:
                self.font = ImageFont.load_default()
        except ImportError:
            self._pil = None

    def frame(self, text=""):
        if not self.ok:
            return
        if self.k % 2 == 0:                          # 50 Hz control -> 25 fps
            self.r.update_scene(self.sim.d, self.cam)
            img = self.r.render()
            if self._pil:
                Image, ImageDraw = self._pil
                im = Image.fromarray(img)
                dr = ImageDraw.Draw(im)
                dr.rectangle([0, 0, self.w, 18], fill=(20, 20, 20))
                dr.text((4, 2), f"{self.label}  {text}", font=self.font, fill=(255, 220, 120))
                img = np.asarray(im)
            self.frames.append(img)
        self.k += 1

    def save(self, path):
        if not self.ok or not self.frames:
            return None
        import imageio.v2 as imageio
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with imageio.get_writer(path, fps=self.fps, codec="libx264", quality=6, macro_block_size=16,
                                pixelformat="yuv420p") as w:
            for f in self.frames:
                w.append_data(f)
        self.frames = []
        return path

    def close(self):
        if self.ok:
            self.r.close()
