"""MuJoCo backend: the duck on the same model sim_test.Sim uses, as the engine's sensors and servos.

Model `scene_flat_terrain_backlash.xml` (train_pg / Open_Duck_Playground), timestep 2 ms, ten
physics steps per 20 ms control tick, position actuators with the MJCF kp (17.11) and the deployable
torque limit of 1.86 N.m, as in render_video.py / eval_policy.py. Only the two foot soles collide
with the floor in this scene, so a duck that falls over sinks through it: that is the model's
limitation, not the engine's (the `_collision` scene variant has body geoms; pass it as `xml`).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from graph_runtime.backends.base import Backend, SensorFrame
from graph_runtime.packages import HOME, JOINTS

PROJECT = Path(__file__).resolve().parents[2]
PG = PROJECT / "sim" / "Open_Duck_Playground" / "playground" / "open_duck_mini_v2"
DEFAULT_XML = PG / "xmls" / "scene_flat_terrain_backlash.xml"
COLLISION_XML = PROJECT / "sim" / "train_pg" / "playground" / "open_duck_mini_v2" / "xmls" / "scene_flat_terrain_backlash_collision.xml"


class MujocoBackend(Backend):
    name = "mujoco"

    def __init__(self, xml: str | os.PathLike = DEFAULT_XML, torque_nm: float = 1.86, timestep: float = 0.002,
                 decimation: int = 10, seed: int | None = None, pose_noise: float = 0.0,
                 render_size: tuple = (480, 400)):
        import mujoco
        self.mujoco = mujoco
        self.xml = str(xml)
        self.m = mujoco.MjModel.from_xml_path(self.xml)
        self.m.opt.timestep = timestep
        self.decim = int(decimation)
        self.dt = timestep * self.decim
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.act_names = [m.actuator(k).name for k in range(m.nu)]
        if self.act_names != JOINTS:
            raise RuntimeError(f"actuator order {self.act_names} differs from the runtime's JOINTS order")
        self.qadr = np.array([m.joint(n).qposadr[0] for n in self.act_names])
        self.vadr = np.array([m.joint(n).dofadr[0] for n in self.act_names])
        s = lambda n: m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)]   # noqa: E731
        self.gyro_a, self.acc_a = s("gyro"), s("accelerometer")
        self.imu_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.home_ctrl = m.keyframe("home").ctrl.copy()
        self.home_qpos = m.keyframe("home").qpos.copy()
        if not np.allclose(self.home_ctrl, HOME, atol=1e-6):
            raise RuntimeError("MJCF home keyframe differs from graph_runtime.packages.HOME")
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.feet = [m.body("foot_assembly").id, m.body("foot_assembly_2").id]
        self.trunk = m.body("trunk_assembly").id
        self.kp0 = m.actuator_gainprm[:, 0].copy()
        self.lo, self.hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        self.torque_nm = torque_nm
        m.actuator_forcerange[:, :] = [-torque_nm, torque_nm]
        self.stiffness = 1.0
        self.push = None                       # (force xyz, ticks left)
        self.renderer = None
        self.render_size = render_size
        self.t = 0.0
        self.reset(seed=seed, pose_noise=pose_noise)

    # ------------------------------------------------------------ Backend interface
    def read(self) -> SensorFrame:
        d = self.d
        gyro = d.sensordata[self.gyro_a:self.gyro_a + 3].copy()
        acc = d.sensordata[self.acc_a:self.acc_a + 3].copy()
        pos, yaw, tilt = self.base_pose()
        gravity = d.site_xmat[self.imu_site].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        return SensorFrame(t=self.t, gyro=gyro, acc=acc, q=d.qpos[self.qadr].copy(), dq=d.qvel[self.vadr].copy(),
                           contacts=np.array(self.contacts()), tilt_deg=tilt, height_m=float(pos[2]),
                           gravity=gravity, extra={"pos": pos, "yaw": yaw})

    def write_targets(self, targets: np.ndarray) -> None:
        t = np.clip(self.check_targets(targets), self.lo, self.hi)
        self.d.ctrl[:] = t
        if self.push is not None:
            force, left = self.push
            self.d.xfrc_applied[self.trunk, :3] = force if left > 0 else 0.0
            self.push = (force, left - 1) if left > 1 else None
            if self.push is None:
                self.d.xfrc_applied[self.trunk, :3] = 0.0
        for _ in range(self.decim):
            self.mujoco.mj_step(self.m, self.d)
        self.t += self.dt

    def set_stiffness(self, scale: float) -> None:
        scale = float(max(scale, 0.0))
        self.stiffness = scale
        self.m.actuator_gainprm[:, 0] = self.kp0 * scale
        self.m.actuator_biasprm[:, 1] = -self.kp0 * scale

    def show(self, channel: str, value) -> None:
        self.last_show = (channel, value)      # nothing to drive in the sim; run_sim overlays it

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    # ------------------------------------------------------------ sim extras
    def reset(self, seed: int | None = None, pose_noise: float = 0.0):
        rng = np.random.default_rng(seed)
        self.d.qpos[:] = self.home_qpos
        self.d.qvel[:] = 0.0
        if pose_noise > 0:
            self.d.qpos[self.qadr] += rng.uniform(-pose_noise, pose_noise, len(self.qadr))
        self.d.ctrl[:] = self.home_ctrl
        self.d.xfrc_applied[:] = 0.0
        self.set_stiffness(1.0)
        self.mujoco.mj_forward(self.m, self.d)
        self.t = 0.0
        self.last_show = None

    def contacts(self):
        out = [0.0, 0.0]
        d, m = self.d, self.m
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for k, f in enumerate(self.feet):
                if (b1 == f and c.geom2 == self.floor) or (b2 == f and c.geom1 == self.floor):
                    out[k] = 1.0
        return out

    def base_pose(self):
        """(position, yaw, tilt_deg) of the floating base, as sim_test.Sim.base_pose."""
        q = self.d.qpos[:7]
        w, x, y, z = q[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        up = 1 - 2 * (x * x + y * y)
        return q[:3].copy(), float(yaw), float(np.degrees(np.arccos(np.clip(up, -1, 1))))

    def apply_push(self, force_xyz, seconds: float):
        """Shove the trunk for `seconds` (as eval_policy.push_test does), starting next tick."""
        self.push = (np.asarray(force_xyz, dtype=float), max(1, int(round(seconds / self.dt))))

    def render(self) -> np.ndarray:
        """RGB frame from a camera tracking the trunk (render_video.py's framing)."""
        mujoco = self.mujoco
        if self.renderer is None:
            w, h = self.render_size
            self.renderer = mujoco.Renderer(self.m, h, w)
            self.cam = mujoco.MjvCamera()
            self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.cam.trackbodyid = self.trunk
            self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.95, 140, -12
        self.renderer.update_scene(self.d, self.cam)
        return self.renderer.render()
