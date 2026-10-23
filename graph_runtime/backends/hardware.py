"""Hardware backend: the engine on the real duck through mini_bdx_runtime (the Pi runtime).

LOCKED. Tonight nothing may run on the robot or the Pi, so the constructor refuses unless BOTH
`allow=True` is passed AND the environment has DUCK_RUNTIME_ALLOW_HARDWARE=1. The imports are
inside the constructor so this module loads anywhere. Everything below maps one engine call onto
the exact runtime calls that scripts/v2_rl_walk_mujoco.py (class RLWalk) makes:

  construction   DuckConfig(duck_config_path)                    ~/duck_config.json: imu_upside_down, joints_offsets, ...
                 HWI(duck_config, serial_port)                   rustypot feetech bus at 1 Mbaud, 14 joints in JOINTS order
                 hwi.set_kps([30]*14 with [8]*4 for the head)    RLWalk.start()
                 hwi.set_kds([0]*14)
                 hwi.turn_on()                                   low kps -> init_pos (= HOME) -> high kps, 2 s
                 Imu(sampling_freq=50, user_pitch_bias=0, upside_down=config.imu_upside_down)   raw_imu.Imu, worker thread
                 FeetContacts()                                  GPIO D22 / D27, pull-up, pressed = contact
  read()         imu.get_data() -> {"gyro": [3], "accelero": [3]}           body frame, same axes the policy was trained on
                 hwi.get_present_positions(ignore=["left_antenna","right_antenna"])  rad, joint offsets already removed
                 hwi.get_present_velocities(ignore=[...])                     rad/s
                 feet_contacts.get() -> [left, right]
                 tilt_deg: None -> the engine estimates tilt from the accelerometer (low-passed). The BNO055 fusion
                 quaternion (mini_bdx_runtime/imu.py Imu, IMUPLUS mode) is the better source and is the first thing
                 to wire when this backend is unlocked.
  write_targets  make_action_dict(targets, list(hwi.joints.keys())) -> hwi.set_position_all(dict)   rad, offsets added inside
                 then sleep to the 20 ms tick (RLWalk.run's loop timing)
  set_stiffness  hwi.set_kps(kps * scale, floor 2 = HWI.low_torque_kps)    0.0 -> hwi.turn_off() (torque disabled)
  show           Antennas.set_position_left/right(0..1), Sounds.play(name), Eyes (blink thread) - optional features
  close          soft kps, then hwi.turn_off()

Joint order, zero pose and the home pose are identical in HWI.joints / HWI.init_pos and
graph_runtime.packages.JOINTS / HOME, so targets pass through unchanged.
"""
from __future__ import annotations

import os
import time

import numpy as np

from graph_runtime.backends.base import Backend, SensorFrame
from graph_runtime.packages import DT, HOME, JOINTS, NUM_JOINTS

LOCK_ENV = "DUCK_RUNTIME_ALLOW_HARDWARE"
ANTENNAS = ["left_antenna", "right_antenna"]


class HardwareLocked(RuntimeError):
    pass


class HardwareBackend(Backend):
    name = "hardware"

    def __init__(self, duck_config_path: str | None = None, serial_port: str = "/dev/ttyACM0",
                 control_freq: float = 50.0, leg_kp: float = 30.0, head_kp: float = 8.0, kd: float = 0.0,
                 allow: bool = False, max_sensor_misses: int = 10):
        if not (allow and os.environ.get(LOCK_ENV) == "1"):
            raise HardwareLocked(
                "The hardware backend is locked: it must not run tonight. To unlock later, pass allow=True "
                f"AND set {LOCK_ENV}=1, and run it on the Pi only after the sim-to-sim gate.")
        # --- imports are deliberately here: the Pi runtime and its drivers are not installed on the dev box
        from mini_bdx_runtime.duck_config import DuckConfig              # noqa: PLC0415
        from mini_bdx_runtime.feet_contacts import FeetContacts          # noqa: PLC0415
        from mini_bdx_runtime.raw_imu import Imu                         # noqa: PLC0415
        from mini_bdx_runtime.rl_utils import make_action_dict           # noqa: PLC0415
        from mini_bdx_runtime.rustypot_position_hwi import HWI           # noqa: PLC0415

        self.dt = 1.0 / control_freq
        self.make_action_dict = make_action_dict
        self.config = DuckConfig(config_json_path=duck_config_path or os.path.expanduser("~/duck_config.json"))
        self.hwi = HWI(self.config, serial_port)
        if list(self.hwi.joints.keys()) != JOINTS:
            raise RuntimeError(f"HWI joint order {list(self.hwi.joints.keys())} differs from JOINTS")
        if not np.allclose(np.array(list(self.hwi.init_pos.values())), HOME, atol=1e-6):
            raise RuntimeError("HWI.init_pos differs from graph_runtime.packages.HOME")
        self.kps = np.array([leg_kp] * NUM_JOINTS, dtype=float)
        self.kps[5:9] = head_kp
        self.hwi.set_kps(list(self.kps))
        self.hwi.set_kds([kd] * NUM_JOINTS)
        self.hwi.turn_on()                                   # low kps, init_pos, high kps (about 2 s)
        self.imu = Imu(sampling_freq=int(control_freq), user_pitch_bias=0, upside_down=self.config.imu_upside_down)
        self.feet = FeetContacts()
        self.antennas = self.sounds = self.eyes = None
        if getattr(self.config, "antennas", False):
            from mini_bdx_runtime.antennas import Antennas               # noqa: PLC0415
            self.antennas = Antennas()
        if getattr(self.config, "speaker", False):
            from mini_bdx_runtime.sounds import Sounds                   # noqa: PLC0415
            self.sounds = Sounds(volume=1.0, sound_directory=os.path.join(os.path.dirname(__file__), "assets"))
        if getattr(self.config, "eyes", False):
            from mini_bdx_runtime.eyes import Eyes                       # noqa: PLC0415
            self.eyes = Eyes()
        self.max_misses = max_sensor_misses
        self.misses = 0
        self.last_q = HOME.copy()
        self.last_dq = np.zeros(NUM_JOINTS)
        self.t = 0.0
        self.next_tick = time.time()

    def read(self) -> SensorFrame:
        imu = self.imu.get_data()
        q = self.hwi.get_present_positions(ignore=ANTENNAS)
        dq = self.hwi.get_present_velocities(ignore=ANTENNAS)
        if q is None or dq is None or len(q) != NUM_JOINTS or len(dq) != NUM_JOINTS:
            self.misses += 1                                 # RLWalk skips the tick; we reuse the last reading
            if self.misses > self.max_misses:
                raise RuntimeError(f"{self.misses} consecutive servo read failures")
            q, dq = self.last_q, self.last_dq
        else:
            self.misses = 0
            self.last_q, self.last_dq = np.asarray(q, dtype=float), np.asarray(dq, dtype=float)
        left, right = self.feet.get()
        return SensorFrame(t=self.t, gyro=np.asarray(imu["gyro"], dtype=float), acc=np.asarray(imu["accelero"], dtype=float),
                           q=self.last_q.copy(), dq=self.last_dq.copy(), contacts=np.array([float(left), float(right)]),
                           tilt_deg=None, height_m=None)

    def write_targets(self, targets: np.ndarray) -> None:
        t = self.check_targets(targets)
        self.hwi.set_position_all(self.make_action_dict(t, list(self.hwi.joints.keys())))
        self.next_tick += self.dt
        now = time.time()
        if now > self.next_tick + self.dt:                   # fell behind by a whole tick: resync, do not burst
            self.next_tick = now
        time.sleep(max(0.0, self.next_tick - now))
        self.t += self.dt

    def set_stiffness(self, scale: float) -> None:
        if scale <= 0.0:
            self.hwi.turn_off()
            return
        soft = np.maximum(self.kps * float(scale), 2.0)      # 2 = HWI.low_torque_kps
        self.hwi.set_kps(list(soft))

    def show(self, channel: str, value) -> None:
        if channel == "antenna_left" and self.antennas:
            self.antennas.set_position_left(float(value))
        elif channel == "antenna_right" and self.antennas:
            self.antennas.set_position_right(float(value))
        elif channel == "sound" and self.sounds:
            self.sounds.play(str(value))

    def close(self) -> None:
        try:
            self.set_stiffness(0.1)
            time.sleep(0.5)
        finally:
            self.hwi.turn_off()
            self.feet.stop()
            if self.antennas:
                self.antennas.stop()
            if self.eyes:
                self.eyes.stop()
