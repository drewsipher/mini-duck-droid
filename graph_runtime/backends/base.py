"""Backend interface: the only thing that differs between the simulator and the robot.

The engine runs the same code on both. A backend turns sensors into a SensorFrame, takes 14 joint
targets in radians (JOINTS order) once per 20 ms tick, and can soften or release the servos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from graph_runtime.packages import DT, JOINTS, NUM_JOINTS


@dataclass
class SensorFrame:
    t: float                         # backend time, s
    gyro: np.ndarray                 # (3,) rad/s, body frame (IMU site)
    acc: np.ndarray                  # (3,) m/s^2, body frame, includes gravity
    q: np.ndarray                    # (14,) joint positions, rad, JOINTS order
    dq: np.ndarray                   # (14,) joint velocities, rad/s
    contacts: np.ndarray             # (2,) left, right foot on the ground (0/1)
    tilt_deg: float | None = None    # angle between the body's up axis and world up, if the backend knows it
    height_m: float | None = None    # trunk height above the floor, if the backend knows it (simulation)
    gravity: np.ndarray | None = None  # (3,) unit gravity direction in the IMU frame (R^T [0,0,-1]) when orientation is known
    extra: dict = field(default_factory=dict)

    @staticmethod
    def tilt_from_accel(acc: np.ndarray) -> float:
        """Fallback tilt estimate from the accelerometer alone (gravity direction vs body z).
        Valid when the body is not accelerating much; the engine low-passes it."""
        n = float(np.linalg.norm(acc))
        if n < 1e-6:
            return 0.0
        return float(np.degrees(np.arccos(np.clip(acc[2] / n, -1.0, 1.0))))


class Backend:
    """Read sensors -> SensorFrame; write 14 joint targets; dt. Subclasses implement the four methods."""
    dt: float = DT
    joints = list(JOINTS)
    name = "base"

    def read(self) -> SensorFrame:
        raise NotImplementedError

    def write_targets(self, targets: np.ndarray) -> None:
        """Command joint position targets (rad) and let one control period (dt) elapse."""
        raise NotImplementedError

    def set_stiffness(self, scale: float) -> None:
        """Scale the servo position gains: 1.0 nominal, small = compliant hold, 0.0 = torque off."""
        raise NotImplementedError

    def show(self, channel: str, value: Any) -> None:
        """Side channel for show functions (antennas, eyes, sound). Backends that have none ignore it."""

    def close(self) -> None:
        pass

    @staticmethod
    def check_targets(targets: np.ndarray) -> np.ndarray:
        t = np.asarray(targets, dtype=float)
        if t.shape != (NUM_JOINTS,) or not np.all(np.isfinite(t)):
            raise ValueError(f"targets must be {NUM_JOINTS} finite radians, got {t!r}")
        return t
