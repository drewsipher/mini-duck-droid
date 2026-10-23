"""Automatic head pose from the drive command, eased with a first-order filter:
  forward / back  -> pitch the head forward / back (neck + head pitch)
  turning         -> yaw the head into the turn
  side-stepping   -> tilt (roll) the head toward the step
Joint signs (from the model): +neck_pitch / +head_pitch tip the head back,
+head_yaw looks left, +head_roll tilts to the duck's left.
Same function goes into the duck runtime."""
import numpy as np

GAINS = dict(neck_fwd=-0.20, neck_back=0.15, pitch_fwd=-0.12, pitch_back=0.12,
             yaw_turn=0.30, roll_side=0.25, tau=0.35)


def target(vx, vy, wz, g=GAINS):
    fx = vx / 0.15
    neck = g["neck_fwd"] * max(fx, 0) + g["neck_back"] * max(-fx, 0)
    pitch = g["pitch_fwd"] * max(fx, 0) + g["pitch_back"] * max(-fx, 0)
    yaw = g["yaw_turn"] * wz
    roll = g["roll_side"] * vy / 0.2
    return np.array([neck, pitch, yaw, roll])


class AutoHead:
    def __init__(self, dt=0.02, g=GAINS):
        self.g = g; self.a = dt / (g["tau"] + dt); self.h = np.zeros(4)
    def __call__(self, vx, vy, wz, extra=np.zeros(4)):
        self.h += self.a * (target(vx, vy, wz, self.g) - self.h)
        out = self.h + extra
        out[2] = np.clip(out[2], -0.35, 0.35)            # the duck's +-20 deg head-yaw clamp
        return out
