"""Open Duck Mini v2 as an mjlab asset, plus a BeyondMimic tracking env config for it.

Mirrors src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py and
src/mjlab/tasks/tracking/config/g1/env_cfgs.py. Nothing here is tuned; the numbers are the
ones our brax/MJX trainer uses (sim_dt 0.002, ctrl 50 Hz, action scale 0.25) and the
humanoid-scale thresholds/stds of the stock task shrunk by the duck's size.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.spec_config import CollisionCfg

PROJECT = Path(os.environ.get("DUCK_PROJECT", "/home/hermes/src/mini-duck-droid"))
DUCK_XML = (
  PROJECT / "sim/train_pg/playground/open_duck_mini_v2/xmls/open_duck_mini_v2_backlash.xml"
)

# The 14 servo joints, in actuator order (= ctrl order of the XML <position> actuators).
# mjlab resolves names with re.match (prefix), so "left_knee" would also hit
# "left_knee_backlash": every pattern is anchored with ^...$.
SERVO_JOINT_NAMES = (
  "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
  "neck_pitch", "head_pitch", "head_yaw", "head_roll",
  "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
SERVO_JOINTS = tuple(f"^{n}$" for n in SERVO_JOINT_NAMES)
# scene_flat_terrain_backlash.xml keyframe "home" (base z=0.15, ctrl = these angles).
HOME_JOINT_POS = {
  f"^{n}$": v
  for n, v in {
    "left_hip_yaw": 0.002, "left_hip_roll": 0.053, "left_hip_pitch": -0.63,
    "left_knee": 1.368, "left_ankle": -0.784,
    "neck_pitch": 0.0, "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": -0.003, "right_hip_roll": -0.065, "right_hip_pitch": 0.635,
    "right_knee": 1.379, "right_ankle": -0.796,
  }.items()
}
FOOT_GEOMS = r"^(left|right)_foot_bottom_tpu$"


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(DUCK_XML))
  # The tracking task reads "robot/imu_lin_vel" and "robot/imu_ang_vel" on the G1; our
  # XML names the same velocimeter/gyro pair "local_linvel"/"gyro". Rename rather than
  # editing the project XML.
  for s in spec.sensors:
    if s.name == "local_linvel":
      s.name = "imu_lin_vel"
    elif s.name == "gyro":
      s.name = "imu_ang_vel"
  return spec


# Option A (used here): keep the XML <position> servos exactly as the MJX trainer uses them
# (class sts3215: kp=17.11, kv=0, forcerange +-3.23, joint damping 0.56, frictionloss
# 0.068, armature 0.027). Option B would be
#   BuiltinPositionActuatorCfg(target_names_expr=SERVO_JOINTS, stiffness=17.11, damping=0.0,
#                              effort_limit=3.23, armature=0.027, frictionloss=0.068,
#                              viscous_damping=0.56)
# after deleting the XML actuators in get_spec(); it is the same MuJoCo element.
DUCK_ACTUATORS = XmlActuatorCfg(
  target_names_expr=SERVO_JOINTS,
  # Command latency like our action_max_delay: lags are in physics steps (2 ms here).
  delay_min_lag=0,
  delay_max_lag=0,
)

DUCK_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(DUCK_ACTUATORS,),
  soft_joint_pos_limit_factor=0.95,
)

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.15),
  joint_pos=dict(HOME_JOINT_POS),  # backlash joints default to 0
  joint_vel={".*": 0.0},
)

# Only the two TPU soles collide, as in the project scene (friction 0.6, condim 3).
FEET_COLLISION = CollisionCfg(
  geom_names_expr=(FOOT_GEOMS,),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)


def get_duck_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FEET_COLLISION,),
    spec_fn=get_spec,
    articulation=DUCK_ARTICULATION,
  )


# Bodies the tracker scores. body_names[0] must be the root (the body carrying the
# freejoint): the command uses index 0 as the root for reference-state initialisation.
TRACK_BODIES = (
  "base",
  "trunk_assembly",
  "left_roll_to_pitch_assembly",
  "knee_and_ankle_assembly_2",
  "foot_assembly",
  "right_roll_to_pitch_assembly",
  "knee_and_ankle_assembly_4",
  "foot_assembly_2",
  "head_assembly",
)
ANCHOR_BODY = "trunk_assembly"
SCALE = 0.3  # duck (0.35 m) vs G1 (1.3 m): shrink metre-valued stds and thresholds.


def duck_flat_tracking_env_cfg(motion_file: str, play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()
  cfg.scene.entities = {"robot": get_duck_robot_cfg()}
  cfg.scene.sensors = ()  # no self-collision sensor: only the soles have collision geoms

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = 0.25  # rad, as in our joystick/standing envs

  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  motion.motion_file = motion_file
  motion.anchor_body_name = ANCHOR_BODY
  motion.body_names = TRACK_BODIES
  motion.pose_range = {k: (a * SCALE, b * SCALE) if k in "xyz" else (a, b) for k, (a, b) in motion.pose_range.items()}
  # Stock RSI / push velocities (+-0.5 m/s, +-0.52 rad/s) are G1-scale; a 2 kg duck falls.
  vel_range = {k: (a * SCALE, b * SCALE) for k, (a, b) in motion.velocity_range.items()}
  motion.velocity_range = vel_range
  cfg.events["push_robot"].params["velocity_range"] = vel_range
  motion.joint_position_range = (-0.1, 0.1)

  # Observations: the 14 servo joints only (the backlash joints are not observable).
  servo = SceneEntityCfg("robot", joint_names=SERVO_JOINTS)
  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    terms["joint_pos"].params = {**terms["joint_pos"].params, "asset_cfg": servo}
    terms["joint_vel"].params = {**terms["joint_vel"].params, "asset_cfg": servo}

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk_assembly",)
  cfg.events["base_com"].params["ranges"] = {0: (-0.01, 0.01), 1: (-0.01, 0.01), 2: (-0.01, 0.01)}

  # Rewards: metre-valued stds shrink with the robot; drop the self-collision cost;
  # joint-limit cost on servo joints only (backlash joints live at their limits).
  for name in ("motion_global_root_pos", "motion_body_pos"):
    cfg.rewards[name].params["std"] *= SCALE
  cfg.rewards["motion_body_lin_vel"].params["std"] *= SCALE
  cfg.rewards.pop("self_collisions")
  cfg.rewards["joint_limit"].params["asset_cfg"] = servo

  cfg.terminations["anchor_pos"].params["threshold"] = 0.25 * SCALE
  cfg.terminations["ee_body_pos"].params["threshold"] = 0.25 * SCALE
  cfg.terminations["ee_body_pos"].params["body_names"] = ("foot_assembly", "foot_assembly_2", "head_assembly")

  cfg.viewer.body_name = ANCHOR_BODY
  cfg.viewer.distance = 1.0

  # Our MJX recipe: 2 ms physics, 50 Hz control. The project XML asks for iterations=1,
  # ls_iterations=5; keep mjlab's 10/20 until the solver is compared.
  cfg.sim.mujoco.timestep = 0.002
  cfg.decimation = 10
  cfg.episode_length_s = 10.0

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion.pose_range = {}
    motion.velocity_range = {}
    motion.sampling_mode = "start"
  return cfg
