# mjlab spike: can MuJoCo Warp + BeyondMimic replace the brax/MJX trainer for the duck?

Date: 2026-10-23 (overnight, GPU owned by the curriculum PPO run most of the time).
Scope: plan section 2 decision 1 and framework 5.3. Everything lives in `~/src/mjlab_spike/`
(clone, venv, duck config, converter, logs); this file is the only project artifact.

## Recommendation (one paragraph)

Stay on brax/MJX for phase 1 and migrate the skill library (episodic clip tracking) to mjlab
in phase 2, starting the port now because nothing blocks it. The spike found no
incompatibility that needs code in mjlab: the duck MJCF loads unchanged, the BAM-tuned
`<position>` servos, joint friction/armature and the backlash joints are preserved exactly
(`XmlActuatorCfg`), the sensors come through, the shipped BeyondMimic tracking task builds
and runs with a converted project clip, and the duck physically follows the bow clip in Warp
when the reference angles are fed as targets. On the RTX 4060 the tracking env runs at
74k env-steps/s with 4096 envs using about 2 GB of VRAM (physics alone 1.0M env-steps/s),
and real PPO iterations measured 32.7k env-steps/s end-to-end at 1024 envs (about 50k
extrapolated to 4096), versus about 42k end-to-end for the current brax curriculum, so a
2 to 3 s clip should train in one to a few hours. What the spike could not do is the one
thing decision 1 asks for: train a clip to convergence (the GPU was busy; the exact command
is below and takes an hour or two). The real porting cost is not the trainer but the
deployment contract: BeyondMimic policies take a clock, a reference anchor orientation and
the reference joint state as observations and export an ONNX that bundles the clip, so the
Pi runtime needs a new policy wrapper and observation builder. Estimated port: 3 to 4 days
to a first clip trained in sim, 8 to 12 days to a first clip on hardware with sim-to-sim
and the strict battery. Locomotion v1/v2 stays on brax until the walking thread is done.

## 1. Versions and installation

| Component | Version | Note |
|---|---|---|
| mjlab | 1.6.0, git `8bb2bc9e` (2026-10-18) | Apache-2.0, `~/src/mjlab_spike/mjlab` |
| mujoco | 3.16.0 | `MjSpec` API everywhere |
| mujoco-warp | 3.16.0 | GPU physics |
| warp-lang | 1.18.0 | from `pypi.nvidia.com` index, CUDA 13 build |
| torch | 2.14.0+cu130 | `--extra cu130`; `cu126` exists for drivers 525+ |
| rsl-rl-lib | 5.5.1 | PPO runner, ONNX export |
| Python | 3.13.13 (uv-managed) | mjlab needs >=3.10; `uv sync --python 3.12` also fine |
| Driver / CUDA | 580.178.04 / 13.0 | docs require driver 580+ for cu130 |

Install (what was run): `git clone https://github.com/mujocolab/mjlab ~/src/mjlab_spike/mjlab &&
cd ~/src/mjlab_spike/mjlab && uv sync --extra cu130` (10 min, venv 6.7 GB). Imports
`mjlab`, `mujoco_warp`, `warp` take 2.3 s. CPU-only work must hide the GPU
(`CUDA_VISIBLE_DEVICES=""`), otherwise Warp creates a CUDA context on import (FAQ); with the
GPU hidden MuJoCo Warp compiles and runs on the CPU (slow, but every check below also ran
there).

## 2. What ships (`list-envs`)

12 tasks: `Mjlab-Cartpole-{Balance,Swingup}`, `Mjlab-Lift-Cube-Yam{,-Depth,-Rgb}`,
`Mjlab-Multi-Cube-Seg-Yam`, `Mjlab-Velocity-{Flat,Rough}-Unitree-{G1,Go1}`,
`Mjlab-Tracking-Flat-Unitree-G1` and `...-No-State-Estimation`. The tracking task is a
re-implementation of BeyondMimic (`tracking_env_cfg.py`, commit f8e20c8 of
whole_body_tracking): adaptive start-frame sampling, RSI with pose/velocity noise, the six
exp-kernel tracking rewards, action-rate, joint-limit and self-collision costs, z-only
anchor/end-effector and anchor-orientation terminations, PPO 512-256-128 with obs
normalisation, 24 steps per env, 30k iterations max.

## 3. How a robot is defined (and the duck recipe)

`EntityCfg(spec_fn, init_state, articulation, collisions, ...)`:
`spec_fn` returns an `mujoco.MjSpec` (the MJCF, unchanged); `init_state` is root pose plus a
`{regex: rad}` joint dict that mjlab turns into a keyframe named `init_state`;
`articulation.actuators` is a tuple of actuator configs matched to joints by regex;
`CollisionCfg` edits contype/conaffinity/condim/priority/friction by geom regex. The scene
attaches the entity under a `robot/` prefix (sensors become `robot/<name>`), adds its own
terrain and overrides `<option>` from `MujocoCfg` (implicitfast, Newton, pyramidal cone).

Duck config (full file: `~/src/mjlab_spike/duck/duck_mjlab_cfg.py`; it builds and compiles):

```python
DUCK_XML = PROJECT / "sim/train_pg/playground/open_duck_mini_v2/xmls/open_duck_mini_v2_backlash.xml"
SERVO = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
         "neck_pitch", "head_pitch", "head_yaw", "head_roll",
         "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")

def get_spec():
  spec = mujoco.MjSpec.from_file(str(DUCK_XML))
  for s in spec.sensors:                      # tracking cfg reads robot/imu_lin_vel, robot/imu_ang_vel
    s.name = {"local_linvel": "imu_lin_vel", "gyro": "imu_ang_vel"}.get(s.name, s.name)
  return spec

robot = EntityCfg(
  spec_fn=get_spec,
  init_state=EntityCfg.InitialStateCfg(            # = scene keyframe "home"
    pos=(0, 0, 0.15),
    joint_pos={f"^{n}$": v for n, v in HOME.items()}),  # anchored! see incompatibility 1
  articulation=EntityArticulationInfoCfg(
    actuators=(XmlActuatorCfg(target_names_expr=tuple(f"^{n}$" for n in SERVO)),),
    soft_joint_pos_limit_factor=0.95),
  collisions=(CollisionCfg(geom_names_expr=(r"^(left|right)_foot_bottom_tpu$",),
                           contype=1, conaffinity=1, condim=3, priority=1, friction=(0.6,)),),
)
```

Verified after compile: 16 bodies, 24 joints (14 servo + 10 backlash), 14 actuators with
`gainprm 17.11 / biasprm -17.11 / forcerange +-3.23`, `left_knee` damping 0.56,
frictionloss 0.068, armature 0.027, backlash range +-0.0087 rad, 15 sensors, 2 collision
geoms, keyframe qpos/ctrl identical to `scene_flat_terrain_backlash.xml` "home".

Tracking env overrides (same file, `duck_flat_tracking_env_cfg`): anchor `trunk_assembly`;
tracked bodies `base` (must be first: the command uses index 0 as the root for RSI),
`trunk_assembly`, both `*_roll_to_pitch_assembly`, `knee_and_ankle_assembly_{2,4}`,
`foot_assembly{,_2}`, `head_assembly`; joint obs and joint-limit cost restricted to the 14
servo joints; `self_collisions` reward and sensor removed (only the soles collide);
metre-valued stds, thresholds, RSI and push velocities scaled by 0.3; action scale 0.25 rad;
`timestep 0.002`, `decimation 10` (50 Hz, as our MJX recipe). Actuator options if we want
more than the XML servo: `BuiltinPositionActuatorCfg` (same element, parameters in Python),
`BuiltinDcMotorActuatorCfg` (native MuJoCo 3.16 `<dcmotor>`: Kt/Ke/R or datasheet
stall/no-load, voltage limit, thermal, LuGre; the natural home for BAM), `DcMotorActuatorCfg`
(explicit PD with a torque-speed clamp), `LearnedMlpActuatorCfg`. All take
`delay_min_lag/max_lag` in physics steps.

Task registration (`duck_tasks.py`, `train_duck.py`, `play_duck.py`): mjlab's CLIs only import
`mjlab.tasks`, so an out-of-tree task is a module that calls `register_mjlab_task` and a
two-line wrapper around `mjlab.scripts.train.main`. `Mjlab-Tracking-Flat-Open-Duck-Mini`
registers and shows up in `list_tasks()`.

## 4. Motion data: what the tracking task wants, and the converter

`MotionLoader` reads one `.npz` with `fps`, `joint_pos (T, nj)`, `joint_vel (T, nj)`,
`body_pos_w (T, nb, 3)`, `body_quat_w (T, nb, 4)` (wxyz), `body_lin_vel_w`, `body_ang_vel_w`
(T, nb, 3). `nj` is every non-free joint of the entity in spec order, so for the duck 24
columns with the 10 backlash joints at 0; `nb` is every body except world in depth-first
spec order (16 for the duck), and the command then picks `body_names` out of it. The shipped
`scripts/csv_to_npz.py` takes Unitree-style CSV (base pos, base quat xyzw, joint angles),
resamples to 50 Hz, plays it through mjlab's own Simulation for forward kinematics and
uploads to a W&B registry; the loader itself only needs the local file, and `train`/`play`
accept `--env.commands.motion.motion-file <npz>` (`--motion-file` for `play`), so W&B is
optional (`--agent.logger tensorboard`).

Converter from our `{"frames": [[t, {joint: rad}], ...], "base"?: [[pos, quat_wxyz], ...]}`
clips: `~/src/mjlab_spike/duck/clip_to_mjlab_npz.py` (plain MuJoCo, 120 lines).
1. Linear-resample the joint dict onto a 50 Hz grid; scatter into the 24-column layout by
   joint name; `joint_vel` by central differences.
2. Root track: use `base` when present (Blender export gives it); otherwise hold the home
   root pose and shift z per frame so the lowest foot-mesh vertex keeps its first-frame
   height (foot pinning). Walking clips need a measured base track
   (`record_policy_clip.py` can supply it).
3. For each frame set qpos/qvel, `mj_forward`, record `xpos`, `xquat` and
   `mj_objectVelocity` for every body.
4. `np.savez`. `test_bow.json` becomes 126 frames x 24 joints x 16 bodies and loads in the env.
For production, run step 3 through mjlab's `Simulation` as `csv_to_npz` does, so velocity
conventions match the trainer exactly, and add the clip checks (`anim/clip_check.py`) before
conversion.

## 5. Incompatibilities and gotchas found

1. Name resolution is prefix matching. `InitialStateCfg.joint_pos` goes through
   `resolve_expr`, which uses `re.match`, so `"left_knee"` also set `left_knee_backlash`
   to 1.368 rad (range +-0.0087): the keyframe was silently wrong. Anchor every pattern
   (`^name$`). `SceneEntityCfg` and actuator regexes use `fullmatch` and are safe.
2. Backlash joints are first-class joints: they count in `joint_names` (24), in the motion
   command (`command` obs is 48-dim, half of it zeros), in the NPZ and in RSI. Observations,
   joint-limit cost and encoder-bias DR must be restricted to the 14 servo joints
   (done through `SceneEntityCfg(joint_names=...)`). Harmless but wasteful; a 24-column
   clip format is the price.
3. Robot-XML `<option>` is ignored on attach (`iterations=1, ls_iterations=5`,
   `eulerdamp` flag): mjlab uses `MujocoCfg` (tracking: implicitfast, Newton, 10/20
   iterations, 5 ms default, set to 2 ms here). Sim-to-sim with plain MuJoCo must copy these.
4. Keyframe: the robot XML has none (it lives in the scene file), mjlab builds `init_state`
   from the config instead; fine. `Entity.reset()` does not teleport to it (events or the
   motion command do), which is a trap for hand-written harnesses, not for the env.
5. Sensors: our `<sensor>` block survives and is exposed as `robot/<name>`; the tracking cfg
   hard-codes `robot/imu_lin_vel` and `robot/imu_ang_vel`, so the velocimeter/gyro are
   renamed in `spec_fn`. The other 13 sensors are computed every step; prune later.
6. Humanoid-scale constants everywhere: pushes and RSI velocities +-0.5 m/s, reward stds
   0.3 m, terminations 0.25 m. Unscaled, a zero-action duck terminated 1.9 times per env in
   4 s; scaled by 0.3 the kinematic replay still tripped the z-only end-effector check once
   per env, so these need a tuning pass against clips, not a constant.
7. Our brax recipe has three things mjlab does not provide as config: the per-step motor
   target rate clamp (`MAX_MOTOR_VEL * dt`), the 8 Hz action low-pass, and the scaffold
   forces. The first two are one custom action term (subclass `JointPositionAction`,
   about 50 lines); action delay maps to `ActuatorCfg.delay_*` (physics steps, 2 ms units,
   ours is in control steps), IMU delay to `ObservationTermCfg.delay_*`, noise to the noise
   cfg, pushes to the interval event, and DR to `dr.body_mass/pd_gains/effort_limits/
   joint_friction/joint_armature/joint_damping/geom_friction/encoder_bias/body_com_offset`.
8. Deployment contract. Actor obs = 105: reference joint pos+vel (48), anchor pos in base
   frame (3), anchor orientation in base frame (6), base lin vel (3), gyro (3), q (14), dq
   (14), last action (14). The `No-State-Estimation` variant drops anchor position and base
   linear velocity and is the one the duck can run (no odometry). The tracking runner's
   ONNX takes `obs[1,105]` and `time_step[1,1]`, bundles the clip, and returns `actions[1,14]`
   plus the reference frame (`joint_pos`, `joint_vel`, tracked `body_*`). So the Pi runtime
   needs a clock-driven wrapper and an observation builder (reference anchor quaternion vs
   IMU quaternion, yaw removed). This, not the trainer, is the bulk of the port.
9. Head joints are actuated and tracked like the legs; decision 6 (head is a show function)
   means the clip author drives them, or they get dropped from action/obs later.
10. Separate toolchain: second venv (torch cu130, no jax), 6.7 GB; W&B default logger but
    tensorboard works; `csv_to_npz` is W&B-only (replaced by our converter).

## 6. Measurements (RTX 4060 8 GB, duck model, logs in `~/src/mjlab_spike/logs/`)

GPU windows were taken between the other agent's runs with an `nvidia-smi` guard before
every command; each run stepped for a few seconds. Figures are env steps per second
(one env step = 10 physics steps at 2 ms); "physics only" is `Simulation.step` with the
servos holding the home pose; "tracking env" is the full BeyondMimic env with a zero agent
(observations, rewards, terminations, DR, no policy network).

| num_envs | physics only (env-steps/s) | tracking env (env-steps/s) | nvidia-smi used, total (MiB) |
|---|---|---|---|
| 256 | 473k | - | - |
| 512 | 666k | 27.4k | 1056 |
| 2048 | 1.00M | 61.8k | 1788 |
| 4096 | 1.04M | 74.3k | 2692 |

Baseline usage before the runs was 494 to 703 MiB (desktop plus Blender), so the mjlab
process takes roughly 0.4 / 1.1 / 2.0 GB at 512 / 2048 / 4096 envs; the memory is Warp's,
torch holds under 50 MiB. Env build takes 0.3 to 4 s once kernels are cached (first CUDA
compile about 40 s, first CPU compile about 15 s). CPU with the GPU hidden: 2 envs run at
450 to 600 physics env-steps/s, the env at 46 to 50 env-steps/s; a 32-env PPO iteration
takes 10 s, enough for functional checks only.

Physics sanity: holding the home pose, root z settles at 0.160 m in mjlab on CPU and GPU
and in plain MuJoCo on the project scene (identical). Kinematic replay of `test_bow`
(reference angles as position targets, play mode): final errors joint 0.058 rad (norm over
24), anchor position 2.3 cm, mean body position 1.6 cm, anchor rotation 0.12 rad. Zero
action in play mode stands the full 2.5 s with no termination. A CPU run of
`MotionTrackingOnPolicyRunner.learn()` for 2 iterations plus ONNX export completed
(`train_smoke_cpu32.log`, `exported/policy.onnx`).

End-to-end PPO, measured in a 60 s GPU window with 1024 envs (the G1 tracking PPO recipe,
`train_smoke_1024.log`): 0.75 s per iteration after warm-up = 32.7k env-steps/s including
policy inference and the update, 1246 MiB total on the card (about 0.75 GB for the process,
torch peak 173 MiB); ONNX export at the end took under a second. The rollout alone at this
size is about 45k env-steps/s, so the PPO update costs about 0.2 s per 24.6k samples.
Extrapolated to 4096 envs (98k samples per iteration): 1.3 s rollout plus about 0.6 s
update, roughly 2 s per iteration and 50k env-steps/s end-to-end, against about 42k for the
brax curriculum (100M steps in 40 min at 4096 envs); 3000 iterations is about 1.7 h.
To confirm at 4096 envs (under 60 s, GPU must be free):

```bash
cd ~/src/mjlab_spike/duck && ../mjlab/.venv/bin/python duck_train_smoke.py \
    --motion ../motions/test_bow_motion.npz --device cuda:0 --num-envs 4096 --iters 4 \
    --log-dir ../logs/train_smoke_4096
```

Stock sanity commands (G1 needs a W&B motion registry; the duck does not):

```bash
cd ~/src/mjlab_spike/mjlab && uv run play Mjlab-Velocity-Flat-Unitree-G1 --agent zero --num-envs 16 --viewer viser
cd ~/src/mjlab_spike/duck && ../mjlab/.venv/bin/python play_duck.py Mjlab-Tracking-Flat-Open-Duck-Mini \
    --agent zero --num-envs 4 --motion-file ../motions/test_bow_motion.npz --viewer viser
# first real training run (tensorboard logs under logs/rsl_rl/duck_tracking/):
cd ~/src/mjlab_spike/duck && ../mjlab/.venv/bin/python train_duck.py Mjlab-Tracking-Flat-Open-Duck-Mini \
    --env.commands.motion.motion-file ../motions/test_bow_motion.npz --env.scene.num-envs 4096 \
    --agent.max-iterations 3000
```

## 7. Effort to port (working days)

| Item | Days |
|---|---|
| Duck asset + tracking cfg from the spike, scaled constants, head policy, DR parity (mass, kp, delays), custom action term for rate clamp and low-pass | 1.5 to 2 |
| Converter through mjlab's Simulation, base track from Blender / recorder, clip checks, 24-column layout | 1 |
| Out-of-tree package, train/play/eval wrappers, VideoRecorder reports, experiment-log hook | 1 |
| Tuning until `test_bow` (then a jump) trains cleanly; multi-seed battery in mjlab | 2 to 4 |
| Runtime: clock-driven ONNX wrapper, observation builder (no-state-estimation variant), sim-to-sim in plain MuJoCo with the deploy contract, hardware trial | 2 to 3 |
| Total to first clip in sim / on hardware | 3 to 4 / 8 to 12 |

Reasons behind the recommendation, in one list: no blocker in loading or physics; servo
model preserved exactly; one MuJoCo file stays the source of truth for training and
sim-to-sim; the tracking recipe we would otherwise re-implement in brax comes for free and
already has adaptive sampling, RSI, the reward set and ONNX export; VRAM is a quarter of the
card at 4096 envs; the costs are a second toolchain, a clip-format and deploy-contract
change, humanoid constants to retune, and the fact that an end-to-end training run has not
been done yet, which one overnight run settles.
