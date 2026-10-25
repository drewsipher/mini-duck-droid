# Skill library

One folder per learned behaviour, one sub-folder per version. This is the hand-off format
between the four teams (framework §2.5, plan §4): the motion-graph runtime loads only complete
packages, the dashboard reads them, the nightly regression re-scores them.

```
skills/<name>/v<N>/
  skill.json              what it is, what it takes, what it may hand off to, the envelope it passed under,
                          and the version record (robot model hash, actuator model, recipe, seed)
  clip.json               reference motion at 50 Hz, Duck Puppet format (below)
  policy.onnx             the 50 Hz policy, input "obs" [1, obs_spec.size], output [1, 14] actions
  report/battery.json     gate G1, tools/battery.py            (required: a package without it is not loadable)
  report/sim2sim.json     gate G2, tools/sim2sim.py
  report/hardware.json    gates G3/G4, written by hand: {"ratio": 0.94, "trials": 50, "setup": "mats", "date": ...}
  report/*.mp4, *.png     one short clip per battery test, battery.png, sim2sim.png
```

`tools/skill_package.py validate <dir>` refuses (exit 1) a package missing `skill.json`,
`clip.json`, `policy.onnx` or `report/battery.json`, or whose `skill.json` fails
`skills/schema.json`, or whose ONNX input width differs from `obs_spec.size`, or whose folder
does not match `name`/`version`. A `robot_model_hash` that is not the current MJCF's is a
warning (`--strict` makes it an error): the skill was trained against an older robot.

## skill.json

Schema: `skills/schema.json` (draft-07). Required fields:

| field | meaning |
|---|---|
| `name`, `type`, `version` | folder name, one of `locomotion` `overlay` `episodic` `recovery`, N of `v<N>` |
| `commands` | `{input: [min, max]}` the runtime inputs and the ranges trained, e.g. `{"vx": [-0.15, 0.15], "vy": [-0.2, 0.2], "wz": [-1, 1]}`; `{}` for a trigger-only clip |
| `entry` | `{from: [names], phase: [lo, hi], window_s}`: which skills may hand off into this one, in which gait-phase window, and how long the reference blend ramps |
| `exit` | `{to: [names], window_s}` |
| `neighbours` | every name in `entry.from` and `exit.to`; the loader checks each exists and passed its battery |
| `timeout_s` | abort to recovery after this long; `null` = none (locomotion), required > 0 for an episodic skill |
| `envelope` | `{torque_nm, max_vel_rad_s, thermal_c, fall_tilt_deg}`: the limits the battery passed under; the runtime enforces the same |
| `robot_model_hash` | sha256 of `sim/train_pg/playground/open_duck_mini_v2/xmls/open_duck_mini_v2_backlash.xml` (`skill_package.py hash-model`) |
| `actuator_model_version`, `recipe_version`, `seed` | the version record |
| `obs_spec` | `{names: [blocks...], size}` in order; known block names are in `tools/skillkit.BLOCK_SIZES` (the walk brain: gyro 3, accelerometer 3, commands 7, joint_pos_rel 14, joint_vel_scaled 14, last_action x3 14, motor_targets 14, feet_contacts 2, gait_phase 2 = 101) |
| `created`, `source {clip, run}`, `status` | ISO time; where the clip and the policy came from; `draft` → `battery_passed` → `sim2sim_passed` → `hardware_passed` → `released` |

Optional, used by the tools: `runtime {action_scale, lpf_hz, head_mode, gait_clock, episodic_cmd}`
(how actions become servo targets; defaults are the deployed walk's except `lpf_hz`, which each
package states explicitly), `thresholds {min_height_m, joint_err_rad, min_tracking_fraction,
battery_pass_ratio, battery_min_test_ratio, sim2sim_rms_rad}`, `hashes {policy_onnx, clip_json}`,
`action_spec`, `status_history`. `create` also writes `policy {kind, action_scale, lpf_hz,
max_vel_rad_s, head_mode add|from_command|policy, accel_bias, gait_period_steps, input_name}`: the
same settings in the vocabulary the motion-graph runtime loads (`graph_runtime/packages.py`
PolicyConfig, which rejects keys it does not know). Keep the two blocks in step until the teams
settle on one name (experiments/questions.md, Software pipeline Q7).

Status moves forward only: `battery.py` sets `battery_passed` on a pass, `sim2sim.py` sets
`sim2sim_passed`, hardware and release are set by hand with `skill_package.py status <dir> <status>`.
A failed re-run never silently downgrades a package; it shows in the regression table.

## clip.json

The Duck Puppet format (`tracker/data/clips/*.json`):
`{"frames": [[t_seconds, {joint: radians}], ...], "interp": "linear", "base": optional [[pos3, quat4], ...]}`
at 50 Hz, 14 joints in the runtime / simulator order `left_hip_yaw, left_hip_roll, left_hip_pitch,
left_knee, left_ankle, neck_pitch, head_pitch, head_yaw, head_roll, right_hip_yaw, right_hip_roll,
right_hip_pitch, right_knee, right_ankle`. A locomotion clip is one gait cycle (start pose = end
pose, from `sim/record_policy_clip.py`); an episodic clip starts and ends in the standing pose.
Extra top-level keys (`recorded_from_policy`, `authored_in`, `tags`) are kept as provenance.

## Tools (run from `sim/` with its venv; rendering wants `MUJOCO_GL=egl`)

```
.venv/bin/python ../tools/skill_package.py create --name bow --type episodic --onnx p.onnx --clip bow.json \
     --entry-from walk --exit-to walk --timeout 4 [--commands vx:-0.15,0.15 ...] [--obs spec.json] [--lpf 8]
.venv/bin/python ../tools/skill_package.py import sim/train_pg/runs_skills/bow/package_draft   # -> skills/bow/v<next>, then runs the battery
.venv/bin/python ../tools/skill_package.py validate ../skills/bow/v1
.venv/bin/python ../tools/skill_package.py list
MUJOCO_GL=egl .venv/bin/python ../tools/battery.py ../skills/bow/v1 --seeds 5 --episodes 10
MUJOCO_GL=egl .venv/bin/python ../tools/sim2sim.py ../skills_archive/upstream_walk_v1
../regression/run.sh                      # every package: validate + battery + sim2sim -> regression/<date>.md
python3 ../experiments/log.py add --title ... --hypothesis ... --command ... --evidence ... --decision ...
```

### battery.py (G1)
Seeds draw the floor friction U(0.5, 1.0), link masses x U(0.9, 1.1), trunk mass +-0.1 kg and the
servo torque ceiling U(0.8, 1.0) x `envelope.torque_nm` (the training recipe's ranges,
`sim/train_pg/playground/common/randomize.py`); episodes jitter the start pose +-0.03 rad, the
settle time 1-2 s and the push direction. A fall is tilt > `envelope.fall_tilt_deg` or base height
< `thresholds.min_height_m` (0.09 m). Rollouts use `sim/sim_test.Sim` (the runtime mirror).

- locomotion: `standing` 3 s, `forward`/`backward`/`turn_left`/`turn_right`/`side_left`/`side_right`
  6 s at the trained range limits, `push_recovery` (15 N for 0.1 s standing, 3 s after). A velocity
  test passes when the duck neither falls nor stands still: at least `min_tracking_fraction` (0.2) of
  the command in the right direction. `no_fall_ratio` is reported next to the strict ratio.
- episodic / overlay / recovery: `clip_tracking`, the clip played from the home pose with the policy
  driving the servos; success = no fall and mean |q - clip| < `thresholds.joint_err_rad` (0.15).
  The observation is built from `obs_spec.names` (`tools/skillkit.PolicyRunner`); a block the
  tools do not know raises instead of guessing (HOOK comment in skillkit.py for the Controls agent's
  tracking.py blocks).

Output `report/battery.json`: `{seeds, episodes, tests: {name: {success, total, ratio, no_fall_ratio,
mean: {...}, episodes: [...]}}, overall_ratio, overall, passed, thresholds, randomisation,
mean_joint_err_rad, created}`; `battery.png`; `battery_<test>.mp4` (first episode of seed 0).
Pass = overall >= 0.9 and every test >= 0.8 (per-package thresholds).

### sim2sim.py (G2)
The package ONNX inside the upstream runtime's own loop (`sim/upstream_runtime/scripts/v2_rl_walk_mujoco.py`
RLWalk.run, `mini_bdx_runtime` HWI / OnnxInfer / LowPassActionFilter / PolyReferenceMotion) with the
servo bus, IMU, foot switches and clock mocked by MuJoCo, against `sim_test.Sim` on the same command
and seed. `lockstep`: both stacks compute targets from the identical state every step (the RMS that
decides `match`); `free_run`: each on its own (falls, joint RMS, achieved velocity). Output
`report/sim2sim.json`: `{match, rms_target_diff_rad, max_diff_rad, rms_clock_aligned_rad,
per_joint_rms_rad, obs_block_rms, free_run, tracking, notes}` and `sim2sim.png`.

Threshold 0.02 rad RMS over 5 s, measured on walk/v1: pipelines that differ only by the servo
rounding (0.001 rad) and the missing speed clip agree to 0.005 rad; the upstream loop's gait clock
running one step behind the scorer's gives 0.020; launching the runtime with an 8 Hz filter the
package does not declare gives 0.056 and the duck stops walking. 0.02 passes noise and catches a
mis-set filter, a wrong action scale or a wrong clock; the one-step clock lag sits exactly on the line
and is reported separately as `rms_clock_aligned_rad` (see experiments/questions.md).

Packages whose `obs_spec` is not the walk layout get `match: null` ("skipped") until the
motion-graph runtime (`graph_runtime/`) can be driven the same way.

### regression/run.sh (G5)
Every `skills/*/v*/`: validate, battery, sim2sim; writes `regression/<YYYY-MM-DD>.md` with the table
(skill, version, type, battery ratio, sim2sim, status, verdict) and a one-line summary; logs in
`regression/logs/<date>/`. Lock file, timeouts, `nice`; `SEEDS`/`EPISODES`/`ONLY`/`NO_VIDEO` env vars.
