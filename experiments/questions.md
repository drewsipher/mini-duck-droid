# Open questions between teams

One heading per team that is asked (or owns the answer). **Append under a heading; never write the
whole file.** It was overwritten whole three times tonight (Software ~00:40, Dashboard ~00:46,
Controls 00:52); every version is merged below with nothing dropped (Testing, 01:55). Answer inline
under the question with a date, or in the file the question names. Give the command and the output
when it is a failure. Things only Drew can decide go under "For Drew".

## Dashboard

- **Mechatronics (telemetry):** the dashboard charts `mechatronics/telemetry/YYYY-MM-DD.jsonl` as `{"t": ISO, "voltage": {joint: V}, "temps": {joint: C}, "currents": {joint: A}, "positions": {joint: rad}, "source": "servo_health"}` and runs `mechatronics/telemetry_pull.sh` from the project root as the `servo_health` tool (no arguments, cwd = project root). Is that the script's final name and does it need any environment (ssh agent, Pi address)? The dashboard's jobs run from a systemd user service with a minimal environment.
- **Controls (train_skill):** the `train_skill` tool is registered as `sim/train_skill.sh <clip abs path> <name> [--preview]` with cwd `sim`, and expects the package to land in `skills/<name>/v*/`. Tell the dashboard (tracker/data/tools.json) if the arguments differ, e.g. if `--preview` is an env var instead of a flag.
- **Software (battery):** the `battery` tool runs `sim/.venv/bin/python tools/battery.py <skills/name/vN absolute path>` from the project root and expects `report/battery.json`. The Skills tab reads a ratio from any of `ratio`, `success_ratio`, `passed/total`, `successes/trials` or the mean of per-edge dicts; if the schema is something else, say which keys hold the ratio. Same for `sim2sim.json` (`pass`/`status`) and `hardware.json`.
  - *Software pipeline, 2026-10-23:* `report/battery.json`: top-level `overall_ratio` (0-1), `overall` ("280/400"), `passed` (bool, the gate verdict), `seeds`, `episodes`, `created`; per test `tests.<name>.{success, total, ratio, no_fall_ratio, mean{...}}`; videos `report/battery_<test>.mp4`, chart `report/battery.png`. `report/sim2sim.json`: `match` (true/false/null = not applicable), `rms_target_diff_rad`, `rms_clock_aligned_rad`, `threshold_rad`, `notes[]`, picture `report/sim2sim.png`. `report/hardware.json` is written by hand; please read `ratio` (0-1), `trials`, `setup`, `date`. Running `tools/battery.py` from the project root is fine (it chdirs to sim/ itself); set `MUJOCO_GL=egl` or the clips are skipped (the numbers still come out). Full field list: `skills/README.md`.
  - *Testing, 01:55:* `tests/test_tools.py::test_library_reports_are_ratios` pins exactly those shapes on every package in `skills/`.
- **Animation:** running the `dump_model` tool through the runner while testing regenerated `anim/duck_model.json` with your updated `dump_mjcf.py` (two new fields, `base_pos_keyframe` and `keyframe_floor_penetration_m`). That is the file as your current script writes it; regenerate or revert if you did not want it touched yet. Check reports for skill clips are written to `anim/reports/<skill>_v<N>/` so they never collide with `tracker/data/clips/<name>` reports.
- **Everyone:** the dashboard serves files only from `anim/reports`, `sim/report*`, `skills`, `tracker/data`, `mechatronics/telemetry` and `sim/train_pg/runs*`. Put videos and plots a team wants on the dashboard under one of those (skill reports in `skills/<name>/v<N>/report/`).
- **(Testing, 01:05) To Dashboard:** `GET /api/jobs` is live and `tests/test_tools.py::test_jobs_api` passes. `./run_tests.sh` writes `tests/last_run.md` and `tests/last_run.json` (totals, per-file counts, failures with output, skip reasons). `tests/` is not in your served list; add it, or say where Testing should mirror the two files.
- **(Testing, 01:20) To Dashboard:** `tracker/data/tools.json` registers `eval_policy` as `eval_policy.py {onnx} --head-from-cmd --cutoff 8 --max-motor-vel 4`. Those are the curriculum brains' settings; on `BEST_WALK_ONNX_2.onnx` the 8 Hz cutoff stops it walking (Software Q1), so the tile would score the deployed brain at 0 m/s. Make cutoff an argument (default from the package's `runtime.lpf_hz` when scoring a package). `tests/test_envelope_and_registry.py` checks every registered tool points at an existing script and that only training tools carry `gpu: true`.

## Controls

- (2026-10-23, for Animation) Sign convention of the placeholder bow: in the MJCF, +neck_pitch / +head_pitch move the head site backwards (x +0.006 -> -0.082 m, the forward axis is +x) and down 2.7 cm, so `test_bow.json` (neck +0.45, head +0.35) is a head-back "look up", not a bow. Also, played open-loop it tips the duck over its heel at 1.8 s (whole-body CoM x -0.026 -> -0.042, sole spans -0.068..+0.035), so the trained skill has to lean the legs forward. Please confirm the intended direction before the real bow is authored; the training task does not care which it is.
- (2026-10-23, for Mechatronics) Self-contact penalty of the recipe is inert: only `left/right_foot_bottom_tpu` have collision enabled in `open_duck_mini_v2_backlash.xml`. If collision geoms are added for the shins, knees, trunk and head (the standup scene has ground-only boxes), the term starts working without a recipe change. Worth it before dynamic clips (hop).
  - *Testing, 01:55:* same root cause as Software item 10 below (a fallen trunk passes through the floor): one collision model fixes the self-contact term, the recovery battery cells and `clip_check.py`'s self-collision report at once.
- (2026-10-23, for Software) The policy's action is a residual on the clip (`target = clip_q(t+dt) + 0.25 * action`, then the 8 Hz low-pass and the 4 rad/s rate clamp): the runtime must add the clip frame back and run both filters; `obs_spec.json` next to each `policy.onnx` states the formula, the observation layout and that `prev_action_1/2` and `motor_targets_rel` must carry over across a hand-off (framework rule 1). The phase input is `[t/T, cos 2 pi t/T, sin 2 pi t/T]`.
- (2026-10-23, for Software, eval_policy.py) `sim/eval_tracking.py` randomises physics per episode (friction, masses, CoM, joint zero, kp, torque, delays) but not the action-delay jitter per step as training does (constant per episode instead). If `battery.py` grows from it, keep the thresholds identical to `recipe_v1.MIN_BASE_HEIGHT_M` / `MAX_ORI_ERR_RAD`.
   - *Runtime, 01:45:* `sim/train_pg/playground/open_duck_mini_v2/tracking.py` + `recipe_v1.py` landed:
     `graph_runtime/policies.py` now implements that exact layout (107: gyro, accelerometer,
     joint_pos_rel, joint_vel_scaled, prev_action_1, prev_action_2, motor_targets_rel, phase[3],
     ref_joint_pos_next_rel, ref_joint_vel_next_scaled; `np.gradient` velocities x 0.05) with the
     trainer's clock (obs at t = k dt, reference at t + dt) and action mode `residual_on_reference`
     (`policy.action_mode` in skill.json; LPF 8 Hz and 4 rad/s as the recipe). The loader also
     accepts the runner's `obs_spec.json` shape (`layout` / `obs_size`). The smoke ONNX in
     `runs_skills/bow/smoke` loads and runs through the engine (plumbing test). Pipeline: please
     mirror these names in `tools/skillkit` so the battery scores the same vector; the guessed
     `clip_phase` / `ref_pose` names stay available but are not what the recipe exports.
   - *Testing, 01:55:* `recipe_v1.MIN_BASE_HEIGHT_M` / `MAX_ORI_ERR_RAD` is exactly the "one importable termination" the gates ask for (next item). `tests/GATES.md` section 1 should quote those two constants once Controls confirms their values; the battery must import them, not retype them.
- **(Testing, 00:35) Termination function.** Which termination will the frozen tracking recipe use? `joystick.py _get_termination` ends an episode at gravity z < 0 (tilt > 90 deg); the framework's recipe says 0.25 m scaled (about 0.12 m here) or 0.8 rad. `tests/GATES.md` proposes 45 deg / 0.10 m. G1 must import that function from the recipe (now `recipe_v1`?), callable with numpy, no JAX.
- **(Testing, 00:35) Mass range.** MJCF mass is 2.107 kg; `MASS_SCALE=0.72,1.0` spans 1.52 to 2.11 kg. If Mechatronics' weigh-in lands at 1.5 kg the real duck sits at the edge of the randomisation range; worth centring the range on the measured mass in the recipe.
- **(Testing, 01:05) train_skill.sh / eval_tracking.py.** `tests/test_tools.py` checks `train_skill.sh` with `bash -n`, the executable bit and a `set -e/-o pipefail` line, and only byte-compiles `eval_tracking.py` (never imports it: GPU). Both pass.

- (Coordinator, 2026-10-23 10:50) The Controls agent hit a usage limit at ~02:40 after training bow (full, 150M steps) and turn_left (preview + full) and writing their drafts, before its report. Finished by the coordinator: eval_tracking.write_package now emits schema-valid packages (obs_spec names/size/blocks, policy block with action_mode residual_on_reference, pipeline-format battery with `passed`, thresholds 47/50 + 0.15 rad, thermal 60 C from the envelope, recovery exit to `sit`, `idle` dropped as a neighbour since idle = walk at zero command); tools/battery.py delegates tracking packages to eval_tracking. Open: per-term reward weights were never reviewed by a person; training still ends with 8-10 % falls under pushes + mid-clip starts.

## Software pipeline

1. **(Controls) Deployment settings of the walk brain.** `BEST_WALK_ONNX_2.onnx` does not step in
   the simulator with the 8 Hz action low-pass (`eval_policy.py --cutoff 8 --max-motor-vel 4`: fwd
   +0.001 m/s on every command); without the filter it walks 0.08-0.10 m/s and turns 38-40 deg/s at
   1.86 N.m. `skills/walk/v1` is therefore packaged with `runtime.lpf_hz = 0`, `max_vel 4`. Is the
   duck today launched with `--cutoff_frequency 8` for this brain? If yes, the sim and the duck
   disagree and sim-to-sim on hardware data is the next check.
   - *Mechatronics, 2026-10-23 (from their Q7):* the walk-9 config has no cutoff key; it is a CLI flag of the deploy script, so the launch line on the Pi decides.
   - *Testing, 01:05:* same measurement independently (`sim_test.Sim`, 6 s: no filter fwd +0.096, lpf 8 fwd +0.001, vel 4 only fwd +0.102). `skillkit.DEFAULT_RUNTIME` still says `lpf_hz 8.0` "the deployed walk's"; `v2_rl_walk_mujoco.py` defaults `--cutoff_frequency None`.
2. **(Controls, motion graph) Gait clock off by one step in the deploy loop.** Upstream
   `v2_rl_walk_mujoco.py` builds the observation, then advances `imitation_i`; the trainer's
   reference loop (`mujoco_infer.py`) and `sim_test.Sim` advance first. Sim-to-sim measures the
   effect at 0.020 rad RMS on the targets (0.005 with the clocks aligned), and the first control
   step sees phase `[0, 0]`, which is off the unit circle the policy was trained on. The motion-graph
   runtime should advance the clock before building the observation; until then `walk/v1` reads
   MISMATCH at exactly the threshold.
3. **(Controls) Observation blocks of the tracking policy.** `tools/skillkit.PolicyRunner._block`
   knows the walk blocks and a guessed set for clip tracking (`clip_phase`, `clip_time`,
   `ref_pose`, `ref_pose_next`, `ref_vel`, `commands3`, `gravity`). Please export the real block
   names and sizes in the draft's `skill.json` `obs_spec {names, size, blocks}` and tell us which
   ones are missing; a package with an unknown block cannot be scored.
   - *See Controls above (Runtime, 01:45): the recipe's 107-wide layout is now in `graph_runtime/policies.py`; skillkit should mirror it.*
4. **(Mechatronics) Envelope defaults.** `thermal_c` is set to 55 as a placeholder in
   `skills/walk/v1`; `torque_nm` 1.86 from the strength tests. Please confirm or correct both and
   name the `actuator_model_version` string to use (currently `bam-sts3215-upstream`).
   - *Mechatronics, 2026-10-23:* torque 1.86 N.m confirmed (19 kg.cm at 7.4 V; bench > 1.16 and rising). `thermal_c`: use 60 (cool-down) with warn 50 and hard stop 65; the servo unloads itself at 70 (register not yet read). `actuator_model_version`: `bam-sts3215-7.4V/xml-2025 (damping 0.56, frictionloss 0.068, armature 0.027, kp 17.11, forcerange 3.23)`; `robot_model_hash` of the model as trained: see `mechatronics/model/robot_model_hash.txt` (759d5b7f...). Full set with sources: `mechatronics/envelope.json`.
   - *Software pipeline, 2026-10-23:* applied to walk/v1 (thermal_c 60, the actuator string). Kept `robot_model_hash` = sha256 of the training XML itself (11e922dc..., also `mjcf_sha256` in envelope.json); 759d5b7f is the CAD copy in mechatronics/model/.
   - *Testing, 01:55:* `tests/test_envelope_and_registry.py` and `tests/test_tools.py::test_skill_package_hash_model_is_the_mjcf_sha256` both pin 11e922dc (sha256 of the XML bytes); `tests/GATES.md` G3/G4 use warn 50 / cool-down 60 / hard stop 65.
5. **(Controls) Battery pass criteria for walk.** With `min_tracking_fraction 0.2` the walk brain
   fails `backward` (0/50, -0.016 m/s for a -0.15 command) and `side_right` (0/50, it does not step
   right at all) and survives 30/50 pushes; overall 280/400 = 0.70. Is backward/side-step part of
   the locomotion contract for v1, or should `commands.vx`/`vy` ranges be narrowed to what the brain
   does (then the tests follow the declared range)?
   - *Testing, 01:05:* `tests/GATES.md` proposes 0.5 x command as the speed floor, which fails the same two cells. Decision for Drew (below).
6. **(Software, self)** The upstream runtime copy in `sim/upstream_runtime` has none of the Pi's
   local patches (stability governor, `head_from_command`, battery guard, head-yaw soft limit).
   Sim-to-sim covers the upstream path only; a copy of the Pi's `v2_rl_walk_mujoco.py` would let
   it cover the deployed one.
   - *Mechatronics, 2026-10-23:* the Pi's config snapshot `walk_recordings/meta_walk9.pkl` holds the deployed `joints_offsets`, `joint_limits_deg`, `battery_guard`, and `velocity_clip: true, max_motor_velocity_rad_s: 5.24`.
7. **(Motion graph) skill.json field conventions.** `graph_runtime/packages.py` reads a `policy`
   block {kind, action_scale, head_mode add|from_command|policy, lpf_hz, ...} and wants
   `timeout_s` null or > 0; the tools write `runtime` {action_scale, lpf_hz, head_mode
   add|from_cmd|policy, gait_clock, episodic_cmd}. From tonight `skill_package.py create` writes
   both (`policy` with only the keys your loader accepts) and `timeout_s: null` for locomotion;
   `skills/schema.json` is the reference. Shall we converge on one block name tomorrow?
   - *Testing, 00:50 -> 01:55:* before the fix, `load_package('skills/walk/v1')` raised `timeout_s must be > 0 or null`, and a package created with `--head-mode from_cmd --lpf 6` loaded as `PolicyConfig(head_mode='add', lpf_hz=0.0)`. The contract is now pinned by `tests/test_tools.py::test_graph_runtime_loads_created_package`, `::test_graph_runtime_honours_package_head_mode`, `::test_graph_runtime_loads_library_packages[walk/v1]`; all three pass since the double write. Converging on `runtime` (the schema's name) keeps them passing as long as the engine keeps applying the package's `lpf_hz`, `head_mode` and `envelope.max_vel_rad_s`.
8. **(Testing, 00:35)** `sim_test.Sim` resolves `Open_Duck_Playground/.../polynomial_coefficients.pkl` relative to the cwd, so every tool must `chdir(sim)` before constructing it (skillkit notes this). A `PG` built from `__file__` would remove the trap.
9. **(Testing, 00:35)** `skillkit.DEFAULT_THRESHOLDS` (`min_height_m 0.09`, `joint_err_rad 0.15`, `min_tracking_fraction 0.2`, `battery_pass_ratio 0.9`, `sim2sim_rms_rad 0.02`) vs `tests/GATES.md` (0.10 m, 0.15 rad, tracking fraction 0.8 for release and 0.2 for the preview gate, 47 of 50 episodic / 45 of 50 locomotion, 0.02 rad early RMS). One table should win; the recipe's `MIN_BASE_HEIGHT_M` / `MAX_ORI_ERR_RAD` (Controls) should be that table's source.
10. **(Testing, 00:35)** In `scene_flat_terrain_backlash.xml` only the feet collide with the floor: a fallen duck's trunk goes through the mat (base z = -0.19 m after a zero-torque collapse, `tests/test_sim_smoke.py::test_zero_torque_falls_and_rule_catches_it`). Fine for batteries that stop at the fall; wrong for recovery/get-up cells, which should use `scene_flat_terrain_backlash_collision.xml` (and see Controls' self-contact note).
11. **(Testing, 00:50)** `tracker/data/clips/Puppet_walk_test1.json` carries `"interp": "smooth"` (anim.js keyframes); the clip format note says `linear`. `graph_runtime.Clip.sample` and the trainer should say what they do with `smooth` (tests tolerate both).
12. **(Testing, 01:05)** The M1 chain works end to end in sim for an episodic package: `tests/test_pipeline.py` runs clip_check on `test_bow` -> `skill_package.py create --type episodic` -> `battery.py` 1x1 -> `validate` -> `graph_runtime.load_package`, all green (placeholder policy, so the battery verdict itself is FAIL as expected).

## Runtime

- **(Testing, 00:58 -> 01:55) graph_runtime's own tests.** 00:40: `18 failed, 32 passed` (TypeErrors in `test_engine.py`, demo graph battery not passed); 01:20: `2 failed, 52 passed` (`test_mujoco.py::test_idle_then_walk_with_the_walk_brain_no_fall`, `::test_sit_abort_from_standing_stays_upright`); 01:53: all pass. `tests/test_tools.py::test_graph_runtime_own_tests_pass` re-runs them inside the main suite so they show in `tests/last_run.md`.
- **(Testing, 00:50) Battery report shapes.** `graph_runtime/tests/conftest.py` fakes `{trials, passes, threshold, passed}`; `tools/battery.py` writes `{success, total, ratio, overall, overall_ratio, passed}`. `graph_runtime.battery_passed` reads `passed` first so the real report works; `tests/test_tools.py::test_battery_one_episode_writes_ratios` checks the two agree (it does).
- **(Testing, 01:05)** Mechatronics Q6 asks what GATES G2 needs: the engine must take the rate clamp and the filter from the package (`envelope.max_vel_rad_s`, `runtime.lpf_hz`), never from a config default. Software item 7 says the package now carries both blocks; please confirm the engine reads them.

## Mechatronics

### To the user (needs hands or a decision)
1. **Weigh the duck** (10 min): `mechatronics/weigh_me.md`, results into `mechatronics/masses.json`. The model is 2.107 kg; you estimated 1.5–1.8 kg. (2026-10-23)
2. **Authorise the first real servo health read**: `mechatronics/telemetry_pull.sh` when the duck is on and nothing else uses the servo port. Read-only. It gives the EEPROM voltage/temperature/current limits and settles the 8.3 V flag. The Pi was unreachable overnight on 2026-10-23 (no mDNS, 192.168.50.126 timed out): was it off?
3. **8.3 V flag**: if the max-voltage register reads 8.0 V (expected), choose: leave the flag (advisory by default), charge to ~8.2 V, or let us raise the register to 8.5–9.0 V on all 14 servos (one EEPROM write each, bus free). Nothing is written without your yes.
4. **Bench session** (45 min, `mechatronics/bench_procedure.md`): step, sine sweep and no-load speed on one servo with and without 100 g. This is the framework's exit gate for the actuator model.
5. **Strength test to the limit**: the ankle test stopped at 1.16 N·m still rising; one more run to 25° pins the stall torque the 1.86 N·m limit assumes.

### To Software
6. The Pi's config snapshot (walk 9, `walk_recordings/meta_walk9.pkl`) has `velocity_clip: true, max_motor_velocity_rad_s: 5.24` while the upstream script has the clip commented out: which runs on the Pi, and will the motion graph runtime take the 4 rad/s clamp from the package envelope rather than a config default? (Answers your Q6 in part: that snapshot also holds the deployed `joints_offsets`, `joint_limits_deg` and `battery_guard` values.)
   - *Software pipeline, 2026-10-23:* the package is the source of truth for the tools: `envelope.max_vel_rad_s` (4 for walk/v1) is what `battery.py` and the sim_test side of `sim2sim.py` clamp at; the runtime side of sim2sim is the unmodified upstream script (no clamp) and that difference is inside the 0.005 rad residual. The motion-graph runtime should read the clamp from the envelope (theirs to confirm).
7. The "8 Hz" action filter is a 1.19 Hz first-order filter (`alpha = (1/fc)/(1/50+1/fc)` puts 1/fc where an RC constant belongs; `mechatronics/envelope.md` §3, measured by `servo_response_sim.py`). Sim and runtime share it, so nothing breaks, but the package envelope should name the real cutoff, and the new runtime must keep the formula byte-identical. On your Q1: the walk-9 config has no cutoff key; it is a CLI flag of the deploy script, so the launch line on the Pi decides.
   - *Software pipeline, 2026-10-23:* `runtime.lpf_hz` in skill.json is the value handed to `LowPassActionFilter(50, cutoff_frequency=...)`, the runtime's parameter, not the physical cutoff; sim2sim checks both sides agree byte for byte.
8. Temperature logging: the battery guard already hands the bus to pyserial every 20 s for voltage (0x3E). Reading 0x3F (temperature) and 0x41 (status flags) in the same handoff is free; the cool-down rule (`envelope.md` §5) needs it. Please log all three per servo.
9. The sit-down abort should set servo P to 2 after reaching the tuck (cool-down hold). Keep `set_kps` in the HWI.

### To Controls and ML
10. At the 1.86 N·m cap the model servo cannot exceed 3.2 rad/s (damping 0.56), but the real knees reach 3.8–4.5 rad/s while walking (`obs_walk7.pkl`). Until the bench decides, keep TORQUE_RANGE 1.6–2.1 and KP_RANGE 1.1–1.8, and expect clips above ~3 rad/s at the knee to look impossible in sim and fine on the duck.
11. The clip checker's speed limit should be the 4 rad/s target-rate clamp, not the servo's physical ceiling.

### To Animation
12. Knees fold only ~8° past standing (deploy range 1.37 → 1.52 rad); left ankle −1.19…0.51; head yaw ±1.22 (soft limit). Author against `mechatronics/envelope.json` → `joints.*.deploy_range_rad`, not the MJCF ranges.
   - *Testing, 01:05:* so `clip_check.py` passes clips the duck cannot do (knee 1.37 -> 1.57 in the MJCF, 1.52 on the duck). Animation/Software: give the checker `--limits mechatronics/envelope.json` or make the deploy range the default. `tests/test_envelope_and_registry.py` checks every deploy range sits inside its MJCF range with home inside it.

### From Testing
- (Testing, 00:35) `mechatronics/servo_health.py --fake` prints one JSON line; `tests/test_tools.py::test_servo_health_fake_reports_14_servos` pins `fake`, `missing == []`, 14 entries in `temps`/`voltage`/`positions`/`status`, `summary.answered == 14`. Tell Testing if names change.
- (Testing, 01:05) Still needed for `tests/GATES.md` G3: the hook point on the shell for the cord.

## Testing

- **Notes (00:20 to 01:55):** pytest 9.1.1 appeared in `sim/.venv` at 00:20 (no pip in that venv; `uv pip install --python sim/.venv/bin/python pytest` if ever missing). The brief said 11 passive backlash joints; the MJCF has 10 (25 = 1 free + 14 + 10), `tests/test_model_dump.py` asserts 10. `anim/blender_duck.py export_clip` writes the `base` track only when the trunk bone is animated since 00:26; the round-trip tests accept both. `Puppet_walk_test1.json` has one 18 ms step among 94 at 20 ms; `tests/test_clips_data.py` tolerates 5 ms. The recorded gait clips (`s2a_*`) start 0.20 rad from home, above the 0.15 rad tolerance, so they only pass `clip_check` as `--loop`.
- **Final run (01:53):** `./run_tests.sh` -> 86 passed, 0 failed, 2 skipped in 13 s (the two skips: the Blender exporter omits the base track for clips without an animated trunk). 00:35 first run: 34 passed, 6 failed, 15 skipped; the failures in between were all real cross-team mismatches (Software item 7) that the owning teams fixed, or in-progress tests of theirs. Nothing in the suite touches the GPU (`JAX_PLATFORMS=cpu`, `CUDA_VISIBLE_DEVICES=""`).

## For Drew

- **(Testing) Gate numbers.** Review `tests/GATES.md`: 45 deg tilt and 0.10 m height as the fall rule, 5 seeds x 10 episodes, 47 of 50 for an episodic skill, 45 of 50 for locomotion cells and graph edges, 9 of 10 held, 18 of 20 free on mats. Say yes, or change the numbers in that file.
- **(Testing) Walk v1 contract.** It honestly fails backward (-0.016 m/s for -0.15) and side-step right (does not step) in the battery (Software Q5, same with GATES' 0.5 x command floor): release it with those two cells recorded as known failures, or narrow `commands.vx/vy` to what the brain does?
- **(Testing) Termination rule.** Training today stops at tilt > 90 deg; the gates propose 45 deg / 0.10 m for the recipe and the battery alike (Controls). Your call before the recipe is frozen.
- **(Controls) Bow direction.** `test_bow.json` is a head-back "look up" in the MJCF sign convention, not a bow (Controls, first item). Say which way the real bow goes before Animation authors it.
- **(Testing) Loop clips.** Should a loop clip carry `"loop": true` in its JSON so the checker and the trainer pick it up without a flag?
- **(Mechatronics)** Items 1 to 5 under Mechatronics, "To the user": weigh the duck, authorise the servo read, the 8.3 V flag, the bench session, the strength test.
- **(Testing)** `./run_tests.sh` any time; it writes `tests/last_run.md` and never touches the GPU.
