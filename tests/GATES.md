# Gates G1 to G5 for the Open Duck Mini: PROPOSED thresholds

**Status: PROPOSED by the Testing team, 2026-10-23, for Drew's review.** Nothing here is agreed
until it is. The framework (`skill-library-framework.md` 4.3) names the five gates; this file
puts numbers on them for this robot. Where another team already wrote a default
(`tools/skillkit.py DEFAULT_THRESHOLDS`) it is quoted so the two can be reconciled into one.

## 0. Numbers measured tonight (what the thresholds are calibrated against)

`sim/sim_test.py Sim`, `BEST_WALK_ONNX_2.onnx`, `scene_flat_terrain_backlash.xml`, torque clamp 1.86 N.m,
MuJoCo 3.15, one seed, no randomisation (`tests/test_sim_smoke.py`, scratch `heights.py`):

| Quantity | Value |
|---|---|
| Standing base height (trunk origin) | 0.153 to 0.165 m (home keyframe 0.15; feet-on-floor kinematic 0.167; legs straight 0.200) |
| Base height when tilt crosses 45 deg in a fall | 0.128 m forward push, 0.110 m backward, 0.177 m sideways |
| Standing 3 s | max tilt 5.9 deg, drift < 2 cm |
| Walking at command 0.15 m/s for 4 s | 0.36 m (0.09 m/s, 60 % of command), max tilt 8.7 deg |
| Two identical rollouts | bit-identical (qpos and targets) |
| MJCF total mass | 2.107 kg. Training `MASS_SCALE=0.72,1.0` spans 1.52 to 2.11 kg; the real duck is quoted at 1.5 kg (Mechatronics to weigh) |
| Floor collision | only the feet collide with the floor in this scene; a fallen trunk passes through (base z = -0.19 m). Use `scene_flat_terrain_backlash_collision.xml` for anything that starts or ends on the floor (recovery) |
| Training termination today (`joystick.py _get_termination`) | gravity z in the trunk frame < 0, i.e. tilt > 90 deg, or NaN |
| Recipe termination (framework 3.3, plan 3.3) | 0.25 m height scaled to the duck (about 0.12 m) or 0.8 rad orientation error |
| Recorded gait clips (`s2a_*`) vs home | 0.20 rad at the start frame, so they need `--loop` in `clip_check.py` (tolerance 0.15) |

## 1. Definitions every gate shares

**Fall (episode failure).** Any one of, at any 50 Hz control step:

| Rule | Threshold | Why this number |
|---|---|---|
| Tilt | angle between the trunk z axis and world up > **45 deg** (`Sim.base_pose()[2]`) | 45 deg = 0.785 rad, the recipe's 0.8 rad orientation termination. The duck cannot recover past it without a get-up policy. |
| Height | trunk origin < **0.10 m** for 5 consecutive steps (0.1 s) | 60 % of standing height, the same fraction the recipe's 0.25 m is of a BeyondMimic humanoid pelvis. Catches a leg collapse that keeps the trunk level. Falls cross 45 deg at 0.11 to 0.18 m, so tilt fires first in a tip-over. `skillkit` default is 0.09 m: pick one. |
| NaN | any NaN in qpos or qvel | the training rule |

**Identical to training.** G1 must terminate on exactly the function the frozen recipe uses, imported
from the recipe module, never re-typed in the battery. Today that function is "tilt > 90 deg", which is
looser than the 45 deg above; the recommendation is that the tracking recipe adopt the 45 deg / 0.10 m
rule and the battery import it. Until then the battery must report which rule it used.

**Seeds and episodes.** **5 seeds x 10 episodes = 50 per cell.** A seed draws the domain-randomisation
set the recipe trained with (friction, mass scale, motor strength, latency, sensor noise, same ranges)
and the perturbation schedule; the 10 episodes inside a seed vary the start (gait phase, entry pose,
command) deterministically. Everything is seeded so a failing episode can be replayed from
`(skill, version, seed, episode)`.

**Ratios.** Always "k of n", with the seeds and the per-seed minimum: `47 of 50 (seeds 0-4, worst seed 8 of 10)`.
Never a percentage alone, never "works".

**Start state.** Home keyframe, 1 s at zero command, then the test. For a graph edge the start is the
neighbour's end state (framework rule 4), sampled at 10 gait phases.

## 2. G1 Sim battery (per skill, per edge; `tools/battery.py`)

### Locomotion skill (walk v1 and later)

| Cell | Command (vx, vy, wz) | Length | Success per episode |
|---|---|---|---|
| stand | 0, 0, 0 | 10 s | no fall; drift < 0.10 m; fewer than 4 foot lifts |
| forward | 0.15, 0, 0 | 10 s | no fall; mean forward speed >= **0.5 x command** (0.075 m/s); lateral drift <= 0.05 m/s; yaw drift <= 15 deg/s |
| backward | -0.15, 0, 0 | 10 s | same, along -x |
| side-step left / right | 0, +-0.2, 0 | 10 s | no fall; mean lateral speed >= 0.5 x command, forward drift <= 0.05 m/s |
| turn left / right | 0, 0, +-1.0 | 10 s | no fall; mean yaw rate >= 0.5 x command; position drift <= 0.3 m |
| push | 0, 0, 0 | 3 s | 8 directions x 15 N for 0.1 s on the trunk (as `eval_policy.push_test`); no fall |
| head overlay | 0, 0, 0 + head swings | 8 s | no fall, fewer than 4 foot lifts (head is a command input) |

Pass: **every cell >= 45 of 50, stand and push >= 48 of 50**. The 0.5 x command speed floor is below the
60 % measured tonight on purpose: it fails a duck that shuffles in place, not one that is slow.

### Episodic skill (bow, wiggle, look_around, hop)

50 episodes entered from idle (locomotion at zero command) at 10 gait phases x 5 seeds. Success:

- no fall during the clip and for **1.0 s** after exit;
- tracking: per-joint RMS error of actuated joints against the reference over the clip <= **0.15 rad**
  (the same number as `clip_check --home-tol` and `skillkit joint_err_rad`), and the fraction of frames
  with max joint error < 0.3 rad >= 0.8 (`skillkit min_tracking_fraction` default 0.2 is too loose for release; 0.2 is fine for the stage 3 preview gate);
- exit pose within 0.15 rad of the declared exit pose at the exit window;
- finishes within clip length + 0.5 s (timeout).

Pass: **>= 47 of 50**. Stage 3 preview gate (nominal behaviour appears): >= 25 of 50 with tracking fraction >= 0.2.

### Edges (walk -> bow -> walk, etc.)

Same as the episodic cell but the start is the neighbour's end state and the success window includes the
hand-off back (1.0 s into the next skill without a fall). Pass **>= 45 of 50** per edge. Every edge in the
motion graph config has a cell; an untested edge does not exist (rule 5).

### Recovery (get-up, sit abort)

50 starts from fallen poses obtained by dropping the duck from the end states of every neighbour (fronts,
backs, both sides), on the collision scene. Success: tilt < 15 deg and height > 0.14 m within 5 s and held for 2 s.
Pass **>= 45 of 50**. Sit abort: 50 starts from every skill mid-episode; success: servos relaxed with
the trunk height falling below 0.10 m within 1 s and no joint beyond its limit.

## 3. G2 Sim-to-sim (`tools/sim2sim.py`)

The same ONNX file through the deploy code (`graph_runtime` policy step, action scale, low-pass, slew limit,
head mode) against `sim_test.Sim` (MuJoCo CPU), compared with the training-side evaluator (MJX).
"Behaviour matches" means all four of:

| Check | Threshold |
|---|---|
| Observation layout | the deploy code builds the same vector size and block order as the package `obs_spec` (101 for walk); no NaN |
| Open loop | replay 200 recorded training-side observations through the deploy policy step: max abs action difference <= **1e-4** (onnxruntime CPU is deterministic; two runs here were bit-identical) |
| Closed loop, early | RMS joint-target difference over the first 2 s (100 steps) from the same start <= **0.02 rad** (`skillkit sim2sim_rms_rad`); for an episodic clip, over the whole clip <= 0.05 rad |
| Closed loop, late | after 2 s trajectories diverge chaotically, so compare the battery instead: the G1 cells re-run in the second simulator with the same seeds drop by **<= 3 of 50** and stay above the G1 pass line |

Record MuJoCo version, XML sha256 and `robot_model_hash` in `report/sim2sim.json`.

`tools/sim2sim.py` (Software, tonight) implements the lockstep comparison with the real upstream runtime code and
argues the 0.02 rad line the same way from the hardware: the servo reports 0.0015 rad ticks and the model carries
+-0.0087 rad of backlash, so differences under about 0.01 rad never reach the output shaft, while a one-step gait-clock
lag or a missing filter shifts targets by 0.03 to 0.1 rad RMS. Measured on the walk package: lockstep RMS 0.008 rad, match.

Software's first sim2sim run also found the deploy loop's gait clock one step behind the trainer's (obs built, then
`imitation_i` advanced): 0.020 rad RMS, exactly on the line, 0.005 rad with the clocks aligned. That is the kind of
pipeline difference the threshold is meant to catch; the fix belongs in the motion-graph runtime, not in the threshold.

**Filters are part of the package.** The deployed runtime (`v2_rl_walk_mujoco.py`) applies no action low-pass unless
`--cutoff_frequency` is passed and its 5.24 rad/s slew clamp is commented out; the curriculum brains train with an 8 Hz
low-pass and 4 rad/s. The same ONNX walks at 0.096 m/s unfiltered and 0.001 m/s under the 8 Hz filter. G1 and G2 are only
valid when the package's `runtime` block states the filters the brain was trained with, and the deploy code applies exactly those.

## 4. G3 Gantry / held (1.5 kg duck)

Rig: a cord from an overhead hook to the top of the shell (Mechatronics to name the hook point) with
**2 to 3 cm of slack**, so it carries nothing while the duck stands and catches it before the shell
touches the mat; for the very first episode of a new skill, a hand under the trunk instead. Mats under it
anyway. Pre-flight, every session:

- `mechatronics/servo_health.py` answers for all 14 servos, no status flags, every temperature < **50 C** (Mechatronics, 2026-10-23: warn 50, cool-down 60, hard stop 65, the servo unloads itself at 70), bus voltage >= **7.4 V**;
- the puppet agent is stopped (the runtime owns the servo port);
- the package passed G1 and G2 (`status` at least `sim2sim_passed`).

Protocol: **10 episodes** per cell (locomotion: forward 5 s, turn 5 s, stand 10 s; episodic: 10 triggers from idle).
Every joint logged at 50 Hz: target, position, load, temperature, voltage, plus IMU, into `experiments/` as one run.
Success per episode: the cord never takes load (observer, and no sudden unloading in the servo loads), IMU tilt < 45 deg,
no servo status flag, no temperature above **60 C**, no load above 80 % for more than 0.5 s.
Pass: **>= 9 of 10**. Stop rules: any servo >= 60 C means a cool-down until it is back under 50 C, 65 C is a hard stop
for the session; any fall is reproduced in sim (same state, same perturbation) and added to `tests/` before the next attempt.

## 5. G4 Free on mats

Mats: **>= 12 mm** EVA foam, at least 1 x 1 m, nothing hard within 0.5 m, shell on. Pre-flight as G3.
**20 episodes** per cell (locomotion: forward 2 m, a full 360 turn, stand 30 s; episodic: 20 triggers from idle;
edges: 20 hand-offs). Success as G3 without the cord.
Pass: **>= 18 of 20** for an episodic skill or an edge; **>= 17 of 20** for locomotion v1 (stated as such in the report).
Session limits: no servo reaches 60 C (cool-down rule as G3), bus voltage never below 7.0 V under load.
Every fall: video, the servo log, and a sim reproduction committed as a test case.
Report line: `18 of 20, 2026-xx-xx, 7.9 V start / 7.5 V end, max 48 C (right_knee)`.

## 6. G5 Library release

- `tools/battery.py` re-run for **every** package in `skills/*/v*/` with status >= `battery_passed`, on the release machine; any ratio below its gate blocks the release.
- Every edge in the motion graph config has a G1 edge cell that passed.
- `./run_tests.sh` reports 0 failed (skips listed in `tests/last_run.md` and accepted).
- `experiments/log.md` has an entry for the release: hypothesis, command, evidence (the battery table), decision.
- The release is a tag: date, git hash, `robot_model_hash`, `actuator_model_version`, `recipe_version`; the folder of packages plus the graph config.
- Any failure seen on hardware since the last release has a sim test case in `tests/` (or a filed reason it cannot have one).

## 7. Report format (`report/battery.json`, proposed)

```json
{"skill": "bow", "version": 1, "date": "2026-10-23T02:10:00-04:00",
 "robot_model_hash": "...", "recipe_version": "v0", "simulator": {"name": "mujoco", "version": "3.15.0", "xml_sha256": "..."},
 "termination": {"tilt_deg": 45, "min_height_m": 0.10, "height_steps": 5, "source": "recipe.termination"},
 "seeds": [0, 1, 2, 3, 4], "episodes_per_seed": 10,
 "cells": {"idle_entry": {"passed": 47, "total": 50, "per_seed": [10, 9, 10, 9, 9],
                          "fail_reasons": {"fall": 2, "tracking": 1, "timeout": 0}}},
 "pass": true}
```

## 8. Decisions needed (see `experiments/questions.md`, "For Drew")

1. 45 deg tilt as the fall rule for the recipe and the battery (training today uses 90 deg).
2. 0.10 m or 0.09 m as the height rule.
3. 5 x 10 episodes; 47 of 50 episodic / 45 of 50 locomotion and edges.
4. The gantry hook point on the shell, and whether the first hardware episodes are hand-held instead.
5. Weigh the duck: the MJCF says 2.1 kg, the brief says 1.5 kg.
