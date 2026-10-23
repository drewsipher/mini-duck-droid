# Skill Library: project plan for the Open Duck Mini

Companion to `skill-library-framework.md`. Written 2026-10-22. The framework says what the shape is; this file says how we get there on this robot, with this machine (one RTX 4060, 8 GB), and with agent teams standing in for the four departments.

Flow map of this plan: http://hermes-agent.local/apps/mermaid-flow-studio/?load=diagrams/skill-library-plan.mmd (source `~/src/mermaid/site/diagrams/skill-library-plan.mmd`).

---

## 0. The one-paragraph version

We stop trying to make one reward-shaped policy do everything. We build a pipeline where a behaviour is a clip, every clip is trained the same way, each trained skill ships as a package, and a motion graph runtime on the Pi blends and hands off between packages. We prove the whole loop first on **one small episodic skill** (a bow or a wiggle authored in the Duck Puppet), not on walking, because that is the cheapest way to find every broken link. Walking becomes just the locomotion skill in the library. Then we add skills on a weekly cycle.

---

## 1. Where we already are

Mapped onto the framework's four layers and six stages. "Exists" means working today; "partial" means usable but not to spec; "build" means new; "find" means evaluate an open tool before building.

| Framework piece | Status | What we have |
|---|---|---|
| L4 robot runtime | exists | `Open_Duck_Mini_Runtime` on the Pi: 50 Hz policy loop, PD servos, IMU, battery guard, stability governor, web UI :8080, gamepad |
| L3 locomotion skill | exists (v1) | `BEST_WALK_ONNX_2.onnx` deployed and demoable. Scaffold-curriculum replacement in progress (`sim/curriculum.sh`, round 8 = stage p2) |
| L3 recovery skill | partial | Upstream stand-up addon in `train_pg/.../standup.py`, `standup_*.npy` references. Never trained or packaged by us |
| L3 episodic skills | none | — |
| L3 overlay layer | partial | Head (4 DoF) is already a command input to the walk policy, not a learned behaviour. Antennas, eyes, sound are classical |
| L2 animation engine, state machine | none | Runtime runs exactly one ONNX. No transitions, no blending, no clip layer |
| L1 operator layer | partial | Web UI and Xbox controller drive velocity commands only |
| Stage 1 author | exists | **Duck Puppet** (`tracker/`): pose, hold, record the real duck; keyframe timeline, mirror, loop, trim, append (`anim.js`); `record_policy_clip.py` records a brain's gait as a clip. Blender is reachable through MCP for later |
| Stage 2 retarget and check | none | No clip checker (ground penetration, self-collision, joint limits, servo speed) |
| Stage 3/4 training | partial | `sim/train_pg` (mujoco_playground, brax PPO on MJX, ~1 h per 100M steps). `ClipReferenceMotion` and `HybridReferenceMotion` track joint angles and contacts from clips. No root-pose tracking, no adaptive sampling, no frozen reward template |
| Stage 5 sim-to-sim | partial | `sim_test.Sim` mirrors the runtime (same filters, torque, obs). Not run through the real deploy code |
| Stage 6 hardware | partial | Hand-held and free tests, no gantry, no ratio reporting |
| Bench actuator ID | exists (upstream) | MJCF carries BAM-fitted STS3215 values (damping 0.56, friction 0.068, armature 0.027, 3.23 N·m peak). Our tests found real walking torque ~1.86 N·m. Servos report a voltage flag at 8.3 V |
| Sim battery and scoring | partial | `eval_policy.py` (single seed), `held_rollout.py`, `render_video.py`, `make_curriculum_plots.py`, tfevents |
| Experiment log | partial | Memory notes and `sim/report*/`. Not append-only, not per run |
| Skill package format | none | — |
| Regression suite | none | — |

---

## 2. Decisions for this robot

1. **Training engine stays brax/MJX for phase 1.** It works, it is tuned to this duck, and it fits the 4060. We add the BeyondMimic tracking template to it rather than porting everything. One parallel spike evaluates **mjlab** (MuJoCo Warp, ships the tracking task): if it trains a duck clip on the 4060 at usable speed we migrate in phase 2, otherwise we stay.
2. **First skill through the pipeline is episodic, not locomotion.** A 2 to 3 second clip that starts and ends in the standing pose. It exercises author, check, train, battery, sim-to-sim, hardware and packaging in a week. Walking is bigger and already has its own thread.
3. **Locomotion v1 is the current best brain, packaged as-is.** The curriculum ladder continues as locomotion v2 under the frozen recipe. The library must never wait on walking being perfect.
4. **Recovery = stand-up + protective sit.** The duck is small and padded by its shell. "Abort" is sit down and relax servos. Get-up is the upstream stand-up skill retrained under our recipe.
5. **Stand is a fallback, not a waypoint.** Episodic clips enter from locomotion at a declared gait phase and exit back into it. Idle is locomotion at zero command.
6. **Head, antennas, eyes, sound are show functions.** They stay out of training. The animation engine schedules them alongside skills.
7. **One MJCF is the source of truth.** `open_duck_mini_v2_backlash*.xml` in `train_pg`. Any mass or actuator change bumps the robot model hash, and every package records the hash it was trained against.
8. **Reports stay visual.** Every milestone ends with videos and curves in the artifact, as before.

---

## 3. Workstreams by team

Each team is a future agent (or a group of agents) with this brief. The hand-off format between all of them is the skill package (section 4).

### 3.1 Mechatronics
Owns the robot model, actuator truth and the safety envelope.
- **Model audit:** weigh the real duck and the major parts, override MJCF masses, record a `robot_model_hash`.
- **Actuator check:** confirm the BAM values against our bench data (`strength_test.py`, `torque_test.py`, `kp_test.py`). Decide the deployable torque (1.86 N·m today) and write it into the package envelope. Resolve the 8.3 V voltage flag.
- **Thermal:** read STS3215 temperature through the agent and the runtime; log it; set a cool-down rule.
- **Test rig:** a string/gantry hook point and a mat area, so gate G3 and G4 are repeatable.
- Exit gate: sim step and chirp responses within tolerance of bench data; envelope written down.

### 3.2 Animation and character
Owns clips and show functions.
- **Author** clips in the Duck Puppet (keyframes or record from the real duck or from a brain). First clips: `bow`, `wiggle`, `look_around`, then `hop`.
- **Tag** every clip with entry window, exit window, gait phase, and the pose it must start and end in.
- **Stage 2 checks** (tool built by Software, used here): no ground penetration, no self-collision, joints inside limits, servo speed inside the envelope, start and end pose match the neighbour.
- Show functions: head gestures, antenna moves, eye and sound cues scheduled by the animation engine.

### 3.3 Controls and ML
Owns the training recipe and skill training.
- **Tracking task** in `train_pg`: body position and orientation tracking against a reference root trajectory, joint tracking, linear and angular velocity, action rate, joint limits, self-contact, with the BeyondMimic weights and σ values from the framework §3.3. Termination 0.25 m scaled to the duck's height (about 0.12 m) and 0.8 rad orientation error.
- **Adaptive sampling:** episode starts in 1 s bins weighted by recent failure.
- **Fixed randomisation set:** friction, mass, motor strength, latency, sensor noise. Same for every skill.
- **Previous two actions in the observation** (transition rule 1) and **blended-start training** (rule 3) and **neighbour end-state starts** (rule 4).
- **Freeze** the recipe after the first two skills converge. A recipe change is a versioned event.
- **mjlab spike:** install, run the tracking example, time a duck clip on the 4060, write the verdict.
- Locomotion v2 continues under the frozen recipe once it exists; until then the curriculum thread runs as today.

### 3.4 Software and runtime
Owns the package format, the motion graph runtime, the tools and the regression suite.
- **Skill package format** (`skill.json` + clip + ONNX + report + version record). Loader refuses a package with a missing part.
- **Clip checker** `clip_check.py`: plays a clip on the MJCF kinematically, reports penetration, self-collision, limit and speed breaches.
- **Motion graph runtime** (new package, runs on the Pi and in the simulator from the same code): loads packages, state machine with explicit edges, reference-level blending with ramps, previous-action continuity, timeout and abort, operator inputs (web UI, gamepad, later show control), show-function scheduler. Replaces the single-policy loop in `v2_rl_walk_mujoco.py` while reusing its hardware layer.
- **Sim battery** `battery.py`: multi-seed, per-skill and per-edge, same thresholds as training, outputs ratios. Grows `eval_policy.py`.
- **Sim-to-sim** `sim2sim.py`: runs the motion graph runtime's own code against `sim_test.Sim`.
- **Experiment log:** append-only `experiments/log.md`, one entry per run: hypothesis, command, evidence, decision.
- **Regression:** nightly script re-runs every package's battery and posts a table.

### 3.5 Testing and release (shared)
- Gates G1 to G5 from the framework §4.3. Ratios, never "works".
- Every hardware failure becomes a sim test case.
- A release is a tagged folder of packages plus the motion graph config.

---

## 4. The skill package

```
skills/<name>/v<N>/
  skill.json        name, type (locomotion|overlay|episodic|recovery), version,
                    commands and trained ranges, entry/exit windows, neighbours,
                    timeout, safety envelope (torque, speed, thermal, fall),
                    robot_model_hash, actuator_model_version, recipe_version, seed
  clip.json         reference motion at 50 Hz (Duck Puppet format)
  policy.onnx       50 Hz, fixed obs/action spec, spec hash in skill.json
  report/           battery.json (ratios), sim2sim.json, hardware.json, videos, curves
```
The motion graph config lists the edges. The loader checks every neighbour named in an edge exists and that both packages passed their battery.

---

## 5. Milestones and gates

| # | Milestone | Done when | Weeks |
|---|---|---|---|
| M0 | Foundation | This plan, the flow map, package spec v0, experiment log started, teams briefed | 1 |
| M1 | One skill end to end | A `bow` clip authored, checked, trained on the tracking task, passes battery and sim-to-sim, plays held on the real duck, lands as `skills/bow/v1` | 2–3 |
| M2 | Locomotion packaged | `BEST_WALK_ONNX_2` is `skills/walk/v1` with envelope and battery; curriculum thread continues as v2 | 3 |
| M3 | Motion graph runtime v1 | On the Pi: idle/walk, `bow` entered from walk and back, sit-down abort, operator from web UI and gamepad; same code passes sim-to-sim | 4–6 |
| M4 | Recovery | Stand-up retrained under the recipe, trained on neighbour end states, hardware ratio stated | 6–8 |
| M5 | Skills at pace | Weekly cycle running; three more episodic skills; head/antenna overlay layer; nightly regression | 8–12 |
| M6 | Consolidation (optional) | mjlab migration or distillation decision, based on data | later |

Gate rule: a failed gate sends the skill back one stage. Most failures are fixed in the clip.

---

## 6. The first two weeks, concretely

Week 1 (M0 and the start of M1)
1. Software: `skills/` folder, `skill.json` schema, `clip_check.py`, `experiments/log.md`.
2. Animation: author `bow` in the Duck Puppet (start and end in the standing pose, ~2.5 s), save to `tracker/data/clips/bow.json`, run the checker.
3. Controls: tracking task in `train_pg` (new env class next to `joystick.py`), 30-minute preview run on `bow`. Start the mjlab spike in parallel on CPU-side install only (GPU stays with the preview run).
4. Mechatronics: weigh the duck, write the envelope, read servo temperatures through the agent.

Week 2 (M1)
5. Controls: full run on `bow` with randomisation and adaptive sampling.
6. Software: `battery.py` multi-seed on `bow`; `sim2sim.py` playing the ONNX through the runtime's own policy step.
7. All: hardware, held by hand, then on the mat; ratio recorded; `skills/bow/v1` written; report with videos in the artifact.

---

## 7. Agent team briefs (for spinning up teams later)

Each brief is the whole context an agent gets, so it names the files.

- **Mechatronics agent:** read `skill-library-framework.md` §3.2 and §5.5, this plan §3.1. Robot model in `sim/train_pg/playground/open_duck_mini_v2/xmls/`, bench scripts in the project root (`strength_test.py`, `torque_test.py`, `kp_test.py`), `strength_data/`. Pi access `bdxv2@bdxv2.local`. Deliver: mass overrides, envelope JSON, temperature logging, a one-page sim-vs-bench comparison.
- **Controls and ML agent:** framework §3.3, plan §3.3. Code in `sim/train_pg/playground/` (`common/clip_reference_motion.py`, `open_duck_mini_v2/joystick.py`, `curriculum.sh`). GPU must be free of ComfyUI. Deliver: tracking env, recipe file, preview and full runs with tfevents, ONNX, per-term reward curves.
- **Software agent:** framework §2.4, §2.5, plan §3.4 and §4. Runtime source `sim/upstream_runtime/` and on the Pi `~/Open_Duck_Mini_Runtime/`. Simulator mirror `sim/sim_test.py`. Deliver: package loader, clip checker, motion graph runtime, battery, sim-to-sim, nightly regression.
- **Animation agent:** framework §2.2, §3.1, plan §3.2. Duck Puppet `tracker/` (`anim.js` for the clip model, `server.py` for the API, clips in `tracker/data/clips/`). Deliver: tagged clips that pass the checker, show-function cue sheets.
- **Testing agent:** framework §4.3, plan §3.5. Deliver: gate reports as ratios, new sim test cases from every hardware failure, the regression table.

Teams talk through files: packages in `skills/`, log entries in `experiments/log.md`, open questions in `experiments/questions.md`.

---

## 8. Risks specific to us

| Risk | Containment |
|---|---|
| 8 GB GPU is too small for mjlab or large tracking batches | brax/MJX stays the default; env count tuned per task; spike reports speed before any migration |
| Tiny servos cannot track a dynamic clip | Clip checker enforces the speed envelope at stage 2; first clips are slow; hop comes after bow |
| Motion graph runtime destabilises the walk we already have | v1 keeps the walk policy's loop untouched inside a state; sim-to-sim gate before any Pi deploy |
| Two programs on one servo port (puppet agent vs runtime) | Runtime owns the port when running; the puppet tool's agent is stopped by the start script and restarted after |
| Reward template drifts back into hand-shaping | Recipe file is versioned; a change triggers full regression; the experiment log records why |
| Thermal and voltage drift on long demos | Temperature logged from week 1; cool-down rule in the runtime; voltage flag resolved by Mechatronics |
