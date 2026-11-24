# Duck restart: a layered animation system (draft 1, 2026-11-05)

Status: draft for Drew. Nothing has been deleted yet; the teardown in §5 waits for a go.
Research behind it: `docs/research/reports/Game animation systems for the duck.md` (notes in
`docs/research/research_notes/`). Repo inventory for the teardown: `docs/teardown-inventory.md`.

## 1. Why restart

The clip-tracking line (one network per clip, output-blended by a gate) stood quietly on the
real duck, but its recovery step made tilt worse on hardware, and every new behaviour cost a
full training run plus a gate. Drew's call on 2026-11-05: rebuild the controller from the inside
out as layers that can each be tested alone, modelled on game animation pipelines (Unreal, Unity,
RAGE/Euphoria), and get the simulation to match the real duck before training or animating.

What the research supports and where it pushes back:

- The layered shape is right. Unreal, Unity and RAGE all run one per-frame pipeline: measure
  intent and state, select an authored clip, blend every switch with one inertializer, shape and
  ground the clip to the measured world, then a physics layer with 0-to-1 dials. Disney's BD-X
  droid (same 14 joints as the duck) runs the same flow with the grounding stage replaced by a
  balance policy.
- Life and realism are separable and budgeted separately. Life is authored clips, idle breaks,
  breathing, attention and gaze, timing knobs, anticipation and reaction. Realism is
  inertialized transitions, warping, foot placement, balance and recovery.
- The replaceable unit is a clip plus a command vector, not a network. The one hardware
  benchmark on switching between separately trained policies got 2 % transition success versus
  100 % when transitions were trained; Disney shipped one policy per clip and lists "one policy
  for all skills" as its own next step. Transitions are where every surveyed robot fell.
- The legs are owned by one foundation. Disney ignores the animator's leg joints at runtime;
  authored leg style enters only through training. The head is the free expression channel the
  duck runtime already layers additively.
- The actuator is the sim-to-real gap, and the STS3215 has an undocumented acceleration cap
  (RAM register 85, factory 50, 6 to 28 rad/s²) that the BAM actuator model was fitted without.
  It leaves quiet standing alone and slows exactly the fast steps that misbehaved on the floor.
  Our Pi runtime never writes registers 41 or 85 at boot; the puppet agent clears 41 only; the
  one-time servo setup wrote 85 = 0 but the register is probably RAM and resets at power-up.

## 2. Target architecture

Five layers. Each one has a fixed interface, a test that runs without the layers above it, and a
hardware precedent. "Owns" means it is the only layer that writes those outputs.

| Layer | Owns | Game-engine analogue | Duck implementation | Tested alone by |
|---|---|---|---|---|
| L0 Servo | joint targets to the bus; per-joint P gain; acceleration registers; setpoint clip; 4 rad/s rate clamp; low-pass; safety envelope (torque, temperature, voltage); same API in sim and on the Pi | Unity ArticulationBody drive, Unreal Physical Animation strength | one Python module with two backends (MuJoCo, rustypot on the Pi); register init that clears 41 and 85 and reads them back | step and chirp tests: hardware versus sim per-joint error |
| L1 Foundation | all ten leg joints; keeps the trunk up; turns a body request (height, pitch, roll, velocity, turn rate, phase) into leg targets | Final IK Grounder, Disney's balance planner, NimbRo's IMU offsets | v0 posture hold plus IMU-driven offsets (fused pitch and roll to hip, ankle and CoM shift); v1 capture-step reflex; v2 one learned tracker with the same interface | passive push limits, then calibrated push battery: falls per 100 pushes |
| L2 Controls | one circumstance each: look, bow, breathe, idle breaks, walk forward, sidestep, turn left, turn right, stunts; each is a clip (Blender, 14-joint rig) plus a command vector plus metadata (entry and exit phase, allowed blend partners, speed and amplitude knobs) | clip library, blend space samples, montages | head and neck clips go to L0 directly (additive, low gain); leg clips are references the foundation tracks; the walk family is a blend space over forward speed, side speed and turn rate | clip checker (limits, velocity, acceleration); tracking error per control on hardware |
| L3 Governor | which control runs; switches only in allowed phase windows; one inertializer for every switch, in command space, velocity-clamped; recovery trigger from capture point or fused tilt plus rate; emergency stop to stand | animation state machine plus Inertialization node; PuppetMaster's supervisor; Choreographer track ownership | a small state machine with a transition table; ramps 0.1 s head, 0.35 s body; tracks: legs, head | transition success rate per (from, to) pair |
| L4 Life | attention and gaze, keep-alive idle, anticipation and reaction clips, timing and amplitude knobs | look-at, additive idle, play rate | emits head offsets and clip triggers to L3; never touches a leg joint | Drew's eye, and a jitter metric |

Interfaces (the whole point: a component reads and writes only these, so it can be swapped):

- Pose request: 14 joint targets with per-track weights (legs, head).
- Body request: trunk height, pitch, roll, forward and side velocity, turn rate, gait phase.
- Balance state: fused tilt and rate, capture point, support phase, foot contact if we add switches.
- Transition rule: from, to, phase window, ramp time, clamp.
- Control package: clip, command vector, metadata, checker report.

How this differs from the first sketch: controls are data; one foundation owns the legs; the
governor blends commands and references, never networks; recovery is an always-on classical
layer in L1 and L3 plus whatever the foundation learns from pushes in training, not a separate
policy cross-faded in. Changing a head clip or a timing knob never retrains anything. Changing
a leg clip means re-running the one standard training job with the new clip in the set
(Disney previews that in 30 minutes); the classical v0 and v1 foundations are the fallback that
stands when everything above is broken.

## 3. Sim-to-real first

Ordered by sim fidelity gained per hour of Drew's time. Pass criteria come from the BAM paper
(10 to 20 mrad trajectory error) and ToddlerBot (1.3° tracking error) where marked; the rest are
estimates. Full table with sources in the report, §5.

| Step | Needs | Measures | Pass |
|---|---|---|---|
| 0 Register dump (15 min, duck on) | the bus | P/I/D, goal acceleration 41, max acceleration 85, torque limits 16 and 48, protection 34, overload 36, return delay 7, at boot and after the runtime's init | decide: clear 85 after 41 at every boot (BAM model then applies), or model the acceleration limiter in sim |
| 1 Weigh (10 min) | kitchen scale | total with battery, battery pack, head assembly, one leg if apart (`mechatronics/weigh_me.md`, `masses.json` is still all null) | model total within 2 % of measured; model says 2.107 kg, Drew's estimate 1.5 to 1.8 |
| 2 Centre of mass (30 min) | ruler edge, string, phone | balance on an edge in two planes; hang from two points for height | centre the CoM randomisation and shrink it from ±5 cm to ±1 to 2 cm |
| 3 Hanging-leg actuator tests (1 h) | duck on its stand, pack voltage logged | at P = 30 and 8: 10°, 30°, 60° steps, 1 Hz sine ±20°, 0.2 to 3 Hz chirp ±15°, at full and at about 7.0 V; goal, position, loop time at 50 Hz | replay the same targets in sim: per-joint error ≤ 10 mrad on sines, ≤ 20 mrad on steps; if hardware peak acceleration ≤ 30 rad/s² the register 85 cap is active |
| 4 Static stiffness and stall (1 h) | scale on a lever arm | torque versus commanded deflection at 2°, 5°, 10°, 20°; 20 s hold for stall and protection fold-back | slope is kp (our training model uses 17.11; measured 1.48× that so far); band width is friction plus backlash (about 0.9° expected); ceiling versus the sim's ±3.23 N·m and the catalog 1.86 |
| 5 Loop latency (1 h) | Pi timestamps, an LED, phone at 240 fps | 50 Hz period histogram over 10 min; command-to-motion onset on one joint; IMU-path delay from a flick of the hanging trunk | sim fixed delay = median, jitter ±1 step kept |
| 6 IMU (45 min) | level surface, wedge of known angle | 60 s static log for bias and noise; mounting offset from mean gravity; a 1 °/s slow-tilt test | if the BNO055 fusion freezes under slow tilt, use raw gyro plus accelerometer with our own filter |
| 7 Foot friction and compliance (30 min) | tilt board, phone inclinometer | static friction on the real floor, clean and dusty, 5 repeats; a 2 to 3 cm drop filmed at 240 fps | narrow the 0.5 to 1.0 friction range; tune contact softness to bounce count and settle time |
| 8 Open-loop replay gate (2 h) | all logs above | replay hanging logs, quiet stand, a 0.2 Hz squat and the failing recovery step from `hardware/logs/2026-10-28` | per-joint error ≤ 10/20 mrad; sim reproduces the hardware 2.1° mean / 7.9° max step error within 20 %; IMU trace within one 20 ms step and 15 % amplitude. If not, fit acceleration cap, delay, kp scale, friction, armature with CMA-ES on the hanging logs |
| 9 BAM pendulum run (optional, 2 h) | a spare servo, arm, weights | Rhoban's pipeline on our 7.4 V units at battery voltage | fitted model versus the Playground's kp and force range |
| 10 Narrow the randomisation | GPU | kp ±15 % around the fit, masses ±10 % around measured, friction around the measured spread; keep wide ranges only for battery, torque, latency, temperature | re-run the recovery step in the corrected sim |
| 11 Standing push battery | known mass on a string, or the scale | calibrated pushes, 3 magnitudes, 2 directions, hardware and sim; tilt peak and settle time | the real counterpart of `sim/fusion/runs/passive_push`; last gate before any new balance work |

Steps 0 and 8 are the difference between a sim that flatters and one that predicts. Steps 0, 1,
2 and 6 need Drew's hands; 3, 4, 5, 7, 8 need the duck on with me driving from hermes.

## 4. Phases and gates

Every phase: sim first with plots or video, then hardware on the floor on Drew's "go". A phase
ends when its gate passes on hardware, not before.

| Phase | Builds | Gate |
|---|---|---|
| 0 Freeze and teardown | tag the repo, back up the real-duck data, remove the nightly cron, move the old method to a branch, repoint the kept scripts, trim the dashboard, rewrite README and NOTICE (§5) | repo builds, puppet and Blender bridge still work, nothing physical lost |
| 1 Real-to-sim | §3 steps 0 to 8 (9 to 11 as they fit), `mechatronics/` becomes the home of every measured number, a scoreboard page of hardware-versus-sim tests that is re-run after every model change | step 8 passes |
| 2 L0 servo layer | one module, two backends, register init, setpoint clip, rate clamp, low-pass, envelope; the puppet agent and the runtime both use it | hardware and sim step and chirp agree within the step 3 tolerances |
| 3 L1 foundation v0 and v1 | posture hold; IMU-offset PD; then capture-step reflex; passive push limits measured first (memory rule), then the calibrated push battery | fewer falls per 100 pushes than the passive duck at the same magnitudes; target flutter no worse than the old stand expert (0.04°/tick) |
| 4 First controls, head only | look, idle breathing and sway, small reactions; all additive on the head track at low gain; Drew authors in Blender, the clip checker enforces velocity and acceleration limits | tracking error under 2° mean on hardware; no visible jitter |
| 5 Governor v0 | three states (idle, look, bow) with the transition table and the inertializer; bow is the first whole-body control: head clip plus a body request through L1 | 20 of 20 transitions clean on hardware, each direction |
| 6 Locomotion | the walk family (Drew's waddle, sidestep, turns) as a blend space; foundation v2: one learned tracker trained once on the whole library with a standard reward template, interface unchanged; phase-gated walk-to-stand | per-control tracking error ≤ 0.12 rad (Disney's worst motion); transition success per pair; no speed-up of falls versus v1 |
| 7 Recovery and self-correction | pushes and sensor delay in the tracker's training; capture-step trigger in L3; emergency stop to stand | push battery at the §3 step 11 magnitudes |
| 8 Life layer | attention engine with habituation, keep-alive idle, anticipation before and reaction after every control, timing and amplitude knobs on every clip, stunts | Drew's eye; expression heavy in social and idle moments, light while stepping |

Later, if wanted: runtime style knobs as tracker inputs (cadence, step height, trunk pitch, as
in Walk These Ways), a cost-biased selector in L3 (the useful half of motion matching), a
learned governor.

## 5. Teardown (waits for a go)

Keep: everything that records a fact about the real duck (`mechatronics/`, `joint_limits/`,
`strength_data/`, `walk_recordings/`, `hardware/`, the IMU and servo scripts, `sim/report1`,
`duck_battery/`, the tuck pose work); the live interaction system (`tracker/` puppet and
`duck_agent.py`, `anim/` Blender bridge, `tracker/data/clips`, `duck_playground/`);
`sim/upstream_runtime` as the reference for the Pi runtime; `official_repo/`.

Retire: `sim/train_pg` (12 GB, own git with three unpushed commits: keep its tracking,
recovery and recipe sources on a branch, drop the venv and run folders), `sim/fusion`,
`sim/eval_tracking.py`, `sim/train_skill.sh`, `sim/curriculum.sh`, `sim/report2`, `sim/report3`,
`skills/`, `skills_archive/`, `graph_runtime/` (keep `backends/hardware.py` as the record of the
runtime calls), `graphs/`, `tools/` skill tooling, `regression/` and its cron line,
`experiments/`, the two skill-library docs, the two arXiv PDFs.

Order: (1) tag `pre-restart-2026-11-05` on both repos; (2) commit `hardware/` and
`walk_recordings/` (54 MB of pickles, the only copy; decide whether the public repo should hold
them or a `data/` folder outside git); (3) remove the 03:30 cron line; (4) repoint
`anim/dump_mjcf.py` and the three `mechatronics/` scripts to `anim/model`; (5) branch
`skill-library` with everything as is, then delete the retired folders from master; (6) trim
`tracker/server.py` (tfevents import, skill-library tabs) and `tools.json`; (7) rewrite README
and NOTICE; (8) new skeleton: `duck/` (L0 to L4 modules), `controls/` (packages), `sim/` kept
small (MuJoCo model, replay and scoreboard tools).

## 6. Decisions for Drew

1. Teardown: branch plus delete from master (recommended) or an `archive/` folder; delete the
   12 GB `sim/train_pg` venv and runs; whether the 54 MB of real walk recordings go into the public
   repo.
2. Foundation v2: accept that the leg-owning foundation will be one learned tracker, re-run as a
   standard job when a leg clip changes, with the classical v0 and v1 as the fallback.
3. The hands-on measurements: weigh, register dump, centre of mass, IMU (about 1.5 h), then a
   session with the duck on its stand for the hanging-leg tests.
4. The puppet stays the live interaction tool; does it also become the front end for
   previewing and triggering controls (the "operator console" Disney has)?

## 7. Open questions

- Is register 85 RAM or EEPROM on our units, and what do 41 and 85 read at boot?
- Real loop latency on the Pi Zero 2 W with 14 servos on one bus.
- Which BNO055 mode the runtime uses, and whether its fusion deadband hides slow tilt.
- Whether the sim's ±3.23 N·m is a stall torque or an overestimate (step 4 decides).
- How much head and trunk offset the foundation tolerates before balance degrades (no
  published number; measure in phase 4).
