# Deployable safety envelope, Open Duck Mini v2 (draft 0.1, 2026-10-23)

Machine-readable twin: `envelope.json` (regenerate with `cd sim && .venv/bin/python ../mechatronics/make_envelope.py`).
Every number carries a status: **verified** (read from a file in this repo), **measured** (on the duck),
**derived** (computed, how is stated), **proposed** (our rule, untested) or **unverified** (manufacturer
figure or assumption). A skill package copies the numbers it was validated under into its
`safety_envelope`; the battery and the clip checker enforce them.

## 1. Torque: 1.86 N·m, not 3.23

| | value | status | source |
|---|---|---|---|
| Deploy limit | **1.86 N·m** | derived | 19 kg·cm rated stall at 7.4 V (README: "STS3215 7.4 V (19 kg·cm)") × 0.0981 N·m per kg·cm. Used by `eval_policy.py --torque`, `curriculum.sh` TORQUE_RANGE, `HR_TORQUE` |
| Model limit | 3.23 N·m | verified | `forcerange` of class `sts3215` in the training XML (BAM fit; the older fit in `joints_properties.xml` says 3.35) |
| Bench | > 1.16 N·m | measured | `strength_data/strength_14_20261014-034321.csv`: right ankle on a kitchen scale, 2.33 kg at 5.1 cm after a 15° press, servo at 21 % load and still rising. The maximum was not reached |
| Stiffness | 25.3 N·m/rad (0.44 N·m/°) | measured | report 1, 1.48× the model's 17.11 |

**Why not 3.23.** The 3.23 is where BAM's fit saturates `kp × error` on a bench supply; it is a peak the
driver can push into a stalled motor for an instant, not a torque the duck can use. The manufacturer's
rating at the pack's nominal voltage is 1.9 N·m, and a 2S pack feeding 14 servos sags (7.23 V at 183 mA
on one servo during the scale test). In simulation the current brain goes from 0.13 m/s forward at 3.35
to 0.10 at 1.86 and the real duck is nearer the latter (report 1, `torque.png`); a policy trained at 3.23
learns to lean on torque the duck does not have. Until the bench says otherwise, 1.86 is the ceiling a
skill must pass under, and training randomises around it (TORQUE_RANGE 1.6–2.1).

## 2. Motor velocity: 4 rad/s target rate limit

| | value | status | source |
|---|---|---|---|
| Deploy clamp on the target step | **4.0 rad/s** | verified | `MAX_MOTOR_VEL=4.0` (curriculum.sh), report 2 "deploy with cutoff_frequency=8 and max motor velocity 4 rad/s". The runtime default is 5.24 (`v2_rl_walk_mujoco.py`; the Pi's config has `velocity_clip: true, max_motor_velocity_rad_s: 5.24`): set 4.0 at deploy |
| Real joints while walking | p95 2.7, p99 3.8, max 4.5 rad/s | measured | `walk_recordings/obs_walk7.pkl` dof_vel, knees fastest |
| Model ceiling at 1.86 N·m | 3.2 rad/s | derived | (1.86 − 0.068) / 0.56: forcerange cap minus frictionloss over damping |
| Datasheet no-load | 4.7 rad/s | unverified | Feetech listing 0.222 s/60° at 7.4 V |

Note the contradiction: capping `forcerange` alone makes the model servo top out at 3.2 rad/s, while
the real knees reach 3.8–4.5 rad/s. The torque cap is right at stall and pessimistic at speed. The bench
no-load speed test (bench_procedure.md) decides whether damping should drop with the cap.

## 3. Action low-pass: "8 Hz" setting, 1.2 Hz in fact

Setting `ACT_LPF_HZ=8` / `--cutoff_frequency 8` (verified: curriculum.sh, `rl_utils.LowPassActionFilter`).
The formula `alpha = (1/fc)/(1/50 + 1/fc) = 0.862` is a first-order filter whose pole at 0.862 per 20 ms
tick is a **1.19 Hz** −3 dB point (derived: `servo_response_sim.py` sweep through the deploy chain;
50/(2π)·(−ln 0.862) = 1.18 Hz). It puts 1/fc where an RC constant belongs. Training, the scorer and the
runtime all share the formula, so sim and duck agree and nothing is broken; but the package envelope
should carry the real cutoff (1.2 Hz) and the clip checker should not expect 8 Hz of bandwidth. Through
the chain a 0.5 rad step rises in 0.30 s with no overshoot (model servo: 0.16 s, 6.6 %).

## 4. Falls and height

| | value | status | source |
|---|---|---|---|
| Fall tilt | **45°** between body z and world z | verified | `sim_test.py run()`: `fell = tilt_max > 45`; `eval_policy.py` same. Framework §3.3's 0.8 rad is 45.8° |
| Training termination | upvector z < 0 (90°) | verified | `joystick.py _get_termination` |
| Standing heights (floor = 0) | base 0.15 m, IMU 0.20, CoM 0.19, head 0.39 | verified | scene keyframe `home`, MuJoCo FK |
| Tuck heights | chin_up base 0.11 m, head_down 0.12 m | derived | `battery_guard.SHUTDOWN_POSES` through FK |
| Minimum base height | **0.09 m** | proposed | 60 % of standing, below both tucks so a deliberate sit is not a fall. Recovery packages declare their own floor |

## 5. Thermal

| | value | status | source |
|---|---|---|---|
| Servo's own limit | 70 °C (EEPROM 0x0D) | unverified | memory-table default; `servo_health.py` reads it |
| Bench abort | 55 °C | verified | `strength_test.py MAX_TEMP` |
| Seen so far | 30 °C at rest | measured | strength CSVs; nothing under walking yet |

**Proposed cool-down rule** (max over 14 servos, read at least every 20 s, logged):
- ≥ 50 °C: warn; no new episodic skill starts.
- ≥ 60 °C: finish the current skill (its timeout applies), go to idle, then the sit-down abort with servo P at 2; stay until every servo is ≤ 45 °C.
- ≥ 65 °C or any `temperature` status flag: sit-down path immediately, motors off.
Why: 70 °C is where the servo unloads itself (unloading condition bit 2, default on); 65 leaves 5 °C for the sampling gap. Framework §5.5 lists thermal drift as a named risk.

## 6. Voltage and the 8.3 V question

| | value | status | source |
|---|---|---|---|
| Nominal | 7.4 V (2S Li-ion, 2 × Molicel 18650) | verified | catalog, README |
| Full charge | 8.4 V | unverified | 4.2 V/cell; BMS listed 7.4/8.4 V 20 A |
| Observed | 8.3 V on all 14 servos with the `voltage` flag | measured | puppet agent, 2026-10-22 |
| Under load | 7.2–7.9 V | measured | strength CSVs |
| Servo max-voltage register (0x0E) | 8.0 V expected | unverified | Feetech default 80; `servo_ping.py` reads it as "voltage limits"; never read on this duck |
| Servo min-voltage register (0x0F) | 4.0 V expected | unverified | default 40 |
| Battery guard | warn 7.0, critical 6.8, hard 6.5 V | verified | `battery_guard.py`, Pi config |
| Operating range | 6.8–8.4 V | proposed | |

**Verdict so far (pending the register read; the Pi was unreachable tonight):** 8.3 V is almost certainly
above the 8.0 V max-voltage register, so the flag is correct and expected right after a charge, and clears
by itself once the pack drops below 8.0 V. It is advisory: by default the unloading condition (0x13 = 44)
does not include voltage, so the servo keeps working and only its LED alarm (0x14 = 47) shows it.
`servo_health.py` reports `unload_on_voltage` per servo to confirm. Options for the user: leave it; charge
to ~8.2 V; or raise 0x0E to 85–90 on all servos, a one-time EEPROM write to authorise explicitly.

## 7. Current

| | value | status | source |
|---|---|---|---|
| Current register unit | 6.5 mA/LSB (0x45) | verified | `duck_agent.py`, `strength_test.py` |
| Servo protection current (0x1C) | 3.25 A expected | unverified | default 500 × 6.5 mA |
| Datasheet stall | 2.7 A at 7.4 V | unverified | listing |
| Bench abort | 2.5 A per servo | verified | `strength_test.py MAX_MA` |
| Static hold | 1.16 N·m at 0.18 A, 7.23 V | measured | strength CSV |
| Proposed budgets | 1.0 A sustained / 2.5 A peak per servo; 8 A sustained / 15 A peak pack | proposed | 25 % under the 20 A BMS; sustained numbers are guesses until a pack-current log exists (`wave_power_test.py`'s `power_log.csv` is not on disk) |

## 8. Joint ranges (rad, runtime frame = MJCF frame, offsets removed)

Rule: deploy range = (MJCF range ∩ measured stops from `joint_limits/real_limits_20261014.json`) minus 3° each side,
clipped by the runtime soft limit (`head_yaw` ±70°, Pi config). The clip checker fails a clip that leaves it or whose
target rate exceeds 4 rad/s. The exact per-joint numbers are in `envelope.json` → `joints`; headline findings:

- Knees: real stops 1.80 (left) / 1.73 (right) with home at 1.37, so only ~0.15 rad (8°) of deeper fold is deployable; the MJCF's ±1.57 was never real.
- Ankles: real −1.25…0.56 (left), −1.18…0.87 (right) against the MJCF's ±1.57.
- Head yaw: soft limit ±1.22 rad (70°) governs; the stops are −1.35…1.75 (cable), the MJCF says ±2.79.
- Head roll: −0.43…0.21 deployable (asymmetric stops).
- Left hip pitch measured stop −2.45 is suspect (report 1); the intersection with the MJCF (−1.22) removes it.
- Home pose is inside every deploy range (checked by `make_envelope.py`).

## 9. Mass and model identity

Model total 2.107 kg (verified, `masses_mjcf.json`); user estimate 1.5–1.8 kg: the largest known model error.
Training covers it with MASS_SCALE and ±10 % per body. `weigh_me.md` gets the real numbers; `apply_masses.py`
writes the measured model and its `robot_model_hash`. The model as trained has `mjcf_sha256
11e922dc…` and the canonical CAD copy hash `759d5b7f…` (`model/robot_model_hash.txt`).

## 10. Control constants a package must match
50 Hz policy (loop 8 ms mean, 9.7 ms p95 measured); servo P 30 on the legs, 8 on the head, EEPROM default 32,
I = D = 0, acceleration 0; `action_scale` 0.25 (0.5 for LEG_ACTION_SCALE brains); IMU mounted upside down
(`imu_upside_down: true`).

## What is still unverified
EEPROM limits (never read); stall torque above 1.16 N·m; no-load speed vs the 3.2 rad/s model ceiling; total
mass; every thermal number under load; current budgets; the 0.09 m height floor.
