# Bench procedure: measure one real STS3215 the way `servo_response_sim.py` simulates it

Goal (framework §3.2 exit gate): the simulated step and chirp responses match the bench within
tolerance. Until then the actuator model is "BAM upstream, unverified on our servos".
Needs the user's hands, about 45 minutes, and a servo that is NOT in the duck (a spare) or the
right ankle with the foot free in the air (duck on its stand, walk program stopped).

## Rig
1. Servo bolted or clamped so the horn axis is **horizontal**. On the duck: the ankle with the foot
   hanging, duck on the stand; the foot is then the "arm".
2. Arm on the horn: the duck's shank + foot (about 0.15 kg, 79 mm) or a printed bar of the same
   length and mass. Zero = hanging straight down.
3. Load variant: a 100 g weight (two AA-size lead/steel slugs, or a 100 g calibration weight)
   taped at the end of the arm.
4. Power: the duck's own pack at 7.6–8.0 V (note the voltage; it matters), or the bench supply at
   7.4 V with a 5 A limit. Servo P = 30 (the walk setting), I = D = 0, acceleration 0, same as the
   walk program sets them.
5. Bus: the Waveshare adapter on `/dev/ttyACM0`, commands at **50 Hz**, present position read each
   tick (the `Bus` class in `tracker/duck_agent.py` does both; a `bench_record.py` will be added when
   the user is ready; the puppet agent's `play` with a clip of the step/sine is enough for a first try).

## Tests (same as the simulation)
| test | command | what to log | sim reference (model, 1.86 N·m) |
|---|---|---|---|
| Step | 0 → 0.5 rad, hold 1.5 s, back to 0; 5 repeats each way | position at 50 Hz, current, voltage, temperature | rise 10–90 % 0.16 s, overshoot 6.6 %, 2 % settle 0.35 s |
| Step + 0.1 kg | same with the weight | same | 0.163 s, 6.2 %, 0.35 s (the model barely notices the load) |
| Sine sweep | 0.05 rad amplitude at 0.5, 0.7, 1, 1.4, 2, 2.8, 4, 5.7, 8, 11, 15 Hz; 5 cycles each (min 2 s) | position | −3 dB at 4.8 Hz (small signal), gain peak 1.15 at 3 Hz |
| Sine sweep, large | 0.2 rad amplitude, same frequencies | position | −3 dB at 3.0 Hz (slew-limited) |
| No-load speed | 0 → 1.5 rad step, measure the constant-velocity part | velocity | model ceiling 3.2 rad/s at the 1.86 cap (real knees reach 4.5 while walking) |
| Stall (optional, with the scale) | repeat `strength_test.py` to 25° or until the servo stops gaining | torque, current | 1.86 N·m rated; bench so far > 1.16 |

Keep the servo under 55 °C (strength_test rule); pause between sweeps.

## Analysis
Fit each sine with least squares at the known frequency (`servo_response_sim.py sine_fit`), take
amplitude ratio and phase; rise time 10–90 %, overshoot, settling into ±2 % of 0.5 rad; velocity
ceiling from the long step. Put the numbers next to `response/response.json` in a `bench_<date>.json`
with the same keys.

## What counts as "matches" (proposal, to agree with Controls)
| quantity | tolerance | why |
|---|---|---|
| rise time 10–90 % | within 25 % of the model | the step is torque-limited; 25 % is the TORQUE_RANGE spread (1.6–2.1 N·m) |
| overshoot | within 5 percentage points | sets kp/damping balance; KP_RANGE 1.1–1.8 is wide already |
| −3 dB bandwidth (0.05 rad) | within 20 % | the policy acts through a 1.2 Hz filter, so 4.8 vs 5.8 Hz is invisible to it; 2 Hz would not be |
| no-load velocity ceiling | within 20 % | decides whether the cap should also lower damping |
| load sensitivity (0 vs 0.1 kg) | rise time change < 10 % in both | tests the armature value (0.027 is 40× the arm's inertia; if the real servo slows with 0.1 kg the model's rotor inertia is too high) |
| stall torque | 1.86 ± 0.3 N·m | the deploy ceiling; outside this band the envelope number changes |

Out of tolerance → change the XML's `sts3215` class (one number at a time, in a COPY under
`mechatronics/model/`), rerun `servo_response_sim.py`, bump `actuator_model_version` in the envelope,
and every package records the new hash.
