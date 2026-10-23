# Mechatronics: robot model truth, actuator facts, safety envelope, telemetry

Owner of the numbers a skill package's `safety_envelope` and `robot_model_hash` come from
(framework §2.5, §3.2; plan §3.1). Everything here was built on 2026-10-23 without touching the
duck: the Pi was unreachable, so the hardware read is a tested dry run waiting for a free port.

## Files
| file | what |
|---|---|
| `servo_health.py` | Read-only health read of the 14 STS3215 (runs on the Pi with the runtime venv): position (runtime frame), speed, load, voltage, temperature, current, status flags, EEPROM limits (0x0D max temp, 0x0E/0x0F voltage, 0x1C protection current, unloading/LED alarm conditions, P/I/D, offset). Refuses a busy port (fuser, /proc, process names), only ever sends PING and READ (`Bus.packet` enforces it). One JSON line. `--fake` for development |
| `telemetry_pull.sh` | scp the script to the Pi, run it, append the line to `telemetry/YYYY-MM-DD.jsonl`, print a one-line summary. `--fake` appends to `telemetry/sample_fake.jsonl` instead. Exit 1 unreachable, 2 port busy |
| `telemetry/` | `sample_fake.jsonl` (schema sample for the dashboard). Real files appear per day |
| `envelope.json`, `envelope.md` | The deployable safety envelope with a source and status for every number: torque 1.86 (why not 3.23), 4 rad/s, the "8 Hz" filter that is 1.2 Hz, 45° fall tilt, 0.09 m height floor, thermal rule, voltage and the 8.3 V verdict, current budgets, per-joint deploy ranges. `make_envelope.py` regenerates the JSON |
| `masses_mjcf.json` | Every body mass of the training MJCF, groups (trunk, head and neck, legs), total 2.107 kg |
| `weigh_me.md`, `masses.json` | Kitchen-scale checklist for the user and the file to fill (grams, nulls ignored) |
| `apply_masses.py` | Writes the measured masses into a COPY under `model/` (inertias scaled), load-tests it with MuJoCo, prints `robot_model_hash`. `--baseline` gives the hash of the model as trained; `dump` rewrites `masses_mjcf.json` |
| `model/` | `open_duck_mini_v2_backlash.cad.xml` (baseline copy, meshdir absolute), `robot_model_hash.txt`, `manifest.json`. The training XML is never edited |
| `servo_response_sim.py`, `response/` | One-joint MuJoCo bench: 0.5 rad step and 0.5–15 Hz sweep/chirp with the XML's BAM actuator at 1.86 N·m, at 3.23, at the measured ×1.48 stiffness, with and without 0.1 kg, and through the deploy chain. `step.png`, `chirp.png`, `response.json` |
| `bench_procedure.md` | How to measure the same on a real servo (needs the user) and the tolerances that count as a match |

## Headline numbers (details and sources in `envelope.md`)
- Torque ceiling **1.86 N·m** (19 kg·cm at 7.4 V); bench showed > 1.16 and was still rising. 3.23 is the BAM saturation, not usable torque.
- Target rate clamp **4 rad/s** (runtime default 5.24; real knees reach 3.8–4.5 rad/s walking). At the 1.86 cap the model's own ceiling is 3.2 rad/s: pessimistic at speed, to be settled on the bench.
- Action filter setting "8 Hz" = **1.19 Hz** −3 dB (formula), identical in training and runtime.
- Fall: tilt > **45°** or base z < **0.09 m** (standing 0.15, chin-up tuck 0.11).
- Thermal: warn 50 °C, cool-down 60, hard stop 65, resume ≤ 45 (servo unloads at 70). No data under load yet.
- Voltage: nominal 7.4, guard 7.0/6.8/6.5; **8.3 V flag** = above the expected 8.0 V max-voltage register, advisory by default (unload condition excludes voltage). Confirmed only once the registers are read.
- Model servo (1.86 N·m, 0.5 rad step): rise 0.16 s, overshoot 6.6 %, settle 0.35 s; −3 dB 4.8 Hz small-signal, 3.0 Hz at 0.2 rad; 0.1 kg on the link changes nothing (armature dominates). Through the deploy chain: rise 0.30 s, 1.2 Hz.
- Mass: model 2.107 kg, estimate 1.5–1.8 kg: weigh it.

## How to run
```
mechatronics/telemetry_pull.sh                 # duck on, port free; read-only
mechatronics/telemetry_pull.sh --fake          # schema sample
cd sim && .venv/bin/python ../mechatronics/apply_masses.py          # after filling masses.json
cd sim && MUJOCO_GL=egl .venv/bin/python ../mechatronics/servo_response_sim.py
cd sim && .venv/bin/python ../mechatronics/make_envelope.py
```

## Open questions
Listed with the other teams' in `experiments/questions.md` → Mechatronics. For the user: weigh the duck,
allow the first health read, decide the 8.3 V handling, book the bench session, finish the stall test.
