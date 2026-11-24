# Teardown inventory (2026-11-05)

Taken before the restart around the layered animation architecture. Columns: path | what it is | size | git state | dependencies.
Git state: T = tracked, T* = tracked with uncommitted changes, U = untracked, I = git-ignored.

**Repo size: 14G total.** About 13G of that is sim/train_pg plus sim/.venv. Everything else is roughly 0.6G.

**Biggest folders:** sim/train_pg 12G (.venv 9.6G, jax cache 1G, runs* ~750M, .git 78M); sim/.venv 1.4G; official_repo 244M; walk_recordings 54M; sim/report2 48M; sim/fusion 19M; .git 15M; sim/upstream_runtime 9.6M; skills 7.3M; sim/report3 6.2M.

**Services and cron:**
- Cron: `30 3 * * * cd …/mini-duck-droid && MUJOCO_GL=egl regression/run.sh >> regression/logs/cron.log` (old methodology; remove first; it rewrites skills/*/v1/report/*.json nightly, which is why those show as modified).
- systemd user service `duck-tracker.service`: active, runs `/usr/bin/python3 %h/src/mini-duck-droid/tracker/server.py --port 8430 --lan` with system Python. No duck timers.

**Uncommitted right now (63 entries):**
- Tracked, modified (12): experiments/log.md, regression/logs/cron.log, sim/eval_tracking.py, sim/fusion/bench.py, sim/fusion/fusion_sim.py, sim/train_skill.sh, skill-library-plan.md, skills/{bow,stand}/v1/report/{battery,sim2sim}.json, skills/walk/v1/report/battery.json.
- Untracked: ax_2604.07984.pdf, ax_2608.00820.pdf, my_ax_*.txt, graph_runtime/run_hw.py, hardware/, regression/2026-10-24…11-05.md + logs/, sim/fusion/{jitter,passive_push,push_video}.py, sim/fusion/runs/{…}, skills/recovery/, skills/stand/v1/ref.json, skills/walk/v1/report/sim2sim.json, tools/clip_ref_table.py, tools/hw_report.py.
- Separate repo sim/train_pg (fork soyeon24/Open_Duck_Playground, branch duck-real-servos): 3 local-only commits (72500c6, 6b6690d, 5c9fd12), uncommitted tracking.py/tracking_runner.py, untracked recovery.py. Only copy of the tracking, fusion and recovery task code.

## KEEP (physical facts)
- imu_axes_check.py | how the BNO055 must sit, from gravity (runs on the Pi) | 8K | T
- imu_mount_check.py | IMU mount check on the assembled duck, gives imu_upside_down | 4K | T
- joint_limits_probe.py | motors limp, records real mechanical stops by hand | 4K | T | writes joint_limits/
- joint_limits/ | measured stops: joint_limits_20261014-165208.json, real_limits_20261014.json | 12K | T | used by sim/render_tuck.py, sim/make_limits_plot.py, battery_guard tuck pose
- strength_test.py + strength_data/ | servo strength by pressing a foot on a kitchen scale; 4 CSVs from 2026-10-14 | 32K | T
- wave_power_test.py | rolling wave through all 14 servos, logs power draw | 12K | T
- stand_check.py | stands the duck in its walk start pose, reports tilt and joint tracking | 8K | T
- servo_ping.py, setup_all_servos.py, zero_servo.py, soft_zero.py, servo_playground.py | servo bus utilities (ping, ID/P=32/zero, offsets, desktop mover) | 64K | T
- head_yaw_soft_limit.patch | head-yaw soft limit for the runtime's duck_config.py | 4K | T
- analyse_obs.py, analyse2.py | analysis of real walk observation pickles | 12K | T | read walk_recordings
- walk_recordings/ | real-duck walk observations obs_walk1-9, meta_walk7-9 .pkl | 54M | I, ONLY COPY | analyse_*.py, sim/make_walk9_plot.py
- hardware/logs/2026-10-28/ | real-duck session: stand held, stand floor, recovery_v2 floor, recovery_v2 taps (jsonl + png) | 3.3M | U | produced by graph_runtime/run_hw.py, read by tools/hw_report.py
- mechatronics/ | safety envelope, masses_mjcf.json, masses.json (all null: never weighed), weigh_me.md, bench_procedure.md, servo_health.py + telemetry_pull.sh, servo_response_sim.py + response/, apply_masses.py, make_envelope.py, model/ | 816K | T | three scripts read the MJCF from sim/train_pg/…/xmls with no fallback (repoint to anim/model, identical files)
- duck_battery/ | battery_guard.py (low-voltage warn, tuck, torque off), installer, patches, duck_off.py, power_watch.py | 72K | T | tuck pose from sim/render_tuck.py + joint_limits
- tracker/docs/ | pinouts, battery pack wiring | 204K | T
- tracker/data/catalog.json, sourcing_2026-09-26.json, tracker/tools/build_catalog.py | parts and Canadian sourcing | ~150K | T
- sim/report1/ | "Duck Training Log": strength, limits, mass, kp, torque, walk9 plots, tuck/shutdown videos | 5.7M | T (mp4 I)
- sim/make_strength_plot.py, make_limits_plot.py, make_walk9_plot.py, make_kp_plot.py | plots of measured data | ~10K | T
- sim/tuck_sim.py, render_tuck.py, shutdown_sim.py, ball_real.json, report_battery/ | tuck/shutdown pose search with real joint limits | ~850K | T (report_battery I) | sim/.venv + playground xmls

## KEEP (puppet / live interaction)
- tracker/server.py | Duck Dashboard at /, puppet at /puppet/, old tracker at /build/; proxies /duck/api to the Pi :8091; deploys duck_agent over ssh | 40K | T | duck-tracker.service; needs official_repo robot model + print/ for STLs; dashboard tabs glob skills/, sim/train_pg/runs*, sim/report*, regression, experiments, tests, graph_runtime/demo; DOCS point at skill-library-*.md
- tracker/duck_agent.py | Pi-side agent: state, cmd, record, clip for the 14 servos, HTTP :8091, --demo | 24K | T | stdlib + pyserial; copied to bdxv2@bdxv2.local:~/duck_agent.py; not used by graph_runtime
- tracker/puppet.html, puppet.js, puppet.css, anim.js, viewer.js, style.css | puppet UI, clip model, 3D viewer | ~85K | T
- tracker/vendor/ | three.js, OrbitControls, STLLoader | 1.4M | T
- tracker/build/, tracker/data/progress.json | old print/shopping tracker | 76K | T
- tracker/data/clips/ | 9 clips: Bow, Stand, Walk, Puppet_walk_test1 (Drew/puppet); s2a_fwd, s2a_turn_left, s2a_turn_right (old RL brain); test_bow, test_look (scripted) | 264K | T | absolute paths in skills/*/skill.json, sim/fusion/train_gate.sh, tests/test_clips_data.py, graph_runtime/make_demo_skills.py
- anim/ | Blender bridge: README, blender_duck.py, duck_tools_addon.py, blender_remote.py, blender.sh, install_addon.sh, duck.blend, duck_model.json, model/ (MJCF copy), dump_mjcf.py, clip_check.py, make_walk.py, make_test_clips.py, reports/ | 1.7M | T | needs sim/.venv (mujoco) for Check clip; dump_mjcf.py has no fallback from sim/train_pg (repoint to anim/model)
- duck_playground/ | bench web page on the Pi (:8090), predecessor of the puppet | 80K | T
- official_repo/ + print symlink | upstream Open_Duck_Mini v2 (STLs, docs, robot model) | 244M | I (own git) | tracker 3D view and /stl/

## ARCHIVE-OR-DELETE (old methodology)
- sim/train_pg/ | Playground fork: tracking.py, tracking_runner.py, recovery.py, fusion_gate*.py, recipe_v1.py, runs*/ (plus older joystick, standup, carry, skate, obstacle, emolo tasks) | 12G | I; own git with unpushed work | anim/dump_mjcf.py, anim/clip_check.py (fallback), 3 mechatronics scripts, tracker/tfevents.py + server.py read it
- sim/fusion/ | expert fusion bench, experts, fusion_sim, train_gate.sh, jitter, passive_push, push_video, runs/ | 19M | T* + U
- sim/eval_tracking.py, sim/train_skill.sh | clip-tracking evaluation and training | 36K | T*
- sim/curriculum.sh, make_curriculum_plots.py | older scaffold walking curriculum | 20K | T | in tracker/data/tools.json
- sim/report2/ | "Teaching the duck to walk in a harness" | 48M | T (mp4 I)
- sim/report3/ | "Duck Overnight Build" (skill library) | 6.2M | T
- sim/record_policy_clip.py | records an RL brain's gait as a clip (source of s2a_* clips) | 8K | T
- skills/ | packages bow/v1, stand/v1, walk/v1, recovery/v1+v2, schema.json, README | 7.3M | T*, recovery U | dashboard, regression, graph_runtime
- skills_archive/ | retired packages | 5.9M | T | in NOTICE.md
- graph_runtime/ | motion-graph engine, packages, policies, mujoco + hardware backends, run_sim, run_hw, demo | 4.7M | T (run_hw.py U) | backends/hardware.py documents the exact mini_bdx_runtime calls (keep as reference)
- graphs/ | motion-graph JSONs | 20K | T
- tools/skill_package.py, skillkit.py, battery.py, sim2sim.py, clip_ref_table.py | skill package tooling, gates | ~90K | T (clip_ref_table U)
- regression/ | nightly re-scoring: run.sh, daily .md 10-23…11-05, logs/ | 844K | T + U | the cron entry
- experiments/ | log.md (RL log 10-23…10-28), questions.md, mjlab_spike.md, log.py | 92K | T* | questions.md "Mechatronics" and "For Drew" sections worth skimming first
- skill-library-plan.md, skill-library-framework.md | old plan and framework | 37K | T*, T | dashboard DOCS
- ax_2604.07984.pdf, ax_2608.00820.pdf (+ my_ax_*.txt) | motion-tracking papers | 16M | U
- tracker/tfevents.py | TensorBoard curves from sim/train_pg/runs* | 4K | T | server.py imports it
- tracker/data/progress_skill_library.json | skill-library milestones | 8K | T | server.py PROGRESS_SL
- tracker/data/jobs/, .pytest_cache/, __pycache__/ | caches/job logs | ~0.3M | I

## UNSURE
- tracker/index.html, dash.js, dash.css | dashboard at "/": Training/Skills/Experiments/tests tabs are old method; Progress, Animations, Telemetry are not | 80K | T
- tracker/jobs.py + tracker/data/tools.json | 9 tools: clip_check, blender, dump_model, servo_health are keep; curriculum, train_skill, battery, eval_policy, render_video are old | 23K | T
- tests/ + run_tests.sh | mixed: test_clips_data, test_blender_roundtrip, test_clip_check, test_model_dump, test_envelope_and_registry cover kept code; test_pipeline, test_tools, test_sim_smoke, GATES.md are old | 504K | T
- sim/.venv | uv venv, Python 3.12, mujoco 3.15 | 1.4G | I | needed by anim and mechatronics; rebuildable
- sim/Open_Duck_Playground | upstream playground checkout | 5.1M | I | sim_test.py reads its xmls
- sim/upstream_runtime | submodule Open_Duck_Mini_Runtime at 376de65, the Pi runtime source | 9.6M | T gitlink | reference for the new architecture
- sim/ walk-policy sim-vs-real scripts (Oct 13-14): sim_test, eval_policy, render_video, experiments, mass_test, torque_test, kp_test, headkp_test, bias_test, filter_test, smooth_cmp(+lib), joint_jitter, startgrid, sweep, sim_routine(+lib), autohead(+test) | encode measured facts (kp ×1.48, 1.86 N·m) | ~70K | T | sim_test used by graph_runtime, tools, tests, make_envelope
- sim/BEST_WALK_ONNX*.onnx, sim_*/simbl_*.pkl | upstream walk brain copies + sim routine observations | 13M | T (pkl I) | duplicates of official_repo
- sim/duck_head_cmd_patch.py, duck_feel/, web_turn_direction.patch | Pi walk-runtime patches (head from command, feel sliders, turn direction) | ~100K | T
- gov.py | replays a stability governor on an observation pickle; imports duck_live/stability_governor which is not in the repo (broken) | 4K | T
- graph_runtime/run_hw.py, backends/hardware.py, tools/hw_report.py | old method, but produced/analyse the 10-28 hardware logs and document the mini_bdx_runtime calls | ~30K | U/T
- archive/hermes_2026-09-23/ | superseded BOM and sourcing notes | 20K | T

## KEEP (repo infra, needs edits)
- README.md, NOTICE.md (name skills/, skills_archive/, graph_runtime/), LICENSE, LICENSE-ASSETS, LICENSES/, .gitignore, .gitmodules.

## Caveats
- Real-duck data at risk: walk_recordings/ (ignored) and hardware/ (untracked) are not in git. Commit or back them up before any cleanup.
- Repoint anim/dump_mjcf.py and mechatronics/{apply_masses,servo_response_sim,make_envelope}.py to anim/model before sim/train_pg goes (identical files, checked with diff).
- Remove the regression cron line before deleting regression/ or skills/.
- tracker/server.py imports tfevents.py; trim the server rather than delete the file.
