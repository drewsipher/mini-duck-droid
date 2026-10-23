# Test run 2026-10-23 10:44:44 EDT

**92 passed, 0 failed, 2 skipped** in 11 s (pytest exit 0).

| File | Pass | Fail | Skip | Seconds |
|---|---|---|---|---|
| tests/test_blender_roundtrip.py | 5 | 0 | 2 | 1.0 |
| tests/test_clip_check.py | 12 | 0 | 0 | 2.5 |
| tests/test_clips_data.py | 7 | 0 | 0 | 0.0 |
| tests/test_envelope_and_registry.py | 7 | 0 | 0 | 0.0 |
| tests/test_model_dump.py | 8 | 0 | 0 | 0.2 |
| tests/test_pipeline.py | 5 | 0 | 0 | 1.3 |
| tests/test_reporter.py | 3 | 0 | 0 | 0.2 |
| tests/test_sim_smoke.py | 6 | 0 | 0 | 0.9 |
| tests/test_tools.py | 39 | 0 | 0 | 5.2 |
| **total** | **92** | **0** | **2** | **11.4** |

## Slowest tests

- 1.3 s  tests/test_tools.py::test_sim2sim_short_run_writes_report
- 0.9 s  tests/test_tools.py::test_battery_one_episode_writes_ratios
- 0.8 s  tests/test_tools.py::test_graph_runtime_own_tests_pass
- 0.3 s  tests/test_tools.py::test_skill_package_draft_validates_without_battery_only
- 0.2 s  tests/test_sim_smoke.py::test_rollout_is_deterministic
- 0.2 s  tests/test_clip_check.py::test_max_vel_flag_is_wired
- 0.2 s  tests/test_clip_check.py::test_feet_not_level_reports_penetration
- 0.2 s  tests/test_tools.py::test_skill_package_refuses_missing_clip_and_skill_json

## Skipped (tools not built yet, or marked slow)

- tests/test_blender_roundtrip.py::test_roundtrip_base_is_standing[s2a_turn_left.json]: exporter omits the base track when the trunk is not animated (clip_check then plants the feet)
- tests/test_blender_roundtrip.py::test_roundtrip_base_is_standing[test_bow.json]: exporter omits the base track when the trunk is not animated (clip_check then plants the feet)
