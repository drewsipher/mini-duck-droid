# Test run 2026-10-23 01:53:05 EDT

**86 passed, 0 failed, 2 skipped** in 13 s (pytest exit 0).

| File | Pass | Fail | Skip | Seconds |
|---|---|---|---|---|
| tests/test_blender_roundtrip.py | 5 | 0 | 2 | 0.9 |
| tests/test_clip_check.py | 12 | 0 | 0 | 3.0 |
| tests/test_clips_data.py | 7 | 0 | 0 | 0.0 |
| tests/test_envelope_and_registry.py | 7 | 0 | 0 | 0.0 |
| tests/test_model_dump.py | 8 | 0 | 0 | 0.3 |
| tests/test_pipeline.py | 5 | 0 | 0 | 1.4 |
| tests/test_reporter.py | 3 | 0 | 0 | 0.2 |
| tests/test_sim_smoke.py | 6 | 0 | 0 | 0.9 |
| tests/test_tools.py | 33 | 0 | 0 | 6.2 |
| **total** | **86** | **0** | **2** | **13.1** |

## Slowest tests

- 1.6 s  tests/test_tools.py::test_sim2sim_short_run_writes_report
- 1.0 s  tests/test_tools.py::test_battery_one_episode_writes_ratios
- 1.0 s  tests/test_tools.py::test_graph_runtime_own_tests_pass
- 0.4 s  tests/test_tools.py::test_skill_package_draft_validates_without_battery_only
- 0.3 s  tests/test_tools.py::test_skill_package_refuses_missing_clip_and_skill_json
- 0.3 s  tests/test_clip_check.py::test_one_frame_jump_fails_on_speed
- 0.3 s  tests/test_clip_check.py::test_not_starting_at_home_fails_without_loop
- 0.3 s  tests/test_clip_check.py::test_knee_beyond_range_fails_on_joint_limits

## Skipped (tools not built yet, or marked slow)

- tests/test_blender_roundtrip.py::test_roundtrip_base_is_standing[s2a_turn_left.json]: exporter omits the base track when the trunk is not animated (clip_check then plants the feet)
- tests/test_blender_roundtrip.py::test_roundtrip_base_is_standing[test_bow.json]: exporter omits the base track when the trunk is not animated (clip_check then plants the feet)
