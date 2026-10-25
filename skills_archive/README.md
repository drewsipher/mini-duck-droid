# Retired skills

Not part of the library: nothing loads packages from here (tools, graphs, dashboard and the nightly
regression only look in `skills/`). Kept for the record.

- `placeholders/test_bow_v1`: trained 2026-10-23 on `test_bow.json`, a script-generated head-only
  curve made to test the pipeline. Never approved as an animation.
- `placeholders/turn_left_v1`: trained 2026-10-23 on `s2a_turn_left.json`, a recording of the old
  s2a walk brain turning. Never approved as an animation.

Rule since then: a skill is only trained on a clip Drew has approved.
- `upstream_walk_v1`: the deployed upstream walk brain (`BEST_WALK_ONNX_2`) packaged on 2026-10-23
  as `skills/walk/v1` under the strict battery (280/400: backward and side-right fail). Moved here
  on 2026-10-25 when Drew made his trained waddle walk `skills/walk/v1`. The motion graph demos
  still run the same brain from `graph_runtime/demo_skills/walk/v1`.
