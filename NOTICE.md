# Licensing

This repository mixes original work with files from other projects. Each file is under the
first rule below that matches it.

## 1. Already licensed by someone else (their terms apply, not ours)

| What | Where | Licence |
|---|---|---|
| Open Duck Mini robot model and meshes (Antoine Pirrone and contributors) | `anim/model/`, `mechatronics/model/*.xml`, the mesh data inside `anim/duck_model.json` and `anim/duck.blend` | Apache 2.0, `LICENSES/Apache-2.0.txt` |
| Open Duck Mini walk policy | `sim/BEST_WALK_ONNX*.onnx`, `skills/walk/v1/policy.onnx`, `graph_runtime/demo_skills/walk/v1/policy.onnx` | Apache 2.0 |
| Modified copy of an Open Duck Mini runtime script | `duck_feel/v2_rl_walk_mujoco.live.py` (the upstream parts) | Apache 2.0 |
| Open Duck Mini runtime | `sim/upstream_runtime` (git submodule, not stored here) | its own repository's terms |
| three.js | `tracker/vendor/*.js` | MIT (see each file's header) |

Changes made here to Apache 2.0 files are original work under rule 2.

## 2. Animations, clips and media: CC BY-NC 4.0

Animations and motion clips (`anim/duck.blend` rig and actions, `tracker/data/clips/`,
`skills/*/*/clip.json`, `graph_runtime/demo_skills/*/*/clip.json`), images, videos, reports and
written documents are licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/)
licence, full text in `LICENSE-ASSETS`.

## 3. Everything else (code, configs, trained skill policies): PolyForm Noncommercial 1.0.0

All other original material is licensed under the
[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0),
full text in `LICENSE`.

Required Notice: Copyright 2026 Drew (https://github.com/drewsipher)

Commercial use of anything under rules 2 or 3 needs written permission from the copyright holder.
