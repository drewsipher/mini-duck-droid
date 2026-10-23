# Experiment log

Append-only. One entry per run: `## <ISO datetime> <title>` then `- hypothesis:` `- command:` `- evidence:` `- decision:`.
Teams write under their own entries; questions for other teams go to `experiments/questions.md`.

## 2026-10-23T00:20-04:00 Controls: clip-tracking task built (recipe v1)
- hypothesis: one fixed BeyondMimic-style tracking recipe (recipe_v1.py) can train a duck skill from one clip without reward engineering; the first clip is the placeholder bow.
- command: new files `sim/train_pg/recipe_v1.py`, `playground/common/clip_tracking_reference.py`, `playground/open_duck_mini_v2/tracking.py`, `tracking_runner.py`, `sim/train_skill.sh`, `sim/eval_tracking.py`. Functional test: `scratchpad/controls/test_env.py` (single env, jitted reset/step, brax wrapper stack, DR).
- evidence: obs 107 = gyro 3, accel 3, joint_pos_rel 14, joint_vel 14, prev_action_1/2 28, motor_targets_rel 14, phase 3, ref next joint pos/vel 28. Open-loop (zero residual) on a 3 s hold-home clip: finishes, reward 0.099/step of 0.100 max, base error 1 cm, tilt 3 deg. Open-loop on `test_bow.json` with pushes and init noise OFF: falls at step 89/125 (1.8 s) every time, tilt 0.83 rad; the bow moves the whole-body CoM 1.6 cm toward the heel (x -0.026 -> -0.042, foot centre -0.016). So the bow is not open-loop feasible; the policy must shift weight with the legs. mujoco_playground's auto-reset copies `info["first_state"]`, never calls reset(): step() now writes a fresh adaptive-sampled start into it every step (see tracking.py docstring).
- decision: keep the recipe as written; run smoke -> preview on the bow; evaluate with `eval_tracking.py` (same thresholds: height < 0.09 m, root tilt > 0.8 rad).

## 2026-10-23T00:48-04:00 Controls: bow smoke + preview launched
- hypothesis: the pipeline runs end to end (64 envs x 5 iterations), then 30M steps show the head tracking the bow with falls decreasing.
- command: `nohup sim/train_skill.sh tracker/data/clips/test_bow.json bow --preview --no-eval` (TORQUE_RANGE=1.6,2.1 KP_RANGE=1.1,1.8 from the recipe; smoke NUM_ENVS=64 BATCH_SIZE=64 NUM_MINIBATCHES=4 NUM_TIMESTEPS=25600; preview PREVIEW=1 NUM_TIMESTEPS=30000000 NUM_ENVS=4096 NUM_EVALS=6 SEED=0). Logs: `sim/train_pg/runs_skills/bow/{smoke,preview}/train.log`, tfevents in the same dirs.
- evidence: smoke ok in 3.4 min (64 envs, 25.6k steps, policy.onnx + obs_spec.json exported). Preview: 30.3M steps in 17.9 min wall (4096 envs, ~47 min per 100M once compiled). Per-step eval means from tfevents (step: fell / finished / base err / joint err / action-rate cost): 0: 0.85 / 0.15 / 3.2 cm / 0.053 rad / 0.73; 6M: 0.13 / 0.87 / 1.5 cm / 0.053 / 0.37; 12M: 0.10 / 0.90 / 1.6 cm / 0.054 / 0.17; 24M: 0.055 / 0.945 / 1.6 cm / 0.050 / 0.074; 30M: 0.125 / 0.875 / 1.6 cm / 0.046 / 0.071 (eval noise: 128 episodes, half start mid-clip). Body terms rose from 0.60-0.78 to 0.93-0.98 of 1.0. Strict battery `eval_tracking.py` (5 seeds x 10 episodes, physics randomised, t = 0 starts, no noise): 50/50, joint err 0.0426 rad (2.44 deg), base err 1.26 cm, tilt err 3.6 deg, every episode runs the full 2.50 s. Video: `sim/train_pg/runs_skills/bow/preview/eval/eval.mp4` (reference ghost left, policy right). Package draft: `runs_skills/bow/preview/package_draft/`.
- decision: gate passed (falls 85 % -> ~10 % in training, 0 falls in the battery, errors falling): full run launched at 01:09.

## 2026-10-23T01:09-04:00 Controls: bow full run (150M steps)
- hypothesis: more steps under the same recipe lower the residual joint error (0.043 rad) and remove the remaining mid-clip falls seen in training eval (5-12 %), without any recipe change.
- command: `nohup sim/train_skill.sh tracker/data/clips/test_bow.json bow --skip-smoke --steps 150000000 --no-eval` (NUM_EVALS=10, seed 0), then `eval_tracking.py ... --package` on `runs_skills/bow/full/policy.onnx`.
- evidence: (pending)
- decision: (pending)
