#!/bin/bash
# Train one skill from one clip on the fixed recipe (recipe_v1.py), then evaluate it strictly.
#
#   sim/train_skill.sh <clip.json> <name> [--preview] [--loop] [--steps N] [--seed S] [--skip-smoke] [--no-eval] [--init CKPT]
#
#   --init  start from another skill's checkpoint (a directory under runs_skills/<skill>/full/) instead of
#           a fresh network: the branched-from-a-shared-base option of the fusion research
#
#   smoke   64 envs, 5 PPO iterations: does it compile and run end to end (~5 min)   -> runs_skills/<name>/smoke/
#   preview --preview: ~30 min (recipe PREVIEW_TIMESTEPS)                             -> runs_skills/<name>/preview/
#   full    default: recipe FULL_TIMESTEPS (~1.5 h)                                   -> runs_skills/<name>/full/
#   eval    sim/eval_tracking.py on the run's policy.onnx: strict battery + video,
#           and the package draft                                                     -> runs_skills/<name>/{eval,package_draft}/
#
# Everything lands under sim/train_pg/runs_skills/<name>/: train.log, tfevents (every reward
# term separately), checkpoints + .onnx per eval, policy.onnx, obs_spec.json, run.json, score.
# GPU: check nvidia-smi first; launch with nohup and poll the log, training metrics are in the
# tfevents (tracking_runner.py prints a one-line digest per eval into the log too).
set -eo pipefail
[[ $# -ge 2 ]] || { sed -n 2,14p "$0"; exit 1; }
CLIP=$(realpath "$1"); NAME=$2; shift 2      # resolve the clip before leaving the caller's directory
CALLER=$PWD
cd "$(dirname "$0")/train_pg"
PY=.venv/bin/python
export PYTHONPATH=.
MODE=full; LOOP=0; STEPS=""; SEED=${SEED:-0}; SMOKE=1; EVAL=1; INIT=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --preview) MODE=preview ;;
    --loop) LOOP=1 ;;
    --steps) STEPS=$2; shift ;;
    --seed) SEED=$2; shift ;;
    --skip-smoke) SMOKE=0 ;;
    --no-eval) EVAL=0 ;;
    --init) INIT=(--restore_checkpoint_path "$(cd "$CALLER" && realpath "$2")"); shift ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
  shift
done
[[ -f "$CLIP" ]] || { echo "no clip at $CLIP" >&2; exit 1; }
OUT=runs_skills/$NAME
mkdir -p "$OUT"
# the recipe's randomisation knobs for randomize.py (same values tracking_runner.py sets)
eval "export $($PY -c 'import recipe_v1 as R; print(" ".join(f"{k}={v}" for k, v in R.as_env_vars().items()))')"
export CLIP LOOP SEED
if nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep -v -i blender | grep -q .; then
  echo "=== warning: another compute process is on the GPU:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
fi
echo "=== $(date '+%F %T') skill $NAME  clip $CLIP  loop $LOOP  mode $MODE  seed $SEED  ${INIT[*]}"

if [[ $SMOKE == 1 ]]; then
  echo "=== $(date '+%T') smoke: 64 envs, 5 iterations -> $OUT/smoke"
  mkdir -p "$OUT/smoke"
  # one PPO iteration = batch_size * num_minibatches * unroll_length env steps = 64 * 4 * 20 = 5120
  NUM_ENVS=64 NUM_EVALS=1 BATCH_SIZE=64 NUM_MINIBATCHES=4 NUM_TIMESTEPS=25600 \
    $PY playground/open_duck_mini_v2/tracking_runner.py --output_dir "$OUT/smoke" > "$OUT/smoke/train.log" 2>&1 \
    || { echo "smoke failed; tail of $OUT/smoke/train.log:"; tail -30 "$OUT/smoke/train.log"; exit 1; }
  [[ -f "$OUT/smoke/policy.onnx" ]] || { echo "smoke produced no policy.onnx" >&2; tail -20 "$OUT/smoke/train.log"; exit 1; }
  echo "=== $(date '+%T') smoke ok: $(grep -c '^\[' "$OUT/smoke/train.log") eval lines, $(tail -1 "$OUT/smoke/train.log")"
fi

RUN=$OUT/$MODE
mkdir -p "$RUN"
if [[ $MODE == preview ]]; then export PREVIEW=1; fi
[[ -n "$STEPS" ]] && export NUM_TIMESTEPS=$STEPS
echo "=== $(date '+%T') $MODE run -> $RUN (steps ${NUM_TIMESTEPS:-recipe default})"
$PY playground/open_duck_mini_v2/tracking_runner.py --output_dir "$RUN" "${INIT[@]}" > "$RUN/train.log" 2>&1 \
  || { echo "$MODE run failed; tail of $RUN/train.log:"; tail -30 "$RUN/train.log"; exit 1; }
echo "=== $(date '+%T') $MODE done: $(tail -1 "$RUN/train.log")"
ln -sfn "$MODE" "$OUT/latest"

if [[ $EVAL == 1 ]]; then
  echo "=== $(date '+%T') strict evaluation -> $OUT/eval, package draft -> $OUT/package_draft"
  loopflag=(); [[ $LOOP == 1 ]] && loopflag=(--loop)
  (cd .. && MUJOCO_GL=egl PYTHONPATH=train_pg train_pg/$PY eval_tracking.py "train_pg/$RUN/policy.onnx" "$CLIP" "${loopflag[@]}" \
      --name "$NAME" --out-dir "train_pg/$OUT" --package 2>&1 | tee "train_pg/$OUT/score")
fi
echo "=== $(date '+%T') all done: $OUT"
