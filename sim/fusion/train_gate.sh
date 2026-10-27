#!/bin/bash
# Train the stand/walk gate over two frozen tracking experts (train_pg/playground/open_duck_mini_v2/fusion_gate.py).
#
#   sim/fusion/train_gate.sh <name> <stand.onnx> <walk.onnx> [--steps N] [--smoke]
#
# Clips: tracker/data/clips/Stand.json and Walk.json (override with STAND_CLIP / WALK_CLIP).
# Output: sim/train_pg/runs_fusion/<name>/ (train.log, tfevents, policy.onnx = the gate, gate_spec.json, run.json).
# GPU: check nvidia-smi first (a tracking run and this do not fit on the 4060 together).
set -eo pipefail
[[ $# -ge 3 ]] || { sed -n 2,8p "$0"; exit 1; }
NAME=$1; STAND=$(realpath "$2"); WALK=$(realpath "$3"); shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
STAND_CLIP=$(realpath "${STAND_CLIP:-$HERE/../../tracker/data/clips/Stand.json}")
WALK_CLIP=$(realpath "${WALK_CLIP:-$HERE/../../tracker/data/clips/Walk.json}")
STEPS=""; SMOKE=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --steps) STEPS=$2; shift ;;
    --smoke) SMOKE=1 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
  shift
done
cd "$HERE/../train_pg"
PY=.venv/bin/python
export PYTHONPATH=. STAND_ONNX=$STAND WALK_ONNX=$WALK STAND_CLIP WALK_CLIP GATE_PRIOR=${GATE_PRIOR:-none}
eval "export $($PY -c 'import recipe_v1 as R; print(" ".join(f"{k}={v}" for k, v in R.as_env_vars().items()))')"
OUT=runs_fusion/$NAME
mkdir -p "$OUT"
[[ -n "$STEPS" ]] && export NUM_TIMESTEPS=$STEPS
if [[ $SMOKE == 1 ]]; then
  export NUM_ENVS=64 NUM_EVALS=1 BATCH_SIZE=64 NUM_MINIBATCHES=4 NUM_TIMESTEPS=25600
fi
echo "=== $(date '+%F %T') gate $NAME  stand $STAND  walk $WALK  steps ${NUM_TIMESTEPS:-default}"
$PY playground/open_duck_mini_v2/fusion_gate_runner.py --output_dir "$OUT" > "$OUT/train.log" 2>&1 \
  || { echo "gate run failed; tail of $OUT/train.log:"; tail -30 "$OUT/train.log"; exit 1; }
echo "=== $(date '+%T') done: $(tail -1 "$OUT/train.log")"
