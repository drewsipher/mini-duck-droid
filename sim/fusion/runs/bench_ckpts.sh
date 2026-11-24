#!/bin/bash
# bench every eval checkpoint of recovery_push05_s50: kicks at 0.4/0.5 and 0.2 s force pushes at 0.3/0.4, 8 directions, 20 runs each
cd "$(dirname "$0")/../.."
export MUJOCO_GL=egl PYTHONPATH=train_pg
PY=train_pg/.venv/bin/python
RUN=train_pg/runs_skills/recovery_push05_s50/full
for ck in $RUN/*_66846720.onnx $RUN/*_83558400.onnx $RUN/*_100270080.onnx $RUN/*_116981760.onnx $RUN/*_133693440.onnx $RUN/policy.onnx; do
  tag=$(basename "$ck" .onnx); tag=${tag##*_}; [[ $tag == policy ]] && tag=150405120
  out=fusion/runs/recovery_ckpts/$tag; mkdir -p "$out"
  echo "=== $(date +%T) checkpoint $tag"
  $PY fusion/bench.py push-dir --stand ../skills/stand/v1/policy.onnx --walk ../skills/walk/v1/policy.onnx --recovery "$ck" --out "$out" --episodes 20 --procs 8 --only recovery --mags 0.4 0.5 2>&1 | grep -E "^recovery|wrote"
  $PY fusion/bench.py push-dir --stand ../skills/stand/v1/policy.onnx --walk ../skills/walk/v1/policy.onnx --recovery "$ck" --out "$out" --episodes 20 --procs 8 --only recovery --mags 0.3 0.4 --force-s 0.2 2>&1 | grep -E "^recovery|wrote"
done
echo "=== $(date +%T) stand expert, same tests"
out=fusion/runs/recovery_ckpts/stand; mkdir -p $out
$PY fusion/bench.py push-dir --stand ../skills/stand/v1/policy.onnx --walk ../skills/walk/v1/policy.onnx --out "$out" --episodes 20 --procs 8 --only stand --mags 0.3 0.4 --force-s 0.2 2>&1 | grep -E "^stand|wrote"
echo "=== $(date +%T) done"
