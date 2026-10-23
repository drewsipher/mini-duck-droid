#!/usr/bin/env bash
# Nightly regression (gate G5 rule: every release re-runs every skill's battery).
# For every skills/<name>/v<N>/: validate, battery, sim-to-sim; writes regression/<YYYY-MM-DD>.md
# (table: skill, version, type, battery ratio, sim2sim, status, verdict) plus a one-line summary.
#
#   regression/run.sh                    # full run (5 seeds x 10 episodes per test)
#   SEEDS=2 EPISODES=3 regression/run.sh # quick
#   ONLY=walk regression/run.sh          # one skill
#
# Safe to run from cron: a lock stops overlapping runs, each step has a timeout, the CPU work is
# niced, nothing touches the GPU beyond EGL off-screen rendering of a few short clips (set
# MUJOCO_GL=osmesa or NO_VIDEO=1 to avoid even that), and nothing is ever written outside
# regression/ and the packages' own report/ folders. Exit 0 = everything passed, 2 = at least one
# package failed a gate, 1 = a tool error (log in regression/logs/<date>/).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/sim/.venv/bin/python"
SEEDS="${SEEDS:-5}"
EPISODES="${EPISODES:-10}"
ONLY="${ONLY:-}"
DATE="$(date +%F)"
STAMP="$(date +%FT%T%z)"
OUT="$ROOT/regression/$DATE.md"
LOGDIR="$ROOT/regression/logs/$DATE"
mkdir -p "$LOGDIR"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
VIDEO_FLAG=""
[ -n "${NO_VIDEO:-}" ] && VIDEO_FLAG="--no-video"

exec 9>"$ROOT/regression/.lock"
if ! flock -n 9; then
  echo "regression: another run holds regression/.lock; exiting"
  exit 3
fi
if [ ! -x "$PY" ]; then
  echo "regression: $PY missing"
  exit 1
fi

cd "$ROOT/sim" || exit 1
t_start=$(date +%s)
rows=()
n=0; n_pass=0; n_bat=0; n_s2s=0; n_err=0; n_invalid=0

json_get() {   # json_get <file> <python expression on d>
  "$PY" - "$1" "$2" <<'EOF' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    v = eval(sys.argv[2], {}, {"d": d})
    print("" if v is None else v)
except Exception:
    print("")
EOF
}

for dir in "$ROOT"/skills/*/v*/; do
  [ -f "$dir/skill.json" ] || continue
  dir="${dir%/}"
  name="$(basename "$(dirname "$dir")")"
  ver="$(basename "$dir")"
  if [ -n "$ONLY" ] && [ "$name" != "$ONLY" ]; then continue; fi
  n=$((n + 1))
  log="$LOGDIR/${name}_${ver}"
  type="$(json_get "$dir/skill.json" 'd.get("type")')"
  echo "== $name/$ver ($type)"

  # 1. validate the package as it is (its battery report may not exist yet on a first run)
  nice -n 10 "$PY" "$ROOT/tools/skill_package.py" validate "$dir" --no-battery >"$log.validate.log" 2>&1
  if [ $? -ne 0 ]; then
    n_invalid=$((n_invalid + 1))
    status="$(json_get "$dir/skill.json" 'd.get("status")')"
    rows+=("| $name | $ver | $type | - | - | ${status:-?} | INVALID (see logs) |")
    echo "   invalid: $(grep -m1 'error:' "$log.validate.log")"
    continue
  fi

  # 2. battery
  nice -n 10 timeout 2400 "$PY" "$ROOT/tools/battery.py" "$dir" --seeds "$SEEDS" --episodes "$EPISODES" $VIDEO_FLAG >"$log.battery.log" 2>&1
  b=$?
  bat_ratio="$(json_get "$dir/report/battery.json" 'd.get("overall_ratio")')"
  bat_n="$(json_get "$dir/report/battery.json" 'd.get("overall")')"
  bat_pass="$(json_get "$dir/report/battery.json" 'd.get("passed")')"
  if [ "$b" -eq 1 ] || [ "$b" -ge 124 ]; then
    n_err=$((n_err + 1)); bat_cell="error (exit $b)"
  elif [ -z "$bat_ratio" ]; then
    n_err=$((n_err + 1)); bat_cell="no report"
  else
    bat_cell="$bat_ratio ($bat_n)"
    [ "$bat_pass" = "True" ] && n_bat=$((n_bat + 1))
  fi

  # 3. sim-to-sim (writes match=null when the package's observation needs the motion-graph runtime)
  nice -n 10 timeout 900 "$PY" "$ROOT/tools/sim2sim.py" "$dir" $VIDEO_FLAG >"$log.sim2sim.log" 2>&1
  s=$?
  s2s_match="$(json_get "$dir/report/sim2sim.json" 'd.get("match")')"
  s2s_rms="$(json_get "$dir/report/sim2sim.json" 'd.get("rms_target_diff_rad")')"
  s2s_al="$(json_get "$dir/report/sim2sim.json" 'd.get("rms_clock_aligned_rad")')"
  if [ "$s" -eq 1 ] || [ "$s" -ge 124 ]; then
    n_err=$((n_err + 1)); s2s_cell="error (exit $s)"
  elif [ -z "$s2s_match" ]; then
    s2s_cell="n/a"
  elif [ "$s2s_match" = "True" ]; then
    s2s_cell="match ($s2s_rms rad)"; n_s2s=$((n_s2s + 1))
  else
    s2s_cell="MISMATCH ($s2s_rms rad; clock-aligned $s2s_al)"
  fi

  # 4. final validity and verdict
  nice -n 10 "$PY" "$ROOT/tools/skill_package.py" validate "$dir" >>"$log.validate.log" 2>&1
  valid=$?
  status="$(json_get "$dir/skill.json" 'd.get("status")')"
  if [ "$valid" -ne 0 ]; then
    verdict="INVALID after run"
  elif [ "$bat_pass" = "True" ] && { [ "$s2s_match" = "True" ] || [ -z "$s2s_match" ]; }; then
    verdict="pass"; n_pass=$((n_pass + 1))
  elif [ "$bat_pass" = "True" ]; then
    verdict="battery ok, sim2sim failed"
  else
    verdict="battery FAILED"
  fi
  rows+=("| $name | $ver | $type | $bat_cell | $s2s_cell | ${status:-?} | $verdict |")
  echo "   battery $bat_cell, sim2sim $s2s_cell, status $status -> $verdict"
done

t_end=$(date +%s)
mins=$(( (t_end - t_start + 30) / 60 ))
summary="$n packages: $n_pass passed both gates, $n_bat passed the battery, $n_s2s matched sim-to-sim, $n_invalid invalid, $n_err tool errors; $SEEDS seeds x $EPISODES episodes; ${mins} min on $STAMP."

{
  echo "# Regression $DATE"
  echo
  echo "$summary"
  echo
  echo "| skill | version | type | battery ratio | sim2sim | status | verdict |"
  echo "|---|---|---|---|---|---|---|"
  for r in "${rows[@]}"; do echo "$r"; done
  echo
  echo "Battery = success/total over all tests (tools/battery.py, thresholds in each skill.json). sim2sim = lockstep target RMS between the upstream runtime code and sim_test (tools/sim2sim.py). Logs: regression/logs/$DATE/."
} >"$OUT"

echo "$summary"
echo "wrote $OUT"
if [ "$n_err" -gt 0 ] || [ "$n_invalid" -gt 0 ]; then exit 1; fi
if [ "$n_pass" -lt "$n" ]; then exit 2; fi
exit 0
