#!/bin/bash
# Scaffold curriculum: hold the duck up like a toddler, then let go a little per stage.
#
#   ./curriculum.sh            run every stage in order, each restored from the one before
#   ./curriculum.sh s2         start at s2 (restores from the newest checkpoint of s1)
#   STEPS=20000000 ./curriculum.sh s0    shorter stages
#
# Every stage shares the same smoothness settings (ACT_LPF_HZ, MAX_MOTOR_VEL, ACT_ACCEL_W):
# those are deployment settings, and the runtime has to run the same low-pass
# (cutoff_frequency = ACT_LPF_HZ). What changes per stage is how much it is held:
#
#   stage  gravity  body weight carried  upright spring  head carried  servos
#   s0     0.7      50-80%               3-6 N.m/rad     50-100%       model (kp 0.9-1.1, 3.2 N.m)
#   s1     0.85     20-50%               1-3             0-50%         model
#   s2     1.0      0-20%                0-1             0             model
#   s3     1.0      none                 none            none          real: 1.6-2.1 N.m, kp 1.1-1.8
#
# Attempt 2 (10-21) showed s1 -> s2 is too big a step: it lost gravity relief, most of the
# support and the balance spring at once and plateaued (steps in place, falls to pushes).
# The s2a..s2d ladder replaces s2: full weight first, balance help kept and then faded,
# like holding a toddler's hands before letting go of them.
#
#   s2a    1.0      20-40%               3-6             0-30%         model
#   s2b    1.0      5-20%                2-4             0-15%         model
#   s2c    1.0      0-10%                1-2             0             model
#   s2d    1.0      0                    0-1             0             model
#
#   RUNS=runs3 RESTORE_FROM=runs/cur_s1/<ckpt> STAGES="s2a s2b s2c s2d s3" ./curriculum.sh s2a
#
# Attempt 3 (10-22 02:00) carried the gait down to a spring of 1 but never push recovery:
# with no spring every episode ended at a push and it marched in place. Round 4 is the
# user's "leading hand": a weak horizontal spring (HAND, 5-10 % of weight) to a hand point
# walking ahead at the commanded speed, the upright spring kept (the hand alone tips it over
# at any strength, 15 %+ drags it), HAND_ROLL rocking it onto the stance foot, and the
# episode ends when the hand gets HAND_MAX_LEAD ahead (HAND_DROP), so standing on a forward
# command now loses the alive reward instead of earning it. Real servos from the start,
# pushes capped at 0.4 m/s, the hand-animated reference gait dropped (IMITATION_W=0).
#
#   RUNS=runs4 RESTORE_FROM=runs3/cur_s2a/<ckpt> STAGES="h1" ./curriculum.sh h1
#   RUNS=runs5 RESTORE_FROM=runs3/cur_s2a/<ckpt> STAGES="h2 h3" ./curriculum.sh h2
#
# "body weight carried" is the fraction of the whole duck's weight the scaffold takes;
# "head carried" only decides how much of it is applied at the head (the strings) instead
# of the trunk (the jumper), so the two never add up to more than SUPPORT.
# Each range is sampled per episode, so s1 already contains s2's easiest cases and so on.
# After each stage eval_policy.py scores the newest .onnx with no scaffold and real servo
# torque (1.86 N.m), so you can see what it does when nobody is holding it.
set -eo pipefail
cd "$(dirname "$0")/train_pg"
PY=.venv/bin/python
export PYTHONPATH=.
trap 'echo "=== stage $s failed; last lines of $RUNS/cur_$s.log:"; tail -25 "$RUNS/cur_$s.log" 2>/dev/null' ERR
STEPS=${STEPS:-100000000}
START=${1:-s0}
RUNS=${RUNS:-runs}            # RUNS=runs_smoke for a trial run
STAGES=(${STAGES:-s0 s1 s2 s3})

# Shared by every stage.
export HEAD_FROM_CMD=1 HEAD_ZERO_P=0.3 CMD_AXIS_ZERO=0.3 MASS_SCALE=0.72,1.0
export ACT_LPF_HZ=${ACT_LPF_HZ:-8} MAX_MOTOR_VEL=${MAX_MOTOR_VEL:-4.0} ACT_ACCEL_W=${ACT_ACCEL_W:--0.3}
export NUM_ENVS=${NUM_ENVS:-4096} NUM_EVALS=${NUM_EVALS:-6}

CLIPS=$(cd .. && pwd)/../tracker/data/clips
stage_env() {
  case $1 in
    s0) echo "GRAVITY_SCALE=0.7  SUPPORT=0.5,0.8 UPRIGHT_K=3,6 HEAD_SUPPORT=0.5,1" ;;
    s1) echo "GRAVITY_SCALE=0.85 SUPPORT=0.2,0.5 UPRIGHT_K=1,3 HEAD_SUPPORT=0,0.5" ;;
    s2) echo "GRAVITY_SCALE=1.0  SUPPORT=0,0.2   UPRIGHT_K=0,1 HEAD_SUPPORT=0" ;;
    s2a) echo "GRAVITY_SCALE=1.0 SUPPORT=0.2,0.4 UPRIGHT_K=3,6 HEAD_SUPPORT=0,0.3" ;;
    s2b) echo "GRAVITY_SCALE=1.0 SUPPORT=0.05,0.2 UPRIGHT_K=2,4 HEAD_SUPPORT=0,0.15" ;;
    s2c) echo "GRAVITY_SCALE=1.0 SUPPORT=0,0.1 UPRIGHT_K=1,2 HEAD_SUPPORT=0" ;;
    s2d) echo "GRAVITY_SCALE=1.0 SUPPORT=0,0 UPRIGHT_K=0,1 HEAD_SUPPORT=0" ;;
    s3) echo "GRAVITY_SCALE=1.0  TORQUE_RANGE=1.6,2.1 KP_RANGE=1.1,1.8" ;;
    h1) echo "GRAVITY_SCALE=1.0 UPRIGHT_K=3,6 HAND=0.03,0.1 HAND_ROLL=0,0.1 HAND_MAX_LEAD=0.3 HAND_DROP=1 IMITATION_W=0 PUSH_MAX=0.4 TORQUE_RANGE=1.6,2.1 KP_RANGE=1.1,1.8" ;;
    # Round 5 (10-22 10:xx, from s2a again): h1 froze, see joystick.py HAND_LEAD_W. Dense lead
    # cost, drop moved out to 1 m, model servos first (h2) and the measured servos after (h3).
    h2) echo "GRAVITY_SCALE=1.0 UPRIGHT_K=3,6 HAND=0.03,0.1 HAND_ROLL=0,0.1 HAND_MAX_LEAD=1.0 HAND_DROP=1 HAND_LEAD_W=-5 IMITATION_W=0 PUSH_MAX=0.4" ;;
    h3) echo "GRAVITY_SCALE=1.0 UPRIGHT_K=3,6 HAND=0.03,0.1 HAND_ROLL=0,0.1 HAND_MAX_LEAD=1.0 HAND_DROP=1 HAND_LEAD_W=-5 IMITATION_W=0 PUSH_MAX=0.4 TORQUE_RANGE=1.6,2.1 KP_RANGE=1.1,1.8" ;;
    # Round 6 (10-22 11:xx): h2 froze like h1. The user's next idea: let the ankles pop
    # (ANKLE_LPF_HZ / ANKLE_MAX_VEL / ANKLE_COST_SCALE), hips and knees stay smooth. Same
    # hand, spring and lead cost as h2; a1 model servos, a2 the measured ones.
    a1) echo "GRAVITY_SCALE=1.0 UPRIGHT_K=3,6 HAND=0.03,0.1 HAND_ROLL=0,0.1 HAND_MAX_LEAD=1.0 HAND_DROP=1 HAND_LEAD_W=-5 IMITATION_W=0 PUSH_MAX=0.4 ANKLE_LPF_HZ=20 ANKLE_MAX_VEL=8 ANKLE_COST_SCALE=0.3" ;;
    a2) echo "GRAVITY_SCALE=1.0 UPRIGHT_K=3,6 HAND=0.03,0.1 HAND_ROLL=0,0.1 HAND_MAX_LEAD=1.0 HAND_DROP=1 HAND_LEAD_W=-5 IMITATION_W=0 PUSH_MAX=0.4 ANKLE_LPF_HZ=20 ANKLE_MAX_VEL=8 ANKLE_COST_SCALE=0.3 TORQUE_RANGE=1.6,2.1 KP_RANGE=1.1,1.8" ;;
    # Round 7 (10-22 15:xx): the user's puppeteered stride (tracker clip Puppet_walk_test1,
    # 1.88 s, both legs, loops) as the imitation reference. Held by the spring and a little
    # lift, real servos, forward commands only, legs allowed +-0.5 rad, ankles freer.
    # p2 (10-22 evening): the hybrid reference. The puppet stride for forward, s2a's own turning
    # (recorded in the sim by record_policy_clip.py, 56 / 53 deg/s, the smoothest turn so far)
    # for left and right. No side-step or backward example exists, so those commands are off.
    p2) echo "GRAVITY_SCALE=1.0 UPRIGHT_K=3,6 SUPPORT=0,0.2 REF_CLIP=$CLIPS/Puppet_walk_test1.json REF_TRIM=auto REF_BLEND=0.15 REF_HYBRID=1 REF_TURN_LEFT=$CLIPS/s2a_turn_left.json REF_TURN_RIGHT=$CLIPS/s2a_turn_right.json LEG_ACTION_SCALE=0.5 LIN_VEL_X_RANGE=0,0.2 LIN_VEL_Y=0 PUSH_MAX=0.4 ANKLE_LPF_HZ=20 ANKLE_MAX_VEL=8 ANKLE_COST_SCALE=0.3 TORQUE_RANGE=1.6,2.1 KP_RANGE=1.1,1.8" ;;
    p1) echo "GRAVITY_SCALE=1.0 UPRIGHT_K=3,6 SUPPORT=0,0.2 REF_CLIP=$(cd .. && pwd)/../tracker/data/clips/Puppet_walk_test1.json REF_TRIM=auto REF_BLEND=0.15 LEG_ACTION_SCALE=0.5 LIN_VEL_X_RANGE=0,0.2 LIN_VEL_Y=0 PUSH_MAX=0.4 ANKLE_LPF_HZ=20 ANKLE_MAX_VEL=8 ANKLE_COST_SCALE=0.3 TORQUE_RANGE=1.6,2.1 KP_RANGE=1.1,1.8" ;;
    *) echo "unknown stage $1" >&2; exit 1 ;;
  esac
}

by_step() {  # sort paths like .../2026_10_13_181627_100761600[.onnx] by that trailing step count
  awk -F_ '{n=$NF; sub(/\.onnx$/, "", n); print n, $0}' | sort -n | tail -1 | cut -d' ' -f2
}
newest_ckpt() {  # newest checkpoint directory of a run (the orbax dir, not the .onnx)
  ls -d "$RUNS/$1"/*_[0-9]* 2>/dev/null | grep -v '\.onnx$' | by_step
}

prev=""
for s in "${STAGES[@]}"; do
  if [[ "$s" < "$START" ]]; then prev=$s; continue; fi
  restore=()
  if [[ -n "$RESTORE_FROM" ]]; then      # explicit checkpoint for the first stage run
    restore=(--restore_checkpoint_path "$RESTORE_FROM"); RESTORE_FROM=""
  elif [[ -n "$prev" ]]; then
    ck=$(newest_ckpt "cur_$prev")
    [[ -n "$ck" ]] || { echo "no checkpoint for cur_$prev to restore from" >&2; exit 1; }
    restore=(--restore_checkpoint_path "$ck")
  fi
  stage_env "$s" > /dev/null || { echo "unknown stage $s, nothing launched" >&2; exit 1; }   # a typo must not start a run with default settings
  echo "=== $(date '+%H:%M') stage $s: $(stage_env $s) ${restore[*]}"
  mkdir -p "$RUNS/cur_$s"
  env $(stage_env $s) $PY playground/open_duck_mini_v2/runner.py \
      --task flat_terrain_backlash --num_timesteps "$STEPS" \
      --output_dir "$RUNS/cur_$s" "${restore[@]}" 2>&1 | tee "$RUNS/cur_$s.log" | grep -E "^STEP|^\[|Saving|Error|error"
  onnx=$(realpath "$(ls "$RUNS/cur_$s"/*.onnx | by_step)")
  echo "=== $(date '+%H:%M') stage $s done: $onnx — scoring without the scaffold at real servo torque"
  ankle=()   # per-joint ankle smoothing of the stage, if any, so the score uses the same filter
  for kv in $(stage_env $s); do
    case $kv in ANKLE_LPF_HZ=*) ankle+=(--ankle-cutoff "${kv#*=}") ;; ANKLE_MAX_VEL=*) ankle+=(--ankle-max-vel "${kv#*=}") ;; LEG_ACTION_SCALE=*) ankle+=(--leg-action-scale "${kv#*=}") ;; esac
  done
  (cd .. && env $(stage_env $s | tr ' ' '\n' | grep '^REF_' | tr '\n' ' ') $PY eval_policy.py "$onnx" --head-from-cmd --cutoff "$ACT_LPF_HZ" --max-motor-vel "$MAX_MOTOR_VEL" "${ankle[@]}" | tee "train_pg/$RUNS/cur_$s.score")
  prev=$s
done
