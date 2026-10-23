#!/usr/bin/env bash
# One read-only servo health sample from the duck, appended to mechatronics/telemetry/.
#
#   mechatronics/telemetry_pull.sh            # real duck: scp servo_health.py to the Pi, run it, append
#   mechatronics/telemetry_pull.sh --fake     # no hardware: run servo_health.py --fake here, append to
#                                             # telemetry/sample_fake.jsonl (never the dated files)
#   PI=bdxv2@192.168.50.126 mechatronics/telemetry_pull.sh    # another host
#
# It refuses to touch the bus if anything holds /dev/ttyACM0 on the Pi (fuser + known program
# names), and servo_health.py checks again itself. Nothing is started or stopped on the Pi;
# the only file left there is ~/servo_health.py.
# Exit codes: 0 sample appended, 1 Pi unreachable, 2 port busy, 3 read failed, 4 bad line.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI="${PI:-bdxv2@bdxv2.local}"
PI_PY="${PI_PY:-~/.virtualenvs/open-duck-mini-runtime/bin/python}"
OUT_DIR="$HERE/telemetry"
mkdir -p "$OUT_DIR"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new "$PI")

summarize() {   # $1 = jsonl line file (one line)
  python3 - "$1" <<'EOF'
import json, sys
r = json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])
s = r.get("summary", {})
lims = r.get("limits", {})
one = next(iter(lims.values()), {})
flags = ", ".join(f"{k}:{len(v)}" for k, v in s.get("flags", {}).items()) or "none"
print(f"{r['t']}  {'FAKE ' if r.get('fake') else ''}{s.get('answered')}/14 servos  "
      f"V {s.get('v_min')}-{s.get('v_max')}  Tmax {s.get('t_max')} C ({s.get('t_max_joint')})  "
      f"I {s.get('i_total_a')} A  torque_on {s.get('any_torque_on')}  flags {flags}  "
      f"limits min_v {one.get('min_v')} max_v {one.get('max_v')} max_temp {one.get('max_temp')} "
      f"Iprot {one.get('protection_current_a')} A unload_on_voltage {s.get('unload_on_voltage')}"
      + ("" if s.get("limits_identical", True) else "  NOTE: limits differ between servos"))
EOF
}

if [[ "${1:-}" == "--fake" ]]; then
  TMP="$(mktemp)"
  if ! python3 "$HERE/servo_health.py" --fake > "$TMP"; then echo "fake read failed"; rm -f "$TMP"; exit 3; fi
  cat "$TMP" >> "$OUT_DIR/sample_fake.jsonl"
  echo -n "appended to telemetry/sample_fake.jsonl: "; summarize "$TMP"; rm -f "$TMP"; exit 0
fi

# 1. reachable?
if ! "${SSH[@]}" true 2>/dev/null; then
  echo "Pi $PI unreachable over ssh (no sample taken)"; exit 1
fi
# 2. port free? (fuser prints PIDs and returns 0 when something holds the device)
BUSY="$("${SSH[@]}" "fuser /dev/ttyACM0 2>/dev/null; pgrep -af 'duck_agent.py|v2_rl_walk|mujoco_infer|strength_test|wave_power' | grep -v -E 'pgrep|servo_health'" 2>/dev/null)"
if [[ -n "$BUSY" ]]; then
  echo "serial port busy on the Pi, leaving it alone:"; echo "$BUSY" | sed 's/^/   /'; exit 2
fi
# 3. copy the reader (the only file we leave there) and run it
if ! scp -q -o BatchMode=yes -o ConnectTimeout=8 "$HERE/servo_health.py" "$PI:~/servo_health.py"; then
  echo "scp failed"; exit 1
fi
TMP="$(mktemp)"; ERR="$(mktemp)"
"${SSH[@]}" "$PI_PY ~/servo_health.py --port /dev/ttyACM0" > "$TMP" 2> "$ERR"
RC=$?
if [[ $RC -ne 0 ]]; then
  echo "servo_health.py exited $RC on the Pi:"; sed 's/^/   /' "$ERR"; rm -f "$TMP" "$ERR"
  [[ $RC -eq 3 ]] && exit 2 || exit 3
fi
if ! python3 -c "import json,sys; d=json.loads(open(sys.argv[1]).read()); assert d.get('source')=='servo_health'" "$TMP" 2>/dev/null; then
  echo "did not get a servo_health JSON line:"; head -c 400 "$TMP"; rm -f "$TMP" "$ERR"; exit 4
fi
DAY="$(date +%F)"
cat "$TMP" >> "$OUT_DIR/$DAY.jsonl"
echo -n "appended to telemetry/$DAY.jsonl: "; summarize "$TMP"
rm -f "$TMP" "$ERR"
