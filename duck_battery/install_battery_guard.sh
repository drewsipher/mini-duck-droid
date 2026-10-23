#!/bin/bash
# Install the low-battery failsafe on the duck (run ON the duck, from the folder holding these files).
# Backs up everything it changes with a timestamp.
set -e
STAMP=$(date +%Y%m%d-%H%M%S)
SCRIPTS=~/Open_Duck_Mini_Runtime/scripts
WALK=$SCRIPTS/v2_rl_walk_mujoco.py
WEBUI=$(python3 - <<'PY'
import glob, os
c = glob.glob(os.path.expanduser("~/Open_Duck_Mini_Runtime/**/webui/index.html"), recursive=True)
print(c[0] if c else "")
PY
)
cp "$WALK" ~/v2_rl_walk_mujoco.py.bak-$STAMP
cp battery_guard.py $SCRIPTS/battery_guard.py
python3 patch_walk_battery.py "$WALK"
if [ -n "$WEBUI" ]; then
  cp "$WEBUI" ~/webui_index.html.bak-$STAMP
  python3 patch_webui_battery.py "$WEBUI" || true
fi
~/.virtualenvs/open-duck-mini-runtime/bin/python -m py_compile "$WALK" $SCRIPTS/battery_guard.py
echo "backups: ~/v2_rl_walk_mujoco.py.bak-$STAMP ~/webui_index.html.bak-$STAMP"
if sudo -n -l /usr/sbin/poweroff >/dev/null 2>&1; then
  echo "poweroff without a password: OK"
else
  echo "poweroff needs a sudo rule. Run once (asks your password):"
  echo "  echo \"$USER ALL=(root) NOPASSWD: /usr/sbin/poweroff\" | sudo tee /etc/sudoers.d/duck-poweroff"
fi
