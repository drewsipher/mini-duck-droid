#!/usr/bin/env bash
# Open the duck rig in Blender on the machine's desktop (display :1) with the MCP server running,
# so Claude can drive it and Drew can animate.   anim/blender.sh [clip.json]
# If a clip is given it is imported as an action on the rig after the file opens.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BLENDER="${BLENDER:-$HOME/apps/blender-5.2.2-linux-x64/blender}"
export DISPLAY="${DISPLAY:-:1}"
LOG="$HERE/blender.log"
if ss -ltn 2>/dev/null | grep -q ':9876 '; then
  echo "Blender is already running with the MCP server on :9876."
  if [ $# -ge 1 ]; then
    "$HERE/../sim/.venv/bin/python" "$HERE/blender_remote.py" import "$1" || echo "could not send the clip to the running Blender"
  fi
  exit 0
fi
EXPR="import bpy, sys; sys.path.insert(0, '$HERE'); bpy.ops.blmcp.server_start()"
if [ $# -ge 1 ]; then
  CLIP="$(readlink -f "$1")"
  EXPR="$EXPR; import blender_duck as bd; bd.import_clip('$CLIP')"
fi
nohup "$BLENDER" "$HERE/duck.blend" --python-expr "$EXPR" > "$LOG" 2>&1 &
echo "Blender started (pid $!), log $LOG"
