#!/usr/bin/env bash
# Link the Duck Tools panel into Blender's add-ons folder (Blender then needs "Duck Tools" enabled once in Preferences > Add-ons).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
V="${1:-5.2}"
mkdir -p ~/.config/blender/$V/scripts/addons
ln -sfn "$HERE/duck_tools_addon.py" ~/.config/blender/$V/scripts/addons/duck_tools.py
echo "linked into ~/.config/blender/$V/scripts/addons/duck_tools.py"
