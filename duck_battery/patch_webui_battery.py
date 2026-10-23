#!/usr/bin/env python3
"""Show the low-battery banner on the web page even while paused (red, like FALLEN).
    python3 patch_webui_battery.py <path to webui/index.html>"""
import sys

path = sys.argv[1]
s = open(path).read()
if "BATTERY" in s:
    sys.exit("already patched")
old = '  else if(s.paused){cls="pause"; txt="❙❙ PAUSED";}\n'
new = ('  else if(s.message && s.message.indexOf("BATTERY")>=0){cls="fall"; txt=s.message;}\n' + old)
if s.count(old) != 1:
    sys.exit("could not find the banner code")
open(path, "w").write(s.replace(old, new))
print("patched", path)
