#!/usr/bin/env python3
"""Add the "Feel" sliders to the web UI Tune panel (walk mode).
    python3 patch_webui_feel.py ~/Open_Duck_Mini_Runtime/mini_bdx_runtime/mini_bdx_runtime/webui/index.html
"""
import sys

path = sys.argv[1]
s = open(path).read()
if "FEEL_CONTROLS" in s:
    sys.exit("already patched")

EDITS = [
    ("""const CAM_CONTROLS=[""", """const FEEL_CONTROLS=[
  {k:"leg_kp",name:"Leg stiffness",min:8,max:32,step:1,
   help:"How hard the leg servos push back when pulled off target (servo P gain, stock 30). <b>Lower</b> = softer, springier, absorbs bumps, but sags and lags. <b>Higher</b> = stiffer and more precise, but more jitter."},
  {k:"head_kp",name:"Head stiffness",min:2,max:32,step:1,
   help:"Same for the 4 head servos (stock 8). <b>Lower</b> = floppier, bobbing head; <b>higher</b> = head held firmly, but its swings shove the body more."},
  {k:"ankle_gain",name:"Ankle gain",min:0.5,max:2,step:0.05,unit:"×",
   help:"Multiplies how much the brain moves the ankles. <b>Higher</b> = he balances and pushes off more with his ankles; <b>lower</b> = flatter-footed. The brain was trained at 1.0, so go in small steps."},
  {k:"knee_gain",name:"Knee gain",min:0.5,max:1.5,step:0.05,unit:"×",
   help:"Same for the knees. <b>Higher</b> = higher foot lift, bouncier; <b>lower</b> = shuffling."},
  {k:"hip_gain",name:"Hip gain",min:0.5,max:1.5,step:0.05,unit:"×",
   help:"Same for the hip pitch (leg swing front-back). <b>Higher</b> = longer strides; <b>lower</b> = shorter."},
  {k:"smoothing",name:"Smoothing",min:0,max:0.8,step:0.05,
   help:"Blends each new motor target with the last one. <b>Higher</b> = softer, rounder moves; too high and he reacts late and can wobble. 0 = off (stock)."},
  {k:"ankle_trim_deg",name:"Ankle lean trim",min:-8,max:8,step:0.5,unit:"°",
   help:"Tilts both feet together. <b>+</b> = he leans forward, <b>−</b> = back. Fix a duck that keeps tipping one way. Saving writes it into the ankle offsets and the slider goes back to 0."},
];
const CAM_CONTROLS=["""),
    ("""    html+=`<div class="tsec"><span class="lbl">Walk tuning</span>`;""",
     """    html+=`<div class="tsec"><span class="lbl">Feel</span>`;
    FEEL_CONTROLS.forEach(spec=>{ const v=readWalk(spec.k,wp,gov); html+=sliderRow("walk",spec,v); });
    html+=`<div class="thint">Live. Saved with the walk tuning below. Reset puts these back to stock.</div></div>`;
    html+=`<div class="tsec"><span class="lbl">Walk tuning</span>`;"""),
]
for old, new in EDITS:
    if s.count(old) != 1:
        sys.exit(f"could not find exactly one copy of:\n{old}")
    s = s.replace(old, new)
open(path, "w").write(s)
print("patched", path)
