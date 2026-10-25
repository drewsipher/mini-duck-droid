# Animating the duck in Blender

Blender 5.2 is the authoring tool for every clip in the skill library. The Duck Puppet page is the
hardware console and dashboard; it plays the clips Blender exports.

## Open it
```
anim/blender.sh                                  # opens anim/duck.blend on the machine's display
anim/blender.sh tracker/data/clips/test_bow.json # ...and imports a clip as an action
```
### On another computer
1. `git clone https://github.com/drewsipher/mini-duck-droid.git`, then open `anim/duck.blend` in Blender 5.2.
2. Install the panel once: **Edit > Preferences > Add-ons**, the **v** menu at the top right, **Install from
   Disk...**, pick `anim/duck_tools_addon.py`, and tick **Duck Tools**. It finds everything else next to
   the open `duck.blend`, so always open the file from the clone.
3. For the **Check clip** button, Python 3 with MuJoCo:
   `python3 -m pip install mujoco numpy imageio imageio-ffmpeg` (Windows: `py -m pip install ...`).
   The panel uses `sim/.venv` if the clone has one, else `python3`/`python` on the PATH; set
   `DUCK_PYTHON` to point it at another. Without `imageio-ffmpeg` the check still runs, just without video.

The file has the **Duck** rig, the floor at z = 0, 50 fps (one frame = one control step), metres.
The sidebar (press N) has a **Duck** tab with every button below.

## The rig
- One bone per servo joint, named like the servo with `.L/.R` (`hip_yaw.L`, `knee.R`, `neck_pitch`,
  `head_yaw`, …). The joint angle is the bone's **Z rotation**, same zero and sign as the real duck.
  X and Y are locked, so you cannot pose a joint the servo cannot reach.
- `base` (red box) is the trunk. Move it to crouch, lean or walk; it is exported as the trunk track.
- `foot_ik.L` / `foot_ik.R` (red circles) place the feet. The **IK legs** slider (0..1) says how
  much the feet follow the circles; at 0 the leg bones are animated directly (FK).
- Joint limits from the MJCF are set as IK limits, so IK never bends a knee the wrong way.
- Blue = leg joints, green = head chain (always FK), red = controls.

## Buttons (sidebar > Duck)
| Button | What it does |
|---|---|
| Rebuild rig | Rebuilds bones and meshes from `anim/duck_model.json` (after `dump_mjcf.py`) |
| Home pose | The standing pose the robot starts from, feet on the floor |
| Snap IK targets | Moves the circles to where the feet are, so turning IK on changes nothing |
| Import clip | A clip JSON (puppet recording, policy recording, or an export) becomes an action |
| Export clip | Bakes the timeline (IK included) to a 50 Hz clip JSON |
| Check clip | Exports and runs the stage 2 gate in MuJoCo: floor penetration, self-collision, joint limits, servo speed, home pose at both ends. Result in the panel, video in `anim/reports/<action>/check.mp4` |
| Mirror pose / action | Left and right swapped with the duck's real sign conventions |
| Make loop | Cycles modifiers on every curve so a gait repeats |

## Rules that make a clip trainable
1. Episodic clips (bow, wiggle, hop) start and end in the home pose, within 0.15 rad per joint.
2. Cyclic clips (gaits, turns) end where they start; check them with "Cyclic clip" ticked.
3. Servo speed stays under 4 rad/s (the checker's `--max-vel`); the servos saturate above that.
4. Feet do not go through the floor when the base is animated; if you leave `base` alone, the
   checker and the trainer put the duck on the floor themselves.
5. Keep the head on the head chain only; the legs and trunk make the balance problem.

## From a clip to a skill
```
cd sim && MUJOCO_GL=egl .venv/bin/python ../anim/clip_check.py ../tracker/data/clips/<clip>.json --video ../anim/reports/<clip>/check.mp4
sim/train_skill.sh tracker/data/clips/<clip>.json <name> --preview     # ~30 min, GPU
sim/.venv/bin/python tools/skill_package.py import sim/train_pg/runs_skills/<name>/package_draft
```
Everything above is also a button on the dashboard (http://192.168.50.109:8430).

## Files
- `dump_mjcf.py` — MJCF → `duck_model.json` (bodies, joints, compiled meshes, home pose). Run with `sim/.venv`.
- `blender_duck.py` — runs inside Blender: `build()`, `import_clip()`, `export_clip()`, `mirror_action()`, `snap_ik_targets()`.
- `duck_tools_addon.py` — the sidebar panel (installed from disk, or linked into `~/.config/blender/5.2/scripts/addons/duck_tools.py` by `install_addon.sh`).
- `model/` — copy of the Open Duck Mini MJCF and meshes the checker uses when the training repo is not cloned (Apache 2.0).
- `blender_remote.py` — drive the running Blender from a shell: `import`, `export`, `snap`, `home`, `exec`.
- `clip_check.py` — the stage 2 gate (exit 0 pass / 1 fail, `--json`, `--video`, `--loop`).
- `make_test_clips.py` — placeholder head clips (`test_bow`, `test_look`) to exercise the pipeline.
- `make_walk.py` — generates a symmetric walking loop (trunk travels, planted feet stay flat, joints from IK on the MuJoCo model); `--stride`, `--period`, `--lift`, `--sway`… Import the result with `blender_duck.import_clip`.
