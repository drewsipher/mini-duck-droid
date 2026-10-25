"""Duck Tools: the N-panel in Blender's 3D view for animating the Open Duck Mini.

Install once: Edit > Preferences > Add-ons > (v menu) > Install from Disk... > this file
(anim/install_addon.sh symlinks it instead on Linux). It finds the rest of anim/ next to the open
duck.blend, so it works from any clone of the repo; DUCK_ANIM_DIR overrides that.
Everything heavy lives in anim/blender_duck.py; this file is only buttons and file dialogs.

Panel (View3D > sidebar > Duck):
  Rig:    Rebuild rig | Home pose | Snap IK | IK legs slider
  Clips:  Import clip | Export clip | Check clip (stage 2 gate in MuJoCo) | Open check video
  Pose:   Mirror pose | Mirror action | Make loop (cycle modifiers)
"""
bl_info = {
    "name": "Duck Tools",
    "author": "mini-duck-droid",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Duck",
    "description": "Rig, clip import/export and stage-2 checks for the Open Duck Mini",
    "category": "Animation",
}
import json
import os
import subprocess
import sys

import bpy
from bpy.props import BoolProperty, IntProperty, StringProperty

def _anim_dir():
    """anim/ of whichever clone is in use: DUCK_ANIM_DIR, else the open .blend's folder, else this file's."""
    cands = [os.environ.get("DUCK_ANIM_DIR"), os.path.dirname(bpy.data.filepath) if bpy.data.filepath else None,
             os.path.dirname(os.path.realpath(__file__))]
    for d in cands:
        if d and os.path.exists(os.path.join(d, "blender_duck.py")):
            return d
    raise RuntimeError("Duck Tools: open anim/duck.blend from the repo (or set DUCK_ANIM_DIR)")


def _paths():
    anim = _anim_dir()
    project = os.path.dirname(anim)
    return anim, project, os.path.join(project, "tracker", "data", "clips")


def _sim_python():
    """Python with mujoco for the checker: DUCK_PYTHON, else the repo's sim/.venv, else python on PATH."""
    project = _paths()[1]
    for c in (os.environ.get("DUCK_PYTHON"),
              os.path.join(project, "sim", ".venv", "bin", "python"),
              os.path.join(project, "sim", ".venv", "Scripts", "python.exe")):
        if c and os.path.exists(c):
            return c
    import shutil
    return shutil.which("python3") or shutil.which("python") or "python3"


def _open_file(path):
    if sys.platform.startswith("win"):
        os.startfile(path)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])


def _bd():
    anim = _anim_dir()
    if anim not in sys.path:
        sys.path.insert(0, anim)
    import importlib
    import blender_duck
    return importlib.reload(blender_duck)


def _arm():
    return bpy.data.objects.get("Duck")


class DUCK_OT_build(bpy.types.Operator):
    bl_idname = "duck.build"
    bl_label = "Rebuild rig"
    bl_description = "Rebuild the Duck armature and meshes from anim/duck_model.json (keeps actions)"

    def execute(self, context):
        bd = _bd()
        bd.build()
        self.report({"INFO"}, "Duck rig rebuilt")
        return {"FINISHED"}


class DUCK_OT_home(bpy.types.Operator):
    bl_idname = "duck.home"
    bl_label = "Home pose"
    bl_description = "Put the rig in the standing (home) pose and snap the foot IK targets to it"

    def execute(self, context):
        bd = _bd()
        m = json.load(open(bd.MODEL))
        arm = _arm()
        bd.set_pose(arm, m["home"]["joints"], base=m["home"])
        bd.snap_ik_targets(arm)
        return {"FINISHED"}


class DUCK_OT_snap(bpy.types.Operator):
    bl_idname = "duck.snap_ik"
    bl_label = "Snap IK targets"
    bl_description = "Move foot_ik.L/.R to where the FK pose puts the feet"

    def execute(self, context):
        _bd().snap_ik_targets(_arm())
        return {"FINISHED"}


class DUCK_OT_import(bpy.types.Operator):
    bl_idname = "duck.import_clip"
    bl_label = "Import clip"
    bl_description = "Load a clip JSON (puppet, recorded or exported) as an action on the rig"
    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = _paths()[2] + os.sep
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        act = _bd().import_clip(self.filepath, arm=_arm())
        self.report({"INFO"}, f"imported {act.name}: frames {context.scene.frame_start}-{context.scene.frame_end}")
        return {"FINISHED"}


class DUCK_OT_export(bpy.types.Operator):
    bl_idname = "duck.export_clip"
    bl_label = "Export clip"
    bl_description = "Bake the timeline (IK included) to a 50 Hz clip JSON for the checker, the trainer and the duck"
    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})
    use_range: BoolProperty(name="Scene frame range", default=True)
    start: IntProperty(name="Start", default=1)
    end: IntProperty(name="End", default=100)

    def invoke(self, context, event):
        if not self.filepath:
            arm = _arm()
            name = arm.animation_data.action.name if arm and arm.animation_data and arm.animation_data.action else "clip"
            self.filepath = os.path.join(_paths()[2], name.replace(".", "_") + ".json")
        self.start, self.end = context.scene.frame_start, context.scene.frame_end
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        s, e = (None, None) if self.use_range else (self.start, self.end)
        info = _bd().export_clip(self.filepath, start=s, end=e, arm=_arm())
        self.report({"INFO"}, f"exported {info['frames']} frames ({info['seconds']:.2f} s) to {self.filepath}")
        return {"FINISHED"}


class DUCK_OT_check(bpy.types.Operator):
    bl_idname = "duck.check_clip"
    bl_label = "Check clip"
    bl_description = "Export the timeline and run the stage-2 gate (anim/clip_check.py) in MuJoCo; result in a popup and anim/reports/<name>/"
    loop: BoolProperty(name="Cyclic clip (end = start)", default=False)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        arm = _arm()
        name = arm.animation_data.action.name if arm.animation_data and arm.animation_data.action else "timeline"
        name = name.replace(".", "_")
        anim, project, _ = _paths()
        out_dir = os.path.join(anim, "reports", name)
        os.makedirs(out_dir, exist_ok=True)
        clip = os.path.join(out_dir, "clip.json")
        _bd().export_clip(clip, arm=arm)
        for old in ("check.json", "check.mp4"):  # never show a stale result if the checker dies
            if os.path.exists(os.path.join(out_dir, old)):
                os.remove(os.path.join(out_dir, old))
        cmd = [_sim_python(), os.path.join(anim, "clip_check.py"), clip, "--json", os.path.join(out_dir, "check.json"),
               "--video", os.path.join(out_dir, "check.mp4")] + (["--loop"] if self.loop else [])
        env = dict(os.environ)
        if sys.platform.startswith("linux"):
            env.setdefault("MUJOCO_GL", "egl")
        try:
            p = subprocess.run(cmd, cwd=anim, env=env, capture_output=True, text=True, timeout=600)
        except OSError as e:
            self.report({"ERROR"}, f"could not start the checker ({cmd[0]}): {e}")
            return {"CANCELLED"}
        try:
            rep = json.load(open(os.path.join(out_dir, "check.json")))
        except Exception:
            out = p.stderr or p.stdout
            err = out[-400:]
            if "No module named" in out:
                err = f"{cmd[0]} is missing a package; run: {cmd[0]} -m pip install mujoco numpy imageio imageio-ffmpeg\n" + err
            self.report({"ERROR"}, f"checker failed: {err}")
            return {"CANCELLED"}
        context.scene["duck_last_check"] = json.dumps({"name": name, "pass": rep["pass"], "fails": rep["fails"],
                                                      "pen_mm": round(1000 * rep["floor_penetration_m"], 1),
                                                      "self": list(rep["self_collisions_m"].keys()),
                                                      "over_speed": rep["over_speed"], "limits": rep["limit_breach_rad"],
                                                      "video": os.path.join(out_dir, "check.mp4")})
        problems = _mark_problems(context.scene, rep, _bd().bone_name)
        msg = ("PASS" if rep["pass"] else "FAIL: " + ", ".join(rep["fails"])) + f"  (penetration {1000 * rep['floor_penetration_m']:.1f} mm)"
        if problems:
            msg += "  |  " + "; ".join(problems) + "  (see the markers on the timeline)"
        if "video_error" in rep:
            msg += f"  [no video: {cmd[0]} -m pip install imageio imageio-ffmpeg]"
        self.report({"INFO"} if rep["pass"] else {"WARNING"}, msg)
        return {"FINISHED"}


MARK = "x "   # prefix of the timeline markers Check clip owns; it clears and redraws them every run


def _runs(frames):
    """Consecutive frame indices grouped into (first, last) runs."""
    out = []
    for f in sorted(frames):
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def _mark_problems(scene, rep, bone_name):
    """Timeline markers at the frames the checker flagged (one per run of frames), plus a short text list.
    Clip frame index 0 is the scene's first frame, which is where Export and Check start."""
    for m in [m for m in scene.timeline_markers if m.name.startswith(MARK)]:
        scene.timeline_markers.remove(m)
    f0 = scene.frame_start
    items = []                                  # (index, marker text, list text)
    def span(a, z):
        return f"{f0 + a}" + (f"-{f0 + z}" if z > a else "")
    for j, idx in rep.get("over_speed_frames", {}).items():
        b, runs = bone_name(j), _runs(idx)
        for a, z in runs:
            items.append((a, f"{MARK}fast {b}", None))
        items.append((runs[0][0], None, f"{b} too fast (peak {rep['over_speed'][j]} rad/s) at frames {', '.join(span(a, z) for a, z in runs)}"))
    for j, idx in rep.get("limit_breach_frames", {}).items():
        b = bone_name(j)
        for a, z in _runs(idx):
            items.append((a, f"{MARK}limit {b}", f"{b} past its limit at frames {span(a, z)}"))
    pen = {int(k): v for k, v in rep.get("floor_penetration_frames", {}).items()}
    for a, z in _runs(pen):
        deep = max(pen[k] for k in range(a, z + 1))
        items.append((a, f"{MARK}floor {1000 * deep:.0f}mm", f"through the floor {1000 * deep:.1f} mm at frames {span(a, z)}"))
    hits = {int(k): v for k, v in rep.get("self_collision_frames", {}).items()}
    for a, z in _runs(hits):
        pairs = sorted({p for k in range(a, z + 1) for p in hits[k]})
        items.append((a, f"{MARK}hit {pairs[0]}", f"{', '.join(pairs)} collide at frames {span(a, z)}"))
    for a, name, _ in items:
        if name:
            scene.timeline_markers.new(name, frame=f0 + a)
    return [t for a, _, t in sorted(items, key=lambda i: i[0]) if t]


class DUCK_OT_open_video(bpy.types.Operator):
    bl_idname = "duck.open_video"
    bl_label = "Open check video"
    bl_description = "Open the last check's MuJoCo playback video"

    def execute(self, context):
        try:
            v = json.loads(context.scene.get("duck_last_check", "{}")).get("video")
        except Exception:
            v = None
        if not v or not os.path.exists(v):
            self.report({"WARNING"}, "no check video yet")
            return {"CANCELLED"}
        _open_file(v)
        return {"FINISHED"}


class DUCK_OT_mirror_pose(bpy.types.Operator):
    bl_idname = "duck.mirror_pose"
    bl_label = "Mirror pose"
    bl_description = "Swap left and right with the duck's real sign conventions (hip pitch/roll/yaw flip, knee/ankle don't)"

    def execute(self, context):
        _bd().mirror_pose(_arm())
        return {"FINISHED"}


class DUCK_OT_mirror_action(bpy.types.Operator):
    bl_idname = "duck.mirror_action"
    bl_label = "Mirror action"
    bl_description = "Create <action>.mirror: the same motion done by the other side"

    def execute(self, context):
        new = _bd().mirror_action(_arm())
        self.report({"INFO"}, f"created {new.name}")
        return {"FINISHED"}


class DUCK_OT_loop(bpy.types.Operator):
    bl_idname = "duck.make_loop"
    bl_label = "Make loop"
    bl_description = ("Add a Cycles modifier to every F-curve of the current action so it repeats; the trunk and "
                      "foot target locations repeat with offset, so a walk keeps travelling instead of snapping back")

    def execute(self, context):
        arm = _arm()
        act = arm.animation_data.action if arm.animation_data else None
        if not act:
            return {"CANCELLED"}
        n = 0
        for fc in _bd().fcurves(act):
            mod = next((m for m in fc.modifiers if m.type == "CYCLES"), None)
            if mod is None:
                mod = fc.modifiers.new("CYCLES")
                n += 1
            travels = fc.data_path.endswith(".location") and ('"base"' in fc.data_path or '"foot_ik.' in fc.data_path)
            mod.mode_before = mod.mode_after = "REPEAT_OFFSET" if travels else "REPEAT"
        self.report({"INFO"}, f"cycles on {n} curves")
        return {"FINISHED"}


class DUCK_PT_panel(bpy.types.Panel):
    bl_label = "Duck"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Duck"

    def draw(self, context):
        l = self.layout
        arm = _arm()
        box = l.box()
        box.label(text="Rig", icon="ARMATURE_DATA")
        row = box.row(align=True)
        row.operator("duck.build")
        row.operator("duck.home")
        box.operator("duck.snap_ik")
        if arm is not None and "ik_legs" in arm:
            box.prop(arm, '["ik_legs"]', text="IK legs", slider=True)
        box = l.box()
        box.label(text="Clips", icon="ACTION")
        if arm is not None and arm.animation_data and arm.animation_data.action:
            box.label(text=f"action: {arm.animation_data.action.name}")
        row = box.row(align=True)
        row.operator("duck.import_clip")
        row.operator("duck.export_clip")
        box.operator("duck.check_clip", icon="CHECKMARK")
        last = context.scene.get("duck_last_check")
        if last:
            try:
                r = json.loads(last)
                box.label(text=("PASS " if r["pass"] else "FAIL ") + r["name"], icon="CHECKMARK" if r["pass"] else "ERROR")
                for f in r["fails"]:
                    box.label(text="  " + f)
                box.label(text=f"  penetration {r['pen_mm']} mm")
                box.operator("duck.open_video", icon="FILE_MOVIE")
            except Exception:
                pass
        box = l.box()
        box.label(text="Pose", icon="MOD_MIRROR")
        row = box.row(align=True)
        row.operator("duck.mirror_pose")
        row.operator("duck.mirror_action")
        box.operator("duck.make_loop")


CLASSES = (DUCK_OT_build, DUCK_OT_home, DUCK_OT_snap, DUCK_OT_import, DUCK_OT_export, DUCK_OT_check,
           DUCK_OT_open_video, DUCK_OT_mirror_pose, DUCK_OT_mirror_action, DUCK_OT_loop, DUCK_PT_panel)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
