"""Duck Tools: the N-panel in Blender's 3D view for animating the Open Duck Mini.

Install once (anim/install_addon.sh does it): copy or symlink this file into
~/.config/blender/5.2/scripts/addons/duck_tools.py and enable "Duck Tools" in Preferences.
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

ANIM = os.environ.get("DUCK_ANIM_DIR", "/home/hermes/src/mini-duck-droid/anim")
PROJECT = os.path.dirname(ANIM)
CLIPS = os.path.join(PROJECT, "tracker", "data", "clips")
SIM_PY = os.path.join(PROJECT, "sim", ".venv", "bin", "python")


def _bd():
    if ANIM not in sys.path:
        sys.path.insert(0, ANIM)
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
            self.filepath = CLIPS + os.sep
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
            self.filepath = os.path.join(CLIPS, name.replace(".", "_") + ".json")
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
        out_dir = os.path.join(ANIM, "reports", name)
        os.makedirs(out_dir, exist_ok=True)
        clip = os.path.join(out_dir, "clip.json")
        _bd().export_clip(clip, arm=arm)
        cmd = [SIM_PY, os.path.join(ANIM, "clip_check.py"), clip, "--json", os.path.join(out_dir, "check.json"),
               "--video", os.path.join(out_dir, "check.mp4")] + (["--loop"] if self.loop else [])
        env = dict(os.environ, MUJOCO_GL="egl")
        p = subprocess.run(cmd, cwd=os.path.join(PROJECT, "sim"), env=env, capture_output=True, text=True, timeout=600)
        try:
            rep = json.load(open(os.path.join(out_dir, "check.json")))
        except Exception:
            self.report({"ERROR"}, f"checker failed: {p.stderr[-400:]}")
            return {"CANCELLED"}
        context.scene["duck_last_check"] = json.dumps({"name": name, "pass": rep["pass"], "fails": rep["fails"],
                                                      "pen_mm": round(1000 * rep["floor_penetration_m"], 1),
                                                      "self": list(rep["self_collisions_m"].keys()),
                                                      "over_speed": rep["over_speed"], "limits": rep["limit_breach_rad"],
                                                      "video": os.path.join(out_dir, "check.mp4")})
        msg = ("PASS" if rep["pass"] else "FAIL: " + ", ".join(rep["fails"])) + f"  (penetration {1000 * rep['floor_penetration_m']:.1f} mm)"
        self.report({"INFO"} if rep["pass"] else {"WARNING"}, msg)
        return {"FINISHED"}


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
        subprocess.Popen(["xdg-open", v])
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
    bl_description = "Add a Cycles modifier to every F-curve of the current action so it repeats"

    def execute(self, context):
        arm = _arm()
        act = arm.animation_data.action if arm.animation_data else None
        if not act:
            return {"CANCELLED"}
        n = 0
        for fc in _bd().fcurves(act):
            if not any(m.type == "CYCLES" for m in fc.modifiers):
                fc.modifiers.new("CYCLES")
                n += 1
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
