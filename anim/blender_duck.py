"""Open Duck Mini in Blender: build the rig from anim/duck_model.json, import and export clips.

Runs INSIDE Blender (5.2). From the Blender MCP or the Python console:

    import sys; sys.path.insert(0, "/home/hermes/src/mini-duck-droid/anim")
    import blender_duck as bd; bd.build()                      # rig + meshes + floor, home pose
    bd.import_clip("/path/clip.json", "s2a_turn_left")          # clip -> action (FK keys, IK off)
    bd.export_clip("/path/out.json", start=1, end=100)          # timeline -> clip JSON at 50 Hz

Rig conventions
- One bone per actuated joint (14) plus "base" (the floating trunk). Bone head = joint pivot,
  bone local Z = hinge axis, X and Y rotation locked, so the joint angle is the bone's Z rotation
  (same sign and zero as MuJoCo). Bone names: left_hip_yaw -> hip_yaw.L, so Blender's mirror works.
- Leg IK: "foot_ik.L/.R" bones at the soles drive 5-bone chains. The armature property "ik_legs"
  (0..1) is the IK influence; imported clips set it to 0 so their FK keys play unchanged.
- Scene runs at 50 fps = the control rate, metres, Z up, floor at z = 0.
- Export reads the solved pose (IK included), so animate however you like.
"""
import json
import math
import os

import bpy
from mathutils import Matrix, Quaternion, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "duck_model.json")
ARM = "Duck"
FPS = 50
BONE_LEN = 0.04


# Left <-> right mirror in the MuJoCo/runtime frame (same table as tracker/anim.js): the two hip
# pitches, rolls and yaws have opposite signs (standing: left_hip_pitch -0.63, right +0.635); knees
# and ankles share a sign; the head yaw and roll flip. Blender's own X-mirror cannot know this.
MIRROR = {
    "left_hip_yaw": ("right_hip_yaw", -1), "right_hip_yaw": ("left_hip_yaw", -1),
    "left_hip_roll": ("right_hip_roll", -1), "right_hip_roll": ("left_hip_roll", -1),
    "left_hip_pitch": ("right_hip_pitch", -1), "right_hip_pitch": ("left_hip_pitch", -1),
    "left_knee": ("right_knee", 1), "right_knee": ("left_knee", 1),
    "left_ankle": ("right_ankle", 1), "right_ankle": ("left_ankle", 1),
    "neck_pitch": ("neck_pitch", 1), "head_pitch": ("head_pitch", 1),
    "head_yaw": ("head_yaw", -1), "head_roll": ("head_roll", -1),
}


def mirror_joints(joints):
    """{joint: rad} -> the same motion done by the other side."""
    out = {}
    for j, v in joints.items():
        mj, s = MIRROR[j]
        out[mj] = s * v
    return out


def mirror_pose(arm):
    """Mirror the current FK pose of the rig (legs and head). Run with IK off or snap afterwards."""
    q, _, _ = joint_angles(arm)
    set_pose(arm, mirror_joints(q))
    snap_ik_targets(arm)


def mirror_action(arm, action=None, name=None):
    """A new action that is the mirror of `action` (default: the rig's current action), sampled per frame."""
    action = action or arm.animation_data.action
    scene = bpy.context.scene
    f0, f1 = (int(action.frame_range[0]), int(action.frame_range[1]))
    frames = []
    cur = scene.frame_current
    for f in range(f0, f1 + 1):
        scene.frame_set(f)
        q, base, _ = joint_angles(arm)
        frames.append((f, mirror_joints(q), base))
    scene.frame_set(cur)
    new = bpy.data.actions.new(name or action.name + ".mirror")
    arm.animation_data.action = new
    try:
        if not new.slots:
            new.slots.new(id_type="OBJECT", name=arm.name)
        arm.animation_data.action_slot = new.slots[0]
    except AttributeError:
        pass
    for f, q, base in frames:
        p, (w, x, y, z) = base
        set_pose(arm, q, base={"base_pos": [p[0], -p[1], p[2]], "base_quat": [w, -x, y, -z]}, frame=f, key=True)
    set_ik(arm, 0.0)
    return new


def bone_name(joint):
    for side, suf in (("left_", ".L"), ("right_", ".R")):
        if joint.startswith(side):
            return joint[len(side):] + suf
    return joint


def joint_name(bone):
    for suf, side in ((".L", "left_"), (".R", "right_")):
        if bone.endswith(suf):
            return side + bone[:-len(suf)]
    return bone


def _q(wxyz):
    return Quaternion((wxyz[0], wxyz[1], wxyz[2], wxyz[3]))


def _clear():
    for o in list(bpy.data.objects):
        if o.name == ARM or o.get("duck_part") or o.name == "DuckFloor":
            bpy.data.objects.remove(o, do_unlink=True)
    for a in list(bpy.data.armatures):
        if a.users == 0:
            bpy.data.armatures.remove(a)


def build(model_path=MODEL, with_meshes=True):
    m = json.load(open(model_path))
    _clear()
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    bodies = {b["name"]: b for b in m["bodies"]}
    act = m["actuator_joints"]
    # body -> its actuated joint (ignore the passive *_backlash twins)
    body_joint = {}
    for b in m["bodies"]:
        for j in b["joints"]:
            if j["name"] in act:
                body_joint[b["name"]] = j
    # world pivot and axis of every actuated joint (rest pose = all joints at 0)
    piv, axis = {}, {}
    for bn, j in body_joint.items():
        b = bodies[bn]
        R = _q(b["xquat"]).to_matrix()
        piv[j["name"]] = Vector(b["xpos"]) + R @ Vector(j["pos"])
        axis[j["name"]] = (R @ Vector(j["axis"])).normalized()
    # parent actuated joint of each actuated joint (walk up the body tree)
    parent_joint = {}
    for bn, j in body_joint.items():
        p = bodies[bn]["parent"]
        while p in bodies and p not in body_joint:
            p = bodies[p]["parent"]
        parent_joint[j["name"]] = body_joint[p]["name"] if p in body_joint else None
    jnames = [j["name"] for j in body_joint.values()]
    children = {jn: [c for c, pj in parent_joint.items() if pj == jn] for jn in jnames}

    arm_data = bpy.data.armatures.new(ARM)
    arm = bpy.data.objects.new(ARM, arm_data)
    scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm_data.display_type = "OCTAHEDRAL"
    arm.show_in_front = True
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones
    base_body = bodies["base"]
    root = eb.new("base")
    root.head = Vector(base_body["xpos"])
    root.tail = root.head + Vector((0, 0, BONE_LEN * 1.5))
    root.roll = 0.0
    for jn, j in ((j["name"], j) for j in body_joint.values()):
        b = eb.new(bone_name(jn))
        a = axis[jn]
        # bone direction: toward the child pivot (or the body's meshes), made perpendicular to the hinge axis
        kids = children.get(jn, [])
        if kids:
            tgt = sum((piv[k] for k in kids), Vector()) / len(kids) - piv[jn]
        else:
            body = next(bb for bb, jj in body_joint.items() if jj["name"] == jn)
            R = _q(bodies[body]["xquat"]).to_matrix()
            gs = bodies[body]["geoms"]
            tgt = (sum((Vector(bodies[body]["xpos"]) + R @ Vector(g["pos"]) for g in gs), Vector()) / len(gs) - piv[jn]) if gs else Vector((0, 0, -1))
        d = tgt - tgt.project(a)
        if d.length < 1e-4:
            d = a.orthogonal()
        d.normalize()
        b.head = piv[jn]
        b.tail = piv[jn] + d * max(BONE_LEN, min(tgt.length, 0.08))
        b.align_roll(a)                           # local Z = hinge axis
        b["mj_joint"] = jn
    for jn in body_joint.values():
        jn = jn["name"]
        eb[bone_name(jn)].parent = eb[bone_name(parent_joint[jn])] if parent_joint[jn] else eb["base"]
        eb[bone_name(jn)].use_connect = False
    # IK targets at the soles, parented to nothing (world), so the body can move over planted feet
    for side in ("L", "R"):
        ankle = eb[f"ankle.{side}"]
        t = eb.new(f"foot_ik.{side}")
        t.head = ankle.tail.copy()                # exactly where the chain tip is at rest, so IK is a no-op until moved
        t.tail = t.head + Vector((0.05, 0, 0))
        t.roll = 0.0
    bpy.ops.object.mode_set(mode="POSE")
    pb = arm.pose.bones
    for jn, j in ((j["name"], j) for j in body_joint.values()):
        p = pb[bone_name(jn)]
        p.rotation_mode = "XYZ"
        p.lock_rotation = (True, True, False)
        p.lock_location = (True, True, True)
        p.lock_scale = (True, True, True)
        p.lock_ik_x = p.lock_ik_y = True
        if j["range"]:
            p.use_ik_limit_z = True
            p.ik_min_z, p.ik_max_z = j["range"]
            p["mj_range"] = list(j["range"])
    pb["base"].rotation_mode = "QUATERNION"
    arm["ik_legs"] = 1.0
    arm.id_properties_ui("ik_legs").update(min=0.0, max=1.0, description="IK influence on both legs (0 = play FK keys)")
    for side in ("L", "R"):
        c = pb[f"ankle.{side}"].constraints.new("IK")
        c.target = arm
        c.subtarget = f"foot_ik.{side}"
        c.chain_count = 5
        c.use_tail = True
        c.iterations = 200
        drv = c.driver_add("influence").driver
        drv.type = "AVERAGE"
        v = drv.variables.new()
        v.name = "ik"
        v.type = "SINGLE_PROP"
        v.targets[0].id = arm
        v.targets[0].data_path = '["ik_legs"]'
        pb[f"foot_ik.{side}"].rotation_mode = "XYZ"
    bpy.ops.object.mode_set(mode="OBJECT")
    # bone collections for the UI
    for name, members in (("Legs", [bone_name(j) for j in act if "hip" in j or "knee" in j or "ankle" in j]),
                          ("Head", [bone_name(j) for j in act if "neck" in j or "head" in j]),
                          ("Controls", ["base", "foot_ik.L", "foot_ik.R"])):
        col = arm_data.collections.new(name)
        for bn in members:
            col.assign(arm_data.bones[bn])
    _controls_look(arm)
    if with_meshes:
        _attach_meshes(arm, m, bodies, body_joint)
    _floor()
    set_pose(arm, m.get("home", {}).get("joints", {}), base=m.get("home", {}))
    snap_ik_targets(arm)
    arm["duck_model"] = model_path
    return arm


def set_ik(arm, value):
    """Set the ik_legs slider from Python. A custom-property change does not re-evaluate the
    drivers on the IK constraints by itself, so tag the armature and update the depsgraph."""
    arm["ik_legs"] = float(value)
    arm.update_tag()
    bpy.context.view_layer.update()


def snap_ik_targets(arm, key=False, frame=None):
    """Move foot_ik.L/.R to where the FK pose puts the feet, so turning IK on changes nothing."""
    ik = arm["ik_legs"]
    set_ik(arm, 0.0)
    tips = {s: (arm.matrix_world @ arm.pose.bones[f"ankle.{s}"].tail).copy() for s in ("L", "R")}
    set_ik(arm, ik)
    for s, tip in tips.items():
        t = arm.pose.bones[f"foot_ik.{s}"]
        rest_head = arm.data.bones[f"foot_ik.{s}"].head_local
        # foot_ik has no parent and roll 0 with the bone along +X: local axes = (-Y?, ...) -> go through the matrix
        t.matrix = Matrix.Translation(tip) @ arm.data.bones[f"foot_ik.{s}"].matrix_local.to_3x3().to_4x4()
        if key:
            t.keyframe_insert("location", frame=frame)
    bpy.context.view_layer.update()


def _shape(name, kind, size):
    """A small wire mesh used as a custom bone shape (hidden helper object)."""
    ob = bpy.data.objects.get(name)
    if ob:
        return ob
    import math as _m
    if kind == "circle":
        n = 24
        verts = [(size * _m.cos(2 * _m.pi * i / n), 0.0, size * _m.sin(2 * _m.pi * i / n)) for i in range(n)]
        edges = [(i, (i + 1) % n) for i in range(n)]
    else:                                        # box
        s = size
        verts = [(x, y, z) for x in (-s, s) for y in (-s, s) for z in (-s, s)]
        edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, edges, [])
    ob = bpy.data.objects.new(name, mesh)
    ob["duck_part"] = True
    bpy.context.scene.collection.objects.link(ob)
    ob.hide_viewport = True
    ob.hide_render = True
    return ob


def _controls_look(arm):
    """Colours and custom shapes so an animator can tell controls from joints at a glance."""
    pb = arm.pose.bones
    for p in pb:
        if p.name.startswith("foot_ik"):
            p.custom_shape = _shape("DuckShape.foot", "circle", 1.0)
            p.custom_shape_scale_xyz = (0.5, 0.5, 0.5)
            p.color.palette = "THEME01"              # red: things you grab
        elif p.name == "base":
            p.custom_shape = _shape("DuckShape.base", "box", 1.0)
            p.custom_shape_scale_xyz = (0.8, 0.8, 0.8)
            p.color.palette = "THEME01"
        elif any(k in p.name for k in ("neck", "head")):
            p.color.palette = "THEME03"              # green: head chain (FK)
        else:
            p.color.palette = "THEME04"              # blue: leg joints (IK or FK)
    arm.data.show_axes = False


def _attach_meshes(arm, m, bodies, body_joint):
    cache = {}
    for b in m["bodies"]:
        # trunk_assembly has no joint and hangs from base; walk up to the nearest bone
        p = b["name"]
        while p in bodies and p not in body_joint and p != "base":
            p = bodies[p]["parent"]
        bone = bone_name(body_joint[p]["name"]) if p in body_joint else "base"
        Rb = _q(b["xquat"]).to_matrix().to_4x4()
        Tb = Matrix.Translation(Vector(b["xpos"])) @ Rb
        for g in b["geoms"]:
            if g["group"] != 2 and g["group"] != 0:
                continue                           # 3 = collision helpers
            key = g["mesh"]
            if key not in cache:
                # compiled (recentred, scaled) vertices from the dump: exactly what MuJoCo renders
                md = m["meshes"][key]
                mesh = bpy.data.meshes.new(key)
                mesh.from_pydata(md["verts"], [], md["faces"])
                mesh.validate()
                mesh.update()
                cache[key] = mesh
            ob = bpy.data.objects.new(f"{b['name']}:{g['name']}", cache[key])
            ob["duck_part"] = True
            bpy.context.scene.collection.objects.link(ob)
            ob.matrix_world = Tb @ Matrix.Translation(Vector(g["pos"])) @ _q(g["quat"]).to_matrix().to_4x4()
            mat_name = "Duck" + ("Dark" if g["rgba"][0] < 0.5 else "Light")
            mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
            mat.diffuse_color = tuple(g["rgba"][:3]) + (1.0,)
            if not ob.data.materials:
                ob.data.materials.append(mat)
            # Parent to the bone. Blender places a bone-parented object in the bone's TAIL frame, so the
            # parent matrix at rest is  arm @ bone.matrix_local @ T(0, length, 0).  Using the REST matrix
            # (not the pose bone's, which IK may already have moved) makes the attachment pose-independent:
            # with parent_inverse = rest_parent^-1 the object's basis IS its rest-world placement.
            ob.parent = arm
            ob.parent_type = "BONE"
            ob.parent_bone = bone
            rest_parent = arm.matrix_world @ arm.data.bones[bone].matrix_local @ Matrix.Translation((0, arm.data.bones[bone].length, 0))
            ob.matrix_parent_inverse = rest_parent.inverted()
            ob.matrix_basis = Tb @ Matrix.Translation(Vector(g["pos"])) @ _q(g["quat"]).to_matrix().to_4x4()


def _floor():
    mesh = bpy.data.meshes.new("DuckFloor")
    mesh.from_pydata([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], [], [(0, 1, 2, 3)])
    ob = bpy.data.objects.new("DuckFloor", mesh)
    ob["duck_part"] = True
    bpy.context.scene.collection.objects.link(ob)
    ob.hide_select = True


def set_pose(arm, joints, base=None, frame=None, key=False):
    """Pose the rig from {mujoco joint: rad}; optionally key it at `frame`."""
    pb = arm.pose.bones
    for jn, v in joints.items():
        p = pb.get(bone_name(jn))
        if p is None:
            continue
        p.rotation_euler = (0.0, 0.0, float(v))
        if key:
            p.keyframe_insert("rotation_euler", index=2, frame=frame)
    if base and base.get("base_pos"):
        b = pb["base"]
        ml = arm.data.bones["base"].matrix_local
        # bone frame moved so its head sits at base_pos, turned by base_quat (MuJoCo base pose)
        P = Matrix.Translation(Vector(base["base_pos"])) @ _q(base["base_quat"]).to_matrix().to_4x4() @ ml.to_3x3().to_4x4()
        b.matrix_basis = ml.inverted() @ P
        if key:
            b.keyframe_insert("location", frame=frame)
            b.keyframe_insert("rotation_quaternion", frame=frame)


def fcurves(action):
    """All F-curves of an action. Blender 5 removed Action.fcurves (layered, slotted actions)."""
    if action is None:
        return []
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    out = []
    for layer in action.layers:
        for strip in layer.strips:
            for cb in strip.channelbags:
                out.extend(cb.fcurves)
    return out


def _frames_of(clip):
    """Accept the plain [[t, {j: rad}]] recording, the saved {frames: ...} dict, or {keys: ...} from anim.js."""
    if isinstance(clip, list):
        return clip
    if "frames" in clip:
        return clip["frames"]
    if "keys" in clip:                             # keyed per joint: sample at 50 Hz with linear interpolation
        keys = clip["keys"]
        end = max((k[-1][0] for k in keys.values() if k), default=0)
        out = []
        for i in range(int(round(end * FPS)) + 1):
            t = i / FPS
            pose = {}
            for jn, ks in keys.items():
                if not ks:
                    continue
                if t <= ks[0][0]:
                    pose[jn] = ks[0][1]
                elif t >= ks[-1][0]:
                    pose[jn] = ks[-1][1]
                else:
                    for (t0, v0), (t1, v1) in zip(ks, ks[1:]):
                        if t0 <= t <= t1:
                            pose[jn] = v0 + (v1 - v0) * (t - t0) / (t1 - t0)
                            break
            out.append([t, pose])
        return out
    raise ValueError("unknown clip format")


def import_clip(path, name=None, arm=None, start=1):
    """Clip JSON -> a new action on the rig (FK keys at 50 Hz, legs IK turned off)."""
    arm = arm or bpy.data.objects[ARM]
    clip = json.load(open(path))
    frames = _frames_of(clip)
    name = name or os.path.splitext(os.path.basename(path))[0]
    action = bpy.data.actions.new(name)
    arm.animation_data_create()
    arm.animation_data.action = action
    try:                                           # Blender 4.4+ slotted actions
        if not action.slots:
            action.slots.new(id_type="OBJECT", name=arm.name)
        arm.animation_data.action_slot = action.slots[0]
    except AttributeError:
        pass
    base_frames = clip.get("base") if isinstance(clip, dict) else None
    for i, (t, pose) in enumerate(frames):
        f = start + int(round(t * FPS))
        base = None
        if base_frames and i < len(base_frames):
            base = {"base_pos": base_frames[i][0], "base_quat": base_frames[i][1]}
        set_pose(arm, pose, base=base, frame=f, key=True)
    set_ik(arm, 0.0)
    arm.keyframe_insert('["ik_legs"]', frame=start)
    scene = bpy.context.scene
    scene.frame_start = start
    scene.frame_end = start + int(round(frames[-1][0] * FPS))
    scene.frame_set(start)
    return action


def joint_angles(arm):
    """{mujoco joint: rad} from the SOLVED pose (constraints and IK included), plus the base pose."""
    out = {}
    leak = 0.0
    for p in arm.pose.bones:
        jn = p.bone.get("mj_joint")
        if not jn:
            continue
        par = p.parent
        rest_rel = par.bone.matrix_local.inverted() @ p.bone.matrix_local
        pose_rel = par.matrix.inverted() @ p.matrix
        B = (rest_rel.inverted() @ pose_rel).to_3x3()
        out[jn] = math.atan2(B[1][0], B[0][0])
        leak = max(leak, abs(1.0 - B[2][2]))
    b = arm.pose.bones["base"]
    M = arm.matrix_world @ b.matrix
    R = M.to_3x3() @ arm.data.bones["base"].matrix_local.to_3x3().inverted()   # strip the bone's rest orientation
    base = [list(M.to_translation()), list(R.to_quaternion())]
    return out, base, leak


def export_clip(path, start=None, end=None, arm=None, meta=None):
    """Bake the timeline to the clip JSON the puppet tool, the simulator and the trainer read."""
    arm = arm or bpy.data.objects[ARM]
    scene = bpy.context.scene
    start = scene.frame_start if start is None else start
    end = scene.frame_end if end is None else end
    cur = scene.frame_current
    frames, bases, worst = [], [], 0.0
    for f in range(start, end + 1):
        scene.frame_set(f)
        q, base, leak = joint_angles(arm)
        worst = max(worst, leak)
        frames.append([round((f - start) / FPS, 3), {k: round(v, 4) for k, v in q.items()}])
        bases.append([[round(x, 4) for x in base[0]], [round(x, 4) for x in base[1]]])
    scene.frame_set(cur)
    # The trunk track is only meaningful when the animator moved the base bone; otherwise leave it out
    # so the checker and the trainer place the duck on the floor themselves.
    act = arm.animation_data.action if arm.animation_data else None
    base_animated = any(fc.data_path.startswith('pose.bones["base"]') for fc in fcurves(act))
    out = {"frames": frames, "interp": "linear",
           **({"base": bases} if base_animated else {}),
           "authored_in": {"tool": "blender", "file": os.path.basename(bpy.data.filepath), "action": getattr(arm.animation_data.action, "name", None) if arm.animation_data else None,
                           "fps": FPS, "off_axis_leak": worst, **(meta or {})}}
    with open(path, "w") as fh:
        json.dump(out, fh)
    return {"frames": len(frames), "seconds": (end - start) / FPS, "off_axis_leak": worst, "path": path}
