#!/usr/bin/env python3
"""Dump the duck's kinematic tree from the MJCF to JSON for the Blender importer (anim/blender_duck.py).

    cd sim && .venv/bin/python ../anim/dump_mjcf.py ../anim/duck_model.json

Bodies in tree order with world-frame rest pose (position, quaternion wxyz), joints (hinge axis in the
body frame, range), geoms (mesh name, file, local pose, scale) and the actuator order the clips use.
"""
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PG = os.path.join(HERE, "..", "sim", "train_pg", "playground", "open_duck_mini_v2")
XML = os.path.join(PG, "xmls", "scene_flat_terrain_backlash.xml")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "duck_model.json")
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)                      # rest pose = qpos0 (keyframe-free)
    meshdir = os.path.join(PG, "xmls", "assets")
    bodies = []
    for b in range(1, m.nbody):                  # 0 = world
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        joints = []
        for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
            jt = m.jnt_type[j]
            joints.append({
                "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j),
                "type": {0: "free", 1: "ball", 2: "slide", 3: "hinge"}[int(jt)],
                "axis": m.jnt_axis[j].tolist(), "pos": m.jnt_pos[j].tolist(),
                "range": m.jnt_range[j].tolist() if m.jnt_limited[j] else None,
                "qposadr": int(m.jnt_qposadr[j]),
            })
        geoms = []
        for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 or m.geom_group[g] in (0, 1, 2):
                pass
            mid = m.geom_dataid[g]
            mname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid)
            geoms.append({
                "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}", "gid": int(g),
                "mesh": mname, "file": os.path.join(meshdir, mname + ".stl"),
                "scale": m.mesh_scale[mid].tolist(),
                "pos": m.geom_pos[g].tolist(), "quat": m.geom_quat[g].tolist(),
                "group": int(m.geom_group[g]), "rgba": m.geom_rgba[g].tolist(),
            })
        bodies.append({
            "name": name, "parent": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.body_parentid[b]),
            "pos": m.body_pos[b].tolist(), "quat": m.body_quat[b].tolist(),          # local, in parent frame
            "xpos": d.xpos[b].tolist(), "xquat": d.xquat[b].tolist(),               # world, rest pose
            "mass": float(m.body_mass[b]), "joints": joints, "geoms": geoms,
        })
    # Compiled mesh geometry. MuJoCo recentres every mesh at compile time (mesh_pos/mesh_quat) and
    # adjusts geom_pos/geom_quat to match, so the raw STL vertices do NOT sit where geom_pos says.
    # The compiled vertices do, which is what the simulator renders, so those are what Blender gets.
    meshes = {}
    for mid in range(m.nmesh):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid)
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        meshes[name] = {"verts": np.round(m.mesh_vert[va:va + vn], 6).tolist(), "faces": m.mesh_face[fa:fa + fn].tolist()}
    # independent check values: world centroid of each mesh geom at rest, straight from MuJoCo
    for b in bodies:
        for g in b["geoms"]:
            gid = g["gid"]                       # geoms are mostly unnamed, never look them up by name
            mid = m.geom_dataid[gid]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            w = d.geom_xpos[gid] + v @ d.geom_xmat[gid].reshape(3, 3).T
            g["rest_world_centroid"] = np.round(w.mean(axis=0), 6).tolist()
            g["rest_world_zrange"] = [float(w[:, 2].min()), float(w[:, 2].max())]
    actuators = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(m.nu)]
    act_joint = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a][0]) for a in range(m.nu)]
    home = {}
    try:                                         # the standing pose the runtime and the trainer start from
        kq = m.keyframe("home").qpos.copy()
        jadr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): int(m.jnt_qposadr[j]) for j in range(m.njnt)}
        # The keyframe's base height (0.15 m) leaves the feet ~17 mm through the floor; the duck settles
        # higher under gravity. Report the height at which the lowest foot point just touches the floor,
        # so a clip authored at the home pose stands ON the floor (the keyframe value is kept alongside).
        d2 = mujoco.MjData(m)
        d2.qpos[:] = kq
        mujoco.mj_kinematics(m, d2)
        mujoco.mj_collision(m, d2)
        floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        pen = max((-d2.contact[i].dist for i in range(d2.ncon) if floor in (d2.contact[i].geom1, d2.contact[i].geom2)), default=0.0)
        settled = kq[:3].copy()
        settled[2] += max(pen, 0.0)
        home = {"joints": {j: float(kq[jadr[j]]) for j in act_joint}, "base_pos": [float(x) for x in settled],
                "base_pos_keyframe": kq[:3].tolist(), "base_quat": kq[3:7].tolist(), "keyframe_floor_penetration_m": float(pen)}
    except KeyError:
        pass
    json.dump({"xml": os.path.relpath(XML, HERE), "bodies": bodies, "actuators": actuators, "meshes": meshes,
               "actuator_joints": act_joint, "qpos0": m.qpos0.tolist(), "home": home}, open(out, "w"))
    print(f"{len(bodies)} bodies, {sum(len(b['joints']) for b in bodies)} joints, "
          f"{sum(len(b['geoms']) for b in bodies)} mesh geoms, {m.nu} actuators -> {out}")
    for b in bodies:
        print(f"  {b['name']:<22} parent {b['parent']:<20} joints {[j['name'] for j in b['joints']]}  geoms {len(b['geoms'])}")


if __name__ == "__main__":
    main()
