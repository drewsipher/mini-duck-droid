#!/usr/bin/env python3
"""Mass audit tool for the Open Duck Mini v2 training model.

    python3 mechatronics/apply_masses.py dump            # -> mechatronics/masses_mjcf.json (CAD masses as trained)
    python3 mechatronics/apply_masses.py                 # apply mechatronics/masses.json -> mechatronics/model/*.xml + hash
    python3 mechatronics/apply_masses.py --baseline      # same copy with no overrides (hash of the model as trained)
    cd sim && .venv/bin/python ../mechatronics/apply_masses.py      # with mujoco on the path the copy is load-tested

It never edits the training XML. The result is a COPY under mechatronics/model/ with the
measured masses written into the <inertial> elements (inertia tensors scaled with the mass),
plus model/manifest.json and model/robot_model_hash.txt. robot_model_hash is the sha256 of the
XML bytes written, which is what a skill package records.

How masses.json is applied (grams in, kilograms in the XML):
  1. bodies_g      a body you weighed on its own: its mass is set to that value.
  2. groups_g      an assembly you weighed as one piece (head_and_neck, left_leg, right_leg,
                   trunk): the measured mass is spread over the group's bodies that were not
                   set in step 1, in the CAD's proportions.
  3. total_g       the whole duck (with_battery, or without_battery + battery_pack_g): every
                   body still untouched is scaled by one factor so the robot's total matches.
Null entries are ignored, so a file with only total_g filled in is already useful.
"""
import argparse
import hashlib
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_MJCF = ROOT / "sim/train_pg/playground/open_duck_mini_v2/xmls/open_duck_mini_v2_backlash.xml"
MASSES = HERE / "masses.json"
OUT_DIR = HERE / "model"

# Bodies that come off the duck as one piece (or can be weighed as one by holding the rest).
GROUPS = {
    "trunk": ["trunk_assembly"],
    "head_and_neck": ["neck_pitch_assembly", "head_pitch_to_yaw", "neck_yaw_assembly", "head_assembly"],
    "left_leg": ["hip_roll_assembly", "left_roll_to_pitch_assembly", "knee_and_ankle_assembly",
                 "knee_and_ankle_assembly_2", "foot_assembly"],
    "right_leg": ["hip_roll_assembly_2", "right_roll_to_pitch_assembly", "knee_and_ankle_assembly_3",
                  "knee_and_ankle_assembly_4", "foot_assembly_2"],
}
PLAIN = {  # what each MJCF body is, in the user's words
    "trunk_assembly": "body shell with the Pi, the battery pack and the two hip-yaw servos",
    "hip_roll_assembly": "left hip: yaw-to-roll bracket with the hip roll servo",
    "left_roll_to_pitch_assembly": "left hip: roll-to-pitch bracket with the hip pitch servo",
    "knee_and_ankle_assembly": "left thigh (hip pitch horn to knee) with the knee servo",
    "knee_and_ankle_assembly_2": "left shank (knee to ankle) with the ankle servo",
    "foot_assembly": "left foot with the TPU sole",
    "neck_pitch_assembly": "neck: pitch bracket with the head pitch servo",
    "head_pitch_to_yaw": "neck: small pitch-to-yaw link",
    "neck_yaw_assembly": "neck: yaw block with the head roll servo",
    "head_assembly": "the head: shell, eyes, speaker, camera mounts",
    "hip_roll_assembly_2": "right hip: yaw-to-roll bracket with the hip roll servo",
    "right_roll_to_pitch_assembly": "right hip: roll-to-pitch bracket with the hip pitch servo",
    "knee_and_ankle_assembly_3": "right thigh (hip pitch horn to knee) with the knee servo",
    "knee_and_ankle_assembly_4": "right shank (knee to ankle) with the ankle servo",
    "foot_assembly_2": "right foot with the TPU sole",
}


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def parse(path):
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(path, parser)


def bodies_of(root):
    """[(name, parent, body_elem, inertial_elem)] in document order."""
    out = []

    def walk(el, parent):
        for ch in el:
            if ch.tag == "body":
                inert = ch.find("inertial")
                out.append((ch.get("name"), parent, ch, inert))
                walk(ch, ch.get("name"))
            else:
                walk(ch, parent)
    walk(root.find("worldbody"), "world")
    return out


def dump(mjcf):
    tree = parse(mjcf)
    bl = bodies_of(tree.getroot())
    masses = {n: float(i.get("mass")) for n, _, _, i in bl if i is not None}
    children = {}
    for n, p, _, _ in bl:
        children.setdefault(p, []).append(n)

    def subtree(n):
        return masses.get(n, 0.0) + sum(subtree(c) for c in children.get(n, []))
    group_of = {b: g for g, bs in GROUPS.items() for b in bs}
    rows = []
    for n, p, _, i in bl:
        rows.append({"name": n, "parent": p, "mass_kg": masses.get(n), "subtree_mass_kg": round(subtree(n), 6),
                     "inertial_pos": i.get("pos") if i is not None else None,
                     "group": group_of.get(n), "what": PLAIN.get(n)})
    groups = {g: {"bodies": bs, "mass_kg": round(sum(masses[b] for b in bs), 6)} for g, bs in GROUPS.items()}
    total = sum(masses.values())
    doc = {
        "source": str(mjcf.relative_to(ROOT)), "source_sha256": sha256(mjcf.read_bytes()),
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "total_kg": round(total, 6),
        "bodies": rows, "groups": groups,
        "notes": [
            "Masses are the onshape-to-robot export with slicer-estimated masses for printed parts (official sim2real.md).",
            "The 'base' body is massless; 'world' is the floor. 15 bodies carry mass.",
            "Each hip_roll_assembly / roll_to_pitch / thigh / shank body carries one STS3215 (about 55-60 g each, unverified), so a leg is five servos plus plastic.",
            "User estimate of the real duck: 1.5-1.8 kg (sim/report1 'Where his weight is'); the model is 2.107 kg. Training randomises mass x0.9-1.1 per body plus MASS_SCALE when set.",
            "Whether the CAD trunk includes the two 18650 cells (about 2 x 47 g) is unknown: weigh the duck with and without the pack.",
        ],
    }
    out = HERE / "masses_mjcf.json"
    out.write_text(json.dumps(doc, indent=1))
    print(f"wrote {out.relative_to(ROOT)}: {len(masses)} bodies, total {total:.3f} kg")
    for g, d in groups.items():
        print(f"  {g:14s} {d['mass_kg']:.3f} kg")
    return 0


def plan_overrides(cad, spec):
    """cad: {body: kg}; spec: masses.json dict. Returns ({body: new_kg}, log lines)."""
    over, log = {}, []
    for b, g in (spec.get("bodies_g") or {}).items():
        if g is None:
            continue
        if b not in cad:
            log.append(f"ignored unknown body '{b}'")
            continue
        over[b] = g / 1000
        log.append(f"body {b}: {cad[b]*1000:.0f} g -> {g:.0f} g (weighed)")
    for gname, g in (spec.get("groups_g") or {}).items():
        if g is None:
            continue
        if gname not in GROUPS:
            log.append(f"ignored unknown group '{gname}'")
            continue
        members = GROUPS[gname]
        fixed = sum(over[b] for b in members if b in over)
        free = [b for b in members if b not in over]
        target = g / 1000 - fixed
        cad_free = sum(cad[b] for b in free)
        if not free:
            log.append(f"group {gname}: every body already set, {g:.0f} g not used")
            continue
        if target <= 0:
            log.append(f"group {gname}: {g:.0f} g is not more than its weighed parts ({fixed*1000:.0f} g), skipped")
            continue
        k = target / cad_free
        for b in free:
            over[b] = cad[b] * k
        log.append(f"group {gname}: {cad_free*1000 + fixed*1000:.0f} g -> {g:.0f} g, free bodies x{k:.3f}")
    tg = spec.get("total_g") or {}
    total = tg.get("with_battery")
    if total is None and tg.get("without_battery") is not None and spec.get("battery_pack_g") is not None:
        total = tg["without_battery"] + spec["battery_pack_g"]
        log.append(f"total = without_battery + battery_pack = {total:.0f} g")
    if total is not None:
        free = [b for b in cad if b not in over]
        target = total / 1000 - sum(over.values())
        cad_free = sum(cad[b] for b in free)
        if free and target > 0:
            k = target / cad_free
            for b in free:
                over[b] = cad[b] * k
            log.append(f"total {sum(cad.values())*1000:.0f} g -> {total:.0f} g: remaining {len(free)} bodies x{k:.3f}")
        elif not free:
            log.append(f"total {total:.0f} g given but every body already set (sum {sum(over.values())*1000:.0f} g)")
        else:
            log.append(f"total {total:.0f} g is below the weighed parts, nothing scaled")
    return over, log


def scale_inertia(inert, k):
    for attr in ("diaginertia", "fullinertia"):
        v = inert.get(attr)
        if v:
            inert.set(attr, " ".join(f"{float(x)*k:.8g}" for x in v.split()))


def apply(mjcf, masses_path, baseline):
    tree = parse(mjcf)
    root = tree.getroot()
    bl = bodies_of(root)
    cad = {n: float(i.get("mass")) for n, _, _, i in bl if i is not None}
    spec, over, log = {}, {}, []
    if not baseline:
        if not masses_path.exists():
            print(f"{masses_path.relative_to(ROOT)} not found: writing the baseline copy (no overrides)")
            baseline = True
        else:
            spec = json.loads(masses_path.read_text())
            over, log = plan_overrides(cad, spec)
            if not over:
                print("masses.json has no numbers yet: writing the baseline copy (no overrides)")
                baseline = True
    changed = {}
    for n, _, _, inert in bl:
        if inert is None or n not in over:
            continue
        new = over[n]
        k = new / cad[n]
        if abs(k - 1) < 1e-9:
            continue
        inert.set("mass", f"{new:.6g}")
        scale_inertia(inert, k)
        changed[n] = {"cad_kg": cad[n], "kg": round(new, 6), "factor": round(k, 4)}
    # make the copy loadable from its new folder: absolute mesh dir
    comp = root.find("compiler")
    assets = (mjcf.parent / (comp.get("meshdir") if comp is not None and comp.get("meshdir") else "")).resolve()
    if comp is not None:
        comp.set("meshdir", str(assets))
    OUT_DIR.mkdir(exist_ok=True)
    stem = mjcf.stem + (".cad" if baseline else ".measured")
    out = OUT_DIR / f"{stem}.xml"
    new_total = sum(over.get(n, m) for n, m in cad.items())
    header = (f"<!-- mechatronics/apply_masses.py {time.strftime('%Y-%m-%d %H:%M')} from {mjcf.relative_to(ROOT)} "
              f"(sha256 {sha256(mjcf.read_bytes())[:16]}). {'No mass overrides.' if baseline else f'{len(changed)} body masses overridden from {masses_path.name}.'} "
              f"Total {new_total:.4f} kg. meshdir made absolute. Do not edit: regenerate. -->\n")
    body = ET.tostring(root, encoding="unicode")
    data = ('<?xml version="1.0"?>\n' + header + body + "\n").encode()
    out.write_bytes(data)
    h = sha256(data)
    (OUT_DIR / "robot_model_hash.txt").write_text(h + "  " + out.name + "\n")
    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": str(mjcf.relative_to(ROOT)),
        "source_sha256": sha256(mjcf.read_bytes()), "output": str(out.relative_to(ROOT)),
        "robot_model_hash": h, "baseline": baseline,
        "masses_file": None if baseline else (str(masses_path.relative_to(ROOT)) if masses_path.resolve().is_relative_to(ROOT) else str(masses_path)),
        "measured_on": None if baseline else spec.get("date"), "cad_total_kg": round(sum(cad.values()), 6),
        "total_kg": round(new_total, 6), "overrides": changed, "log": log,
    }
    # load test when mujoco is importable (run with the sim venv)
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(out))
        manifest["mujoco_load"] = {"ok": True, "version": mujoco.__version__, "nbody": m.nbody,
                                   "total_kg": round(float(m.body_subtreemass[m.body("base").id]), 6)}
    except ImportError:
        manifest["mujoco_load"] = {"ok": None, "note": "mujoco not importable here; run with sim/.venv to load-test"}
    except Exception as e:   # noqa: BLE001
        manifest["mujoco_load"] = {"ok": False, "error": str(e)}
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for line in log:
        print("  " + line)
    print(f"wrote {out.relative_to(ROOT)}  total {new_total:.4f} kg (CAD {sum(cad.values()):.4f})")
    print(f"mujoco load: {manifest['mujoco_load']}")
    print(f"robot_model_hash = {h}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="apply", choices=["apply", "dump"])
    ap.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    ap.add_argument("--masses", type=Path, default=MASSES)
    ap.add_argument("--baseline", action="store_true", help="copy with no overrides (hash of the model as trained)")
    a = ap.parse_args()
    if not a.mjcf.exists():
        sys.exit(f"MJCF not found: {a.mjcf}")
    if a.cmd == "dump":
        return dump(a.mjcf)
    return apply(a.mjcf, a.masses, a.baseline)


if __name__ == "__main__":
    sys.exit(main())
