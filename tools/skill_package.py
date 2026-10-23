#!/usr/bin/env python3
"""Skill package tool: create, validate, import, list and hash (skills/README.md has the layout).

    cd sim && .venv/bin/python ../tools/skill_package.py validate ../skills/walk/v1
    .venv/bin/python ../tools/skill_package.py create --name walk --type locomotion \
        --onnx BEST_WALK_ONNX_2.onnx --clip /path/clip.json --commands vx:-0.15,0.15 vy:-0.2,0.2 wz:-1,1
    .venv/bin/python ../tools/skill_package.py import sim/train_pg/runs_skills/bow/package_draft
    .venv/bin/python ../tools/skill_package.py list
    .venv/bin/python ../tools/skill_package.py hash-model
    .venv/bin/python ../tools/skill_package.py status ../skills/walk/v1 hardware_passed

validate exits 1 when a required file is missing or skill.json fails the schema, so it can
gate a pipeline. jsonschema is used when installed; otherwise the small validator below
covers the subset of draft-07 that skills/schema.json uses.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import skillkit as sk  # noqa: E402

DEFAULT_PACKAGE_DIR = None


# ----------------------------------------------------------------------------- schema
def _mini_validate(schema, obj, path="$"):
    """Enough of JSON schema draft-07 for skills/schema.json: type, required, properties,
    additionalProperties, items, minItems, maxItems, enum, minimum, maximum, pattern."""
    errs = []
    t = schema.get("type")
    if t:
        types = t if isinstance(t, list) else [t]
        ok = any(_is_type(obj, x) for x in types)
        if not ok:
            return [f"{path}: expected {t}, got {type(obj).__name__}"]
    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{path}: {obj!r} not one of {schema['enum']}")
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if "minimum" in schema and obj < schema["minimum"]:
            errs.append(f"{path}: {obj} < minimum {schema['minimum']}")
        if "maximum" in schema and obj > schema["maximum"]:
            errs.append(f"{path}: {obj} > maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and obj <= schema["exclusiveMinimum"]:
            errs.append(f"{path}: {obj} must be > {schema['exclusiveMinimum']}")
    if isinstance(obj, str) and "pattern" in schema and not re.search(schema["pattern"], obj):
        errs.append(f"{path}: {obj!r} does not match {schema['pattern']}")
    if isinstance(obj, dict):
        for k in schema.get("required", []):
            if k not in obj:
                errs.append(f"{path}: missing required field '{k}'")
        props = schema.get("properties", {})
        for k, v in obj.items():
            if k in props:
                errs += _mini_validate(props[k], v, f"{path}.{k}")
            else:
                ap = schema.get("additionalProperties", True)
                if ap is False:
                    errs.append(f"{path}: unexpected field '{k}'")
                elif isinstance(ap, dict):
                    errs += _mini_validate(ap, v, f"{path}.{k}")
    if isinstance(obj, list):
        if "minItems" in schema and len(obj) < schema["minItems"]:
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(obj) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for i, v in enumerate(obj):
                errs += _mini_validate(schema["items"], v, f"{path}[{i}]")
    return errs


def _is_type(obj, t):
    if t == "object":
        return isinstance(obj, dict)
    if t == "array":
        return isinstance(obj, list)
    if t == "string":
        return isinstance(obj, str)
    if t == "integer":
        return isinstance(obj, int) and not isinstance(obj, bool)
    if t == "number":
        return isinstance(obj, (int, float)) and not isinstance(obj, bool)
    if t == "boolean":
        return isinstance(obj, bool)
    if t == "null":
        return obj is None
    return True


def schema_errors(skill):
    schema = sk.load_json(sk.SCHEMA)
    try:
        import jsonschema
        v = jsonschema.Draft7Validator(schema)
        return [f"$.{'.'.join(str(p) for p in e.absolute_path)}: {e.message}" if e.absolute_path else f"$: {e.message}"
                for e in sorted(v.iter_errors(skill), key=lambda e: list(e.absolute_path))]
    except ImportError:
        return _mini_validate(schema, skill)


# ----------------------------------------------------------------------------- validate
def validate_package(skill_dir, require_battery=True, strict_hash=False):
    """Returns (errors, warnings, skill_dict_or_None)."""
    errors, warnings = [], []
    skill_dir = os.path.abspath(skill_dir)
    if not os.path.isdir(skill_dir):
        return [f"{skill_dir}: not a directory"], warnings, None
    required = [f for f in sk.REQUIRED_FILES if require_battery or not f.endswith("battery.json")]
    for f in required:
        if not os.path.isfile(os.path.join(skill_dir, f)):
            errors.append(f"missing {f}")
    if "missing skill.json" in errors:
        return errors, warnings, None
    try:
        skill = sk.load_skill(skill_dir)
    except Exception as e:
        return errors + [f"skill.json unreadable: {e}"], warnings, None
    errors += schema_errors(skill)
    if errors and any("missing required" in e or "expected" in e for e in errors):
        return errors, warnings, skill
    # cross-checks beyond the schema
    parent, leaf = os.path.split(skill_dir)
    m = re.fullmatch(r"v(\d+)", leaf)
    if m and os.path.dirname(parent) == os.path.abspath(sk.SKILLS):
        if int(m.group(1)) != skill.get("version"):
            errors.append(f"version {skill.get('version')} does not match folder {leaf}")
        if os.path.basename(parent) != skill.get("name"):
            errors.append(f"name '{skill.get('name')}' does not match folder {os.path.basename(parent)}")
    nb = set(skill.get("neighbours", []))
    linked = set(skill.get("entry", {}).get("from", [])) | set(skill.get("exit", {}).get("to", []))
    missing = sorted(linked - nb)
    if missing:
        errors.append(f"entry.from/exit.to name skills not in neighbours: {missing}")
    for n in sorted(nb):
        if not os.path.isdir(os.path.join(sk.SKILLS, n)):
            warnings.append(f"neighbour '{n}' has no folder in skills/ yet")
    if "clip.json" not in " ".join(errors):
        try:
            times, q, _ = sk.load_clip(os.path.join(skill_dir, "clip.json"))
            if q.shape[1] != 14:
                errors.append("clip.json: expected 14 joints")
            if len(times) > 1:
                dt = float((times[-1] - times[0]) / (len(times) - 1))
                if abs(dt - 0.02) > 0.005:
                    warnings.append(f"clip.json: mean frame spacing {dt * 1000:.1f} ms, expected 20 ms (50 Hz)")
        except Exception as e:
            errors.append(f"clip.json unreadable: {e}")
    onnx_path = os.path.join(skill_dir, "policy.onnx")
    if os.path.isfile(onnx_path):
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            shape = sess.get_inputs()[0].shape
            width = shape[-1] if shape else None
            size = skill.get("obs_spec", {}).get("size")
            if isinstance(width, int) and size is not None and width != size:
                errors.append(f"policy.onnx input width {width} != obs_spec.size {size}")
            out = sess.get_outputs()[0].shape
            if out and isinstance(out[-1], int) and out[-1] != 14:
                warnings.append(f"policy.onnx outputs {out[-1]} actions, the duck has 14 servos")
        except Exception as e:
            errors.append(f"policy.onnx unreadable: {e}")
    try:
        current = sk.model_hash()
        if skill.get("robot_model_hash") != current:
            msg = f"robot_model_hash {str(skill.get('robot_model_hash'))[:12]}.. != current MJCF {current[:12]}.. (robot model changed since training)"
            (errors if strict_hash else warnings).append(msg)
    except OSError as e:
        warnings.append(f"cannot hash the robot model: {e}")
    hashes = skill.get("hashes", {})
    for key, fname in (("policy_onnx", "policy.onnx"), ("clip_json", "clip.json")):
        p = os.path.join(skill_dir, fname)
        if key in hashes and os.path.isfile(p) and sk.sha256_file(p) != hashes[key]:
            errors.append(f"{fname} does not match hashes.{key}")
    rep = os.path.join(skill_dir, "report", "battery.json")
    if os.path.isfile(rep):
        try:
            b = sk.load_json(rep)
            for k in ("tests", "overall_ratio", "seeds", "episodes"):
                if k not in b:
                    errors.append(f"report/battery.json missing '{k}'")
        except Exception as e:
            errors.append(f"report/battery.json unreadable: {e}")
    return errors, warnings, skill


def cmd_validate(a):
    errors, warnings, _ = validate_package(a.dir, require_battery=not a.no_battery, strict_hash=a.strict)
    for w in warnings:
        print(f"  warning: {w}")
    if errors:
        for e in errors:
            print(f"  error: {e}")
        print(f"INVALID {a.dir}")
        return 1
    print(f"OK {a.dir}")
    return 0


# ----------------------------------------------------------------------------- create
def parse_commands(items):
    out = {}
    for it in items or []:
        name, rng = it.split(":")
        lo, hi = (float(x) for x in rng.split(","))
        out[name] = [lo, hi]
    return out


def next_version(name):
    d = os.path.join(sk.SKILLS, name)
    vs = [int(m.group(1)) for x in (os.listdir(d) if os.path.isdir(d) else []) for m in [re.fullmatch(r"v(\d+)", x)] if m]
    return max(vs, default=0) + 1


def build_skill_json(a, onnx_path, clip_path):
    obs = {"names": list(sk.WALK_OBS_NAMES), "size": sk.WALK_OBS_SIZE} if a.obs == "walk" else sk.load_json(a.obs)
    entry_from = a.entry_from or []
    exit_to = a.exit_to or []
    skill = {
        "name": a.name,
        "type": a.type,
        "version": a.version,
        "description": a.description or "",
        "commands": parse_commands(a.commands),
        "entry": {"from": entry_from, "phase": [float(x) for x in a.entry_phase.split(",")], "window_s": a.entry_window},
        "exit": {"to": exit_to, "window_s": a.exit_window},
        "neighbours": sorted(set(entry_from) | set(exit_to) | set(a.neighbours or [])),
        "timeout_s": a.timeout,
        "envelope": {"torque_nm": a.torque, "max_vel_rad_s": a.max_vel, "thermal_c": a.thermal, "fall_tilt_deg": a.fall_tilt},
        "robot_model_hash": sk.model_hash(),
        "actuator_model_version": a.actuator_model,
        "recipe_version": a.recipe,
        "seed": a.seed,
        "obs_spec": obs,
        "action_spec": {"size": 14, "joints": list(sk.JOINTS)},
        "runtime": {"action_scale": a.action_scale, "lpf_hz": a.lpf, "head_mode": a.head_mode,
                    "accel_bias": 0.0, "gait_clock": a.gait_clock, "episodic_cmd": [0.0, 0.0, 0.0]},
        # the same settings in the motion-graph runtime's vocabulary (graph_runtime/packages.py PolicyConfig)
        "policy": {"kind": "onnx", "action_scale": a.action_scale, "lpf_hz": a.lpf, "max_vel_rad_s": a.max_vel,
                   "head_mode": {"add": "add", "from_cmd": "from_command", "policy": "policy"}[a.head_mode],
                   "accel_bias": 0.0, "gait_period_steps": 27, "input_name": "obs"},
        "thresholds": {"min_height_m": a.min_height, "joint_err_rad": a.joint_err},
        "created": sk.now_iso(),
        "source": {"clip": a.clip_source or os.path.relpath(clip_path, sk.ROOT), "run": a.run or os.path.relpath(onnx_path, sk.ROOT),
                   "onnx": os.path.relpath(onnx_path, sk.ROOT)},
        "status": "draft",
        "hashes": {"policy_onnx": sk.sha256_file(onnx_path), "clip_json": sk.sha256_file(clip_path)},
    }
    return skill


def cmd_create(a):
    onnx_path, clip_path = os.path.abspath(a.onnx), os.path.abspath(a.clip)
    for p in (onnx_path, clip_path):
        if not os.path.isfile(p):
            print(f"error: {p} not found")
            return 1
    if a.version is None:
        a.version = next_version(a.name)
    out = os.path.abspath(a.out) if a.out else os.path.join(sk.SKILLS, a.name, f"v{a.version}")
    if os.path.exists(out) and not a.force:
        print(f"error: {out} exists (use --force to overwrite)")
        return 1
    if a.type == "episodic" and not a.timeout:
        print("error: an episodic skill needs --timeout > 0 (the engine aborts it to recovery after that)")
        return 1
    skill = build_skill_json(a, onnx_path, clip_path)
    errs = schema_errors(skill)
    if errs:
        for e in errs:
            print(f"  error: {e}")
        return 1
    os.makedirs(os.path.join(out, "report"), exist_ok=True)
    shutil.copyfile(onnx_path, os.path.join(out, "policy.onnx"))
    shutil.copyfile(clip_path, os.path.join(out, "clip.json"))
    sk.dump_json(skill, os.path.join(out, "skill.json"))
    errors, warnings, _ = validate_package(out, require_battery=False)
    for w in warnings:
        print(f"  warning: {w}")
    if errors:
        for e in errors:
            print(f"  error: {e}")
        return 1
    print(f"created {out} (status draft). Next: tools/battery.py {out}")
    return 0


# ----------------------------------------------------------------------------- import
def cmd_import(a):
    draft = os.path.abspath(a.draft_dir)
    errors, warnings, skill = validate_package(draft, require_battery=False)
    for w in warnings:
        print(f"  warning: {w}")
    if errors:
        for e in errors:
            print(f"  error: {e}")
        print(f"REFUSED {draft}")
        return 1
    name = a.name or skill["name"]
    version = next_version(name)
    dest = os.path.join(sk.SKILLS, name, f"v{version}")
    os.makedirs(os.path.join(dest, "report"), exist_ok=True)
    for f in os.listdir(draft):
        src = os.path.join(draft, f)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(dest, f), dirs_exist_ok=True)
        else:
            shutil.copyfile(src, os.path.join(dest, f))
    skill["name"], skill["version"] = name, version
    skill.setdefault("source", {})["draft"] = os.path.relpath(draft, sk.ROOT)
    skill["imported"] = sk.now_iso()
    skill["status"] = "draft" if not os.path.isfile(os.path.join(dest, "report", "battery.json")) else skill.get("status", "draft")
    skill.setdefault("hashes", {})
    skill["hashes"]["policy_onnx"] = sk.sha256_file(os.path.join(dest, "policy.onnx"))
    skill["hashes"]["clip_json"] = sk.sha256_file(os.path.join(dest, "clip.json"))
    sk.dump_json(skill, os.path.join(dest, "skill.json"))
    print(f"imported {draft} -> {dest}")
    if a.no_battery:
        return 0
    cmd = [sk.PYTHON, os.path.join(HERE, "battery.py"), dest, "--seeds", str(a.seeds), "--episodes", str(a.episodes)]
    print("running", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=sk.SIM, env={**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")})
    if rc:
        print(f"battery failed (exit {rc}); the package stays a draft without report/battery.json")
    errors, _, _ = validate_package(dest)
    print("VALID" if not errors else "INVALID: " + "; ".join(errors), dest)
    return 1 if errors else 0


# ----------------------------------------------------------------------------- list
def iter_packages():
    if not os.path.isdir(sk.SKILLS):
        return
    for name in sorted(os.listdir(sk.SKILLS)):
        d = os.path.join(sk.SKILLS, name)
        if not os.path.isdir(d):
            continue
        for v in sorted(os.listdir(d), key=lambda x: int(x[1:]) if re.fullmatch(r"v\d+", x) else -1):
            if re.fullmatch(r"v\d+", v) and os.path.isfile(os.path.join(d, v, "skill.json")):
                yield name, v, os.path.join(d, v)


def package_summary(path):
    skill = sk.load_skill(path)
    rep = os.path.join(path, "report")
    out = {"name": skill.get("name"), "version": skill.get("version"), "type": skill.get("type"),
           "status": skill.get("status"), "created": skill.get("created", "")[:10],
           "battery": None, "sim2sim": None, "hardware": None}
    b = os.path.join(rep, "battery.json")
    if os.path.isfile(b):
        bj = sk.load_json(b)
        out["battery"] = bj.get("overall_ratio")
        out["battery_n"] = sum(t.get("total", 0) for t in bj.get("tests", {}).values())
        out["battery_ok"] = sum(t.get("success", 0) for t in bj.get("tests", {}).values())
    s = os.path.join(rep, "sim2sim.json")
    if os.path.isfile(s):
        sj = sk.load_json(s)
        out["sim2sim"] = sj.get("match")
        out["sim2sim_rms"] = sj.get("rms_target_diff_rad")
    h = os.path.join(rep, "hardware.json")
    if os.path.isfile(h):
        hj = sk.load_json(h)
        out["hardware"] = hj.get("ratio", hj.get("overall_ratio"))
    return out


def cmd_list(a):
    rows = [package_summary(p) for _, _, p in iter_packages()]
    if a.json:
        import json
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print("no packages in skills/")
        return 0
    print(f"{'skill':<14}{'ver':<5}{'type':<12}{'status':<16}{'battery':<17}{'sim2sim':<18}{'hardware':<10}created")
    for r in rows:
        bat = "-" if r["battery"] is None else f"{r['battery']:.2f} ({r.get('battery_ok', 0)}/{r.get('battery_n', 0)})"
        s2s = "-" if r["sim2sim"] is None else (("match" if r["sim2sim"] else "MISMATCH") + f" {r.get('sim2sim_rms', float('nan')):.4f}")
        hw = "-" if r["hardware"] is None else f"{r['hardware']}"
        print(f"{r['name']:<14}v{r['version']:<4}{r['type']:<12}{r['status']:<16}{bat:<17}{s2s:<18}{hw:<10}{r['created']}")
    return 0


def cmd_hash(a):
    print(sk.model_hash(a.xml or sk.MODEL_XML))
    return 0


def cmd_status(a):
    path = os.path.join(a.dir, "skill.json")
    skill = sk.load_json(path)
    cur, new = skill.get("status", "draft"), a.status
    if sk.STATUS_ORDER.index(new) < sk.STATUS_ORDER.index(cur) and not a.force:
        print(f"refusing to move {cur} -> {new} without --force")
        return 1
    skill["status"] = new
    skill.setdefault("status_history", []).append({"at": sk.now_iso(), "from": cur, "to": new, "by": a.by or "skill_package.py"})
    sk.dump_json(skill, path)
    print(f"{a.dir}: {cur} -> {new}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("validate", help="check a package (exit 1 when it must not be loaded)")
    p.add_argument("dir")
    p.add_argument("--no-battery", action="store_true", help="do not require report/battery.json (drafts)")
    p.add_argument("--strict", action="store_true", help="a robot_model_hash mismatch is an error, not a warning")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("create", help="build skills/<name>/v<N>/ from an ONNX and a clip")
    p.add_argument("--name", required=True)
    p.add_argument("--type", required=True, choices=["locomotion", "overlay", "episodic", "recovery"])
    p.add_argument("--onnx", required=True)
    p.add_argument("--clip", required=True)
    p.add_argument("--version", type=int)
    p.add_argument("--out", help="package folder (default skills/<name>/v<next>)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--description", default="")
    p.add_argument("--commands", nargs="*", help="name:min,max ...  e.g. vx:-0.15,0.15 vy:-0.2,0.2 wz:-1,1")
    p.add_argument("--entry-from", nargs="*")
    p.add_argument("--exit-to", nargs="*")
    p.add_argument("--neighbours", nargs="*")
    p.add_argument("--entry-phase", default="0,1")
    p.add_argument("--entry-window", type=float, default=0.3)
    p.add_argument("--exit-window", type=float, default=0.3)
    p.add_argument("--timeout", type=float, default=None, help="seconds before the engine aborts the skill; omit for locomotion (null), required > 0 for episodic")
    p.add_argument("--torque", type=float, default=1.86)
    p.add_argument("--max-vel", type=float, default=4.0)
    p.add_argument("--thermal", type=float, default=55.0)
    p.add_argument("--fall-tilt", type=float, default=45.0)
    p.add_argument("--min-height", type=float, default=0.09)
    p.add_argument("--joint-err", type=float, default=0.15)
    p.add_argument("--actuator-model", default="bam-sts3215-upstream")
    p.add_argument("--recipe", default="legacy-joystick")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--obs", default="walk", help="'walk' (101-wide deployed layout) or a JSON file with {names, size[, blocks]}")
    p.add_argument("--action-scale", type=float, default=0.25)
    p.add_argument("--lpf", type=float, default=8.0)
    p.add_argument("--head-mode", default="add", choices=["add", "from_cmd", "policy"])
    p.add_argument("--gait-clock", default="poly", choices=["poly", "clip", "hybrid"])
    p.add_argument("--clip-source", help="where the clip came from (free text)")
    p.add_argument("--run", help="training run the ONNX came from (free text)")
    p.set_defaults(fn=cmd_create)

    p = sub.add_parser("import", help="copy a draft into skills/<name>/v<next>/ after validation, then run its battery")
    p.add_argument("draft_dir")
    p.add_argument("--name")
    p.add_argument("--no-battery", action="store_true")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--episodes", type=int, default=10)
    p.set_defaults(fn=cmd_import)

    p = sub.add_parser("list", help="table of every package with status and ratios")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("hash-model", help="sha256 of the robot MJCF")
    p.add_argument("--xml")
    p.set_defaults(fn=cmd_hash)

    p = sub.add_parser("status", help="set a package's status (forward only unless --force)")
    p.add_argument("dir")
    p.add_argument("status", choices=sk.STATUS_ORDER)
    p.add_argument("--force", action="store_true")
    p.add_argument("--by")
    p.set_defaults(fn=cmd_status)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
