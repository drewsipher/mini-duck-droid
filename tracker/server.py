#!/usr/bin/env python3
"""Duck Dashboard - the local web server behind http://localhost:8430 (LAN: :8430 on every interface).

Pages
  /            the dashboard (progress, animations, training, skills, tools, telemetry, experiments)
  /puppet/     the Duck Puppet: pose, hold, record and replay the real duck's 14 servos
  /build/      the old print-and-shopping tracker
Data
  /api/clips, /api/progress, /api/agent, /duck/api/*   the puppet's own API (unchanged)
  /api/jobs, /api/tools      run registered tools (data/tools.json) as subprocesses, tail their logs
  /api/animations /api/runs /api/reports /api/skills /api/telemetry /api/experiments
  /api/progress_skill_library (GET, POST)   /api/gpu   /api/status
  /files/<path>   files under whitelisted project folders only (reports, skills, runs, telemetry)
Standard library only.  Usage:  python3 server.py [--port 8430] [--lan]
"""
import argparse
import glob
import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA = HERE / "data"
CLIPS = DATA / "clips"
PROGRESS = DATA / "progress.json"
PROGRESS_SL = DATA / "progress_skill_library.json"
TOOLS = DATA / "tools.json"
PI = "bdxv2@bdxv2.local"
PI_PY = "~/.virtualenvs/open-duck-mini-runtime/bin/python"
AGENT = "http://bdxv2.local:8091"
ROBOT = PROJECT / "official_repo" / "mini_bdx" / "robots" / "open_duck_mini_v2"
ROUTES = {  # url prefix -> directory served read-only
    "/robot/": ROBOT, "/stl/": PROJECT / "print",
    "/build/robot/": ROBOT, "/build/stl/": PROJECT / "print", "/build/vendor/": HERE / "vendor",
    "/build/docs/": HERE / "docs",
}
# /files/<path> serves only what lies under these project folders (globs allowed)
FILE_ROOTS = ["anim/reports", "sim/report*", "skills", "tracker/data", "mechatronics", "sim/train_pg/runs*", "tests", "graph_runtime/demo", "regression", "experiments"]
DOCS = {"plan": PROJECT / "skill-library-plan.md", "framework": PROJECT / "skill-library-framework.md",
        "log": PROJECT / "experiments" / "log.md", "questions": PROJECT / "experiments" / "questions.md"}
MAX_BODY = 8_000_000
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,60}$")
STEP_LINE = re.compile(r"STEP:\s*(\d+)\s+reward:\s*([-\d.eE+]+)(?:\s+reward_std:\s*([-\d.eE+]+))?")
TRAILING_STEP = re.compile(r"_(\d+)(?:\.onnx)?$")

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("model/stl", ".stl")
mimetypes.add_type("application/json", ".jsonl")
mimetypes.add_type("text/markdown", ".md")

import sys  # noqa: E402
sys.path.insert(0, str(HERE))
from jobs import JobError, Jobs, clipname  # noqa: E402
from tfevents import scalars as read_scalars  # noqa: E402

JOBS = Jobs(PROJECT, DATA, TOOLS)


# ---------------------------------------------------------------------------- helpers
def rel(p):
    return str(Path(p).resolve().relative_to(PROJECT))


def servable(relpath):
    """Resolve a /files/ path; None unless it is a file under one of FILE_ROOTS."""
    try:
        target = (PROJECT / relpath).resolve()
    except (OSError, ValueError):
        return None
    for pat in FILE_ROOTS:
        for d in PROJECT.glob(pat):
            if d.is_dir() and target.is_relative_to(d.resolve()):
                return target if target.is_file() else None
    return None


def file_url(p):
    return "/files/" + rel(p)


def with_urls(files):
    out = []
    for f in files:
        if servable(f["path"]):
            f["url"] = "/files/" + f["path"]
        out.append(f)
    return out


def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def tail_text(p, nbytes=16384):
    p = Path(p)
    if not p.exists():
        return ""
    size = p.stat().st_size
    with open(p, "rb") as f:
        if size > nbytes:
            f.seek(size - nbytes)
        data = f.read()
    text = data.decode("utf-8", "replace")
    if size > nbytes and "\n" in text:
        text = text[text.index("\n") + 1:]
    return text


def clip_path(name):
    name = unquote(name)
    if not NAME.match(name):
        return None
    return CLIPS / (name + ".json")


def clip_summary(p):
    """A clip file is a plain recording [[t, {joint: rad}], ...] or an edited clip
    {"keys": {joint: [{t, v}]}, "interp": ..., "frames": [...]} (frames sampled from the keys)."""
    try:
        data = json.loads(p.read_text())
        frames = data if isinstance(data, list) else data.get("frames", [])
        return {"name": p.stem, "seconds": frames[-1][0] if frames else 0, "frames": len(frames),
                "keys": None if isinstance(data, list) else sum(len(k) for k in data.get("keys", {}).values()),
                "saved": int(p.stat().st_mtime)}
    except Exception:
        return {"name": p.stem, "seconds": 0, "frames": 0, "saved": int(p.stat().st_mtime), "broken": True}


def pi(cmd, timeout=25):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", PI, cmd],
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr).strip()


def agent_control(action):
    """start / stop / deploy the agent on the Pi over ssh. Returns a JSON-able dict."""
    try:
        if action in ("start", "deploy"):
            rc, out = pi("fuser /dev/ttyACM0 2>/dev/null; pgrep -af 'duck_agent.py|v2_rl_walk|mujoco_infer' | grep -v pgrep")
            if "v2_rl_walk" in out or "mujoco_infer" in out:
                return {"ok": False, "msg": "the walk program is using the servo port on the Pi; stop it first"}
            if action == "deploy" or "duck_agent.py" not in out:
                subprocess.run(["scp", "-q", str(HERE / "duck_agent.py"), PI + ":~/duck_agent.py"],
                               check=True, timeout=25)
                # two ssh calls: a pkill in the same shell as the start line would match itself
                pi("pkill -f '[d]uck_agent.py'; sleep 0.5")
                pi(f"nohup {PI_PY} ~/duck_agent.py --http 8091 > ~/duck_agent.log 2>&1 & sleep 2; tail -3 ~/duck_agent.log")
            return {"ok": True, "msg": "agent running on the duck"}
        if action == "stop":
            pi("pkill -f '[d]uck_agent.py'")
            return {"ok": True, "msg": "agent stopped, motors off"}
        if action == "log":
            rc, out = pi("tail -20 ~/duck_agent.log")
            return {"ok": True, "msg": out}
    except subprocess.TimeoutExpired:
        return {"ok": False, "msg": "the Pi did not answer over ssh"}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "msg": f"ssh/scp failed: {e}"}
    return {"ok": False, "msg": f"unknown action {action}"}


# ---------------------------------------------------------------------------- GPU
_gpu_cache = {"t": 0, "v": None}


def gpu_status():
    if time.time() - _gpu_cache["t"] < 2:
        return _gpu_cache["v"]
    out = {"ok": False, "util": None, "mem_used": None, "mem_total": None, "procs": []}
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4)
            name, util, used, total, temp = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
            out.update(ok=True, name=name, util=float(util), mem_used=float(used), mem_total=float(total), temp=float(temp))
            r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=4)
            for line in r.stdout.strip().splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 3:
                    out["procs"].append({"pid": parts[0], "name": Path(parts[1]).name, "mem": float(parts[2]) if parts[2].replace(".", "").isdigit() else parts[2]})
        except Exception as e:
            out["error"] = str(e)
    _gpu_cache.update(t=time.time(), v=out)
    return out


_agent_cache = {"t": 0, "ok": False}


def agent_alive():
    if time.time() - _agent_cache["t"] < 5:
        return _agent_cache["ok"]
    ok = False
    try:
        with urllib.request.urlopen(AGENT + "/api/state", timeout=1.5) as r:
            ok = r.status == 200
    except Exception:
        ok = False
    _agent_cache.update(t=time.time(), ok=ok)
    return ok


# ---------------------------------------------------------------------------- animations
def clip_info(p, source):
    p = Path(p)
    info = {"name": p.stem if source != "skill" else f"{p.parent.parent.name} {p.parent.name}", "source": source,
            "path": rel(p), "url": file_url(p), "saved": int(p.stat().st_mtime), "seconds": 0, "frames": 0, "keys": None,
            "interp": None, "has_base": False, "meta": {}}
    data = read_json(p)
    if data is None:
        info["broken"] = True
    elif isinstance(data, list):
        info.update(frames=len(data), seconds=round(float(data[-1][0]), 3) if data else 0, interp="linear", kind="recording")
    else:
        frames = data.get("frames") or []
        keys = data.get("keys") or {}
        info.update(frames=len(frames), keys=sum(len(k) for k in keys.values()) or None,
                    interp=data.get("interp"), has_base=bool(data.get("base")), kind="keyed" if keys else "frames")
        if frames:
            info["seconds"] = round(float(frames[-1][0]), 3)
        elif keys:
            info["seconds"] = round(max((k["t"] if isinstance(k, dict) else k[0]) for ks in keys.values() for k in ks), 3)
        info["meta"] = {k: v for k, v in data.items() if k not in ("frames", "keys", "base", "interp")}
    cname = clipname(p)
    info["check_name"] = cname
    rep_dir = PROJECT / "anim" / "reports" / cname
    rep = read_json(rep_dir / "check.json")
    if rep is not None:
        info["report"] = rep
        info["report_url"] = file_url(rep_dir / "check.json")
        info["report_time"] = int((rep_dir / "check.json").stat().st_mtime)
        info["report_stale"] = info["report_time"] < info["saved"]
    if (rep_dir / "check.mp4").exists():
        info["video_url"] = file_url(rep_dir / "check.mp4")
    if source == "skill":
        info["skill"] = f"{p.parent.parent.name}/{p.parent.name}"
    return info


def animations():
    out = []
    CLIPS.mkdir(parents=True, exist_ok=True)
    for p in sorted(CLIPS.glob("*.json")):
        out.append(clip_info(p, "puppet"))
    for p in sorted(PROJECT.glob("skills/*/v*/clip.json")):
        out.append(clip_info(p, "skill"))
    out.sort(key=lambda c: -c["saved"])
    return out


# ---------------------------------------------------------------------------- training runs
_scalar_cache = {}


def run_scalars(run_dir):
    run_dir = Path(run_dir)
    key = str(run_dir)
    mtime = max([f.stat().st_mtime for f in run_dir.glob("events*")] or [0])
    hit = _scalar_cache.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    sc = read_scalars(str(run_dir))
    _scalar_cache[key] = (mtime, sc)
    return sc


def step_of(name):
    m = TRAILING_STEP.search(name)
    return int(m.group(1)) if m else -1


def score_summary(text):
    """The useful lines of a .score file: the ones after the ONNX path, minus the [ref] chatter."""
    lines = [l.rstrip() for l in text.splitlines()]
    keep = [l for l in lines if l.startswith("  ") and l.strip()]
    pushes = re.search(r"pushes survived (\d+/\d+)", text)
    falls = "no falls" if "no falls" in text else (re.search(r"(\d+ falls?)", text).group(1) if re.search(r"(\d+ falls?)", text) else None)
    speed = re.search(r"speed:\s*fwd\s*([+-]?[\d.]+)", text)
    return {"lines": keep, "pushes": pushes.group(1) if pushes else None, "falls": falls,
            "speed_fwd": float(speed.group(1)) if speed else None}


def run_entry(rd, run):
    log = rd / f"{run.name}.log"
    onnx = sorted(run.glob("*.onnx"), key=lambda p: step_of(p.name), reverse=True)
    ckpts = sorted([d for d in run.iterdir() if d.is_dir() and TRAILING_STEP.search(d.name)], key=lambda p: step_of(p.name), reverse=True)
    events = list(run.glob("events*"))
    scores = sorted(rd.glob(f"{run.name}*.score"))
    mt = [run.stat().st_mtime] + [p.stat().st_mtime for p in onnx[:1] + events] + ([log.stat().st_mtime] if log.exists() else [])
    e = {"id": f"{rd.name}/{run.name}", "runs_dir": rd.name, "name": run.name, "stage": run.name[4:] if run.name.startswith("cur_") else run.name,
         "path": rel(run), "mtime": int(max(mt)), "has_events": bool(events), "log_url": file_url(log) if log.exists() else None,
         "log_size": log.stat().st_size if log.exists() else 0,
         "onnx": [{"file": p.name, "step": step_of(p.name), "url": file_url(p), "size": p.stat().st_size} for p in onnx],
         "checkpoints": [c.name for c in ckpts], "steps": None, "reward": None, "reward_std": None, "scores": []}
    for s in scores:
        txt = s.read_text(errors="replace")
        parsed = None
        if txt.lstrip().startswith("{"):
            parsed = read_json(s)
        e["scores"].append({"file": s.name, "url": file_url(s), "json": parsed, **(score_summary(txt) if parsed is None else {"lines": []})})
    if log.exists():
        hits = STEP_LINE.findall(tail_text(log, 40000))
        if hits:
            st, rw, sd = hits[-1]
            e.update(steps=int(st), reward=float(rw), reward_std=float(sd) if sd else None)
        e["running"] = time.time() - log.stat().st_mtime < 180
    if e["steps"] is None and events:
        sc = run_scalars(run).get("eval/episode_reward")
        if sc:
            e.update(steps=int(sc[-1][0]), reward=float(sc[-1][1]))
    return e


def runs_list():
    out = []
    for rd in sorted(PROJECT.glob("sim/train_pg/runs*")):
        if not rd.is_dir():
            continue
        for run in sorted(rd.iterdir()):
            if run.is_dir() and (any(run.glob("events*")) or any(run.glob("*.onnx"))):
                out.append(run_entry(rd, run))
    out.sort(key=lambda e: -e["mtime"])
    return out


def report_media():
    out = []
    for d in sorted(PROJECT.glob("sim/report*")):
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p.suffix.lower() in (".mp4", ".png", ".jpg", ".html", ".json") and p.is_file():
                out.append({"name": p.name, "dir": d.name, "path": rel(p), "url": file_url(p), "size": p.stat().st_size,
                            "mtime": int(p.stat().st_mtime), "kind": p.suffix.lower()[1:]})
    out.sort(key=lambda f: -f["mtime"])
    return out


def run_detail(runs_dir, name):
    rd = PROJECT / "sim" / "train_pg" / runs_dir
    run = rd / name
    if not (rd.resolve().is_relative_to((PROJECT / "sim" / "train_pg").resolve()) and runs_dir.startswith("runs") and run.is_dir()):
        return None
    e = run_entry(rd, run)
    sc = run_scalars(run)
    e["curves"] = {k: v for k, v in sc.items() if not k.endswith("_std")}
    e["log_tail"] = tail_text(rd / f"{name}.log", 12000)
    for s in e["scores"]:
        s["text"] = (rd / s["file"]).read_text(errors="replace")
    tok = re.escape(e["stage"])
    pat = re.compile(rf"(^|[^a-z0-9]){tok}([^a-z0-9]|$)", re.I)
    e["media"] = [m for m in report_media() if pat.search(m["name"]) and m["kind"] in ("mp4", "png")]
    return e


# ---------------------------------------------------------------------------- skills
def ratio_of(doc):
    """A success ratio out of a report. battery.json (tools/battery.py): overall_ratio + overall
    "280/400" + passed; hardware.json: ratio + trials; otherwise common key pairs or the mean of
    per-test dicts."""
    if not isinstance(doc, dict):
        return None
    if isinstance(doc.get("overall_ratio"), (int, float)) and not isinstance(doc["overall_ratio"], bool):
        return {"value": float(doc["overall_ratio"]), "text": str(doc.get("overall") or f"{doc['overall_ratio']:.2f}"),
                "passed": doc.get("passed") if isinstance(doc.get("passed"), bool) else None}
    if isinstance(doc.get("ratio"), (int, float)) and not isinstance(doc["ratio"], bool) and doc.get("trials"):
        return {"value": float(doc["ratio"]), "text": f"{round(doc['ratio'] * doc['trials'])}/{doc['trials']}"}
    for k in ("ratio", "success_ratio", "pass_ratio", "success_rate", "pass_rate"):
        v = doc.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return {"value": float(v), "text": f"{v:.2f}"}
    for num, den in (("passed", "total"), ("successes", "trials"), ("success", "trials"), ("ok", "n"), ("pass", "runs"),
                     ("passes", "episodes"), ("passed", "episodes"), ("survived", "trials")):
        a, b = doc.get(num), doc.get(den)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and b:
            return {"value": a / b, "text": f"{a}/{b}"}
    subs = [r for r in (ratio_of(v) for v in doc.values() if isinstance(v, dict)) if r]
    if subs:
        vals = [s["value"] for s in subs]
        return {"value": sum(vals) / len(vals), "n": len(subs),
                "text": subs[0]["text"] if len(subs) == 1 else f"{min(vals):.2f}–{max(vals):.2f} ({len(subs)})"}
    if isinstance(doc.get("pass"), bool):
        return {"value": 1.0 if doc["pass"] else 0.0, "text": "pass" if doc["pass"] else "fail"}
    return None


def status_of(doc):
    """sim2sim.json: match true/false/null (null = not applicable); else status-like keys."""
    if not isinstance(doc, dict):
        return None
    if "match" in doc:
        return "match" if doc["match"] is True else "mismatch" if doc["match"] is False else "n/a"
    for k in ("status", "result", "verdict"):
        if isinstance(doc.get(k), str):
            return doc[k]
    for k in ("pass", "passed", "ok", "match"):
        if isinstance(doc.get(k), bool):
            return "pass" if doc[k] else "fail"
    r = ratio_of(doc)
    return r["text"] if r else None


def skill_entry(d, full=False):
    sj = read_json(d / "skill.json", {}) or {}
    rep = d / "report"
    battery, sim2sim, hardware = read_json(rep / "battery.json"), read_json(rep / "sim2sim.json"), read_json(rep / "hardware.json")
    e = {"id": f"{d.parent.name}/{d.name}", "name": sj.get("name", d.parent.name), "dir": d.parent.name, "version": d.name,
         "type": sj.get("type"), "created": sj.get("created"), "path": rel(d), "skill": sj,
         "parts": {"skill.json": (d / "skill.json").exists(), "clip.json": (d / "clip.json").exists(),
                   "policy.onnx": (d / "policy.onnx").exists(), "battery": battery is not None,
                   "sim2sim": sim2sim is not None, "hardware": hardware is not None},
         "battery": ratio_of(battery), "sim2sim": status_of(sim2sim), "hardware": ratio_of(hardware),
         "clip_url": file_url(d / "clip.json") if (d / "clip.json").exists() else None,
         "onnx_url": file_url(d / "policy.onnx") if (d / "policy.onnx").exists() else None,
         "mtime": int(max([d.stat().st_mtime] + [p.stat().st_mtime for p in d.rglob("*") if p.is_file()]))}
    e["complete"] = all(e["parts"][k] for k in ("skill.json", "clip.json", "policy.onnx"))
    if full:
        e["reports"] = {"battery": battery, "sim2sim": sim2sim, "hardware": hardware}
        e["media"] = sorted(({"name": p.name, "url": file_url(p), "kind": p.suffix.lower()[1:], "size": p.stat().st_size}
                             for p in rep.glob("*") if p.suffix.lower() in (".mp4", ".png", ".jpg")), key=lambda m: m["name"])
        e["files"] = sorted(rel(p) for p in d.rglob("*") if p.is_file())
        if (d / "clip.json").exists():
            e["clip"] = clip_info(d / "clip.json", "skill")
    return e


def skills_list():
    out = [skill_entry(d) for d in sorted(PROJECT.glob("skills/*/v*")) if d.is_dir()]
    out.sort(key=lambda e: (e["dir"], -step_of("_" + e["version"][1:]) if e["version"][1:].isdigit() else 0))
    return out


# ---------------------------------------------------------------------------- telemetry
def telemetry(days=3):
    files = sorted(PROJECT.glob("mechatronics/telemetry/*.jsonl"), reverse=True)[:max(1, days)]
    rows = []
    for p in reversed(files):
        for line in p.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    series = {"t": [], "voltage": {}, "temps": {}, "currents": {}}
    stride = max(1, len(rows) // 1500)
    for i, r in enumerate(rows):
        if i % stride and i != len(rows) - 1:
            continue
        series["t"].append(r.get("t"))
        for metric in ("voltage", "temps", "currents"):
            m = r.get(metric) or {}
            for j, v in m.items():
                series[metric].setdefault(j, [None] * (len(series["t"]) - 1)).append(v)
            for j in series[metric]:
                if len(series[metric][j]) < len(series["t"]):
                    series[metric][j].append(None)
    return {"files": [rel(p) for p in files], "count": len(rows), "last": rows[-1] if rows else None, "series": series}


# ---------------------------------------------------------------------------- experiments
def experiments():
    log = DOCS["log"].read_text(errors="replace") if DOCS["log"].exists() else ""
    q = DOCS["questions"].read_text(errors="replace") if DOCS["questions"].exists() else ""
    parts = re.split(r"(?m)^(?=## )", log)
    intro = parts[0] if parts and not parts[0].startswith("## ") else ""
    entries = [p for p in parts if p.startswith("## ")]
    out = []
    for p in entries:
        head, _, body = p.partition("\n")
        out.append({"title": head[3:].strip(), "body": body.strip()})
    out.reverse()
    return {"intro": intro.strip(), "entries": out, "questions": q, "log_exists": DOCS["log"].exists(),
            "questions_exists": DOCS["questions"].exists()}


# ---------------------------------------------------------------------------- progress
def update_progress(body):
    cur = read_json(PROGRESS_SL, {}) or {}
    if "milestones" in body and isinstance(body["milestones"], list):
        cur["milestones"] = body["milestones"]
    if "workstreams" in body and isinstance(body["workstreams"], list):
        cur["workstreams"] = body["workstreams"]
    m = body.get("milestone")
    if isinstance(m, dict) and m.get("id"):
        for row in cur.setdefault("milestones", []):
            if row.get("id") == m["id"]:
                for k in ("status", "notes", "title", "weeks"):
                    if k in m:
                        row[k] = m[k]
                break
        else:
            cur["milestones"].append({k: m.get(k, "") for k in ("id", "title", "status", "weeks", "notes")})
    w = body.get("workstream")
    if isinstance(w, dict) and w.get("item"):
        for row in cur.setdefault("workstreams", []):
            if row.get("team") == w.get("team") and row.get("item") == w["item"]:
                for k in ("status", "notes"):
                    if k in w:
                        row[k] = w[k]
                break
        else:
            cur["workstreams"].append({k: w.get(k, "") for k in ("team", "item", "status", "notes")})
    cur["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if PROGRESS_SL.exists():
        shutil.copy2(PROGRESS_SL, PROGRESS_SL.with_suffix(".json.bak"))
    fd, tmp = tempfile.mkstemp(dir=PROGRESS_SL.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(cur, f, indent=1, ensure_ascii=False)
    os.replace(tmp, PROGRESS_SL)
    return cur


# ---------------------------------------------------------------------------- handler
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(HERE), **kw)

    def log_message(self, fmt, *args):  # keep the journal quiet: errors only
        if args and str(args[1]).startswith(("4", "5")) and "/duck/api/" not in str(args[0]):
            super().log_message(fmt, *args)

    def translate_path(self, path):
        path = unquote(urlparse(path).path)
        for prefix, root in ROUTES.items():
            if path.startswith(prefix):
                target = (root / path[len(prefix):]).resolve()
                if target.is_relative_to(root.resolve()):
                    return str(target)
                return str(root / "__forbidden__")
        if path in ("/puppet", "/puppet/", "/puppet/index.html"):
            return str(HERE / "puppet.html")
        if path.startswith("/build/data/"):
            path = "/data/" + path[len("/build/data/"):]
        if path.startswith("/data/") and not path.endswith("catalog.json"):
            return str(HERE / "__forbidden__")  # progress and clips only via /api/ and /files/
        if path.endswith(".py") or "/__pycache__/" in path:
            return str(HERE / "__forbidden__")  # the server's own sources are not a page
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, code, text, ctype="text/plain; charset=utf-8"):
        body = text.encode("utf-8", "replace")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        length = int(self.headers.get("Content-Length", 0))
        if not 0 <= length <= MAX_BODY:
            raise ValueError("bad size")
        return self.rfile.read(length)

    def json_body(self):
        raw = self.body()
        return json.loads(raw) if raw else {}

    # ---- whitelisted project files, with byte ranges so phones can seek the videos
    def send_file(self, relpath, head=False, download=False):
        p = servable(relpath)
        if p is None:
            return self.send_json(404, {"error": "not a servable file"})
        size = p.stat().st_size
        ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        start, end, code = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes=") and size:
            a, _, b = rng[6:].partition("-")
            try:
                if a:
                    start, end = int(a), (int(b) if b else size - 1)
                else:
                    start, end = max(0, size - int(b)), size - 1
            except ValueError:
                start, end = 0, size - 1
            end = min(end, size - 1)
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            code = 206
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(p.stat().st_mtime))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{p.name}"')
        self.end_headers()
        if head:
            return
        with open(p, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(65536, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    # ---- the duck's agent, proxied so the page has one origin
    def proxy(self, method):
        url = AGENT + urlparse(self.path).path[len("/duck"):]
        data = self.body() if method == "POST" else None
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=4) as r:
                payload = r.read()
        except urllib.error.URLError as e:
            return self.send_json(502, {"error": "no agent on the duck", "detail": str(e.reason)})
        except Exception as e:
            return self.send_json(502, {"error": "agent request failed", "detail": str(e)})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_HEAD(self):
        path = urlparse(self.path).path
        if path.startswith("/files/"):
            return self.send_file(unquote(path[len("/files/"):]), head=True)
        return super().do_HEAD()

    def do_GET(self):
        url = urlparse(self.path)
        path, q = url.path, parse_qs(url.query)
        try:
            if path.startswith("/duck/api/"):
                return self.proxy("GET")
            if path.startswith("/files/"):
                return self.send_file(unquote(path[len("/files/"):]), download="download" in q)
            if path == "/puppet":
                self.send_response(302); self.send_header("Location", "/puppet/"); self.end_headers(); return
            if path.startswith("/api/"):
                return self.api_get(path[5:], q)
            return super().do_GET()
        except BrokenPipeError:
            pass
        except Exception as e:  # a bug in one endpoint must not take the page down
            try:
                self.send_json(500, {"error": f"{type(e).__name__}: {e}"})
            except Exception:
                pass

    def api_get(self, ep, q):
        if ep == "progress":
            return self.send_json(200, read_json(PROGRESS, {}) or {})
        if ep == "clips":
            CLIPS.mkdir(parents=True, exist_ok=True)
            clips = sorted((clip_summary(p) for p in CLIPS.glob("*.json")), key=lambda c: -c["saved"])
            return self.send_json(200, clips)
        if ep.startswith("clips/"):
            p = clip_path(ep[len("clips/"):])
            if p is None or not p.exists():
                return self.send_json(404, {"error": "no such clip"})
            return self.send_json(200, json.loads(p.read_text()))
        if ep == "status":
            g = gpu_status()
            running = JOBS.running()
            return self.send_json(200, {"time": time.strftime("%H:%M:%S"), "gpu": g, "jobs_running": len(running),
                                        "gpu_busy": any(m.get("gpu") for m in running),
                                        "running": [{"id": m["id"], "tool_name": m["tool_name"], "gpu": m["gpu"], "started": m["started"]} for m in running],
                                        "agent": agent_alive()})
        if ep == "gpu":
            return self.send_json(200, gpu_status())
        if ep == "tools":
            return self.send_json(200, JOBS.tools_for_ui())
        if ep == "jobs":
            return self.send_json(200, [{k: v for k, v in m.items() if k != "env"} for m in JOBS.list()])
        if ep.startswith("jobs/"):
            jid, _, what = ep[5:].partition("/")
            if what == "log":
                text, size = JOBS.log_tail(jid, int(q.get("tail", [65536])[0]))
                if text is None:
                    return self.send_json(404, {"error": "no such job"})
                return self.send_text(200, text)
            m = JOBS.get(jid, with_urls)
            if m is None:
                return self.send_json(404, {"error": "no such job"})
            return self.send_json(200, m)
        if ep == "animations":
            return self.send_json(200, animations())
        if ep == "runs":
            return self.send_json(200, runs_list())
        if ep.startswith("runs/"):
            rd, _, name = ep[5:].partition("/")
            e = run_detail(rd, name)
            return self.send_json(200, e) if e else self.send_json(404, {"error": "no such run"})
        if ep == "reports":
            return self.send_json(200, report_media())
        if ep == "skills":
            return self.send_json(200, skills_list())
        if ep.startswith("skills/"):
            name, _, ver = ep[7:].partition("/")
            d = PROJECT / "skills" / name / ver
            if not (NAME.match(name or "") and NAME.match(ver or "") and d.is_dir()):
                return self.send_json(404, {"error": "no such skill package"})
            return self.send_json(200, skill_entry(d, full=True))
        if ep == "telemetry":
            return self.send_json(200, telemetry(int(q.get("days", [3])[0])))
        if ep == "experiments":
            return self.send_json(200, experiments())
        if ep == "progress_skill_library":
            return self.send_json(200, read_json(PROGRESS_SL, {}) or {})
        if ep == "doc":
            p = DOCS.get(q.get("name", [""])[0])
            if p is None or not p.exists():
                return self.send_json(404, {"error": "no such document"})
            return self.send_text(200, p.read_text(errors="replace"), "text/markdown; charset=utf-8")
        return self.send_json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path.startswith("/duck/api/"):
                return self.proxy("POST")
            if path == "/api/agent":
                try:
                    action = self.json_body().get("action", "start")
                except Exception:
                    return self.send_json(400, {"error": "invalid json"})
                return self.send_json(200, agent_control(action))
            if path == "/api/jobs":
                try:
                    body = self.json_body()
                    assert isinstance(body, dict)
                except Exception:
                    return self.send_json(400, {"error": "invalid json"})
                try:
                    m = JOBS.start(body.get("tool_id", ""), body.get("args") or {})
                except JobError as e:
                    return self.send_json(e.code, {"error": e.msg})
                return self.send_json(200, {"id": m["id"], "job": {k: v for k, v in m.items() if k != "env"}})
            if path.startswith("/api/jobs/") and path.endswith("/kill"):
                jid = path[len("/api/jobs/"):-len("/kill")]
                try:
                    return self.send_json(200, JOBS.kill(jid))
                except JobError as e:
                    return self.send_json(e.code, {"error": e.msg})
            if path == "/api/progress_skill_library":
                try:
                    body = self.json_body()
                    assert isinstance(body, dict)
                except Exception:
                    return self.send_json(400, {"error": "invalid json"})
                return self.send_json(200, update_progress(body))
            return self.send_json(404, {"error": "not found"})
        except BrokenPipeError:
            pass
        except Exception as e:
            try:
                self.send_json(500, {"error": f"{type(e).__name__}: {e}"})
            except Exception:
                pass

    def do_PUT(self):
        path = urlparse(self.path).path
        if path == "/api/progress":
            return self.save_json(PROGRESS, backup=True)
        if path.startswith("/api/clips/"):
            p = clip_path(path[len("/api/clips/"):])
            if p is None:
                return self.send_json(400, {"error": "bad clip name"})
            return self.save_json(p, backup=False, kind=(list, dict))
        return self.send_json(404, {"error": "not found"})

    def do_DELETE(self):
        path = urlparse(self.path).path
        if path.startswith("/api/clips/"):
            p = clip_path(path[len("/api/clips/"):])
            if p is None or not p.exists():
                return self.send_json(404, {"error": "no such clip"})
            p.unlink()
            return self.send_json(200, {"ok": True})
        return self.send_json(404, {"error": "not found"})

    def save_json(self, target, backup, kind=dict):
        try:
            state = json.loads(self.body())
            assert isinstance(state, kind)
        except Exception:
            return self.send_json(400, {"error": "invalid json"})
        target.parent.mkdir(parents=True, exist_ok=True)
        if backup and target.exists():
            shutil.copy2(target, target.with_suffix(".json.bak"))
        fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=None if kind is list else 1)
        os.replace(tmp, target)
        return self.send_json(200, {"ok": True})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8430)
    ap.add_argument("--lan", action="store_true", help="listen on all interfaces (phone access)")
    args = ap.parse_args()
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    print(f"Duck dashboard on http://{'localhost' if not args.lan else host}:{args.port}", flush=True)
    srv = ThreadingHTTPServer((host, args.port), Handler)
    srv.daemon_threads = True
    srv.serve_forever()


if __name__ == "__main__":
    main()
