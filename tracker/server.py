#!/usr/bin/env python3
"""Duck Puppet - the local web server behind http://localhost:8430.

Serves the puppet page (pose, hold, record and replay the real duck's 14 servos), proxies
it to the agent running on the duck's Pi (tracker/duck_agent.py, deployed over ssh), keeps
recorded clips in data/clips/, and still serves the old build tracker at /build/.
Standard library only.  Usage:  python3 server.py [--port 8430] [--lan]
"""
import argparse
import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA = HERE / "data"
CLIPS = DATA / "clips"
PROGRESS = DATA / "progress.json"
PI = "bdxv2@bdxv2.local"
PI_PY = "~/.virtualenvs/open-duck-mini-runtime/bin/python"
AGENT = "http://bdxv2.local:8091"
ROBOT = PROJECT / "official_repo" / "mini_bdx" / "robots" / "open_duck_mini_v2"
ROUTES = {  # url prefix -> directory served read-only
    "/robot/": ROBOT, "/stl/": PROJECT / "print",
    "/build/robot/": ROBOT, "/build/stl/": PROJECT / "print", "/build/vendor/": HERE / "vendor",
    "/build/docs/": HERE / "docs",
}
MAX_BODY = 8_000_000
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,60}$")

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("model/stl", ".stl")


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


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(HERE), **kw)

    def log_message(self, fmt, *args):  # keep the journal quiet: errors only
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)

    def translate_path(self, path):
        path = unquote(urlparse(path).path)
        for prefix, root in ROUTES.items():
            if path.startswith(prefix):
                target = (root / path[len(prefix):]).resolve()
                if target.is_relative_to(root.resolve()):
                    return str(target)
                return str(root / "__forbidden__")
        if path.startswith("/build/data/"):
            path = "/data/" + path[len("/build/data/"):]
        if path.startswith("/data/") and not path.endswith("catalog.json"):
            return str(HERE / "__forbidden__")  # progress and clips only via /api/
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

    def body(self):
        length = int(self.headers.get("Content-Length", 0))
        if not 0 <= length <= MAX_BODY:
            raise ValueError("bad size")
        return self.rfile.read(length)

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

    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith("/duck/api/"):
            return self.proxy("GET")
        if path == "/api/progress":
            try:
                return self.send_json(200, json.loads(PROGRESS.read_text()))
            except FileNotFoundError:
                return self.send_json(200, {})
        if path == "/api/clips":
            CLIPS.mkdir(parents=True, exist_ok=True)
            clips = sorted((clip_summary(p) for p in CLIPS.glob("*.json")), key=lambda c: -c["saved"])
            return self.send_json(200, clips)
        if path.startswith("/api/clips/"):
            p = clip_path(path[len("/api/clips/"):])
            if p is None or not p.exists():
                return self.send_json(404, {"error": "no such clip"})
            return self.send_json(200, json.loads(p.read_text()))
        return super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if path.startswith("/duck/api/"):
            return self.proxy("POST")
        if path == "/api/agent":
            try:
                action = json.loads(self.body() or b"{}").get("action", "start")
            except Exception:
                return self.send_json(400, {"error": "invalid json"})
            return self.send_json(200, agent_control(action))
        return self.send_json(404, {"error": "not found"})

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
    print(f"Duck puppet on http://{'localhost' if not args.lan else host}:{args.port}", flush=True)
    ThreadingHTTPServer((host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
