"""Jobs runner for the Duck Dashboard: run a registered tool as a subprocess, keep its log.

Registry: tracker/data/tools.json, a list of
  {id, name, group, cwd, command: ["tok", "{arg}", ...], args: [{name, label, type, default,
   choices?, glob?, derive?, optional?}], gpu: bool, outputs: [globs], help?}
Arg types: text | number | bool | choice | file | derived.
  - bool renders as the arg's `flag` (default "--<name>") when true, nothing when false
  - file values are paths relative to the project root; they must exist under it and are
    passed absolute, so the tool's cwd does not matter
  - derived args never appear in the form; `derive` is one of
        stem:<arg>       basename of <arg> without its extension
        clipname:<arg>   stem, except skills/<s>/v<N>/clip.json -> <s>_v<N>
        first:<arg>      first whitespace-separated word of <arg>
        now              timestamp YYYYMMDD-HHMMSS
    (an ordinary arg may also carry `derive`: used when the field is left blank)
  - a command token that renders to "" is dropped, so optional args vanish cleanly
  - leading tokens "env" and KEY=VALUE become the job's environment
Every job has data/jobs/<id>.json (metadata) and data/jobs/<id>.log. At most one job with
gpu=true runs at a time.
"""
import glob
import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

PLACEHOLDER = re.compile(r"\{(\w+)\}")
ENV_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
TYPES = ("text", "number", "bool", "choice", "file", "derived")


class JobError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code, self.msg = code, msg


def clipname(path):
    """Report folder name for a clip: its stem, except a skill package's clip.json is
    named after the package (skills/bow/v1/clip.json -> bow_v1) so reports never collide."""
    p = Path(path)
    if p.name == "clip.json" and p.parent.parent.parent.name == "skills":
        return f"{p.parent.parent.name}_{p.parent.name}"
    return p.stem


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Jobs:
    def __init__(self, project, data_dir, tools_path):
        self.project = Path(project).resolve()
        self.dir = Path(data_dir) / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tools_path = Path(tools_path)
        self._tools, self._tools_mtime = [], None
        self.lock = threading.Lock()
        self.procs = {}          # id -> Popen while running
        self.meta = {}           # id -> dict (every job we know about)
        for p in sorted(self.dir.glob("*.json")):
            try:
                m = json.loads(p.read_text())
            except Exception:
                continue
            if m.get("ended") is None:       # the server restarted under it
                m["ended"] = now_iso(); m["exit"] = None; m["status"] = "lost"
                self._save(m)
            self.meta[m["id"]] = m

    # ---------------------------------------------------------------- registry
    def tools(self):
        try:
            mtime = self.tools_path.stat().st_mtime
        except FileNotFoundError:
            return []
        if mtime != self._tools_mtime:
            self._tools = json.loads(self.tools_path.read_text())
            self._tools_mtime = mtime
        return self._tools

    def tool(self, tool_id):
        for t in self.tools():
            if t["id"] == tool_id:
                return t
        return None

    def options(self, arg):
        """Choices for a file arg: paths relative to the project root matching its glob(s)."""
        pats = arg.get("glob") or ""
        pats = pats if isinstance(pats, list) else [p for p in pats.split(";") if p]
        found = []
        for pat in pats:
            for hit in glob.glob(str(self.project / pat), recursive=True):
                try:
                    rel = str(Path(hit).resolve().relative_to(self.project))
                except ValueError:
                    continue
                found.append((os.path.getmtime(hit), rel))
        found.sort(key=lambda x: -x[0])
        seen, out = set(), []
        for _, rel in found:
            if rel not in seen:
                seen.add(rel); out.append(rel)
        return out

    def tools_for_ui(self):
        out = []
        for t in self.tools():
            t = json.loads(json.dumps(t))
            for a in t.get("args", []):
                if a.get("type") == "file":
                    a["options"] = self.options(a)
            out.append(t)
        return out

    # ---------------------------------------------------------------- rendering
    def _derive(self, spec, values):
        kind, _, src = spec.partition(":")
        v = values.get(src, "")
        if kind == "stem":
            return Path(v).stem if v else ""
        if kind == "clipname":
            return clipname(v) if v else ""
        if kind == "first":
            return v.split()[0] if v.split() else ""
        if kind == "now":
            return datetime.now().strftime("%Y%m%d-%H%M%S")
        raise JobError(500, f"unknown derive rule {spec!r} in tools.json")

    def render(self, tool, args):
        """-> (argv, env, cwd, values). Raises JobError on a bad argument."""
        args = args or {}
        values = {}
        for a in tool.get("args", []):
            name, typ = a["name"], a.get("type", "text")
            if typ not in TYPES:
                raise JobError(500, f"tools.json: arg {name} has unknown type {typ}")
            raw = args.get(name, a.get("default"))
            if typ == "derived":
                continue
            if typ == "bool":
                on = raw in (True, 1, "1", "true", "on", "yes")
                values[name] = a.get("flag", f"--{name}") if on else ""
                continue
            s = "" if raw is None else str(raw).strip()
            if not s and a.get("derive"):
                s = self._derive(a["derive"], values)
            if typ == "number":
                if s:
                    try:
                        float(s)
                    except ValueError:
                        raise JobError(400, f"{a.get('label', name)} must be a number")
            elif typ == "choice":
                if s and s not in [str(c) for c in a.get("choices", [])]:
                    raise JobError(400, f"{a.get('label', name)}: {s!r} is not one of the choices")
            elif typ == "file":
                if s:
                    p = (self.project / s).resolve()
                    if not p.is_relative_to(self.project) or not p.exists():
                        raise JobError(400, f"{a.get('label', name)}: {s!r} is not a file inside the project")
                    s = str(p)
            if not s and not a.get("optional") and typ != "text":
                raise JobError(400, f"{a.get('label', name)} is required")
            if not s and typ == "text" and a.get("required"):
                raise JobError(400, f"{a.get('label', name)} is required")
            values[name] = s
        for a in tool.get("args", []):         # derived args, after everything they may depend on
            if a.get("type") == "derived":
                values[a["name"]] = self._derive(a["derive"], values)

        def sub(tok):
            return PLACEHOLDER.sub(lambda m: str(values.get(m.group(1), "")), tok)

        argv, env = [], {}
        head = True
        for tok in tool["command"]:
            had = bool(PLACEHOLDER.search(tok))
            s = sub(tok)
            if head and tok == "env":
                continue
            if head and ENV_TOKEN.match(tok):
                k, _, v = s.partition("=")
                env[k] = v
                continue
            head = False
            if had and not s.strip():
                continue
            argv.append(s)
        if not argv:
            raise JobError(500, "tools.json: command is empty after rendering")
        cwd = (self.project / (tool.get("cwd") or ".")).resolve()
        return argv, env, cwd, values

    # ---------------------------------------------------------------- lifecycle
    def running(self):
        return [m for m in self.meta.values() if m.get("ended") is None]

    def start(self, tool_id, args):
        tool = self.tool(tool_id)
        if tool is None:
            raise JobError(404, f"no tool {tool_id!r}")
        argv, env_extra, cwd, values = self.render(tool, args)
        if not cwd.is_dir():
            raise JobError(400, f"the tool's working directory is missing: {cwd}")
        with self.lock:
            if tool.get("gpu"):
                busy = [m for m in self.running() if m.get("gpu")]
                if busy:
                    b = busy[0]
                    raise JobError(409, f"the GPU is busy with job {b['id']} ({b['tool_name']}); wait for it or kill it first")
            base = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + re.sub(r"[^a-z0-9]+", "", tool_id.lower())[:16]
            jid, n = base, 1
            while jid in self.meta:
                n += 1
                jid = f"{base}_{n}"
            env = dict(os.environ)
            env.update({k: str(v) for k, v in (tool.get("env") or {}).items()})
            env.update(env_extra)
            env.setdefault("PYTHONUNBUFFERED", "1")
            log_path = self.dir / f"{jid}.log"
            meta = {"id": jid, "tool_id": tool_id, "tool_name": tool.get("name", tool_id), "group": tool.get("group"),
                    "gpu": bool(tool.get("gpu")), "args": values, "command": argv, "command_str": shlex.join(argv),
                    "env": env_extra, "cwd": str(cwd), "started": now_iso(), "ended": None, "exit": None,
                    "status": "running", "pid": None, "log": log_path.name,
                    "outputs": [PLACEHOLDER.sub(lambda m: str(values.get(m.group(1), "")), g) for g in tool.get("outputs", [])]}
            with open(log_path, "ab") as log:
                log.write((f"$ cd {cwd}\n$ " + (" ".join(f"{k}={shlex.quote(v)}" for k, v in env_extra.items()) + " " if env_extra else "")
                           + shlex.join(argv) + "\n\n").encode())
                log.flush()
                try:
                    proc = subprocess.Popen(argv, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except OSError as e:
                    meta.update(ended=now_iso(), exit=127, status="failed", error=str(e))
                    log.write(f"could not start: {e}\n".encode())
                    self.meta[jid] = meta
                    self._save(meta)
                    raise JobError(400, f"could not start {argv[0]}: {e}")
            meta["pid"] = proc.pid
            self.meta[jid] = meta
            self.procs[jid] = proc
            self._save(meta)
        threading.Thread(target=self._reap, args=(jid, proc), daemon=True).start()
        return meta

    def _reap(self, jid, proc):
        rc = proc.wait()
        with self.lock:
            m = self.meta[jid]
            m["ended"] = now_iso()
            m["exit"] = rc
            m["status"] = "done" if rc == 0 else ("killed" if m.get("status") == "killing" or rc < 0 else "failed")
            self.procs.pop(jid, None)
            self._save(m)

    def kill(self, jid):
        proc = self.procs.get(jid)
        m = self.meta.get(jid)
        if m is None:
            raise JobError(404, "no such job")
        if proc is None or proc.poll() is not None:
            return {"ok": False, "msg": "the job is not running"}
        m["status"] = "killing"
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

        def hammer():
            time.sleep(8)
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        threading.Thread(target=hammer, daemon=True).start()
        return {"ok": True, "msg": "stop signal sent"}

    def _save(self, m):
        tmp = self.dir / f"{m['id']}.json.tmp"
        tmp.write_text(json.dumps(m, indent=1))
        os.replace(tmp, self.dir / f"{m['id']}.json")

    # ---------------------------------------------------------------- queries
    def list(self):
        return sorted(self.meta.values(), key=lambda m: m["started"], reverse=True)

    def get(self, jid, outputs_fn=None):
        m = self.meta.get(jid)
        if m is None:
            return None
        m = dict(m)
        if m.get("ended") and m.get("started"):
            try:
                m["seconds"] = round((datetime.fromisoformat(m["ended"]) - datetime.fromisoformat(m["started"])).total_seconds(), 1)
            except ValueError:
                pass
        files = []
        for pat in m.get("outputs", []):
            for hit in sorted(glob.glob(str(self.project / pat), recursive=True)):
                p = Path(hit)
                if p.is_file():
                    try:
                        rel = str(p.resolve().relative_to(self.project))
                    except ValueError:
                        continue
                    files.append({"path": rel, "size": p.stat().st_size, "mtime": int(p.stat().st_mtime)})
        m["files"] = outputs_fn(files) if outputs_fn else files
        return m

    def log_tail(self, jid, tail=65536):
        p = self.dir / f"{jid}.log"
        if not p.exists():
            return None, 0
        size = p.stat().st_size
        with open(p, "rb") as f:
            if size > tail:
                f.seek(size - tail)
                data = f.read()
                nl = data.find(b"\n")
                data = data[nl + 1:] if nl >= 0 else data
            else:
                data = f.read()
        return data.decode("utf-8", "replace"), size
