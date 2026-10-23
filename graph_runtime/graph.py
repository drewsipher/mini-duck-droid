"""Motion graph config: nodes are skill packages, edges are the only transitions that exist.

graphs/<name>.json:
    {
      "name": "demo",
      "root": "graph_runtime/demo_skills",          # package dirs are relative to this (or absolute)
      "nodes": {"walk": "walk/v1", "bow": "bow/v1", "sit": "sit/v1"},
      "start": "walk",
      "idle": {"node": "walk", "command": [0, 0, 0]},   # idle is locomotion at zero command, not a node
      "recovery": {"node": "sit", "window_s": 0.3},     # the abort target
      "edges": [
        {"from": "walk", "to": "bow", "phase": [0, 1], "contact": "double", "speed": [0, 0.03],
         "window_s": 0.4, "wait_timeout_s": 3.0},
        ...
      ]
    }

Validation (framework rules 2 and 5): only listed edges exist; every edge names existing packages
whose battery passed and whose own entry/exit declarations allow the edge; the recovery node is a
recovery package reachable from every node through listed edges; episodic nodes have a return edge.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from graph_runtime.packages import JOINTS, PackageError, SkillPackage, load_package

CONTACT_MODES = ("any", "double", "left", "right", "flight")
HEAD_JOINTS = JOINTS[5:9]


class GraphError(Exception):
    pass


@dataclass
class Edge:
    src: str
    dst: str
    phase: list                      # [lo, hi] window on the OUTGOING skill's phase; lo > hi wraps
    contact: str = "any"
    speed: list | None = None        # [lo, hi] on the commanded planar speed (m/s), None = any
    window_s: float = 0.3            # blend duration
    wait_timeout_s: float = 2.0      # give up waiting for the window after this
    kind: str = "normal"             # normal | return | recovery

    def phase_open(self, phase: float) -> bool:
        lo, hi = self.phase
        if lo <= hi:
            return lo <= phase <= hi
        return phase >= lo or phase <= hi

    def contact_open(self, contacts) -> bool:
        l, r = bool(contacts[0] > 0.5), bool(contacts[1] > 0.5)
        return {"any": True, "double": l and r, "left": l and not r, "right": r and not l,
                "flight": not l and not r}[self.contact]

    def speed_open(self, speed: float) -> bool:
        return self.speed is None or (self.speed[0] - 1e-9 <= speed <= self.speed[1] + 1e-9)

    def unmet(self, phase, contacts, speed) -> list:
        out = []
        if not self.phase_open(phase):
            out.append(f"phase {phase:.2f} not in {self.phase}")
        if not self.contact_open(contacts):
            out.append(f"contact {self.contact}")
        if not self.speed_open(speed):
            out.append(f"speed {speed:.2f} not in {self.speed}")
        return out


@dataclass
class MotionGraph:
    name: str
    path: Path | None
    nodes: dict                      # name -> SkillPackage
    edges: list                      # [Edge]
    start: str
    idle_node: str
    idle_command: list
    recovery_node: str
    recovery_window_s: float
    raw: dict = field(default_factory=dict)

    def edge(self, src: str, dst: str) -> Edge | None:
        for e in self.edges:
            if e.src == src and e.dst == dst:
                return e
        return None

    def edges_from(self, src: str) -> list:
        return [e for e in self.edges if e.src == src]

    def is_overlay(self, node: str) -> bool:
        return node in self.nodes and self.nodes[node].type == "overlay"

    def abort_edge(self, node: str) -> Edge | None:
        """The edge an abort from `node` takes: the listed one, else an implied one when both packages allow it."""
        if node == self.recovery_node:
            return None
        e = self.edge(node, self.recovery_node)
        if e is not None:
            return e
        rec, src = self.nodes[self.recovery_node], self.nodes[node]
        if rec.allows_entry_from(node) and src.allows_exit_to(self.recovery_node):
            return Edge(node, self.recovery_node, [0.0, 1.0], "any", None, self.recovery_window_s, 0.0, "recovery")
        return None

    def return_edge(self, node: str) -> Edge | None:
        """Where an episodic or recovery skill goes when it is over: its first exit target with an edge."""
        pkg = self.nodes[node]
        for dst in pkg.exit.get("to", []):
            if dst == "*":
                continue
            e = self.edge(node, dst)
            if e is not None and e.kind != "recovery":      # falling back into recovery is an abort, not a return
                return e
        return None


def _resolve(root: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (root / q)


def load_graph(path: str, project_root: str | Path | None = None, require_battery: bool = True) -> MotionGraph:
    path = Path(path)
    try:
        cfg = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise GraphError(f"{path}: cannot read graph ({e})")
    return build_graph(cfg, path=path, project_root=project_root, require_battery=require_battery)


def build_graph(cfg: dict, path: Path | None = None, project_root=None, require_battery: bool = True) -> MotionGraph:
    where = str(path or "<graph>")
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    root = _resolve(Path(project_root), cfg.get("root", "."))
    nodes_cfg = cfg.get("nodes")
    if not isinstance(nodes_cfg, dict) or not nodes_cfg:
        raise GraphError(f"{where}: 'nodes' must map node names to package dirs")
    nodes: dict[str, SkillPackage] = {}
    for name, pdir in nodes_cfg.items():
        try:
            pkg = load_package(_resolve(root, pdir))
        except PackageError as e:
            raise GraphError(f"{where}: node {name!r}: {e}")
        if require_battery and not pkg.battery_passed:
            raise GraphError(f"{where}: node {name!r}: battery report did not pass ({pkg.path}/report/battery.json)")
        nodes[name] = pkg

    edges: list[Edge] = []
    seen = set()
    for i, e in enumerate(cfg.get("edges", [])):
        src, dst = e.get("from"), e.get("to")
        if src not in nodes or dst not in nodes:
            raise GraphError(f"{where}: edge {i} {src!r}->{dst!r} names a node that is not listed")
        if src == dst:
            raise GraphError(f"{where}: edge {i} {src!r}->{dst!r} is a self loop")
        if (src, dst) in seen:
            raise GraphError(f"{where}: edge {src!r}->{dst!r} listed twice")
        seen.add((src, dst))
        if nodes[src].type == "overlay":
            raise GraphError(f"{where}: edge {src!r}->{dst!r}: overlays are layered, they do not transition anywhere")
        if not nodes[dst].allows_entry_from(src):
            raise GraphError(f"{where}: edge {src!r}->{dst!r}: package {nodes[dst].name} does not allow entry from {src!r} "
                             f"(entry.from = {nodes[dst].entry.get('from')})")
        if not nodes[src].allows_exit_to(dst):
            raise GraphError(f"{where}: edge {src!r}->{dst!r}: package {nodes[src].name} does not allow exit to {dst!r} "
                             f"(exit.to = {nodes[src].exit.get('to')})")
        contact = e.get("contact", "any")
        if contact not in CONTACT_MODES:
            raise GraphError(f"{where}: edge {src!r}->{dst!r}: contact must be one of {CONTACT_MODES}")
        phase = e.get("phase", nodes[dst].entry.get("phase", [0.0, 1.0]))
        if not (isinstance(phase, list) and len(phase) == 2 and all(0.0 <= float(v) <= 1.0 for v in phase)):
            raise GraphError(f"{where}: edge {src!r}->{dst!r}: phase must be [lo, hi] in [0, 1]")
        speed = e.get("speed", nodes[dst].entry.get("speed"))
        if speed is not None and not (isinstance(speed, list) and len(speed) == 2 and float(speed[0]) <= float(speed[1])):
            raise GraphError(f"{where}: edge {src!r}->{dst!r}: speed must be [lo, hi]")
        window = float(e.get("window_s", nodes[dst].entry.get("window_s", 0.3)))
        if window < 0:
            raise GraphError(f"{where}: edge {src!r}->{dst!r}: window_s must be >= 0")
        if nodes[src].type == "overlay":
            raise GraphError(f"{where}: edge {src!r}->{dst!r}: overlays are layered, they do not transition anywhere")
        if nodes[dst].type == "overlay":
            kind = "overlay"
            if nodes[src].type != "locomotion":
                raise GraphError(f"{where}: edge {src!r}->{dst!r}: an overlay can only be layered on a locomotion node")
            for j in nodes[dst].overlay["mask"]:
                if j not in HEAD_JOINTS:
                    raise GraphError(f"{where}: edge {src!r}->{dst!r}: overlay drives {j!r}; the locomotion policy only takes head targets as commands")
                if j not in nodes[src].commands:
                    raise GraphError(f"{where}: edge {src!r}->{dst!r}: {nodes[src].name} has no {j!r} command input for the overlay to use")
        elif nodes[dst].type == "recovery":
            kind = "recovery"
        elif nodes[src].type in ("episodic", "recovery"):
            kind = "return"
        else:
            kind = "normal"
        edges.append(Edge(src, dst, [float(phase[0]), float(phase[1])], contact,
                          None if speed is None else [float(speed[0]), float(speed[1])],
                          window, float(e.get("wait_timeout_s", 2.0)), kind))

    start = cfg.get("start")
    if start not in nodes:
        raise GraphError(f"{where}: 'start' must name a node")
    idle = cfg.get("idle", {})
    idle_node = idle.get("node", start)
    if idle_node not in nodes or nodes[idle_node].type != "locomotion":
        raise GraphError(f"{where}: idle.node must be a locomotion node (idle is locomotion at zero command)")
    idle_cmd = list(idle.get("command", [0.0, 0.0, 0.0]))
    if len(idle_cmd) != 3:
        raise GraphError(f"{where}: idle.command must be [vx, vy, wz]")
    rec = cfg.get("recovery")
    if not isinstance(rec, dict) or rec.get("node") not in nodes:
        raise GraphError(f"{where}: 'recovery' must name a node")
    rec_node = rec["node"]
    if nodes[rec_node].type != "recovery":
        raise GraphError(f"{where}: recovery node {rec_node!r} is a {nodes[rec_node].type} package, not a recovery one")
    rec_window = float(rec.get("window_s", 0.3))

    g = MotionGraph(cfg.get("name", where), path, nodes, edges, start, idle_node, idle_cmd, rec_node, rec_window, cfg)
    # Abort is an immediate hand-off, so every node needs a DIRECT edge into the recovery node:
    # either listed, or implied by the recovery package accepting entry from "*" and the node allowing exit to it.
    unreachable = [n for n in nodes if n != rec_node and nodes[n].type != "overlay" and g.abort_edge(n) is None]
    if unreachable:
        raise GraphError(f"{where}: the abort target {rec_node!r} cannot be entered directly from {unreachable}; "
                         f"list those edges or let the packages allow them")
    for n, pkg in nodes.items():
        if pkg.type in ("episodic",) and g.return_edge(n) is None:
            raise GraphError(f"{where}: episodic node {n!r} has no return edge to one of its exit.to {pkg.exit.get('to')}")
        if pkg.type == "overlay" and not any(e.dst == n for e in edges):
            raise GraphError(f"{where}: overlay node {n!r} is not layered on any locomotion node (no edge into it)")
    return g


def reaches(g: MotionGraph, src: str, dst: str) -> bool:
    seen, todo = {src}, deque([src])
    while todo:
        n = todo.popleft()
        for e in g.edges_from(n):
            if e.dst == dst:
                return True
            if e.dst not in seen:
                seen.add(e.dst)
                todo.append(e.dst)
    return False


def describe(g: MotionGraph) -> str:
    lines = [f"graph {g.name}: start {g.start}, idle = {g.idle_node} @ {g.idle_command}, recovery = {g.recovery_node}"]
    for n, p in g.nodes.items():
        lines.append(f"  node {n:8s} {p.type:10s} v{p.version} {p.policy.kind:5s} clip {p.clip.duration:.2f}s "
                     f"battery {'pass' if p.battery_passed else 'FAIL'} status {p.status!r}")
    for e in g.edges:
        lines.append(f"  edge {e.src} -> {e.dst} [{e.kind}] phase {e.phase} contact {e.contact} speed {e.speed} "
                     f"window {e.window_s}s wait {e.wait_timeout_s}s")
    return "\n".join(lines)
