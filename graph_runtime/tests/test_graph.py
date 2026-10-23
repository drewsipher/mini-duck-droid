"""Motion graph validation: only listed edges, packages exist and passed, abort reachable."""
import json

import pytest

from graph_runtime.graph import GraphError, build_graph, load_graph
from graph_runtime.tests.conftest import PROJECT, graph_cfg, write_json


def test_valid_graph_loads(skill_root):
    g = build_graph(graph_cfg(skill_root))
    assert set(g.nodes) == {"walk", "bow", "sit", "nod"}
    assert g.edge("walk", "bow") is not None and g.edge("bow", "walk").kind == "return"
    assert g.edge("walk", "nod").kind == "overlay" and g.is_overlay("nod") and g.abort_edge("nod") is None
    assert g.abort_edge("walk").dst == "sit" and g.abort_edge("bow").dst == "sit"
    assert g.return_edge("bow").dst == "walk"
    assert g.idle_node == "walk" and g.recovery_node == "sit"


def test_unlisted_edge_does_not_exist(skill_root):
    g = build_graph(graph_cfg(skill_root))
    assert g.edge("sit", "bow") is None
    assert g.edge("walk", "walk") is None


def test_edge_to_missing_node_refused(skill_root):
    cfg = graph_cfg(skill_root, extra_edges=[{"from": "walk", "to": "dance"}])
    with pytest.raises(GraphError, match="not listed"):
        build_graph(cfg)


def test_missing_package_refused(skill_root):
    cfg = graph_cfg(skill_root)
    cfg["nodes"]["dance"] = "dance/v1"
    with pytest.raises(GraphError, match="dance"):
        build_graph(cfg)


def test_failed_battery_refused(skill_root):
    write_json(skill_root / "bow" / "v1" / "report" / "battery.json", {"trials": 10, "passes": 3, "threshold": 0.9})
    with pytest.raises(GraphError, match="battery"):
        build_graph(graph_cfg(skill_root))
    g = build_graph(graph_cfg(skill_root), require_battery=False)       # the batteries themselves need this
    assert "bow" in g.nodes


def test_edge_must_be_allowed_by_both_packages(skill_root):
    # bow only accepts entry from walk; sit -> bow is not declared by bow
    cfg = graph_cfg(skill_root, extra_edges=[{"from": "sit", "to": "bow"}])
    with pytest.raises(GraphError, match="does not allow entry"):
        build_graph(cfg)


def test_abort_must_be_direct_from_every_node(skill_root):
    cfg = graph_cfg(skill_root)
    cfg["edges"] = [e for e in cfg["edges"] if not (e["from"] == "bow" and e["to"] == "sit")]
    # sit accepts "*" and bow allows exit to sit, so the abort edge is implied: still valid
    g = build_graph(cfg)
    assert g.abort_edge("bow").kind == "recovery" and g.abort_edge("bow").window_s == 0.1
    # now bow refuses to exit to sit: no abort route -> refused
    d = skill_root / "bow" / "v1" / "skill.json"
    s = json.loads(d.read_text())
    s["exit"]["to"] = ["walk"]
    write_json(d, s)
    with pytest.raises(GraphError, match="cannot be entered directly from \\['bow'\\]"):
        build_graph(cfg)


def test_recovery_node_must_be_recovery_type(skill_root):
    cfg = graph_cfg(skill_root)
    cfg["recovery"]["node"] = "bow"
    with pytest.raises(GraphError, match="not a recovery one"):
        build_graph(cfg)


def test_idle_must_be_locomotion(skill_root):
    cfg = graph_cfg(skill_root)
    cfg["idle"]["node"] = "bow"
    with pytest.raises(GraphError, match="idle"):
        build_graph(cfg)


def test_episodic_needs_return_edge(skill_root):
    cfg = graph_cfg(skill_root)
    cfg["edges"] = [e for e in cfg["edges"] if not (e["from"] == "bow" and e["to"] == "walk")]
    with pytest.raises(GraphError, match="no return edge"):
        build_graph(cfg)


def test_duplicate_edge_refused(skill_root):
    cfg = graph_cfg(skill_root, extra_edges=[{"from": "walk", "to": "bow"}])
    with pytest.raises(GraphError, match="twice"):
        build_graph(cfg)


def test_overlay_rules(skill_root):
    cfg = graph_cfg(skill_root)
    # an overlay cannot transition anywhere
    with pytest.raises(GraphError, match="layered"):
        build_graph(graph_cfg(skill_root, extra_edges=[{"from": "nod", "to": "walk"}]))
    # it must be layered on something
    cfg2 = graph_cfg(skill_root)
    cfg2["edges"] = [e for e in cfg2["edges"] if e["to"] != "nod"]
    with pytest.raises(GraphError, match="not layered on any"):
        build_graph(cfg2)
    # only on a locomotion node
    cfg3 = graph_cfg(skill_root)
    d = skill_root / "nod" / "v1" / "skill.json"
    s = json.loads(d.read_text())
    s["entry"]["from"] = ["walk", "bow"]
    write_json(d, s)
    b = skill_root / "bow" / "v1" / "skill.json"
    bs = json.loads(b.read_text())
    bs["exit"]["to"] = ["walk", "sit", "nod"]
    write_json(b, bs)
    with pytest.raises(GraphError, match="only be layered on a locomotion"):
        build_graph(graph_cfg(skill_root, extra_edges=[{"from": "bow", "to": "nod"}]))
    # the masked joints must be command inputs of the locomotion policy
    s["overlay"]["mask"] = ["neck_pitch", "left_knee"]
    write_json(d, s)
    with pytest.raises(GraphError, match="only takes head targets"):
        build_graph(cfg)
    w = skill_root / "walk" / "v1" / "skill.json"
    ws = json.loads(w.read_text())
    del ws["commands"]["head_pitch"]
    write_json(w, ws)
    s["overlay"]["mask"] = ["neck_pitch", "head_pitch"]
    write_json(d, s)
    with pytest.raises(GraphError, match="no 'head_pitch' command"):
        build_graph(cfg)


def test_demo_graph_if_built():
    demo = PROJECT / "graphs" / "demo.json"
    root = PROJECT / "graph_runtime" / "demo_skills"
    if not (root / "walk" / "v1" / "policy.onnx").exists():
        pytest.skip("demo skills not built")
    for n in ("walk", "bow_overlay", "sit"):
        if json.loads((root / n / "v1" / "report" / "battery.json").read_text()).get("pending"):
            pytest.skip(f"{n} battery still pending (make_demo_skills.py running?)")
    g = load_graph(demo, project_root=PROJECT)
    assert g.nodes["walk"].policy.kind == "onnx"
    assert all(p.battery_passed for p in g.nodes.values())
    assert g.is_overlay("bow")
    # the open-loop episodic bow failed its battery: the gate must refuse that graph
    with pytest.raises(GraphError, match="battery"):
        load_graph(PROJECT / "graphs" / "demo_openloop.json", project_root=PROJECT)
    g2 = load_graph(PROJECT / "graphs" / "demo_openloop.json", project_root=PROJECT, require_battery=False)
    assert g2.nodes["bow"].type == "episodic" and g2.nodes["bow"].policy.open_loop
