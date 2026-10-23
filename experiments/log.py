#!/usr/bin/env python3
"""Append-only experiment log (experiments/log.md): hypothesis, command, evidence, decision.

    python3 experiments/log.py add --title "walk/v1 battery" --hypothesis "..." \
        --command "tools/battery.py skills/walk/v1" --evidence "overall 0.86 (344/400)" --decision "..."
    python3 experiments/log.py list [--last 10] [--grep walk] [--json]

Format of one entry (other agents append the same by hand; `list` tolerates extra lines,
missing bullets and either "- Key:" or "**Key:**" bullets):

    ## 2026-10-23T01:02:03-07:00 walk/v1 battery
    - Hypothesis: ...
    - Command: ...
    - Evidence: ...
    - Decision: ...
"""
import argparse
import datetime
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "log.md")
FIELDS = ["hypothesis", "command", "evidence", "decision"]
HEADER = "# Experiment log\n\nAppend-only. One entry per run: `## <ISO datetime> <title>` and four bullets " \
         "(Hypothesis, Command, Evidence, Decision). Add with `experiments/log.py add ...` or by hand in the same shape.\n"


def now_iso():
    return datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()


def add(a):
    title = a.title.strip().replace("\n", " ")
    lines = [f"## {a.when or now_iso()} {title}"]
    for f in FIELDS:
        v = (getattr(a, f) or "").strip().replace("\r", "")
        v = v.replace("\n", "\n  ")                       # continuation lines stay inside the bullet
        lines.append(f"- {f.capitalize()}: {v}")
    if a.by:
        lines.append(f"- By: {a.by}")
    entry = "\n".join(lines) + "\n"
    new = not os.path.exists(LOG) or os.path.getsize(LOG) == 0
    with open(LOG, "a") as fh:
        if new:
            fh.write(HEADER)
        fh.write("\n" + entry)
    print(entry, end="")
    return 0


def parse(text):
    entries, cur = [], None
    for raw in text.splitlines():
        m = re.match(r"^##\s+(\S+)\s*(.*)$", raw)
        if m and not raw.startswith("###"):
            cur = {"when": m.group(1), "title": m.group(2).strip(), "fields": {}, "_last": None}
            entries.append(cur)
            continue
        if cur is None:
            continue
        b = re.match(r"^[-*]\s+\**([A-Za-z ]+?)\**\s*:\**\s*(.*)$", raw)
        if b:
            key = b.group(1).strip().lower()
            cur["fields"][key] = b.group(2).strip()
            cur["_last"] = key
        elif raw.strip() and cur["_last"]:
            cur["fields"][cur["_last"]] += " " + raw.strip()
    for e in entries:
        e.pop("_last", None)
    return entries


def list_entries(a):
    if not os.path.exists(LOG):
        print("no log yet:", LOG)
        return 0
    entries = parse(open(LOG).read())
    if a.grep:
        pat = re.compile(a.grep, re.I)
        entries = [e for e in entries if pat.search(e["title"]) or any(pat.search(v) for v in e["fields"].values())]
    if a.last:
        entries = entries[-a.last:]
    if a.json:
        print(json.dumps(entries, indent=2))
        return 0
    for e in entries:
        print(f"{e['when']}  {e['title']}")
        for f in FIELDS:
            if f in e["fields"]:
                v = e["fields"][f]
                print(f"    {f:<11}{v if a.full or len(v) <= 110 else v[:107] + '...'}")
    print(f"({len(entries)} entries)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add")
    p.add_argument("--title", required=True)
    for f in FIELDS:
        p.add_argument(f"--{f}", default="")
    p.add_argument("--by", help="agent or person")
    p.add_argument("--when", help="ISO datetime (default now)")
    p.set_defaults(fn=add)
    p = sub.add_parser("list")
    p.add_argument("--last", type=int)
    p.add_argument("--grep")
    p.add_argument("--json", action="store_true")
    p.add_argument("--full", action="store_true")
    p.set_defaults(fn=list_entries)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
