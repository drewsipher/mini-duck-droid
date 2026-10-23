#!/usr/bin/env python3
"""Talk to the running Blender (the MCP add-on's socket on localhost:9876) from a shell or the dashboard.

    python3 anim/blender_remote.py import tracker/data/clips/test_bow.json   # clip -> action on the rig
    python3 anim/blender_remote.py export out.json [start end]              # timeline -> clip JSON
    python3 anim/blender_remote.py snap                                     # foot IK targets to the feet
    python3 anim/blender_remote.py home                                     # standing pose
    python3 anim/blender_remote.py exec "result = {'n': len(bpy.data.objects)}"

Exit 2 when Blender is not running (start it with anim/blender.sh).
"""
import json
import os
import socket
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("BLENDER_MCP_PORT", "9876"))


def run(code, timeout=120):
    s = socket.create_connection(("127.0.0.1", PORT), timeout=timeout)
    s.sendall((json.dumps({"type": "execute", "code": code, "strict_json": True}) + "\0").encode())
    buf = b""
    while not buf.endswith(b"\0"):
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    s.close()
    return json.loads(buf.rstrip(b"\0").decode())


PRELUDE = f"import bpy, sys, importlib\nsys.path.insert(0, {HERE!r})\nimport blender_duck as bd\nimportlib.reload(bd)\n"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "import":
        path = os.path.abspath(args[0])
        code = PRELUDE + f"act = bd.import_clip({path!r})\nresult = {{'action': act.name, 'frames': [bpy.context.scene.frame_start, bpy.context.scene.frame_end]}}"
    elif cmd == "export":
        path = os.path.abspath(args[0])
        rng = f"start={int(args[1])}, end={int(args[2])}" if len(args) >= 3 else ""
        code = PRELUDE + f"result = bd.export_clip({path!r}, {rng})"
    elif cmd == "snap":
        code = PRELUDE + "bd.snap_ik_targets(bpy.data.objects[bd.ARM]); result = {'ok': True}"
    elif cmd == "home":
        code = PRELUDE + "import json; m = json.load(open(bd.MODEL)); arm = bpy.data.objects[bd.ARM]; bd.set_pose(arm, m['home']['joints'], base=m['home']); bd.snap_ik_targets(arm); result = {'ok': True}"
    elif cmd == "exec":
        code = "import bpy\n" + args[0]
    else:
        print("unknown command", cmd)
        sys.exit(1)
    try:
        r = run(code)
    except (ConnectionRefusedError, socket.timeout, OSError) as e:
        print(f"Blender is not reachable on :{PORT} ({e}); start it with anim/blender.sh")
        sys.exit(2)
    print(json.dumps(r, indent=1))
    sys.exit(0 if r.get("status") == "ok" else 1)


if __name__ == "__main__":
    main()
