"""Read scalar curves out of TensorBoard tfevents files with no dependencies.

The dashboard server runs on the system python3 (no tensorboardX), so this decodes just enough
protobuf by hand: Event{step=2 (varint), summary=5}, Summary{value=1 repeated},
Summary.Value{tag=1 (string), simple_value=2 (float32)}. Same result as the `scalars`
function in sim/make_curriculum_plots.py (checked against it on runs8/cur_p2).

    scalars("sim/train_pg/runs8/cur_p2") -> {"eval/episode_reward": [[step, value], ...], ...}
"""
import glob
import os
import struct


def _varint(buf, i):
    shift, out = 0, 0
    while True:
        b = buf[i]
        i += 1
        out |= (b & 0x7F) << shift
        if not b & 0x80:
            return out, i
        shift += 7


def _fields(buf):
    """Yield (field_number, wire_type, value) for one protobuf message."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 1:
            v, i = buf[i:i + 8], i + 8
        elif wt == 2:
            ln, i = _varint(buf, i)
            v, i = buf[i:i + ln], i + ln
        elif wt == 5:
            v, i = buf[i:i + 4], i + 4
        else:  # groups (3, 4) never appear in tfevents
            return
        yield fn, wt, v


def _event(buf, out):
    step = 0
    summary = None
    for fn, wt, v in _fields(buf):
        if fn == 2 and wt == 0:
            step = v if v < 1 << 63 else v - (1 << 64)
        elif fn == 5 and wt == 2:
            summary = v
    if summary is None:
        return
    for fn, wt, val in _fields(summary):
        if fn != 1 or wt != 2:
            continue
        tag, simple = None, None
        for f2, w2, v2 in _fields(val):
            if f2 == 1 and w2 == 2:
                tag = v2.decode("utf-8", "replace")
            elif f2 == 2 and w2 == 5:
                simple = struct.unpack("<f", v2)[0]
        if tag is not None and simple is not None:
            out.setdefault(tag, []).append([step, simple])


def scalars(run_dir, tags=None):
    """{tag: [[step, value], ...]} from every events* file in run_dir (sorted by name).
    `tags` (a set) keeps only those tags; None keeps all."""
    out = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "events*"))):
        with open(path, "rb") as f:
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                (n,) = struct.unpack("<Q", hdr)
                f.read(4)                       # crc of the length
                rec = f.read(n)
                f.read(4)                       # crc of the record
                if len(rec) < n:
                    break                       # file still being written
                try:
                    _event(rec, out)
                except (IndexError, struct.error):
                    break
    if tags is not None:
        out = {k: v for k, v in out.items() if k in tags}
    return out


if __name__ == "__main__":
    import json
    import sys
    sc = scalars(sys.argv[1])
    print(json.dumps({k: (len(v), v[-1]) for k, v in sorted(sc.items())}, indent=1))
