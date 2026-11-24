#!/usr/bin/env python3
"""Read run_hw.py logs: find the pushes, measure each one, compare runs.

    python3 tools/hw_report.py logs/stand_held.jsonl logs/recovery_mat.jsonl [--out report.png] [--json report.json]

A push is a spike of horizontal acceleration (|acc_xy| above --spike m/s^2 after a quiet 0.3 s), or a
{"mark": ...} line typed by the operator. For each push: its direction in the duck's frame (where the
duck was thrown: forward / backward / left / right from the first 60 ms of acceleration), its size as the
acceleration impulse (integral of |acc_xy| over 0.3 s, m/s), the peak tilt within 1.5 s, the settle time
(tilt back under --settle degrees for 0.5 s), the peak gyro, and whether the run ended in a fall within
3 s. The plot shows tilt over time for every run with the pushes marked.
"""
import argparse
import json
import os

import numpy as np

DT = 0.02


def load(path):
    rows, marks, meta, summary = [], [], {}, {}
    for line in open(path):
        r = json.loads(line)
        if "meta" in r:
            meta = r["meta"]
        elif "summary" in r:
            summary = r["summary"]
        elif "mark" in r:
            marks.append(r)
        else:
            rows.append(r)
    return meta, rows, marks, summary


def direction(ax, ay):
    ang = np.degrees(np.arctan2(ay, ax)) % 360          # body x forward, y left
    labels = ["forward", "left-fwd", "left", "left-back", "backward", "right-back", "right", "right-fwd"]
    return labels[int(((ang + 22.5) % 360) // 45)]


def analyse(path, spike, settle_deg):
    meta, rows, marks, summary = load(path)
    t = np.array([r["t"] for r in rows])
    acc = np.array([r["acc"] for r in rows])
    gyro = np.array([r["gyro"] for r in rows])
    tilt = np.array([r["tilt"] for r in rows])
    # the proper acceleration minus gravity along the body axes: at rest acc ~ (0, 0, 9.81) in a level body
    axy = np.linalg.norm(acc[:, :2], axis=1)
    rest = float(np.median(axy)) if len(axy) else 0.0        # pushes are brief, so the median is the quiet level
    pushes, i, quiet_n = [], int(0.5 / DT), int(0.3 / DT)     # skip the start-up transient
    while i < len(rows):
        # a push: the horizontal acceleration rises above the quiet level by `spike`, the 0.3 s before it
        # were calmer than that (a stepping policy is never fully quiet, so "calmer" = below the spike)
        if axy[i] - rest > spike and (i < quiet_n or np.max(axy[i - quiet_n:i] - rest) < spike):
            pushes.append(i)
            i += int(1.0 / DT)                               # one push per second at most
        else:
            i += 1
    out = []
    fell_at = None
    if summary.get("stopped", "").startswith("tilt"):
        fell_at = float(summary.get("seconds", t[-1] if len(t) else 0))
    for i in pushes:
        w = slice(i, min(i + 3, len(rows)))
        # a hand shove accelerates the trunk in the push direction, so the first 60 ms of acceleration give the
        # direction the duck is thrown. (A simulated velocity KICK reads inverted: the body is already moving and
        # the feet brake it, so trust the {"mark"} line for sim kicks.)
        a0 = acc[w, :2].mean(axis=0)
        d = direction(a0[0], a0[1])
        w03 = slice(i, min(i + int(0.3 / DT), len(rows)))
        impulse = float(np.sum(np.clip(axy[w03] - rest, 0, None)) * DT)
        w15 = slice(i, min(i + int(1.5 / DT), len(rows)))
        peak = float(tilt[w15].max())
        gpk = float(np.linalg.norm(gyro[w15], axis=1).max())
        settled = None
        for j in range(i + 5, len(rows) - int(0.5 / DT)):
            if np.all(tilt[j:j + int(0.5 / DT)] < settle_deg):
                settled = float(t[j] - t[i])
                break
        fell = fell_at is not None and 0 <= fell_at - t[i] <= 3.0
        if impulse < 0.1:                                    # a blip, not a push
            continue
        out.append({"t": float(t[i]), "direction": d, "impulse_ms": round(impulse, 3), "peak_tilt_deg": round(peak, 1),
                    "peak_gyro_rad_s": round(gpk, 2), "settle_s": None if settled is None else round(settled, 2), "fell": fell})
    return {"file": os.path.basename(path), "skill": meta.get("skill"), "backend": meta.get("backend"), "seconds": summary.get("seconds"),
            "stopped": summary.get("stopped"), "peak_tilt_deg": summary.get("peak_tilt_deg"),
            "pushes": out, "marks": marks, "t": t, "tilt": tilt}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--spike", type=float, default=1.0, help="m/s^2 of horizontal acceleration above the quiet level that counts as a push")
    ap.add_argument("--settle", type=float, default=3.0, help="deg: settled when tilt stays under this for 0.5 s")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    reports = [analyse(p, a.spike, a.settle) for p in a.logs]
    for r in reports:
        print(f"\n{r['file']}: {r['skill']} on {r['backend']}, {r['seconds']} s, stopped: {r['stopped']}, peak tilt {r['peak_tilt_deg']} deg, "
              f"{len(r['pushes'])} pushes")
        for p in r["pushes"]:
            print(f"  t {p['t']:6.2f}  {p['direction']:10s} impulse {p['impulse_ms']:.2f} m/s  peak tilt {p['peak_tilt_deg']:5.1f} deg  "
                  f"gyro {p['peak_gyro_rad_s']:.2f}  settle {p['settle_s']}  {'FELL' if p['fell'] else ''}")
        for m in r["marks"]:
            print(f"  mark t {m['t']:6.2f}: {m['mark']}")
    if a.json:
        json.dump([{k: v for k, v in r.items() if k not in ("t", "tilt")} for r in reports], open(a.json, "w"), indent=1)
    if a.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(len(reports), 1, figsize=(10, 2.2 * len(reports) + 0.5), sharex=True, squeeze=False)
        for ax, r in zip(axs[:, 0], reports):
            ax.plot(r["t"], r["tilt"], color="#3a7bd5", lw=1)
            for p in r["pushes"]:
                ax.axvline(p["t"], color="#d5503a" if p["fell"] else "#e08a2e", ls=":", lw=1)
                ax.text(p["t"], ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 1, p["direction"], fontsize=6, rotation=90, va="top")
            ax.set_ylabel("tilt deg"); ax.set_title(f"{r['file']}: {r['skill']} ({r['backend']})", fontsize=9, loc="left")
        axs[-1, 0].set_xlabel("t (s); dotted = push (red: fell)")
        fig.tight_layout()
        fig.savefig(a.out, dpi=110)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
