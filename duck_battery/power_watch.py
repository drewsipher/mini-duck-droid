#!/usr/bin/env python3
"""Log the Pi's power-trouble flags so a sudden reboot leaves evidence.
Every second: read `vcgencmd get_throttled`; write a line to ~/power_log.csv when the
flags change, plus a heartbeat every 10 s (fsynced, so the last line before a power
cut survives). Bit 0 = under-voltage right now, bit 16 = under-voltage since boot,
bit 2/18 = throttled, bit 1/17 = frequency capped."""
import os
import subprocess
import time

LOG = os.path.expanduser("~/power_log.csv")


def flags():
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2).stdout
        return int(out.strip().split("=")[1], 16)
    except Exception:  # noqa: BLE001
        return -1


def write(f, line):
    f.write(line + "\n")
    f.flush()
    os.fsync(f.fileno())


def main():
    new = not os.path.exists(LOG)
    with open(LOG, "a") as f:
        if new:
            write(f, "time,uptime_s,event,throttled_hex,undervolt_now,undervolt_since_boot")
        last, last_beat = None, 0.0
        while True:
            fl, now = flags(), time.time()
            up = float(open("/proc/uptime").read().split()[0])
            ev = "boot" if last is None else "change" if fl != last else "beat" if now - last_beat >= 10 else None
            if ev:
                write(f, f"{time.strftime('%Y-%m-%d %H:%M:%S')},{up:.0f},{ev},{fl:#x},{fl & 1},{(fl >> 16) & 1}")
                last_beat = now
            last = fl
            time.sleep(1)


if __name__ == "__main__":
    main()
