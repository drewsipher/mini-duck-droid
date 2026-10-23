#!/usr/bin/env python3
"""Duck Servo Playground: see and move Feetech STS servos on a Waveshare Bus Servo Adapter.

  python3 servo_playground.py                  # real servos, port picked automatically
  python3 servo_playground.py --port /dev/ttyACM0
  python3 servo_playground.py --demo           # pretend servos, no hardware needed

Needs: pyserial (pip install pyserial) and Tkinter (sudo apt install python3-tk).
Bench use only: it lets every servo turn through its full circle, which would break
parts once the servos are built into the duck.
Everything this program writes goes to the servos' working memory (RAM): target, speed
and strength. It never changes their saved settings (ID, limits), and a power cycle
undoes all of it.
"""
import argparse
import math
import random
import time
import tkinter as tk
from tkinter import ttk

try:
    import serial
    from serial.tools import list_ports
except ImportError:  # demo mode still works without pyserial
    serial = None

NAMES = {
    10: "right hip yaw", 11: "right hip roll", 12: "right hip pitch", 13: "right knee", 14: "right ankle",
    20: "left hip yaw", 21: "left hip roll", 22: "left hip pitch", 23: "left knee", 24: "left ankle",
    30: "neck pitch", 31: "head pitch", 32: "head yaw", 33: "head roll", 1: "brand new (not configured)",
}

# STS3215 registers, all in the RAM area
TORQUE_ENABLE = 40  # only ever write 0 or 1 here (128 would re-zero the servo)
GOAL = 42           # goal position (2 bytes), goal time (2), goal speed (2)
TORQUE_LIMIT = 48   # strength, 0-1000 = 0-100 %. Reset from saved max at power-up
PRESENT = 56        # 56 position, 58 speed, 60 load, 62 volt, 63 temp, 65 errors, 69 current
PRESENT_LEN = 15
ERRORS = {0: "supply voltage out of range", 1: "position sensor error", 2: "too hot",
          3: "too much current", 4: "angle limit", 5: "overloaded: pushed too hard"}
DEG_PER_STEP = 360 / 4096
PING, READ, WRITE = 0x01, 0x02, 0x03
TICK_MS = 100


def angle_of(pos):
    """Encoder steps (0-4095) to degrees, with the middle (2048) as 0 degrees."""
    return (pos - 2048) * 360 / 4096


def pos_of(deg):
    return max(0, min(4095, round(2048 + deg * 4096 / 360)))


def signed(v, bit):
    """Feetech sign-magnitude: the given bit is the sign, the bits below it the size."""
    mag = v & ((1 << bit) - 1)
    return -mag if v >> bit & 1 else mag


# ---------------------------------------------------------------- serial bus
class Bus:
    """Feetech packet protocol: FF FF id length instruction params checksum."""

    def __init__(self, port):
        self.port = port

    @staticmethod
    def packet(sid, instr, params=b""):
        body = bytes([sid, len(params) + 2, instr]) + bytes(params)
        return b"\xff\xff" + body + bytes([~sum(body) & 0xFF])

    def _reply(self, sid, wait):
        buf = bytearray()
        end = time.monotonic() + wait
        while time.monotonic() < end:
            n = self.port.in_waiting
            if n:
                buf += self.port.read(n)
                i = buf.find(b"\xff\xff")
                if i >= 0 and len(buf) >= i + 4 and len(buf) >= i + 4 + buf[i + 3]:
                    ln = buf[i + 3]
                    pkt = buf[i + 2:i + 4 + ln]
                    if pkt[0] == sid and pkt[-1] == (~sum(pkt[:-1]) & 0xFF):
                        return bytes(pkt[3:-1])  # parameters after the error byte
                    return None
            else:
                time.sleep(0.0005)
        return None

    def _txn(self, sid, instr, params=b"", wait=0.03):
        self.port.reset_input_buffer()
        self.port.write(self.packet(sid, instr, params))
        return self._reply(sid, wait)

    def ping(self, sid, wait=0.015):
        return self._txn(sid, PING, wait=wait) is not None

    def status(self, sid):
        d = self._txn(sid, READ, bytes([PRESENT, PRESENT_LEN]))
        if d is None or len(d) < PRESENT_LEN:
            return None
        u16 = lambda i: d[i] | d[i + 1] << 8
        return {"pos": u16(0),
                "speed": signed(u16(2), 15) * DEG_PER_STEP,     # degrees per second
                "load": abs(signed(u16(4), 10)) / 10,            # % of full motor drive
                "volt": d[6] / 10, "temp": d[7], "errors": d[9],
                "current": (u16(13) & 0x7FFF) * 6.5}             # milliamps

    def read_u16(self, sid, addr):
        d = self._txn(sid, READ, bytes([addr, 2]))
        return None if d is None or len(d) < 2 else d[0] | d[1] << 8

    def strength(self, sid, permille):
        v = max(0, min(1000, int(permille)))
        self._txn(sid, WRITE, bytes([TORQUE_LIMIT, v & 0xFF, v >> 8]), wait=0.01)

    def torque(self, sid, on):
        self._txn(sid, WRITE, bytes([TORQUE_ENABLE, 1 if on else 0]), wait=0.01)

    def goal(self, sid, pos, speed):
        pos, speed = int(pos), int(speed)
        self._txn(sid, WRITE, bytes([GOAL, pos & 0xFF, pos >> 8, 0, 0, speed & 0xFF, speed >> 8]), wait=0.01)

    def close(self):
        self.port.close()


class FakePort:
    """Pretend servos for --demo. With the motor off, they drift as if a hand turned them."""

    def __init__(self, ids=(10, 11, 12)):
        self.s = {i: {"pos": 2048.0 + random.randint(-400, 400), "goal": 2048, "speed": 600, "torque": 0,
                      "limit": 1000, "vel": 0.0, "phase": random.random() * 6} for i in ids}
        self.out = bytearray()
        self.t = time.monotonic()

    def _physics(self):
        now = time.monotonic()
        dt, self.t = now - self.t, now
        for v in self.s.values():
            before = v["pos"]
            if v["torque"]:
                step = (v["speed"] or 3400) * dt
                d = v["goal"] - v["pos"]
                v["pos"] += max(-step, min(step, d))
            else:
                v["pos"] = 2048 + 900 * math.sin(now * 0.8 + v["phase"])
            v["vel"] = (v["pos"] - before) / dt if dt else 0

    @property
    def in_waiting(self):
        return len(self.out)

    def read(self, n):
        d, self.out = bytes(self.out[:n]), self.out[n:]
        return d

    def reset_input_buffer(self):
        self.out.clear()

    def close(self):
        pass

    def write(self, b):
        self._physics()
        sid, instr, params = b[2], b[4], b[5:-1]
        v = self.s.get(sid)
        if v is None:
            return
        data = b""
        if instr == READ:
            pos = int(v["pos"])
            moving = abs(v["goal"] - v["pos"]) > 5 and v["torque"]
            load = random.randint(150, 300) if moving else (random.randint(20, 60) if v["torque"] else 0)
            load = load * v["limit"] // 1000
            vel = int(abs(v["vel"])) | (0x8000 if v["vel"] < 0 else 0)
            cur = load // 4 + 20  # units of 6.5 mA: about 0.1 A idle, 0.6 A moving
            regs = bytes([pos & 0xFF, pos >> 8, vel & 0xFF, vel >> 8, load & 0xFF, load >> 8, 72, 25,
                          0, 0, int(moving), 0, 0, cur & 0xFF, cur >> 8])
            if params[0] == TORQUE_LIMIT:
                regs, base = bytes([v["limit"] & 0xFF, v["limit"] >> 8]), TORQUE_LIMIT
            else:
                base = PRESENT
            data = regs[params[0] - base:][:params[1]]
        elif instr == WRITE and params[0] == TORQUE_ENABLE:
            v["torque"] = params[1]
        elif instr == WRITE and params[0] == TORQUE_LIMIT:
            v["limit"] = params[1] | params[2] << 8
        elif instr == WRITE and params[0] == GOAL:
            v["goal"] = params[1] | params[2] << 8
            v["speed"] = params[5] | params[6] << 8
        self.out += Bus.packet(sid, 0, data)  # a status reply has error byte 0 where the instruction sits


# ---------------------------------------------------------------- one servo card
class ServoCard(ttk.Frame):
    SIZE, R = 200, 84

    def __init__(self, parent, app, sid):
        super().__init__(parent, padding=10, relief="groove", borderwidth=2)
        self.app, self.sid = app, sid
        self.pos = None       # where the servo is now
        self.target = None    # where we asked it to go
        self.dirty = False    # target not sent yet
        self.torque = False
        self.stream_speed = None
        self.strength_dirty = False
        self.watts = self.amps = 0.0
        self._syncing = False

        ttk.Label(self, text=f"ID {sid}", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(self, text=NAMES.get(sid, ""), style="Muted.TLabel").grid(row=0, column=1, sticky="e")

        self.dial = tk.Canvas(self, width=self.SIZE, height=self.SIZE, bg="#fbfaf7",
                              highlightthickness=0, cursor="hand2")
        self.dial.grid(row=1, column=0, columnspan=2, pady=4)
        self.dial.bind("<Button-1>", self.on_dial)
        self.dial.bind("<B1-Motion>", self.on_dial)

        self.slider_var = tk.DoubleVar(value=0)
        ttk.Scale(self, from_=-180, to=180, variable=self.slider_var, command=self.on_slider).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(0, 6))

        self.readout = ttk.Label(self, text="waiting…", style="Readout.TLabel", justify="left")
        self.readout.grid(row=3, column=0, columnspan=2, sticky="w")
        self.warning = ttk.Label(self, text="", style="Warn.TLabel", wraplength=250)
        self.warning.grid(row=4, column=0, columnspan=2, sticky="w")

        ttk.Label(self, text="Effort", style="Muted.TLabel").grid(row=5, column=0, sticky="w")
        self.effort = ttk.Progressbar(self, maximum=100, length=120)
        self.effort.grid(row=5, column=1, sticky="ew", pady=2)

        # speed: how fast it travels to a new target (steps per second)
        self.speed = tk.IntVar(value=600)
        self.speed_lbl = ttk.Label(self, style="Muted.TLabel")
        self.speed_lbl.grid(row=6, column=0, sticky="w")
        ttk.Scale(self, from_=50, to=3000, variable=self.speed, command=lambda v: self.on_speed()).grid(
            row=6, column=1, sticky="ew", pady=2)
        # strength: torque limit, how hard it is allowed to push
        self.strength = tk.IntVar(value=100)
        self.strength_lbl = ttk.Label(self, style="Muted.TLabel")
        self.strength_lbl.grid(row=7, column=0, sticky="w")
        ttk.Scale(self, from_=5, to=100, variable=self.strength, command=lambda v: self.on_strength()).grid(
            row=7, column=1, sticky="ew", pady=2)
        self.on_speed(send=False)
        self.on_strength(send=False)

        self.torque_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="Motor on (holds its position)", variable=self.torque_var,
                        command=lambda: app.set_torque(self, self.torque_var.get())).grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Button(self, text="Go to centre", command=lambda: self.set_target(0)).grid(
            row=9, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.columnconfigure(1, weight=1)
        self.draw_face()

    # -- input
    def on_dial(self, e):
        c = self.SIZE / 2
        if math.hypot(e.x - c, e.y - c) < 12:
            return
        self.set_target(math.degrees(math.atan2(e.x - c, c - e.y)))

    def on_slider(self, value):
        if not self._syncing:
            self.set_target(float(value), from_slider=True)

    def on_speed(self, send=True):
        self.speed_lbl.config(text=f"Speed {self.speed.get() * DEG_PER_STEP:4.0f}°/s")
        if send and self.torque and self.target is not None:
            self.dirty = True  # resend the goal with the new speed

    def on_strength(self, send=True):
        self.strength_lbl.config(text=f"Strength {self.strength.get():3d} %")
        if send:
            self.strength_dirty = True

    def set_target(self, deg, from_slider=False):
        self.target, self.dirty, self.stream_speed = pos_of(deg), True, None
        if not from_slider:
            self.sync_slider(deg)
        if not self.torque:
            self.app.set_torque(self, True)
        self.draw()

    def sync_slider(self, deg):
        self._syncing = True
        self.slider_var.set(deg)
        self._syncing = False

    # -- display
    def point(self, deg, r):
        c = self.SIZE / 2
        a = math.radians(deg)
        return c + r * math.sin(a), c - r * math.cos(a)

    def draw_face(self):
        c, r = self.SIZE / 2, self.R
        self.dial.create_oval(c - r, c - r, c + r, c + r, outline="#c9c4b8", width=2, fill="#ffffff")
        for deg in range(-180, 180, 15):
            inner = r - (10 if deg % 90 == 0 else 6 if deg % 45 == 0 else 3)
            self.dial.create_line(*self.point(deg, inner), *self.point(deg, r), fill="#a39e93")
        for deg, txt in ((0, "0°"), (90, "90°"), (-90, "-90°"), (180, "180°")):
            self.dial.create_text(*self.point(deg, r - 22), text=txt, fill="#7a756b",
                                  font=("TkDefaultFont", 9))

    def draw(self):
        d = self.dial
        d.delete("dyn")
        c = self.SIZE / 2
        if self.torque and self.target is not None:
            d.create_line(c, c, *self.point(angle_of(self.target), self.R - 4), fill="#e8922a",
                          width=3, dash=(6, 4), tags="dyn")
        if self.pos is not None:
            colour = "#2f5fb3" if self.torque else "#6b6b6b"
            x, y = self.point(angle_of(self.pos), self.R - 14)
            d.create_line(c, c, x, y, fill=colour, width=8, capstyle="round", tags="dyn")
            d.create_oval(x - 7, y - 7, x + 7, y + 7, fill=colour, outline="", tags="dyn")
        d.create_oval(c - 9, c - 9, c + 9, c + 9, fill="#393838", outline="", tags="dyn")
        label = "MOTOR ON" if self.torque else "motor off: turn me by hand"
        d.create_text(c, self.SIZE - 2, text=label, anchor="s", tags="dyn",
                      fill="#2f5fb3" if self.torque else "#7a756b", font=("TkDefaultFont", 8, "bold"))

    def show(self, st):
        if st is None:
            self.readout.config(text="no reply from this servo", style="Warn.TLabel")
            self.warning.config(text="")
            self.watts = self.amps = 0.0
            return
        self.pos = st["pos"]
        deg = angle_of(self.pos)
        if not self.torque:
            self.sync_slider(deg)
        self.amps = st["current"] / 1000
        self.watts = self.amps * st["volt"]
        self.readout.config(style="Readout.TLabel", text=(
            f"angle {deg:+5.0f}°   moving {abs(st['speed']):4.0f}°/s\n"
            f"supply {st['volt']:4.1f} V   temp {st['temp']:3d} °C\n"
            f"current {st['current']:5.0f} mA  power {self.watts:4.1f} W"))
        errs = [txt for bit, txt in ERRORS.items() if st["errors"] >> bit & 1]
        self.warning.config(text=("⚠ " + ", ".join(errs)) if errs else "")
        self.effort["value"] = st["load"]
        self.draw()


# ---------------------------------------------------------------- the app
class App:
    def __init__(self, root, port_name, demo):
        self.root, self.demo = root, demo
        self.bus = None
        self.cards = {}
        self.frames, self.recording, self.playing = [], False, False
        self.play_i, self.approach_until = 0, 0
        root.title("Duck Servo Playground")
        root.geometry("960x860")
        self.styles()

        top = ttk.Frame(root, padding=(10, 8))
        top.pack(fill="x")
        ttk.Label(top, text="Port").pack(side="left")
        ports = ["demo"] if demo else self.find_ports()
        self.port_var = tk.StringVar(value=port_name or (ports[0] if ports else "/dev/ttyACM0"))
        ttk.Combobox(top, textvariable=self.port_var, values=ports, width=18).pack(side="left", padx=6)
        ttk.Button(top, text="Connect and find servos", command=self.connect).pack(side="left")
        ttk.Button(top, text="ALL MOTORS OFF  (space)", style="Stop.TButton",
                   command=self.all_off).pack(side="right")

        bar = ttk.Frame(root, padding=(10, 0, 10, 8))
        bar.pack(fill="x")
        ttk.Button(bar, text="● Record", command=self.record).pack(side="left")
        ttk.Button(bar, text="■ Stop", command=self.stop).pack(side="left", padx=4)
        ttk.Button(bar, text="▶ Play", command=self.play).pack(side="left")
        self.loop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Loop", variable=self.loop_var).pack(side="left", padx=6)
        self.rec_lbl = ttk.Label(bar, text="", style="Muted.TLabel")
        self.rec_lbl.pack(side="left", padx=6)
        self.total_lbl = ttk.Label(bar, text="", style="Total.TLabel")
        self.total_lbl.pack(side="right")

        self.status = ttk.Label(root, padding=(10, 0, 10, 6), style="Muted.TLabel",
                                text="Pick the port and press Connect. Bench supply on, jumper on B.")
        self.status.pack(fill="x")

        wrap = ttk.Frame(root)
        wrap.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(wrap, highlightthickness=0, bg="#f1efea")
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg="#f1efea")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((8, 8), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.bind("<Configure>", lambda e: self.layout())
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        root.bind("<space>", lambda e: self.all_off())
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.after(TICK_MS, self.tick)
        if demo or port_name:
            self.root.after(200, self.connect)

    def styles(self):
        s = ttk.Style()
        if "clam" in s.theme_names():
            s.theme_use("clam")
        base = ("TkDefaultFont", 11)
        s.configure(".", font=base)
        s.configure("Title.TLabel", font=("TkDefaultFont", 15, "bold"))
        s.configure("Muted.TLabel", foreground="#6b665d")
        s.configure("Readout.TLabel", font=("TkFixedFont", 12))
        s.configure("Total.TLabel", font=("TkFixedFont", 12, "bold"))
        s.configure("Warn.TLabel", foreground="#b3261e", font=("TkDefaultFont", 11, "bold"))
        s.configure("Stop.TButton", font=("TkDefaultFont", 11, "bold"), foreground="#ffffff",
                    background="#b3261e")
        s.map("Stop.TButton", background=[("active", "#8c1d17")])

    @staticmethod
    def find_ports():
        if serial is None:
            return []
        found = [p.device for p in list_ports.comports()]
        return sorted(found, key=lambda d: ("ACM" not in d and "usbmodem" not in d, d))

    def say(self, text):
        self.status.config(text=text)
        self.root.update_idletasks()

    # -- connection
    def connect(self):
        self.disconnect()
        name = self.port_var.get()
        try:
            if self.demo or name == "demo":
                port = FakePort()
            else:
                if serial is None:
                    return self.say("pyserial is missing. Run: pip install pyserial")
                port = serial.Serial(name, 1_000_000, timeout=0)
        except Exception as e:
            return self.say(f"Could not open {name}: {e}")
        self.bus = Bus(port)
        found = []
        for sid in range(254):
            if sid % 16 == 0:
                self.say(f"Looking for servos… checking ID {sid} of 253")
            if self.bus.ping(sid):
                found.append(sid)
        for sid in found:
            self.bus.torque(sid, False)
            card = self.cards[sid] = ServoCard(self.inner, self, sid)
            limit = self.bus.read_u16(sid, TORQUE_LIMIT)
            if limit is not None:
                card.strength.set(max(5, round(limit / 10)))
                card.on_strength(send=False)
        self.layout()
        if found:
            self.say(f"Found {len(found)} servo(s): {', '.join(map(str, found))}. Drag a dial or the "
                     f"slider to move it. Motors start off, so you can turn them by hand.")
        else:
            self.say("No servos answered. Check the bench supply is on, the jumper is on B, and the cable.")

    def disconnect(self):
        self.stop()
        if self.bus:
            for sid in self.cards:
                try:
                    self.bus.torque(sid, False)
                except Exception:
                    pass
            self.bus.close()
        self.bus = None
        for c in self.cards.values():
            c.destroy()
        self.cards.clear()

    def layout(self):
        cols = max(1, (self.canvas.winfo_width() - 16) // 290)
        for i, card in enumerate(self.cards.values()):
            card.grid(row=i // cols, column=i % cols, padx=6, pady=6, sticky="n")

    # -- motor control
    def set_torque(self, card, on):
        card.torque = on
        card.torque_var.set(on)
        if self.bus:
            if on:  # hold where it is first, so switching on never makes it jump
                here = card.pos if card.pos is not None else 2048
                self.bus.goal(card.sid, here, card.speed.get())
                if card.target is None or not card.dirty:
                    card.target = here
            self.bus.torque(card.sid, on)
        card.draw()

    def all_off(self):
        self.stop()
        for card in self.cards.values():
            card.dirty = False
            self.set_torque(card, False)
        self.say("All motors off. Turn them by hand.")

    # -- record and play
    def record(self):
        if not self.cards:
            return
        self.stop()
        for card in self.cards.values():
            self.set_torque(card, False)
        self.frames, self.recording = [], True
        self.say("Recording. Move the servos by hand, then press Stop.")

    def stop(self):
        if self.recording:
            self.say(f"Recorded {len(self.frames) * TICK_MS / 1000:.1f} s. Press Play.")
        elif self.playing:
            self.say("Stopped.")
        self.recording = self.playing = False

    def play(self):
        if not self.frames:
            return self.say("Nothing recorded yet. Press Record first.")
        self.stop()
        self.playing, self.play_i = True, -1
        self.approach_until = time.monotonic() + 3
        for sid, p in self.frames[0].items():
            card = self.cards.get(sid)
            if card:
                card.target, card.dirty, card.stream_speed = p, True, None
                if not card.torque:
                    self.set_torque(card, True)
        self.say("Playing back…")

    def advance_playback(self):
        if self.play_i < 0:  # moving to the first pose at the normal speed
            close = all(c.pos is not None and abs(c.pos - self.frames[0].get(s, c.pos)) < 40
                        for s, c in self.cards.items())
            if close or time.monotonic() > self.approach_until:
                self.play_i = 0
            return
        self.play_i += 1
        if self.play_i >= len(self.frames):
            if self.loop_var.get():
                return self.play()
            self.playing = False
            return self.say("Done. Press Play to see it again.")
        prev, cur = self.frames[self.play_i - 1], self.frames[self.play_i]
        for sid, p in cur.items():
            card = self.cards.get(sid)
            if card:
                step = abs(p - prev.get(sid, p))
                card.target, card.dirty = p, True
                card.stream_speed = max(50, min(3400, step * 1000 / TICK_MS * 1.3))

    # -- main loop
    def tick(self):
        t0 = time.monotonic()
        if self.bus and self.cards:
            try:
                if self.playing:
                    self.advance_playback()
                for card in self.cards.values():
                    if card.strength_dirty:
                        self.bus.strength(card.sid, card.strength.get() * 10)
                        card.strength_dirty = False
                    if card.dirty and card.torque and card.target is not None:
                        self.bus.goal(card.sid, card.target, card.stream_speed or card.speed.get())
                        card.dirty = False
                    card.show(self.bus.status(card.sid))
                amps = sum(c.amps for c in self.cards.values())
                watts = sum(c.watts for c in self.cards.values())
                self.total_lbl.config(text=f"All servos: {amps:4.2f} A  {watts:4.1f} W")
                if self.recording:
                    self.frames.append({s: c.pos for s, c in self.cards.items() if c.pos is not None})
                    self.rec_lbl.config(text=f"{len(self.frames) * TICK_MS / 1000:.1f} s recorded")
            except (OSError, getattr(serial, "SerialException", OSError)) as e:
                self.cards_lost(e)
        spent = int((time.monotonic() - t0) * 1000)
        self.root.after(max(10, TICK_MS - spent), self.tick)

    def cards_lost(self, e):
        self.bus = None
        for c in self.cards.values():
            c.destroy()
        self.cards.clear()
        self.say(f"Lost the connection ({e}). Plug it back in and press Connect.")

    def quit(self):
        self.disconnect()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port, e.g. /dev/ttyACM0")
    ap.add_argument("--demo", action="store_true", help="use pretend servos, no hardware needed")
    args = ap.parse_args()
    root = tk.Tk()
    App(root, args.port, args.demo)
    root.mainloop()


if __name__ == "__main__":
    main()
