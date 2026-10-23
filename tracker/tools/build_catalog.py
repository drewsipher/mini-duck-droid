"""Generate tracker/data/catalog.json (print parts, buy list, 3D mesh map)."""
import json
import sys
from pathlib import Path

OUT = Path.home() / "src/mini-duck-droid/tracker/data/catalog.json"
BUY_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else None

# id, name, qty, material, section, in 3D model
PRINT = [
    ("foot_top", "Foot top", 2, "PLA", "Feet", True),
    ("foot_side", "Foot side", 2, "PLA", "Feet", True),
    ("foot_bottom_pla", "Foot bottom (PLA)", 2, "PLA", "Feet", True),
    ("foot_bottom_tpu", "Foot sole (TPU)", 2, "TPU", "Feet", True),
    ("knee_to_ankle_left_sheet", "Knee-to-ankle sheet, left", 4, "PLA", "Legs", True),
    ("knee_to_ankle_right_sheet", "Knee-to-ankle sheet, right", 4, "PLA", "Legs", True),
    ("leg_spacer", "Leg spacer", 4, "PLA", "Legs", True),
    ("left_cache", "Knee cover, left", 1, "PLA", "Legs", True),
    ("right_cache", "Knee cover, right", 1, "PLA", "Legs", True),
    ("left_roll_to_pitch", "Hip roll-to-pitch, left", 1, "PLA", "Hips", True),
    ("right_roll_to_pitch", "Hip roll-to-pitch, right", 1, "PLA", "Hips", True),
    ("roll_motor_bottom", "Roll motor bottom", 2, "PLA", "Hips", True),
    ("roll_motor_top", "Roll motor top", 2, "PLA", "Hips", True),
    ("trunk_bottom", "Trunk bottom", 1, "PLA", "Trunk & body", True),
    ("trunk_top", "Trunk top", 1, "PLA", "Trunk & body", True),
    ("body_front", "Body front", 1, "PLA", "Trunk & body", True),
    ("body_middle_bottom", "Body middle bottom", 1, "PLA", "Trunk & body", True),
    ("body_middle_top", "Body middle top", 1, "PLA", "Trunk & body", True),
    ("body_back", "Body back", 1, "PLA", "Trunk & body", True),
    ("battery_pack_lid", "Battery pack lid", 1, "PLA", "Trunk & body", True),
    ("neck_left_sheet", "Neck sheet, left", 1, "PLA", "Neck & head", True),
    ("neck_right_sheet", "Neck sheet, right", 1, "PLA", "Neck & head", True),
    ("head_pitch_to_yaw", "Head pitch-to-yaw", 1, "PLA", "Neck & head", True),
    ("head_yaw_to_roll", "Head yaw-to-roll", 1, "PLA", "Neck & head", True),
    ("head_roll_mount", "Head roll mount", 1, "PLA", "Neck & head", True),
    ("head", "Head", 1, "PLA", "Neck & head", True),
    ("head_bot_sheet", "Head bottom sheet", 1, "PLA", "Neck & head", True),
    ("left_antenna_holder", "Antenna holder, left", 1, "PLA", "Neck & head", True),
    ("right_antenna_holder", "Antenna holder, right", 1, "PLA", "Neck & head", True),
    ("left_eye", "Eye, left", 1, "PLA", "Expression (optional)", False),
    ("right_eye", "Eye, right", 1, "PLA", "Expression (optional)", False),
    ("bulb", "Projector bulb", 1, "PLA", "Expression (optional)", False),
    ("flash_light_module", "Flashlight module", 1, "PLA", "Expression (optional)", False),
    ("flash_reflector_interface", "Flash reflector interface", 1, "PLA", "Expression (optional)", False),
    ("speaker_interface", "Speaker interface", 1, "PLA", "Expression (optional)", False),
    ("speaker_stand", "Speaker stand", 1, "PLA", "Expression (optional)", False),
]

# Designer's colours from the Onshape model, merged into a few filament choices.
PALETTE = [
    {"id": "white", "name": "White PLA", "hex": "#eaeaea"},
    {"id": "yellow", "name": "Yellow PLA", "hex": "#fab601"},
    {"id": "orange", "name": "Orange PLA", "hex": "#e8922a"},
    {"id": "charcoal", "name": "Charcoal PLA", "hex": "#393838"},
    {"id": "grey", "name": "Grey PLA", "hex": "#a5a5a5"},
    {"id": "blue", "name": "Blue PLA", "hex": "#3b61b4"},
    {"id": "black_tpu", "name": "Black TPU", "hex": "#4e4c47"},
]
DESIGNER = {
    "white": ["body_back", "body_front", "body_middle_bottom", "body_middle_top", "head",
              "left_cache", "right_cache", "roll_motor_top"],
    "yellow": ["foot_side", "foot_top", "left_antenna_holder", "right_antenna_holder"],
    "orange": ["left_roll_to_pitch", "right_roll_to_pitch"],
    "charcoal": ["knee_to_ankle_left_sheet", "knee_to_ankle_right_sheet", "neck_left_sheet",
                 "neck_right_sheet", "trunk_bottom", "trunk_top", "head_pitch_to_yaw",
                 "head_bot_sheet", "head_roll_mount"],
    "grey": ["leg_spacer", "roll_motor_bottom", "head_yaw_to_roll"],
    "blue": ["battery_pack_lid"],
    "black_tpu": ["foot_bottom_pla", "foot_bottom_tpu"],
}

# URDF mesh file -> tracker item.  Print parts map to themselves by name.
MESH_MAP = {
    "left_knee_to_ankle_left_sheet": "print:knee_to_ankle_left_sheet",
    "left_knee_to_ankle_right_sheet": "print:knee_to_ankle_right_sheet",
    "wj-wk00-0122topcabinetcase_95": "buy:servo_sts3215",
    "wj-wk00-0123middlecase_56": "buy:servo_sts3215",
    "wj-wk00-0124bottomcase_45": "buy:servo_sts3215",
    "drive_palonier": "buy:servo_sts3215",
    "passive_palonier": "buy:servo_sts3215",
    "sg90": "buy:servo_9g",
    "raspberrypizerow": "buy:pi_zero_2w",
    "bno055": "buy:imu_bno055",
    "cell": "buy:cell_18650",
    "holder": "buy:holder_2s",
    "bms": "buy:bms_2s",
    "power_switch": "buy:power_switch",
    "usb_c_charger": "buy:usbc_charger",
    "roll_bearing": "buy:bearing_6804",
    "board": "other:PCB (servo adapter / regulator)",
    "antenna": "other:Antenna",
}
for pid, *_ in PRINT:
    MESH_MAP.setdefault(pid, f"print:{pid}")

# Official BOM (EUR) - replaced by verified Canadian sourcing when BUY_FILE is given.
OFFICIAL_BUY = [
    ("servo_sts3215", "Feetech STS3215 servo, 7.4 V (19 kg·cm)", 14, "Servos & control", False),
    ("servo_adapter", "Waveshare Bus Servo Adapter (A)", 1, "Servos & control", False),
    ("pi_zero_2w", "Raspberry Pi Zero 2 W", 1, "Servos & control", False),
    ("microsd", "microSD card 32 GB", 1, "Servos & control", False),
    ("imu_bno055", "BNO055 9-DOF IMU", 1, "Servos & control", False),
    ("foot_switch", "Foot contact switch (SS-10)", 4, "Servos & control", False),
    ("servo_9g", "9 g micro servo (antennas)", 2, "Servos & control", False),
    ("cell_18650", "18650 cell, high-drain", 2, "Power", False),
    ("holder_2s", "2S 18650 holder", 1, "Power", False),
    ("bms_2s", "2S BMS", 1, "Power", False),
    ("ubec_5v", "5 V regulator (UBEC)", 1, "Power", False),
    ("power_switch", "Power switch", 1, "Power", False),
    ("usbc_charger", "USB-C 2S charger board", 1, "Power", False),
    ("barrel_jack", "2.1 mm barrel jack", 2, "Power", False),
    ("xt30", "XT30 connector pair", 1, "Power", False),
    ("bearing_6804", "Bearing 6804 (20×32×7 mm)", 3, "Power", False),
    ("cable_sleeve", "Braided cable sleeve", 1, "Power", False),
    ("usb_15cm", "Micro-USB to USB-C cable, 15 cm", 1, "Power", False),
    ("usb_50cm", "Micro-USB to USB-C cable, 50 cm", 1, "Power", False),
    ("m3_inserts", "M3 heat-set inserts", 1, "Hardware", False),
    ("m3_screws", "M3 screw assortment", 1, "Hardware", False),
    ("loctite_243", "Loctite 243 threadlocker", 1, "Hardware", False),
    ("wire", "Silicone hookup wire", 1, "Hardware", False),
    ("pla", "PLA filament (~500 g)", 1, "Hardware", False),
    ("tpu", "TPU 95A filament", 1, "Hardware", False),
    ("reflector", "Projector reflector", 1, "Expression (optional)", True),
    ("speaker", "Speaker", 1, "Expression (optional)", True),
    ("leds", "LEDs (eyes + projector)", 3, "Expression (optional)", True),
    ("mic", "I2S microphone", 1, "Expression (optional)", True),
    ("eye_diffusers", "Eye diffusers", 4, "Expression (optional)", True),
    ("amp", "MAX98357A I2S amplifier", 1, "Expression (optional)", True),
    ("camera", "Pi camera (Zero-compatible)", 1, "Expression (optional)", True),
]

buy = [{"id": i, "name": n, "qty": q, "category": c, "optional": o} for i, n, q, c, o in OFFICIAL_BUY]
if BUY_FILE:
    extra = {b["id"]: b for b in json.loads(BUY_FILE.read_text())}
    for b in buy:
        b.update({k: v for k, v in extra.get(b["id"], {}).items() if k not in ("id", "category", "optional")})

catalog = {
    "print": [{"id": i, "name": n, "file": f"{i}.stl", "qty": q, "material": m, "infill": "40%" if m == "TPU" else "15%",
               "section": s, "in_model": im} for i, n, q, m, s, im in PRINT],
    "buy": buy,
    "palette": PALETTE,
    "designer_colours": {pid: cid for cid, pids in DESIGNER.items() for pid in pids},
    "mesh_map": MESH_MAP,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(catalog, indent=1))
print("wrote", OUT, len(catalog["print"]), "print parts,", len(buy), "buy items")
