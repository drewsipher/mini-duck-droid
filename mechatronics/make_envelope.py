#!/usr/bin/env python3
"""Build mechatronics/envelope.json: the deployable safety envelope with a source for every
number. Numbers that nobody has measured are marked "unverified"; numbers from the robot or
the bench are "measured"; numbers read from a file in this repo are "verified" (file named).

    cd sim && .venv/bin/python ../mechatronics/make_envelope.py        # needs mujoco for the joint ranges

Per-joint deploy range = (MJCF range ∩ measured stops) shrunk by 3 deg each side, then clipped
by the runtime's soft limits (duck_config.json: head_yaw ±70 deg). The 3 deg margin is what
the tuck poses in battery_guard.py keep from the stops.
"""
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
XML_DIR = ROOT / "sim/train_pg/playground/open_duck_mini_v2/xmls"
SCENE = XML_DIR / "scene_flat_terrain_backlash.xml"
ROBOT = XML_DIR / "open_duck_mini_v2_backlash.xml"
REAL_STOPS = ROOT / "joint_limits/real_limits_20261014.json"
RESPONSE = HERE / "response/response.json"
MARGIN = math.radians(3.0)
SOFT_LIMITS_DEG = {"head_yaw": [-70.0, 70.0]}      # ~/duck_config.json on the Pi (walk 9 config snapshot, 2026-10-14)
JOINT_ORDER = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch", "head_pitch",
               "head_yaw", "head_roll", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
# joints_offsets from the Pi's duck_config.json as recorded in walk_recordings/meta_walk9.pkl (2026-10-14)
OFFSETS = {"left_hip_yaw": 0.062, "left_hip_roll": -0.051, "left_hip_pitch": 0.035, "left_knee": 0.078, "left_ankle": 0.0785,
           "neck_pitch": 0.09, "head_pitch": -0.01, "head_yaw": -0.067, "head_roll": 0.055, "right_hip_yaw": -0.005,
           "right_hip_roll": 0.016, "right_hip_pitch": 0.056, "right_knee": -0.052, "right_ankle": 0.0}


def q(value, unit, status, source, note=None):
    d = {"value": value, "unit": unit, "status": status, "source": source}
    if note:
        d["note"] = note
    return d


def joints():
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    k = m.key("home")
    home = {m.actuator(i).name: float(k.ctrl[i]) for i in range(m.nu)}
    real = json.loads(REAL_STOPS.read_text())
    out = {}
    for n in JOINT_ORDER:
        j = m.joint(n)
        lo, hi = (float(x) for x in m.jnt_range[j.id])
        rlo, rhi = real[n]
        dlo, dhi = max(lo, rlo) + MARGIN, min(hi, rhi) - MARGIN
        soft = SOFT_LIMITS_DEG.get(n)
        if soft:
            dlo, dhi = max(dlo, math.radians(soft[0])), min(dhi, math.radians(soft[1]))
        out[n] = {
            "mjcf_range_rad": [round(lo, 4), round(hi, 4)],
            "measured_stops_rad": [rlo, rhi],
            "soft_limit_rad": [round(math.radians(soft[0]), 4), round(math.radians(soft[1]), 4)] if soft else None,
            "deploy_range_rad": [round(dlo, 3), round(dhi, 3)],
            "home_rad": round(home[n], 3),
            "home_inside": dlo <= home[n] <= dhi,
            "offset_rad": OFFSETS[n],
            "max_velocity_rad_s": 4.0,
        }
    return out


def main():
    resp = json.loads(RESPONSE.read_text()) if RESPONSE.exists() else {}
    v = resp.get("variants", {})
    model = v.get("model, 1.86 N·m", {})
    deploy = v.get("model, 1.86 N·m, deploy chain 8 Hz + 4 rad/s", {})
    env = {
        "name": "Open Duck Mini v2: deployable safety envelope",
        "version": "0.1 (draft, 2026-10-23, Mechatronics)",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status_legend": {
            "verified": "read from a file in this repo (named in source)",
            "measured": "measured on the real duck or its servos (file named)",
            "derived": "computed from verified/measured numbers (how, in note)",
            "proposed": "a rule we propose; not yet tested on the robot",
            "unverified": "manufacturer figure or assumption; nobody has checked it on this duck",
        },
        "robot_model": {
            "mjcf": str(ROBOT.relative_to(ROOT)),
            "mjcf_sha256": q("11e922dc9524716e9f54b2fd8be54133fa65efbc61de96ddc846d991de1c9d38", "sha256", "verified", "sha256sum of the training XML, 2026-10-23"),
            "robot_model_hash_cad_copy": q("759d5b7fb4b0c78a2f0038bec50e6526e6d65fbf0eb51297745df9a8c9687108", "sha256", "verified", "mechatronics/model/robot_model_hash.txt (apply_masses.py --baseline)"),
            "actuator_model_version": q("bam-sts3215-7.4V/xml-2025 (damping 0.56, frictionloss 0.068, armature 0.027, kp 17.11, forcerange 3.23)", "", "verified",
                                        "class sts3215 in the training XML; the older fit in joints_properties.xml is 0.60/0.052/0.028/17.8/3.35"),
            "total_mass_kg": q(2.107, "kg", "verified", "mechatronics/masses_mjcf.json (sum of 15 bodies)",
                               "the real duck is estimated at 1.5-1.8 kg (sim/report1); mass is the biggest known model error -> weigh_me.md"),
        },
        "torque": {
            "deploy_limit_nm": q(1.86, "N.m", "derived", "19 kg.cm rated stall at 7.4 V (README: 'STS3215 7.4 V (19 kg.cm)') x 0.0981 = 1.864; used by eval_policy.py, curriculum TORQUE_RANGE, HR_TORQUE",
                                 "every score and every held test since report 1 ran at this ceiling; a skill passes its battery at this value or it does not ship"),
            "model_limit_nm": q(3.23, "N.m", "verified", "forcerange in the training XML (BAM fit)",
                                "why not 3.23: it is the fit's saturation of kp*error on a bench supply, not a sustained torque; the manufacturer rates 1.9 N.m at 7.4 V; on a 2S pack shared by 14 servos the supply sags; the current brain walks forward at 0.13 m/s in sim at 3.35 but 0.10 at 1.86 and the real duck is nearer the latter (report 1 torque.png). Train at 3.23 and the policy learns to lean on torque the duck does not have"),
            "bench_demonstrated_nm": q(1.16, "N.m", "measured", "strength_data/strength_14_20261014-034321.csv (right ankle on a scale, 15 deg press, still rising at 21 % load)",
                                       "the maximum was not reached; so the real ceiling is > 1.16 and the 1.86 figure is unverified above that"),
            "stiffness_nm_per_rad": q(25.3, "N.m/rad", "measured", "report 1: 0.44 N.m/deg real vs 0.30 sim (x1.48); sim/make_strength_plot.py", "training KP_RANGE 1.1-1.8 covers it"),
            "training_randomisation": q("TORQUE_RANGE U(1.6, 2.1) N.m per joint when set; KP_RANGE U(1.1, 1.8)", "", "verified", "sim/train_pg/playground/common/randomize.py, curriculum.sh stage s3"),
        },
        "velocity": {
            "max_motor_velocity_rad_s": q(4.0, "rad/s", "verified", "MAX_MOTOR_VEL=4.0 in sim/curriculum.sh and report 2 'deploy with ... max motor velocity 4 rad/s'; joystick.py clips the target step to max_motor_velocity*dt",
                                          "the runtime default is 5.24 rad/s (v2_rl_walk_mujoco.py, duck_config 'max_motor_velocity_rad_s': 5.24); set 4.0 at deploy"),
            "runtime_default_rad_s": q(5.24, "rad/s", "verified", "sim/upstream_runtime/scripts/v2_rl_walk_mujoco.py; Pi config snapshot in walk_recordings/meta_walk9.pkl has velocity_clip true"),
            "real_walk_joint_speed_rad_s": q({"p95": 2.7, "p99": 3.8, "max": 4.5}, "rad/s", "measured", "walk_recordings/obs_walk7.pkl dof_vel (knees are the fastest), 155 s of walking on 2026-10-14"),
            "model_ceiling_at_deploy_torque_rad_s": q(3.2, "rad/s", "derived", "(1.86 - 0.068) / 0.56 from the XML's forcerange cap, frictionloss and damping (mechatronics/response/response.json velocity_ceiling)",
                                                       "lower than the 3.8-4.5 rad/s the real knees reach: capping forcerange alone makes the model slower than the duck at speed. The bench no-load speed test settles it"),
            "datasheet_no_load_rad_s": q(4.7, "rad/s", "unverified", "Feetech listing 0.222 s/60 deg at 7.4 V (no document on disk)"),
        },
        "action_filter": {
            "setting_hz": q(8.0, "Hz", "verified", "ACT_LPF_HZ=8 in curriculum.sh; runtime --cutoff_frequency 8 (LowPassActionFilter in mini_bdx_runtime/rl_utils.py)"),
            "formula": "alpha = (1/fc) / (1/50 + 1/fc) = 0.862; y = alpha*y + (1-alpha)*x at 50 Hz (same in joystick.py and sim_test.py)",
            "effective_minus3db_hz": q(1.19, "Hz", "derived", "mechatronics/servo_response_sim.py sweep through the deploy chain; also 50/(2*pi) * -ln(0.862) = 1.18 Hz",
                                       "the '8 Hz' setting is a 1.2 Hz first-order filter: the formula uses 1/fc where an RC constant belongs. Sim and runtime share the formula, so training and the duck agree; name the package envelope by the real cutoff"),
            "step_rise_through_chain_s": q(deploy.get("step", {}).get("rise_time_s"), "s", "derived", "response.json, 0.5 rad step, 10-90 %"),
        },
        "control": {
            "policy_hz": q(50, "Hz", "verified", "control_freq in v2_rl_walk_mujoco.py; measured loop 8 ms mean / 9.7 ms p95 (meta_walk9.pkl tick)"),
            "servo_p_gain": q({"legs": 30, "head": 8, "eeprom_default": 32}, "firmware units", "verified", "v2_rl_walk_mujoco.py pid=[30,0,0], kps[5:9]=8; setup_all_servos.py writes P=32 I=0 D=0"),
            "action_scale_rad": q(0.25, "rad", "verified", "runtime action_scale; LEG_ACTION_SCALE=0.5 brains need the matching runtime value (memory note)"),
            "servo_acceleration_register": q(0, "", "verified", "setup_all_servos.py: acceleration 0 (no internal ramp)"),
        },
        "fall": {
            "tilt_deg": q(45, "deg", "verified", "sim_test.py run(): fell = tilt_max > 45 (angle between body z and world z); eval_policy.py uses the same",
                          "framework 3.3 says 0.8 rad (45.8 deg) orientation error; training itself only terminates at upvector z < 0 (90 deg)"),
            "stability_governor": q({"tilt_lo_deg": 10, "tilt_hi_deg": 25, "enabled": False}, "deg", "verified", "Pi duck_config snapshot in meta_walk9.pkl", "a command scaler, not a fall rule; off"),
            "min_base_height_m": q(0.09, "m", "proposed", "60 % of the standing base height; below the chin-up tuck (0.11 m) so a deliberate sit does not count as a fall",
                                   "frame: base body z with the floor at 0, the number sim_test.base_pose() returns. Standing 0.15 m (scene keyframe 'home'), IMU 0.20 m, CoM 0.19 m, head 0.39 m; tuck chin_up base 0.11 m, head_down 0.12 m (make_envelope.py / MuJoCo FK)"),
        },
        "thermal": {
            "servo_max_temperature_register_c": q(70, "C", "unverified", "STS3215 memory table default for EEPROM 0x0D; read it with servo_health.py (Pi unreachable 2026-10-23)"),
            "bench_abort_c": q(55, "C", "verified", "strength_test.py MAX_TEMP"),
            "measured_rest_c": q(30, "C", "measured", "strength_data CSVs (14 Oct, servos idle-holding, room temperature)", "no reading under walking yet; the telemetry pull supplies it"),
            "proposed_rule": {
                "warn_c": 50, "cool_down_c": 60, "hard_stop_c": 65, "resume_below_c": 45,
                "rule": "max over 14 servos, read at least every 20 s. >= 50: banner, no new episodic skill starts. >= 60: finish the current skill (timeout applies), go to idle, then sit-down abort and servo P to 2 (hold, no stiffness); stay until every servo <= 45. >= 65 or any 'temperature' status flag: motors off via the sit-down path immediately. Log every crossing to the experiment log.",
                "status": "proposed",
                "why": "70 C is where the servo unloads itself (default unloading condition bit 2); 65 leaves 5 C for the sampling gap; the Olaf character robot feeds temperature to its policy and our risk table (framework 5.5) names thermal drift",
            },
        },
        "voltage": {
            "nominal_v": q(7.4, "V", "verified", "2S Li-ion pack: 2 x Molicel 18650 (tracker/data/catalog.json), 7.4 V servo version (README)"),
            "full_charge_v": q(8.4, "V", "unverified", "4.2 V/cell, BMS listed as 7.4 V/8.4 V 20 A (archive print_vs_buy.csv)"),
            "observed_after_charge_v": q(8.3, "V", "measured", "puppet agent / duck_agent.py state on 2026-10-22: every servo reports 8.3 V with the 'voltage' status flag"),
            "strength_test_v": q([7.2, 7.9], "V", "measured", "strength_data CSVs volt column, 14 Oct (sagging to 7.23 V at 183 mA on one servo while holding)"),
            "servo_max_voltage_register_v": q(8.0, "V", "unverified", "EEPROM 0x0E; Feetech default 80 (0.1 V units) for the 7.4 V STS3215 per the memory table; servo_ping.py reads it as 'voltage limits'; NOT read yet on this duck"),
            "servo_min_voltage_register_v": q(4.0, "V", "unverified", "EEPROM 0x0F default 40"),
            "battery_guard": q({"warn_v": 7.0, "crit_v": 6.8, "hard_v": 6.5}, "V", "verified", "duck_battery/battery_guard.py DEFAULTS and the Pi config snapshot"),
            "operating_range_v": q([6.8, 8.4], "V", "proposed", "battery guard critical to full charge"),
            "verdict_8p3": {
                "status": "pending the register read",
                "most_likely": "8.3 V > 8.0 V max-voltage register, so the flag is correct and expected on a fresh charge, clears by itself once the pack is below 8.0 V (first minutes of use).",
                "consequence": "harmless unless bit 0 of the unloading condition (0x13, default 44 = 0b101100: temperature, current, overload) is set; by default a voltage alarm only lights the LED (0x14 default 47 includes it). servo_health.py reports unload_on_voltage per servo.",
                "options_for_the_user": ["leave it (flag is advisory)", "stop charging at ~8.2 V (storage charge)", "raise 0x0E to 85-90 on all 14 servos: a one-time EEPROM write the user must authorise and do with the bus free; the servos already run at 8.3 V"],
            },
        },
        "current": {
            "present_current_unit": q(6.5, "mA per LSB", "verified", "register 0x45; duck_agent.py, strength_test.py MA_PER_UNIT"),
            "servo_protection_current_register_a": q(3.25, "A", "unverified", "EEPROM 0x1C default 500 x 6.5 mA; read with servo_health.py"),
            "datasheet_stall_a": q(2.7, "A", "unverified", "Feetech listing at 7.4 V"),
            "bench_abort_ma": q(2500, "mA", "verified", "strength_test.py MAX_MA"),
            "measured_static_hold": q({"nm": 1.16, "ma": 183, "v": 7.23}, "", "measured", "strength_14_20261014-034321.csv last row"),
            "bms_rating_a": q(20, "A", "unverified", "BOM: 2S BMS 7.4/8.4 V 20 A"),
            "pi_supply": q("5 V from the pack through a regulator; under-voltage flags logged by duck_battery/power_watch.py", "", "verified", "power_watch.py"),
            "proposed": {"per_servo_sustained_a": 1.0, "per_servo_peak_a": 2.5, "pack_sustained_a": 8.0, "pack_peak_a": 15.0, "status": "proposed",
                         "why": "peak = the bench abort; pack numbers leave 25 % under the BMS rating; sustained numbers are guesses until the wave_power_test.py log (power_log.csv, not on disk) or a walking telemetry series exists"},
        },
        "joints": joints(),
        "joint_rules": {
            "deploy_range": "MJCF range ∩ measured stops (joint_limits/real_limits_20261014.json, joint_limits_probe.py by hand on 2026-10-14), minus 3 deg each side, clipped by the runtime soft limits",
            "clip_checker": "a clip whose joint targets leave deploy_range_rad, or whose target rate exceeds max_velocity_rad_s, fails stage 2",
            "suspect": "left_hip_pitch measured stop -2.45 rad (-140 deg) is probably the leg folding past the probe (report 1); the intersection with the MJCF makes it harmless",
        },
        "servo_response_model": {
            "source": "mechatronics/response/response.json (servo_response_sim.py)",
            "model_1p86": {"rise_10_90_s": model.get("step", {}).get("rise_time_s"), "overshoot_pct": model.get("step", {}).get("overshoot_pct"),
                           "settle_2pct_s": model.get("step", {}).get("settling_2pct_s"), "bandwidth_3db_hz_small": model.get("sweep", {}).get("0.05", {}).get("bandwidth_3db_hz"),
                           "bandwidth_3db_hz_0p2rad": model.get("sweep", {}).get("0.2", {}).get("bandwidth_3db_hz")},
            "bench_tolerance": "see bench_procedure.md: rise time within 25 %, overshoot within 5 points, -3 dB within 20 %",
        },
        "unverified_summary": [
            "servo EEPROM limits (0x0D, 0x0E, 0x0F, 0x13, 0x1C): never read on this duck; servo_health.py does it, Pi was unreachable tonight",
            "stall torque 1.86 N.m: bench reached 1.16 and was still rising",
            "no-load speed and the model's 3.2 rad/s ceiling at 1.86 N.m vs the real 4.5 rad/s",
            "total mass 2.107 kg vs 1.5-1.8 kg estimate",
            "thermal numbers (no temperature series under load exists)",
            "current budgets (no pack-current log on disk)",
            "min base height 0.09 m (proposed from FK, untested)",
        ],
    }
    out = HERE / "envelope.json"
    out.write_text(json.dumps(env, indent=1, ensure_ascii=False))
    print(f"wrote {out.relative_to(ROOT)}")
    for n, j in env["joints"].items():
        print(f"  {n:16s} deploy {j['deploy_range_rad']}  home {j['home_rad']:+.3f} {'ok' if j['home_inside'] else 'HOME OUTSIDE'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
