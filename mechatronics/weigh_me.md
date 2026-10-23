# Weigh the duck (10 minutes with a kitchen scale)

The training model says the duck weighs **2.107 kg**. Your estimate is 1.5 to 1.8 kg
(report 1). That gap is the largest known error in the model, and it changes how hard every
servo has to work in simulation. This checklist gets the real numbers into the model without
taking the duck apart more than you want to.

Write every reading into `mechatronics/masses.json` (grams, whole numbers are fine), then run

    cd ~/src/mini-duck-droid/sim && .venv/bin/python ../mechatronics/apply_masses.py

It writes a copy of the model with your masses to `mechatronics/model/` and prints the
`robot_model_hash` that skill packages record. The training XML itself is not touched.

## Before you start
- Kitchen scale, 1 g resolution, tared with nothing on it. Weigh something known (a full
  500 ml bottle is about 520 g) once to check it.
- Duck off, motors limp. Hold the head so it does not flop into the scale.
- Write the date and the scale into the file's `date` and `scale` fields.

## Level 1: no tools (do this tonight if you can)
| # | Weigh | Write into | Notes |
|---|---|---|---|
| 1 | The whole duck, battery pack in, exactly as it walks | `total_g.with_battery` | Stand him on the scale or lay him on it; make sure no cable hangs on the table. |
| 2 | The battery pack alone (holder, both cells, BMS, its leads) | `battery_pack_g` | Lift the pack out through the lid. |
| 3 | The duck without the pack | `total_g.without_battery` | Check: 1 ≈ 2 + 3 within a few grams. If not, something was hanging. |

With just these three numbers the tool scales every body in the model to the real total.
That already fixes the 15 to 30 % error in the gravity and inertia the policy fights.

## Level 2: one screwdriver (when the head is off anyway)
| # | Weigh | Write into | Notes |
|---|---|---|---|
| 4 | The head assembly: everything above the head-roll servo horn (shell, eyes, camera mount, speaker if fitted) | `bodies_g.head_assembly` | The model says 407 g; it is 20 % of the duck and sits 35 cm up, so this one matters most. |
| 5 | Head plus neck: everything above the neck-pitch servo (4 bodies) | `groups_g.head_and_neck` | Only if the neck comes off the trunk easily. |

## Level 3: next time a leg is apart
| # | Weigh | Write into |
|---|---|---|
| 6 | A whole leg (hip yaw horn down to the foot) | `groups_g.left_leg` or `right_leg` |
| 7 | Foot alone (with the TPU sole) | `bodies_g.foot_assembly` / `foot_assembly_2` |
| 8 | Shank alone (knee to ankle, ankle servo inside) | `bodies_g.knee_and_ankle_assembly_2` / `_4` |
| 9 | One bare STS3215 servo (no horn) | `servo_g` (reference only; the model says nothing, Feetech lists about 55 g) |

## What the names mean
Body names are the CAD's; `masses_mjcf.json` lists all 15 with a plain description and the
CAD mass. The groups the tool knows: `trunk` (1 body), `head_and_neck` (4), `left_leg` (5),
`right_leg` (5).

## Rules the tool follows
1. A body you weighed is set to that mass.
2. A group you weighed is spread over its bodies that you did not weigh, in the CAD's proportions.
3. The total (with battery) scales everything else so the whole duck matches.
4. Inertia tensors are scaled with the mass (same shape, new mass).
5. Nulls are ignored. Any single number is better than none.

## Open point to settle while weighing
Whether the CAD trunk (699 g) includes the two 18650 cells (about 2 × 47 g) is unknown.
Readings 2 and 3 answer it: if `without_battery` is close to 2107 − battery, the CAD had the
cells; if it is close to 2107, it did not.
