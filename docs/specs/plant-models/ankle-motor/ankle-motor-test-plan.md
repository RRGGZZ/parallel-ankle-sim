# Ankle motor plant — test plan

| ID | Test | Expected result |
|---|---|---|
| M01 | Workbook units and one-time inertia reflection | 0.61e-4 × 30² = 0.0549 kg·m²; unknown reducer input remains unknown |
| M02 | Explicit body allocation and compiled mass matrix | Existing 0.0002 + allocated 0.0547 = 0.0549 per shaft; no duplicate passive armature |
| M03 | Native static guardrails for all four motors | Control, scalar actuator force and joint actuator force support the same peak brake envelope |
| M04 | Direction/velocity/request grid | Positive and negative velocity × both torque signs, zero, knot neighbors, 130/160/190 rpm, both modes agree with shared policy |
| M05 | Unsaturated request | +10 N·m at +130 rpm remains +10; no erroneous multiplicative derating |
| M06 | Overspeed braking | Same-direction drive goes to zero; opposite-direction braking remains; qvel is never clipped |
| M07 | Rated-current cap | Interval intersects ±16, not a fabricated complete thermal curve; expected driving crossover is 136 rpm |
| M08 | Actual vs commanded speed | Limiter uses the measured state even when target velocity has the opposite sign |
| M09 | Callback absence and ownership | Missing hook yields zero; conflicts are rejected; close cannot erase a later third-party callback |
| M10 | Multiple data/model instances | Per-data modes do not cross-contaminate; parameters are not changed during force evaluation |
| M11 | Native batched and RK evaluations | Callback is evaluated during actual integration and substeps, not only before the outer loop |
| M12 | Invalid model/profile/ABI/mode | Bind fails clearly or invalid actuation fails closed; non-finite data do not generate unsafe torque |
| M13 | Unicode paths, includes and left/right extraction | Native virtual assets load; general actuators, userdata and metadata are retained |
| M14 | Cache staleness | Read-only profile/XML check fails if any generated motor parameter drifts |
| M15 | Integrated trajectory | Recorded finite qpos/qvel/qacc and actual generalized force; no passive external driving; replay/torque-bound and tracking/tendon consistency checks |
| M16 | Numerical sensitivity | Repeat with smaller integration step; compare states, tracking and force validity without claiming real hardware calibration |

Reference limits: drive 40 N·m, brake 44 N·m (+10% estimate), X1/X2 100/160 rpm; default operating cap 16 N·m. Zero-speed current-limit interval is conservatively ±40. Tests assess agreement with the approved reference model, not the missing electrical, thermal or CAD evidence.
