# Ankle motor plant — system specification

## Purpose and boundary

Make the four explicit parallel-ankle motors, analytical capability plots and MuJoCo actuation use the same user-approved workbook reference. Only the ankle motor family changes; unrelated robot joints, the serial-joint URDF variants and optional Pinocchio tools are not silently reinterpreted as motor-side models.

## Operating scenarios

- Fixed-shin, gravity-on, unloaded tracking; continuous/rated-current operation is the default.
- Explicit current-limit drive and estimated braking probes, including reversal and externally imposed overspeed.
- Full MJCF loading and independent left/right benches; multiple data instances may have different modes.

## Controller interface

| Class | Signal | Units / timing |
|---|---|---|
| u | Requested gearbox-output torque per motor; explicit operating mode | N·m; mode set outside integration callbacks |
| w | Gravity, external load and mechanical transmission reactions | SI generalized forces; no invented electrical disturbance inputs |
| y | Actual shaft position, velocity and actuator generalized force | rad, rad/s, N·m; same-state forward evaluation |
| z | Request, lower/upper bounds, limited/applied torque, saturation, power, inertia ledger | SI; sampled together before the next step |

Nominal trajectory sampling and mechanical integration are 1 ms. The limiter is evaluated at every native actuation evaluation, not merely at controller sample times.

## Evidence and acceptance

The user approved main-sheet row 10 and the embedded directional reference diagram. Simulation-derived inertia, measured torque constant, calculated current limits and estimated braking have distinct evidence labels. Missing gearbox inertia, friction, electrical/thermal and regeneration data remain explicitly unvalidated. Raw private workbook data are not redistributed.

Acceptance means executable software agrees with this reference contract and records actual integrated motor-driven states. It does not mean the full robot CAD, real motor thermal rating or regenerating power bus is physically certified.

See the [public evidence contract](../../../motor_model.md) and [architecture](ankle-motor-architecture.md).
