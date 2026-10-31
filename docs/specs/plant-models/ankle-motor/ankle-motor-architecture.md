# Ankle motor plant — architecture

## Subsystems and dependency direction

1. **Reference profile:** selected workbook inputs with units, cell provenance and missing-data flags; no raw private workbook or unrelated device configuration.
2. **Pure policy:** immutable output-shaft specification, directional torque inequalities and inertia-budget allocation. No simulator dependency.
3. **Mechanical realization:** cached MJCF with explicit rigid bodies plus only the unrepresented portion of the selected shaft budget. Passive axes do not receive a second reflected motor inertia.
4. **Native actuation adapter:** fail-closed gain/bias definition plus process-global, marker-dispatched pure callback. Actual transmission velocity selects the torque interval.
5. **Experiment/telemetry:** integrated fixed-shin bench and same-state requested/limited/applied torque records. Analytical graphs call the same policy.

The profile feeds both XML generation and runtime validation; a read-only cache check prevents hand-edited copies from becoming a second truth source.

## State and concurrency

Mechanical states belong to MuJoCo. Operating modes belong to each data instance, not to mutable shared model force ranges. Callback evaluation never advances a thermal timer or modifies model parameters. Global callback ownership is explicit; compilation and supported native evaluation are serialized while the Python hook is temporarily unavailable. Uncoordinated external native stepping during compilation is unsupported.

## Fidelity choices

The actuation interface is an output-shaft torque request, not voltage. No unidentifiable electrical or friction state is invented. The explicit-body inertia ledger is a documented budget policy, not evidence of CAD component identity. Existing structural compliance remains separate from the motor capability envelope.

See [implementation](ankle-motor-implementation-plan.md) and [test plan](ankle-motor-test-plan.md).
