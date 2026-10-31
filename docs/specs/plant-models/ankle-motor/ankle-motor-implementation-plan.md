# Ankle motor plant — implementation plan

1. Extract only the approved row and its defining metadata. Preserve the supplied workbook unchanged and retain a digest for provenance.
2. Define the immutable, unit-checked reference interface before integrating simulation entry points.
3. Implement signed torque bounds and explicit inertia accounting; reject malformed profiles or negative remaining inertia budgets.
4. Generate the cached MJCF actuator parameters and allocation from that interface. Retain the original non-motor mechanism and restrict changes to the four ankle motors/passive armature placeholders.
5. Bind actual-speed native actuation with fail-closed behavior, per-data modes and explicit callback ownership. Handle model compilation without invoking Python on incomplete native objects.
6. Migrate supported loaders and analytical plots. Add a motor-driven recorder that does not externally torque the passive ankle.
7. Run independent policy, adapter, whole-model and integrated trajectory tests; inspect generated graphs and all intended changes.
8. Publish only reviewed source, selected public parameters, documentation and compact verification evidence. Commit and perform a normal non-forced push after local verification; do not modify remote history or publish the workbook.

Execution results belong in a separate validation record, not in this plan.
