"""Verify actual MuJoCo torque against the shared motor model and replay data."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
import mujoco
import numpy as np
from pndbotics_ankle import CustomAnkleKinematics
from pndbotics_ankle.motor import DEFAULT_MOTOR
from pndbotics_ankle.mujoco_io import load_mujoco_model, build_fixed_ankle_model
from pndbotics_ankle.mujoco_motor import MotorRuntime


def probe_native_forces(out):
    spec = DEFAULT_MOTOR
    model = load_mujoco_model()
    data = mujoco.MjData(model)
    rpms = [-200., -190., -160.000001, -160., -159.999999, -145., -136., -130., -100.000001,
            -100., -99.999999, -50., 0., 50., 99.999999, 100., 100.000001, 130., 136., 145.,
            159.999999, 160., 160.000001, 190., 200.]
    requests = [-100., -44., -40., -10., 0., 10., 40., 44., 100.]
    rows = []
    max_error = 0.
    with MotorRuntime(model) as runtime:
        for mode in ("continuous", "current_limit"):
            runtime.set_mode(data, mode)
            for rpm in rpms:
                speed = rpm*np.pi/30
                data.qvel[runtime.dof_ids] = speed
                for request in requests:
                    data.ctrl[runtime.actuator_ids] = request
                    telemetry = runtime.telemetry(data)
                    expected = spec.limit_torque(request, speed, mode)
                    error = float(np.max(np.abs(telemetry["applied"]-expected)))
                    max_error = max(max_error, error)
                    np.testing.assert_allclose(telemetry["applied"], expected, rtol=0, atol=1e-10)
                    np.testing.assert_allclose(telemetry["actuator_force"], expected, rtol=0, atol=1e-10)
                    for i, name in enumerate(runtime.names):
                        rows.append([name, mode, rpm, request, telemetry["lower"][i], telemetry["upper"][i],
                                     telemetry["applied"][i], expected])
        ledger = runtime.inertia_ledger
        mass = np.empty((model.nv, model.nv))
        mujoco.mj_fullM(model, mass, data.qM)
        block = mass[np.ix_(runtime.dof_ids, runtime.dof_ids)]
        np.testing.assert_allclose(block, np.eye(4)*spec.output_inertia, rtol=0, atol=1e-12)
    with (out/"native_torque_probes.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["motor", "mode", "actual_rpm", "requested_Nm", "lower_Nm", "upper_Nm", "actual_generalized_Nm", "expected_Nm"])
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, mode in zip(axes, ("continuous", "current_limit")):
        grid = np.linspace(-200, 200, 801)
        bounds = np.array([spec.torque_bounds(float(rpm*np.pi/30), mode) for rpm in grid])
        ax.plot(grid, bounds[:, 0], color="#2477aa", lw=1.4, label="shared policy")
        ax.plot(grid, bounds[:, 1], color="#2477aa", lw=1.4)
        for request, marker in ((-100., "o"), (100., "x")):
            selected = [r for r in rows if r[0] == "ankleMotor_Left_1" and r[1] == mode and r[3] == request]
            ax.scatter([r[2] for r in selected], [r[6] for r in selected], marker=marker, s=23,
                       color="#dd7927", label=f"native force, request {request:g}")
        ax.set(title=mode, xlabel="Prescribed actual shaft speed (rpm)", ylabel="Motor generalized torque (N m)")
        ax.axhline(0, color="gray", lw=.5)
        ax.axvline(0, color="gray", lw=.5)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle(f"{spec.motor_model}: native force evaluation vs reference\nImposed speed states; not a speed-tracking experiment")
    fig.savefig(out/"mujoco_tn_verified.png", dpi=160)
    plt.close(fig)
    return {"evaluated_motor_cases": len(rows), "max_generalized_torque_error_Nm": max_error,
            "motor_inertia_ledger": ledger, "compiled_motor_mass_block_kg_m2": block.tolist()}


def verify_trajectory(folder):
    spec = DEFAULT_MOTOR
    model = build_fixed_ankle_model()
    data = mujoco.MjData(model)
    names = ("ankleMotor_Left_1", "ankleMotor_Left_2")
    maximum = {"force": 0., "acceleration": 0., "tendon": 0., "reflected_inertia": 0.}
    with np.load(folder/"telemetry.npz") as archive:
        # NpzFile does not cache decompressed members: read each array once.
        saved = {key: archive[key] for key in archive.files}
        for key in saved:
            if np.issubdtype(saved[key].dtype, np.number) and not np.isfinite(saved[key]).all():
                raise AssertionError(f"Nonfinite saved field {key}")
        times = saved["time"]
        dt = float(times[1]-times[0])
        model.opt.timestep = dt
        dofs = saved["motor_dofs"]
        mode = "continuous" if int(saved["mode_code"]) == 0 else "current_limit"
        max_all_samples_error = 0.
        for k in range(len(times)):
            for i in (0, 1):
                expected = spec.limit_torque(float(saved["requested_torque"][k, i]), float(saved["qvel"][k, dofs[i]]), mode)
                max_all_samples_error = max(max_all_samples_error, abs(expected-float(saved["motor_torque"][k, i])))
        if max_all_samples_error > 1e-10:
            raise AssertionError("Recorded motor torque violates the policy")
        kin = CustomAnkleKinematics()
        with MotorRuntime(model, names=names) as runtime:
            runtime.set_mode(data, mode)
            for k in np.linspace(0, len(times)-1, min(102, len(times)), dtype=int):
                data.qpos[:] = saved["qpos"][k]
                data.qvel[:] = saved["qvel"][k]
                data.ctrl[:] = saved["control"][k]
                data.qfrc_applied[:] = 0
                telemetry = runtime.telemetry(data)
                maximum["force"] = max(maximum["force"], float(np.max(np.abs(telemetry["applied"]-saved["motor_torque"][k]))))
                maximum["acceleration"] = max(maximum["acceleration"], float(np.max(np.abs(data.qacc-saved["qacc"][k]))))
                maximum["tendon"] = max(maximum["tendon"], float(np.max(np.abs(data.ten_length-saved["tendon_length"][k]))))
                j = kin.calc_Jvel(*saved["actual_roll_pitch"][k])
                reflected = j.T@np.diag([spec.output_inertia]*2)@j
                maximum["reflected_inertia"] = max(maximum["reflected_inertia"], float(np.max(np.abs(reflected-saved["Mref"][k]))))
            k = min(len(times)-2, len(times)//2)
            data.qpos[:] = saved["qpos"][k]
            data.qvel[:] = saved["qvel"][k]
            data.ctrl[:] = saved["control"][k]
            data.qfrc_applied[:] = 0
            runtime.step(data)
            next_error = float(np.max(np.abs(data.qpos-saved["qpos"][k+1])))
        if maximum["force"] > 1e-10 or maximum["acceleration"] > 1e-7 or maximum["tendon"] > 1e-12 or maximum["reflected_inertia"] > 1e-12 or next_error > 1e-11:
            raise AssertionError(f"Recorded-state replay failed: {maximum}, next={next_error}")
        integration_residual = float(np.max(np.abs(np.diff(saved["qpos"], axis=0)/dt-saved["qvel"][1:])))
        if integration_residual > 1e-8:
            raise AssertionError("Recorded positions disagree with integrated velocities")
        return {"samples": len(times), "duration_s": float(times[-1]), "dt_s": dt,
                "telemetry_sha256": hashlib.sha256((folder/"telemetry.npz").read_bytes()).hexdigest(),
                "all_samples_max_torque_error_Nm": max_all_samples_error,
                "reconstructed_state_max_errors": maximum, "single_step_next_qpos_error_rad": next_error,
                "position_integration_residual_rad_s": integration_residual}


def compare_steps(coarse, fine):
    with np.load(coarse/"telemetry.npz") as a, np.load(fine/"telemetry.npz") as b:
        ratio = int(round((a["time"][1]-a["time"][0])/(b["time"][1]-b["time"][0])))
        if ratio < 2:
            raise AssertionError("Fine trajectory must use a smaller step")
        np.testing.assert_allclose(a["time"], b["time"][::ratio], atol=1e-12, rtol=0)
        position_error = float(np.max(np.abs(a["qpos"]-b["qpos"][::ratio])))
        torque_error = float(np.max(np.abs(a["motor_torque"]-b["motor_torque"][::ratio])))
        if position_error > .001 or torque_error > .1:
            raise AssertionError("Step refinement exceeds the stated demonstration tolerances")
        return {"step_ratio": ratio, "max_position_difference_rad": position_error,
                "max_motor_torque_difference_Nm": torque_error,
                "position_tolerance_rad": .001, "torque_tolerance_Nm": .1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT/"artifacts"/"motor_validation")
    parser.add_argument("--fine-results", type=Path)
    parser.add_argument("--source-workbook", type=Path)
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    target = args.results/"motor_validation.json"
    target.write_text('{"status":"RUNNING"}\n', encoding="utf-8")
    try:
        result = {"motor_profile": DEFAULT_MOTOR.as_dict(), "mujoco_version": mujoco.__version__,
                  "source_xml_sha256": hashlib.sha256((ROOT/"adam_pro"/"adam_pro.xml").read_bytes()).hexdigest(),
                  "native_force_probes": probe_native_forces(args.results), "trajectory": verify_trajectory(args.results)}
        if args.fine_results:
            result["fine_trajectory"] = verify_trajectory(args.fine_results)
            result["step_refinement"] = compare_steps(args.results, args.fine_results)
        if args.source_workbook:
            source_hash = hashlib.sha256(args.source_workbook.read_bytes()).hexdigest()
            if source_hash != DEFAULT_MOTOR.source_sha256:
                raise AssertionError("Original workbook digest changed or the wrong source was supplied")
            result["original_workbook_unchanged"] = True
        result["status"] = "PASS"
    except Exception as error:
        target.write_text(json.dumps({"status": "FAIL", "error_type": type(error).__name__, "error": str(error)}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
