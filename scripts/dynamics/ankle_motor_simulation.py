"""Record a real motor-driven ankle simulation with the shared motor model.

Only mj_resetData initializes pose. Every recorded movement comes from mj_step;
passive ankle qfrc_applied is always zero. Controls are requested output-shaft
Nm, not normalized voltages. T-N clipping occurs inside MuJoCo actuation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import mujoco
import numpy as np
from pndbotics_ankle import CustomAnkleKinematics
from pndbotics_ankle.motor import DEFAULT_MOTOR
from pndbotics_ankle.mujoco_io import build_fixed_ankle_model, camera
from pndbotics_ankle.mujoco_motor import MotorRuntime, explicit_axis_inertia

MOTOR_NAMES = ("ankleMotor_Left_1", "ankleMotor_Left_2")
JOINT_NAMES = MOTOR_NAMES + ("anklePitch_Left", "ankleRoll_Left")


def wrap(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def target_trajectory(t):
    result = np.zeros((len(t), 2))
    chapters = np.zeros(len(t), dtype=np.int8)
    def smooth(x):
        x = np.clip(x, 0, 1)
        return x*x*x * (10 + x*(-15 + 6*x))
    for start, chapter in ((2., 1), (10., 2), (18., 3)):
        selected = (t >= start) & (t < start + 8)
        u = t[selected] - start
        envelope = smooth(u/1.2) * smooth((8-u)/1.2)
        phase = 2*np.pi*u/4
        if chapter == 1:
            result[selected, 1] = envelope*(.32*np.sin(phase) - .05*(1-np.cos(phase)))
        elif chapter == 2:
            result[selected, 0] = envelope*.27*np.sin(phase)
        else:
            result[selected, 0] = envelope*.24*np.sin(phase)
            result[selected, 1] = envelope*(.23*np.sin(phase+np.pi/3)-.03)
        chapters[selected] = chapter
    chapters[t >= 26] = 4
    return result, chapters


def simulate(out: Path, *, model_path: Path | None = None, duration: float = 28., dt: float = .001,
             warmup: float = 1., mode: str = "continuous") -> dict:
    if not np.isfinite([duration, dt, warmup]).all() or duration <= 0 or not 0 < dt <= .002 or warmup < 0:
        raise ValueError("Require positive duration, 0 < dt <= .002 and nonnegative warmup")
    if abs(duration/dt - round(duration/dt)) > 1e-8:
        raise ValueError("duration must be an integer number of simulation steps")
    if round(duration/dt) < 2:
        raise ValueError("At least two steps / three samples are required for trajectory derivatives")
    if abs(warmup/dt - round(warmup/dt)) > 1e-8:
        raise ValueError("warmup must be an integer number of simulation steps")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    spec = DEFAULT_MOTOR
    kin = CustomAnkleKinematics()
    model = build_fixed_ankle_model(model_path, output=out / "derived_left_ankle.xml")
    model.opt.timestep = dt
    data = mujoco.MjData(model)
    dofs = np.array([model.joint(name).dofadr[0] for name in JOINT_NAMES], dtype=int)
    qadr = np.array([model.joint(name).qposadr[0] for name in JOINT_NAMES], dtype=int)
    aids = np.array([model.actuator(name).id for name in MOTOR_NAMES], dtype=int)
    explicit = np.array([explicit_axis_inertia(model, model.joint(name).id) for name in MOTOR_NAMES])
    armature = model.dof_armature[dofs[:2]].copy()
    total_motor_inertia = explicit + armature
    if not np.allclose(total_motor_inertia, spec.output_inertia, rtol=0, atol=1e-12):
        raise AssertionError("Motor inertia budget is not the source profile budget")
    n = int(round(duration/dt)) + 1
    t = np.arange(n)*dt
    targets, chapters = target_trajectory(t)
    target_velocity = np.gradient(targets, dt, axis=0, edge_order=2)
    home = np.array(kin.home_position)
    motor_target = np.array([wrap(-(np.array(kin.IK(*pose))-home)) for pose in targets])
    motor_velocity_target = np.gradient(motor_target, dt, axis=0, edge_order=2)
    motor_acceleration_target = np.gradient(motor_velocity_target, dt, axis=0, edge_order=2)
    limits = model.jnt_range[[model.joint(name).id for name in MOTOR_NAMES]]
    if not (np.all(motor_target > limits[:, 0]+.01) and np.all(motor_target < limits[:, 1]-.01)):
        raise ValueError("Requested trajectory is outside motor position limits")
    KP, KD = 110., 2.4
    arrays = {name: np.empty((n, width)) for name, width in (
        ("qpos", model.nq), ("qvel", model.nv), ("qacc", model.nv), ("control", model.nu),
        ("requested_torque", 2), ("limited_torque", 2), ("motor_torque", 2), ("actuator_force", 2),
        ("torque_lower", 2), ("torque_upper", 2), ("actual_roll_pitch", 2), ("actual_velocity", 2),
        ("tendon_length", 2), ("motor_ik_error", 2), ("velocity_jacobian_error", 2),
        ("mapped_joint_torque", 2), ("motor_armature_torque", 2), ("motor_inertial_torque", 2))}
    arrays["Mref"] = np.empty((n, 2, 2))
    arrays["Mref_armature_only"] = np.empty((n, 2, 2))
    saturated = np.zeros((n, 2), dtype=bool)
    overspeed = np.zeros((n, 2), dtype=bool)
    max_force_error = 0.
    max_bound_violation = 0.
    with MotorRuntime(model, spec=spec, names=MOTOR_NAMES) as runtime:
        mujoco.mj_resetData(model, data)
        runtime.set_mode(data, mode)
        for _ in range(int(round(warmup/dt))):
            data.ctrl[aids] = -KP*data.qpos[qadr[:2]] - KD*data.qvel[dofs[:2]]
            data.qfrc_applied[:] = 0
            runtime.step(data)
        origin = data.time
        for k in range(n):
            # Submit the raw request. The callback, not a caller-side clip, is
            # responsible for measured-speed/direction-dependent saturation.
            request = (KP*(motor_target[k]-data.qpos[qadr[:2]]) +
                       KD*(motor_velocity_target[k]-data.qvel[dofs[:2]]) +
                       total_motor_inertia*motor_acceleration_target[k])
            data.ctrl[aids] = request
            data.qfrc_applied[:] = 0
            telemetry = runtime.telemetry(data)  # same-state mj_forward
            arrays["qpos"][k], arrays["qvel"][k], arrays["qacc"][k] = data.qpos, data.qvel, data.qacc
            arrays["control"][k] = data.ctrl
            arrays["requested_torque"][k] = telemetry["requested"]
            arrays["limited_torque"][k] = telemetry["limited"]
            arrays["motor_torque"][k] = telemetry["applied"]
            arrays["actuator_force"][k] = telemetry["actuator_force"]
            arrays["torque_lower"][k], arrays["torque_upper"][k] = telemetry["lower"], telemetry["upper"]
            saturated[k], overspeed[k] = telemetry["saturated"], telemetry["overspeed"]
            max_force_error = max(max_force_error, float(np.max(np.abs(telemetry["applied"]-telemetry["limited"]))))
            max_bound_violation = max(max_bound_violation, float(np.max(np.maximum(
                telemetry["applied"]-telemetry["upper"], telemetry["lower"]-telemetry["applied"]))))
            actual = data.qpos[[qadr[3], qadr[2]]]
            actual_velocity = data.qvel[[dofs[3], dofs[2]]]
            arrays["actual_roll_pitch"][k], arrays["actual_velocity"][k] = actual, actual_velocity
            jacobian = kin.calc_Jvel(*actual)
            arrays["Mref"][k] = jacobian.T @ np.diag(total_motor_inertia) @ jacobian
            arrays["Mref_armature_only"][k] = jacobian.T @ np.diag(armature) @ jacobian
            predicted_motor = wrap(-(np.array(kin.IK(*actual))-home))
            arrays["motor_ik_error"][k] = wrap(data.qpos[qadr[:2]]-predicted_motor)
            arrays["velocity_jacobian_error"][k] = data.qvel[dofs[:2]] + jacobian @ actual_velocity
            arrays["mapped_joint_torque"][k] = (-jacobian).T @ telemetry["applied"]
            arrays["motor_armature_torque"][k] = armature*data.qacc[dofs[:2]]
            arrays["motor_inertial_torque"][k] = total_motor_inertia*data.qacc[dofs[:2]]
            arrays["tendon_length"][k] = data.ten_length
            if k < n-1:
                runtime.step(data)
            if k % max(1, n//4) == 0:
                print(f"Integrated sample {k}/{n-1}: {t[k]:.3f}s", flush=True)
        if abs((data.time-origin)-duration) > 1e-7:
            raise AssertionError("Wrong integrated duration")
        mass = np.empty((model.nv, model.nv))
        mujoco.mj_fullM(model, mass, data.qM)
        compiled_diagonal = np.diag(mass)[dofs[:2]].copy()
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise AssertionError(f"Nonfinite {name}")
    if max_force_error > 1e-10 or max_bound_violation > 1e-10:
        raise AssertionError("Actual MuJoCo generalized torque violates the motor model")
    np.testing.assert_allclose(compiled_diagonal, total_motor_inertia, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(model.dof_armature[dofs[2:]], 0)
    np.testing.assert_array_equal(model.dof_damping[dofs[:2]], 0)
    error = arrays["actual_roll_pitch"]-targets
    active = (t >= 2) & (t <= 26)
    if not np.any(active):
        active = np.ones(n, dtype=bool)
    rmse = np.rad2deg(np.sqrt(np.mean(error[active]**2, axis=0)))
    tendon_deviation_mm = np.max(np.abs(arrays["tendon_length"]-np.array([.280, .208])), axis=0)*1000
    # Do not persist a PASS summary before the last acceptance gates.
    if np.max(rmse) > 1 or np.max(tendon_deviation_mm) > .5:
        raise AssertionError("Tracking or tendon consistency check failed")
    summary = {
        "status": "PASS", "motor_model": spec.motor_model,
        "motor_profile": spec.as_dict(),
        "derived_model_sha256": hashlib.sha256((out/"derived_left_ankle.xml").read_bytes()).hexdigest(),
        "profile_source": "user-approved workbook row 10 and embedded reference diagram",
        "mode": mode, "mujoco_version": mujoco.__version__, "numpy_version": np.__version__,
        "python_version": platform.python_version(), "dt_s": dt, "duration_s": duration, "warmup_s": warmup, "samples": n,
        "drive": "two motor actuators through native actuation callback; passive qfrc_applied=0",
        "proximal_shin": "fixed", "gravity_m_s2": model.opt.gravity.tolist(), "contact": False,
        "inertia_policy": spec.inertia_policy, "motor_total_output_inertia_kg_m2": total_motor_inertia.tolist(),
        "motor_explicit_axis_inertia_kg_m2": explicit.tolist(), "motor_additional_armature_kg_m2": armature.tolist(),
        "motor_compiled_mass_diagonal_kg_m2": compiled_diagonal.tolist(),
        "ankle_extra_armature_pitch_roll_kg_m2": model.dof_armature[dofs[2:]].tolist(),
        "torque_model": {"rated_current_torque_cap_Nm": spec.rated_torque, "drive_current_limit_Nm": spec.drive_torque,
                         "braking_estimate_Nm": spec.brake_torque, "corner_rpm": spec.corner_rpm, "drive_zero_rpm": spec.no_load_rpm},
        "max_applied_vs_limited_torque_error_Nm": max_force_error, "max_TN_violation_Nm": max(0., max_bound_violation),
        "tracking_rmse_active_roll_pitch_deg": rmse.tolist(),
        "tracking_max_abs_roll_pitch_deg": np.rad2deg(np.max(np.abs(error), axis=0)).tolist(),
        "motor_torque_abs_max_Nm": np.max(np.abs(arrays["motor_torque"]), axis=0).tolist(),
        "motor_speed_abs_max_rad_s": np.max(np.abs(arrays["qvel"][:, dofs[:2]]), axis=0).tolist(),
        "motor_power_abs_max_W": np.max(np.abs(arrays["motor_torque"]*arrays["qvel"][:, dofs[:2]]), axis=0).tolist(),
        "saturated_samples": np.sum(saturated, axis=0).tolist(), "overspeed_samples": np.sum(overspeed, axis=0).tolist(),
        "tendon_length_max_abs_deviation_mm": tendon_deviation_mm.tolist(),
        "elapsed_wall_seconds": time.perf_counter()-started,
        "limitations": ["Reference output-shaft torque envelope, not an electrical or thermal motor model.",
                        "44 Nm braking is the workbook's +10% estimate, not a measured regenerative power limit.",
                        "100 rpm was reported at rated current; the full-current knee is an idealized reference diagram assumption.",
                        "16 Nm is an inferred rated-current operating cap, not a validated continuous thermal curve.",
                        "Missing gearbox inertia and Fs/Fd are not identified as physical zero.",
                        "Explicit body allocation is a bookkeeping policy; spatial inertia/CAD and full robot dynamics are not calibrated.",
                        "mapped_joint_torque is the ideal Jacobian mapping, not measured net torque delivered to the foot.",
                        "No contact or external load; original compliant tendons remain."]}
    np.savez_compressed(out/"telemetry.npz", time=t, **arrays, saturated=saturated, overspeed=overspeed,
                        target_roll_pitch=targets, target_velocity=target_velocity, motor_target=motor_target,
                        motor_target_velocity=motor_velocity_target, chapter=chapters, motor_dofs=dofs[:2], motor_qadr=qadr[:2],
                        model_armature=armature, explicit_axis_inertia=explicit, total_motor_inertia=total_motor_inertia,
                        mode_code=np.array(0 if mode == "continuous" else 1, dtype=np.int8))
    columns = ["time_s", "roll_target_rad", "pitch_target_rad", "roll_actual_rad", "pitch_actual_rad"]
    for i in (1, 2):
        columns.extend([f"motor{i}_{name}" for name in ("speed_rad_s", "request_Nm", "lower_Nm", "upper_Nm", "applied_Nm", "power_W", "saturated")])
    with (out/"telemetry.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for k in range(n):
            row = [t[k], *targets[k], *arrays["actual_roll_pitch"][k]]
            for i in range(2):
                speed = arrays["qvel"][k, dofs[i]]
                torque = arrays["motor_torque"][k, i]
                row.extend([speed, arrays["requested_torque"][k, i], arrays["torque_lower"][k, i],
                            arrays["torque_upper"][k, i], torque, torque*speed, int(saturated[k, i])])
            writer.writerow(row)
    (out/"simulation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"artifacts"/"motor_validation")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--duration", type=float, default=28.)
    parser.add_argument("--dt", type=float, default=.001)
    parser.add_argument("--warmup", type=float, default=1.)
    parser.add_argument("--mode", choices=("continuous", "current_limit"), default="continuous")
    args = parser.parse_args()
    simulate(args.out, model_path=args.model, duration=args.duration, dt=args.dt, warmup=args.warmup, mode=args.mode)


if __name__ == "__main__":
    main()
