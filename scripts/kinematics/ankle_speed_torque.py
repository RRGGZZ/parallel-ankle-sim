"""Directional output-shaft T-N and joint envelopes from the canonical profile.

The source workbook reference diagram distinguishes driving and braking. Rated
current is a cap intersecting that diagram, not a second measured thermal curve.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from pndbotics_ankle import CustomAnkleKinematics
from pndbotics_ankle.motor import DEFAULT_MOTOR

# Compatibility aliases; no independent numerical parameters live here.
TAU_PEAK = DEFAULT_MOTOR.drive_torque
TAU_CONT = DEFAULT_MOTOR.rated_torque
RPM_CORNER = DEFAULT_MOTOR.corner_rpm
RPM_MAX = DEFAULT_MOTOR.no_load_rpm
W_CORNER = DEFAULT_MOTOR.corner_speed
W_MAX = DEFAULT_MOTOR.no_load_speed


def rpm2rads(rpm: float) -> float:
    return rpm*math.tau/60


def _mode_from_flat(tau_flat):
    if tau_flat is None or math.isclose(tau_flat, TAU_PEAK):
        return "current_limit"
    if math.isclose(tau_flat, TAU_CONT):
        return "continuous"
    raise ValueError("tau_flat must be a limit derived from the selected motor profile")


def motor_torque_limit(omega: float, tau_flat: float | None = None) -> float:
    """Legacy helper: non-negative *driving* bound, not a symmetric brake cap."""
    return DEFAULT_MOTOR.drive_limit(omega, _mode_from_flat(tau_flat))


def joint_torque_bounds(J_vel, J_tau, axis: int, joint_speed: float, *, mode="continuous", spec=DEFAULT_MOTOR):
    """Intersect both signed motor inequalities for a pure joint-axis torque.

    For each motor, speed=J_vel[i,axis]*joint_speed and
    torque=J_tau[i,axis]*joint_torque. The signs decide driving vs braking;
    an absolute-speed-only available-torque formula loses this distinction.
    Analytical crank and XML coordinates reverse BOTH speed and torque, so
    this centrally symmetric four-quadrant envelope is invariant to that map.
    """
    velocity = np.asarray(J_vel, dtype=float)
    torque = np.asarray(J_tau, dtype=float)
    if axis not in (0, 1) or velocity.shape != (2, 2) or torque.shape != (2, 2):
        raise ValueError("Expect two 2x2 Jacobians and axis 0 or 1")
    if not np.isfinite(velocity).all() or not np.isfinite(torque).all() or not math.isfinite(joint_speed):
        raise ValueError("Jacobians and speed must be finite")
    lower, upper = -math.inf, math.inf
    for i in range(2):
        gain = torque[i, axis]
        lo, hi = spec.torque_bounds(float(velocity[i, axis]*joint_speed), mode)
        if abs(gain) < 1e-12:
            continue
        limits = sorted((lo/gain, hi/gain))
        lower, upper = max(lower, limits[0]), min(upper, limits[1])
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError("No finite joint-torque interval")
    return lower, upper


def joint_speed_torque_curve(J_vel, J_tau, axis: int, tau_flat=None, n_pts=300, *, mode=None, spec=DEFAULT_MOTOR):
    """Positive joint-speed/driving-torque curve; keep the former call shape."""
    if not isinstance(n_pts, int) or n_pts < 2:
        raise ValueError("n_pts must be an integer >= 2")
    selected = _mode_from_flat(tau_flat) if mode is None else mode
    velocity_gains = np.asarray(J_vel, dtype=float)[:, axis]
    candidates = [spec.no_load_speed/abs(gain) for gain in velocity_gains if abs(gain) > 1e-12]
    if not candidates:
        raise ValueError("No motor-speed transmission along the axis")
    speed = np.linspace(0, min(candidates), n_pts)
    torque = np.array([joint_torque_bounds(J_vel, J_tau, axis, float(w), mode=selected, spec=spec)[1] for w in speed])
    return speed, torque


def generate_plots(out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    spec = DEFAULT_MOTOR
    rpm = np.linspace(-190, 190, 761)
    omega = rpm*math.tau/60
    fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
    colors = {"current_limit": "#de7c29", "continuous": "#2375b5"}
    for mode in ("current_limit", "continuous"):
        bounds = np.array([spec.torque_bounds(float(w), mode) for w in omega])
        label = "Current-limit reference (44 Nm braking estimate)" if mode == "current_limit" else "Rated-current operating cap (16 Nm)"
        ax.fill_between(rpm, bounds[:, 0], bounds[:, 1], color=colors[mode], alpha=.13)
        ax.plot(rpm, bounds[:, 0], color=colors[mode], lw=1.8, label=label)
        ax.plot(rpm, bounds[:, 1], color=colors[mode], lw=1.8)
    ax.axhline(0, color="#333333", lw=.7)
    ax.axvline(0, color="#333333", lw=.7)
    for value in (-spec.corner_rpm, spec.corner_rpm):
        ax.axvline(value, color="gray", ls="--", lw=.7)
    for value in (-spec.no_load_rpm, spec.no_load_rpm):
        ax.axvline(value, color="gray", ls=":", lw=.9)
    ax.set(xlabel="Actual gearbox-output speed (rpm)", ylabel="Allowed output-shaft torque (N m)",
           title=f"{spec.motor_model}: directional T-N reference envelope")
    ax.grid(alpha=.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(out/"motor_tn_envelope.png", dpi=170)
    plt.close(fig)
    kin = CustomAnkleKinematics()
    velocity, torque = kin.calc_Jacobians(0, 0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for axis, ax in enumerate(axes):
        speed, _ = joint_speed_torque_curve(velocity, torque, axis, mode="current_limit")
        for mode in ("current_limit", "continuous"):
            bounds = np.array([joint_torque_bounds(velocity, torque, axis, float(w), mode=mode) for w in speed])
            ax.fill_between(speed, bounds[:, 0], bounds[:, 1], color=colors[mode], alpha=.12)
            ax.plot(speed, bounds[:, 1], color=colors[mode], label=mode)
            ax.plot(speed, bounds[:, 0], color=colors[mode], ls="--")
        ax.set(title=("Roll" if axis == 0 else "Pitch")+" at home pose", xlabel="Joint speed (rad/s)", ylabel="Joint torque (N m)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Both motor constraints intersected; dashed = joint braking torque")
    fig.savefig(out/"ankle_joint_tn_envelope.png", dpi=170)
    plt.close(fig)
    print(f"Motor: {spec.motor_model}; {spec.rated_torque:g} Nm rated-current cap; {spec.drive_torque:g}/{spec.brake_torque:g} Nm drive/brake reference")
    for axis, name in enumerate(("Roll", "Pitch")):
        print(name, "home standstill current-limit bounds:", joint_torque_bounds(velocity, torque, axis, 0, mode="current_limit"))
    return out/"motor_tn_envelope.png", out/"ankle_joint_tn_envelope.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"docs")
    for path in generate_plots(parser.parse_args().out):
        print("Saved", path)


if __name__ == "__main__":
    main()
