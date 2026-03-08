"""
Ankle Joint Armature Calculator
==============================

Computes the reflected motor inertia (armature) for each joint axis
using the Jacobian matrix of the parallel ankle mechanism.

Formula:
    I_arm = Σ_i (k_vel_i)² × I_motor
    k_vel = J_vel @ e_axis

Usage:
    python scripts/kinematics/ankle_armature.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# Use non-interactive backend for headless environments
import matplotlib
matplotlib.use('Agg')

import numpy as np
import math
import matplotlib.pyplot as plt

from pndbotics_ankle import CustomAnkleKinematics


# ══════════════════════════════════════════════════════════════════════
# User Parameters
# ══════════════════════════════════════════════════════════════════════
I_MOTOR = 0.0549        # kg·m²  Motor rotor inertia (single motor)

ROLL_RANGE_RAD  = (-0.35, 0.35)
PITCH_RANGE_RAD = (-1.0, 0.35)
N_SCAN = 120                  # Number of scan points


# ══════════════════════════════════════════════════════════════════════
# Armature computation.
# ══════════════════════════════════════════════════════════════════════
def compute_armature(J_vel: np.ndarray, I_motor: float, axis: int):
    """
    Compute reflected armature for one joint axis.

    Parameters
    ----------
    J_vel : np.ndarray
        2x2 velocity Jacobian (`motor_speed = J_vel @ joint_speed`).
    I_motor : float
        Rotor inertia of a single motor [kg*m^2].
    axis : int
        0 for roll (tx), 1 for pitch (ty).

    Returns
    -------
    tuple[np.ndarray, float]
        Per-motor speed ratios (length 2) and reflected armature [kg*m^2].
    """
    e = np.zeros(2)
    e[axis] = 1.0
    k_vel = J_vel @ e                           # Speed transmission ratios.
    I_arm = float(np.sum(k_vel**2) * I_motor)   # Σ k_i² × I_motor
    return k_vel, I_arm


def scan_armature_2d(robot, I_motor, roll_range_rad, pitch_range_rad, n):
    """
    Run a 2D armature scan over all (roll, pitch) pairs.

    Returns grid coordinates and armature maps.
    Invalid points (IK failure or singular Jacobian) are set to `NaN`.
    """
    roll_arr  = np.linspace(*roll_range_rad,  n)
    pitch_arr = np.linspace(*pitch_range_rad, n)

    # Grid layout: rows are roll, columns are pitch (all in radians).
    PITCH_grid, ROLL_grid = np.meshgrid(pitch_arr, roll_arr)
    ARM_roll  = np.full_like(PITCH_grid, np.nan)
    ARM_pitch = np.full_like(PITCH_grid, np.nan)

    for i, tx in enumerate(roll_arr):
        for j, ty in enumerate(pitch_arr):
            q1, q2 = robot.IK(tx, ty)
            if q1 is None:
                continue
            Jv = robot.calc_Jvel(tx, ty, q1, q2)
            if Jv is None:
                continue
            _, Ir = compute_armature(Jv, I_motor, 0)
            _, Ip = compute_armature(Jv, I_motor, 1)
            ARM_roll[i, j]  = Ir
            ARM_pitch[i, j] = Ip

    return ROLL_grid, PITCH_grid, ARM_roll, ARM_pitch


# ══════════════════════════════════════════════════════════════════════
# Main routine.
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    robot = CustomAnkleKinematics()

    roll_rad = ROLL_RANGE_RAD
    pitch_rad = PITCH_RANGE_RAD

    # ── Home-pose details ──────────────────────────────────────────────
    q1h, q2h = robot.IK(0.0, 0.0)
    Jv_home = robot.calc_Jvel(0.0, 0.0, q1h, q2h)

    print("=" * 60)
    print(f"  Motor rotor inertia I_motor = {I_MOTOR} kg*m^2")
    print("=" * 60)
    print(f"\nHome J_vel =\n{np.round(Jv_home, 4)}\n")

    home_results = {}
    for axis, name in [(0, "Roll (tx)"), (1, "Pitch (ty)")]:
        k_vel, I_arm = compute_armature(Jv_home, I_MOTOR, axis)
        home_results[name] = I_arm
        print(f"  {name}:")
        print(f"    Motor1: k = {k_vel[0]:+.4f}  contribution = {k_vel[0]**2 * I_MOTOR:.5f} kg*m^2")
        print(f"    Motor2: k = {k_vel[1]:+.4f}  contribution = {k_vel[1]**2 * I_MOTOR:.5f} kg*m^2")
        print(f"    -> Armature @ home = {I_arm:.5f} kg*m^2\n")

    # ── 2D grid scan ───────────────────────────────────────────────────
    print("Running 2D grid scan, please wait...")
    ROLL_G, PITCH_G, ARM_ROLL, ARM_PITCH = scan_armature_2d(
        robot, I_MOTOR, roll_rad, pitch_rad, N_SCAN
    )

    cons_roll  = float(np.nanmax(ARM_ROLL))
    cons_pitch = float(np.nanmax(ARM_PITCH))

    print("=" * 60)
    print("  Conservative estimate (max over full motion range, recommended for simulators)")
    print("=" * 60)
    print(f"  Roll  armature = {cons_roll:.5f} kg*m^2")
    print(f"  Pitch armature = {cons_pitch:.5f} kg*m^2")

    # ── 3D surface plots ───────────────────────────────────────────────
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib.cm as cm

    fig = plt.figure(figsize=(15, 6))
    fig.patch.set_facecolor('white')

    cfg = [
        (fig.add_subplot(121, projection='3d'),
         ARM_ROLL  * 1e3, 'Roll Axis Armature', cm.Blues,  cons_roll  * 1e3),
        (fig.add_subplot(122, projection='3d'),
         ARM_PITCH * 1e3, 'Pitch Axis Armature', cm.Reds,  cons_pitch * 1e3),
    ]

    for ax, Z, title, cmap, cons_val in cfg:
        ax.set_facecolor('white')

        surf = ax.plot_surface(
            PITCH_G, ROLL_G, Z,
            cmap=cmap, alpha=0.90,
            linewidth=0, antialiased=True,
        )

        # Project filled contours onto the bottom plane.
        ax.contourf(
            PITCH_G, ROLL_G, Z,
            zdir='z', offset=np.nanmin(Z) - 1,
            cmap=cmap, alpha=0.35, levels=15,
        )

        # Mark maximum point.
        idx = np.unravel_index(np.nanargmax(Z), Z.shape)
        max_p = PITCH_G[idx]
        max_r = ROLL_G[idx]
        max_z = Z[idx]
        ax.scatter([max_p], [max_r], [max_z],
                   color='black', s=40, zorder=5)
        ax.text(max_p, max_r, max_z + 0.5,
                f' max={max_z:.1f}\n({max_p:.2f},{max_r:.2f})rad', fontsize=7, color='black')

        cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08)
        cbar.set_label('Armature (×10⁻³ kg·m²)', fontsize=8)

        ax.set_xlabel('Pitch (rad)', fontsize=9, labelpad=6)
        ax.set_ylabel('Roll  (rad)', fontsize=9, labelpad=6)
        ax.set_zlabel('Armature\n(×10⁻³ kg·m²)', fontsize=8, labelpad=6)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=-5)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=28, azim=-50)

    fig.suptitle(
        f'Reflected Motor Inertia (Armature) — Parallel Ankle Mechanism  |  '
        f'I_motor = {I_MOTOR} kg·m²\n'
        f'Conservative:  Roll = {cons_roll*1e3:.1f}×10⁻³ kg·m²    '
        f'Pitch = {cons_pitch*1e3:.1f}×10⁻³ kg·m²',
        fontsize=11, fontweight='bold',
    )

    plt.tight_layout()
    out_png = 'ankle_armature_3d.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"\nImage saved to: {out_png}")
    plt.show()
