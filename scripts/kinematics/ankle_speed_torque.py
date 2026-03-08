"""
Motor Speed-Torque Curve Mapping to Joint Space
==============================================

Maps motor speed-torque characteristics to joint space capabilities
for the parallel ankle mechanism.

Usage:
    python scripts/kinematics/ankle_speed_torque.py
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
import matplotlib.patches as mpatches

from pndbotics_ankle import CustomAnkleKinematics


# ── Motor Parameters (from motor datasheet) ────────────────────────────
TAU_PEAK       = 46.0   # Nm  Peak torque
TAU_CONT       = 18.0   # Nm  Continuous torque
RPM_CORNER     = 55.0   # RPM Corner speed
RPM_MAX        = 80.0   # RPM Maximum speed


def rpm2rads(rpm: float) -> float:
    """Convert RPM to rad/s."""
    return rpm * 2 * math.pi / 60


W_CORNER = rpm2rads(RPM_CORNER)   # ≈ 5.76 rad/s
W_MAX    = rpm2rads(RPM_MAX)       # ≈ 8.38 rad/s


def motor_torque_limit(omega: float, tau_flat: float) -> float:
    """
    Compute maximum motor torque at given speed using linear speed-torque model.
    
    Parameters
    ----------
    omega : float
        Motor angular velocity (rad/s)
    tau_flat : float
        Maximum torque at low speed (Nm)
    
    Returns
    -------
    float
        Available torque at given speed (Nm)
    """
    omega = abs(omega)
    if omega <= W_CORNER:
        return tau_flat
    elif omega <= W_MAX:
        return tau_flat * (W_MAX - omega) / (W_MAX - W_CORNER)
    else:
        return 0.0


# ── Map motor speed-torque curve to joint space ───────────────────────
def joint_speed_torque_curve(J_vel, J_tau, axis: int, tau_flat: float, n_pts=300):
    """
    给定雅可比矩阵和电机扭矩限制，计算关节速度-扭矩边界曲线。

    axis: 0 = roll (tx), 1 = pitch (ty)
    返回: (omega_joint_array, tau_joint_array)

    推导：
      对于关节速度方向 e_axis = [1,0] 或 [0,1]，
        q_dot = J_vel @ (omega_j * e_axis)  → 各电机转速与 omega_j 成比例
      对于关节扭矩方向 e_axis，
        tau_motor = J_tau @ (tau_j * e_axis) → 各电机扭矩与 tau_j 成比例

      在给定 omega_j 下，每个电机可提供的最大扭矩 tau_avail_i(omega_j)，
      关节扭矩受限于：
        tau_j = min_i [ tau_avail_i(q_dot_i) / |J_tau[i, axis]| ]
    """
    e = np.zeros(2); e[axis] = 1.0

    # 每个电机速度系数（J_vel第i行·e_axis）
    k_vel = J_vel @ e          # shape (2,)

    # 每个电机扭矩系数（J_tau第i行·e_axis）
    k_tau = J_tau @ e          # shape (2,)

    # 关节最大速度：所有电机都不超过 W_MAX
    # |k_vel_i| * omega_j <= W_MAX → omega_j <= W_MAX / |k_vel_i|
    omega_j_max = min(W_MAX / abs(k) for k in k_vel if abs(k) > 1e-9)

    omega_arr = np.linspace(0, omega_j_max, n_pts)
    tau_arr   = np.zeros(n_pts)

    for idx, omega_j in enumerate(omega_arr):
        tau_j_limit = np.inf
        for i in range(2):
            q_dot_i = k_vel[i] * omega_j
            tau_avail_i = motor_torque_limit(q_dot_i, tau_flat)
            if abs(k_tau[i]) > 1e-9:
                tau_j_limit = min(tau_j_limit, tau_avail_i / abs(k_tau[i]))
        tau_arr[idx] = tau_j_limit if np.isfinite(tau_j_limit) else 0.0

    return omega_arr, tau_arr


# ── 主程序 ──────────────────────────────────────────────────────────
robot = CustomAnkleKinematics()
q1_home, q2_home = robot.IK(0.0, 0.0)
J_vel, J_tau = robot.calc_Jacobians(0.0, 0.0, q1=q1_home, q2=q2_home)

print(f"Home J_vel:\n{J_vel}")
print(f"Home J_tau:\n{J_tau}")

# 计算四条曲线（roll/pitch × peak/continuous）
w_roll_peak,  tau_roll_peak  = joint_speed_torque_curve(J_vel, J_tau, axis=0, tau_flat=TAU_PEAK)
w_roll_cont,  tau_roll_cont  = joint_speed_torque_curve(J_vel, J_tau, axis=0, tau_flat=TAU_CONT)
w_pitch_peak, tau_pitch_peak = joint_speed_torque_curve(J_vel, J_tau, axis=1, tau_flat=TAU_PEAK)
w_pitch_cont, tau_pitch_cont = joint_speed_torque_curve(J_vel, J_tau, axis=1, tau_flat=TAU_CONT)

# ── 绘图 ────────────────────────────────────────────────────────────
plt.style.use('default')
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor('white')

COLORS = {'peak': '#e74c3c', 'cont': '#922b21'}

titles = ['Ankle Roll Joint', 'Ankle Pitch Joint']
datasets = [
    (w_roll_peak,  tau_roll_peak,  w_roll_cont,  tau_roll_cont),
    (w_pitch_peak, tau_pitch_peak, w_pitch_cont, tau_pitch_cont),
]

for ax, title, (wp, tp, wc, tc) in zip(axes, titles, datasets):
    ax.set_facecolor('white')

    ax.fill_between(wp, tp, alpha=0.30, color=COLORS['peak'], label='Limited Duty Region')
    ax.fill_between(wc, tc, alpha=0.65, color=COLORS['cont'], label='Continuous Duty Region')

    ax.plot(wp, tp, color=COLORS['peak'], lw=2.0, ls='--')
    ax.plot(wc, tc, color=COLORS['cont'],  lw=2.0)

    ax.annotate(f"{tp[0]:.1f} Nm", xy=(0, tp[0]), xytext=(8, tp[0]+0.5),
                color=COLORS['peak'], fontsize=9, va='bottom')
    ax.annotate(f"{tc[0]:.1f} Nm", xy=(0, tc[0]), xytext=(8, tc[0]+0.5),
                color=COLORS['cont'], fontsize=9, va='bottom')
    ax.annotate(f"{wp[-1]:.2f} rad/s", xy=(wp[-1], 0),
                xytext=(wp[-1]-0.5, 1.5),
                color=COLORS['peak'], fontsize=9)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Angular Velocity (rad/s)', fontsize=10)
    ax.set_ylabel('Torque (Nm)', fontsize=10)
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
    ax.grid(True, color='#dddddd', linewidth=0.6)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    ax.legend(fontsize=9, loc='upper right')

fig.suptitle('PND-50-6F5S-30-P → Ankle Joint Speed–Torque Envelope\n'
             '(via Parallel Mechanism Jacobian, Home Position)',
             fontsize=12, fontweight='bold', y=1)

plt.tight_layout()
import os as _os
_save_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'ankle_joint_speed_torque.png')
plt.savefig(_save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print(f"图像已保存至: {_save_path}")

# ── 打印关键参数 ─────────────────────────────────────────────────────
print("\n── 关节能力汇总（零位姿态）──")
for name, wp, tp, wc, tc in [
    ('Roll',  w_roll_peak,  tau_roll_peak,  w_roll_cont,  tau_roll_cont),
    ('Pitch', w_pitch_peak, tau_pitch_peak, w_pitch_cont, tau_pitch_cont),
]:
    print(f"  {name}:")
    print(f"    Peak  stall torque = {tp[0]:.2f} Nm,  max speed = {wp[-1]:.2f} rad/s")
    print(f"    Cont. stall torque = {tc[0]:.2f} Nm,  max speed = {wc[-1]:.2f} rad/s")