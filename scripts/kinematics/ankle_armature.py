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
#  Armature 计算
# ══════════════════════════════════════════════════════════════════════
def compute_armature(J_vel: np.ndarray, I_motor: float, axis: int):
    """
    计算给定关节轴的等效 armature。

    参数
    ----
    J_vel   : 2×2 速度雅可比（电机角速度 = J_vel @ 关节角速度）
    I_motor : 单个电机转子惯量 [kg·m²]
    axis    : 0 = Roll (tx), 1 = Pitch (ty)

    返回
    ----
    k_vel   : 各电机的速度传动比（长度 2 的数组）
    I_arm   : 等效反射惯量（armature）[kg·m²]
    """
    e = np.zeros(2)
    e[axis] = 1.0
    k_vel = J_vel @ e                           # 速度传动比
    I_arm = float(np.sum(k_vel**2) * I_motor)   # Σ k_i² × I_motor
    return k_vel, I_arm


def scan_armature_2d(robot, I_motor, roll_range_rad, pitch_range_rad, n):
    """
    2D 网格扫描：遍历所有 (roll, pitch) 组合，
    返回网格坐标和对应的 armature 矩阵。
    无效点（IK失败/奇异）填 NaN。
    """
    roll_arr  = np.linspace(*roll_range_rad,  n)
    pitch_arr = np.linspace(*pitch_range_rad, n)

    # 网格：行 = roll，列 = pitch（坐标直接用弧度）
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
#  主程序
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    robot = CustomAnkleKinematics()

    roll_rad = ROLL_RANGE_RAD
    pitch_rad = PITCH_RANGE_RAD

    # ── 零位详细分析 ──────────────────────────────────────────────────
    q1h, q2h = robot.IK(0.0, 0.0)
    Jv_home = robot.calc_Jvel(0.0, 0.0, q1h, q2h)

    print("=" * 60)
    print(f"  电机转子惯量 I_motor = {I_MOTOR} kg·m²")
    print("=" * 60)
    print(f"\n零位 J_vel =\n{np.round(Jv_home, 4)}\n")

    home_results = {}
    for axis, name in [(0, "Roll (tx)"), (1, "Pitch (ty)")]:
        k_vel, I_arm = compute_armature(Jv_home, I_MOTOR, axis)
        home_results[name] = I_arm
        print(f"  {name}:")
        print(f"    电机1: k = {k_vel[0]:+.4f}  贡献 = {k_vel[0]**2 * I_MOTOR:.5f} kg·m²")
        print(f"    电机2: k = {k_vel[1]:+.4f}  贡献 = {k_vel[1]**2 * I_MOTOR:.5f} kg·m²")
        print(f"    ➜ Armature @ home = {I_arm:.5f} kg·m²\n")

    # ── 2D 网格扫描 ──────────────────────────────────────────────────
    print("正在进行 2D 网格扫描，请稍候...")
    ROLL_G, PITCH_G, ARM_ROLL, ARM_PITCH = scan_armature_2d(
        robot, I_MOTOR, roll_rad, pitch_rad, N_SCAN
    )

    cons_roll  = float(np.nanmax(ARM_ROLL))
    cons_pitch = float(np.nanmax(ARM_PITCH))

    print("=" * 60)
    print("  保守估计（全运动范围最大值，建议填入仿真器）")
    print("=" * 60)
    print(f"  Roll  armature = {cons_roll:.5f} kg·m²")
    print(f"  Pitch armature = {cons_pitch:.5f} kg·m²")

    # ── 3D 曲面图 ────────────────────────────────────────────────────
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

        # 等高线投影到底部
        ax.contourf(
            PITCH_G, ROLL_G, Z,
            zdir='z', offset=np.nanmin(Z) - 1,
            cmap=cmap, alpha=0.35, levels=15,
        )

        # 标注最大值点
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
    print(f"\n图像已保存至: {out_png}")
    plt.show()
