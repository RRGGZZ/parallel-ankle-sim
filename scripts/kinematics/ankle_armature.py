"""
Ankle Joint Armature Calculator (Parallel Mechanism)
=====================================================
通过雅可比矩阵将电机转子惯量反射到关节空间，
计算并联踝关节机构各轴的等效 armature。

公式：
    I_arm = Σ_i (k_vel_i)² × I_motor
    k_vel = J_vel @ e_axis   （每个电机对该关节轴的速度传动比）
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════
#  用户参数
# ══════════════════════════════════════════════════════════════════════
I_MOTOR = 0.0549        # kg·m²  电机转子惯量（单个电机）

rad2deg = 180.0 / np.pi
ROLL_RANGE_DEG  = (-0.35 * rad2deg, 0.35 * rad2deg)   # Roll  机械限位 ±0.35 rad (±20°)
PITCH_RANGE_DEG = (-1.0  * rad2deg, 0.35 * rad2deg)   # Pitch 机械限位 -1~+0.35 rad (-57.3°~+20°)
N_SCAN = 120                  # 扫描点数


# ══════════════════════════════════════════════════════════════════════
#  并联踝关节运动学
# ══════════════════════════════════════════════════════════════════════
class CustomAnkleKinematics:
    def __init__(self):
        # 机构几何参数（单位 mm，内部计算时统一量纲不影响比值）
        self.r1, self.l1, self.h1, self.d1 = 40.00, 280.00, 15.00, 37.10
        self.r2, self.l2, self.h2, self.d2 = 40.00, 208.00, 33.00, 35.00
        self.h3, self.d3 = 31.50, 47.00
        self.th1, self.th2, self.th3, self.th4 = 0.51, 1.83, 0.57, 1.92

        # 连接点 C1、C2 在本体坐标系中的位置
        self.C1_w = np.array([self.d1,  (self.d3 + self.d2) / 4, -self.h1])
        self.C2_w = np.array([self.d1, -(self.d2 + self.d3) / 4, -self.h1])

        # 电机连接点 A1、A2
        alpha1 = np.pi - self.th2
        B1_zero = self.C1_w + np.array([-self.l1 * np.cos(alpha1), 0,
                                          self.l1 * np.sin(alpha1)])
        self.A1 = B1_zero + np.array([self.r1 * np.cos(self.th1), 0,
                                       self.r1 * np.sin(self.th1)])

        alpha2 = np.pi - self.th4
        B2_zero = self.C2_w + np.array([-self.l2 * np.cos(alpha2), 0,
                                          self.l2 * np.sin(alpha2)])
        self.A2 = B2_zero + np.array([self.r2 * np.cos(self.th3), 0,
                                       self.r2 * np.sin(self.th3)])

    # ── 旋转矩阵（Roll=tx, Pitch=ty）─────────────────────────────────
    def calc_R(self, tx, ty):
        cx, sx = np.cos(tx), np.sin(tx)
        cy, sy = np.cos(ty), np.sin(ty)
        return np.array([[ cy,  sx * sy,  cx * sy],
                          [  0,       cx,      -sx],
                          [-sy,  cy * sx,  cx * cy]])

    # ── 逆运动学：给定关节角 → 电机角 ───────────────────────────────
    def IK(self, theta_x, theta_y):
        R = self.calc_R(theta_x, theta_y)
        C1_O = R @ self.C1_w
        C2_O = R @ self.C2_w
        q_out = []
        for A, C, r, l in [(self.A1, C1_O, self.r1, self.l1),
                            (self.A2, C2_O, self.r2, self.l2)]:
            U = A[0] - C[0]
            V = A[1] - C[1]
            W = A[2] - C[2]
            K1 = 2 * r * W
            K2 = 2 * r * U
            K3 = l**2 - r**2 - U**2 - V**2 - W**2
            sv = K1**2 + K2**2 - K3**2
            if sv < 0:
                return None, None
            q = math.atan2(K3, -math.sqrt(sv)) - math.atan2(K2, K1)
            q_out.append(q)
        return q_out[0], q_out[1]

    # ── 速度雅可比（关节速度 → 电机角速度）──────────────────────────
    def calc_Jvel(self, tx, ty, q1=None, q2=None):
        """
        返回 J_vel (2×2)，满足：
            [q1_dot, q2_dot]^T = J_vel @ [tx_dot, ty_dot]^T
        """
        if q1 is None or q2 is None:
            q1, q2 = self.IK(tx, ty)
        if q1 is None:
            return None

        cx, sx = np.cos(tx), np.sin(tx)
        cy, sy = np.cos(ty), np.sin(ty)
        R = self.calc_R(tx, ty)

        C1_O = R @ self.C1_w
        C2_O = R @ self.C2_w

        B1 = self.A1 + np.array([self.r1 * np.cos(q1), 0, self.r1 * np.sin(q1)])
        B2 = self.A2 + np.array([self.r2 * np.cos(q2), 0, self.r2 * np.sin(q2)])
        L1 = B1 - C1_O
        L2 = B2 - C2_O

        v_B1 = np.array([-self.r1 * np.sin(q1), 0, self.r1 * np.cos(q1)])
        v_B2 = np.array([-self.r2 * np.sin(q2), 0, self.r2 * np.cos(q2)])

        a11 = np.dot(L1, v_B1)
        a22 = np.dot(L2, v_B2)
        if abs(a11) < 1e-9 or abs(a22) < 1e-9:
            return None   # 接近奇异点

        dR_dtx = np.array([[0,  cx * sy, -sx * sy],
                            [0,      -sx,      -cx],
                            [0,  cx * cy, -sx * cy]])
        dR_dty = np.array([[-sy,  sx * cy,  cx * cy],
                            [  0,        0,        0],
                            [-cy, -sx * sy, -cx * sy]])

        dC1_dtx = dR_dtx @ self.C1_w
        dC1_dty = dR_dty @ self.C1_w
        dC2_dtx = dR_dtx @ self.C2_w
        dC2_dty = dR_dty @ self.C2_w

        B_mat = np.array([
            [np.dot(L1, dC1_dtx), np.dot(L1, dC1_dty)],
            [np.dot(L2, dC2_dtx), np.dot(L2, dC2_dty)],
        ])

        J_vel = np.array([
            [B_mat[0, 0] / a11, B_mat[0, 1] / a11],
            [B_mat[1, 0] / a22, B_mat[1, 1] / a22],
        ])
        return J_vel


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

    roll_rad  = tuple(np.radians(ROLL_RANGE_DEG))
    pitch_rad = tuple(np.radians(PITCH_RANGE_DEG))

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