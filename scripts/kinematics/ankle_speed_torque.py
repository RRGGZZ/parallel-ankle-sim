import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ── 电机参数（从图中读取）────────────────────────────────────────────
TAU_PEAK       = 46.0   # Nm  峰值扭矩
TAU_CONT       = 18.0   # Nm  持续扭矩
RPM_CORNER     = 55.0   # RPM 拐点转速
RPM_MAX        = 80.0   # RPM 最大转速

# RPM → rad/s
def rpm2rads(rpm):
    return rpm * 2 * math.pi / 60

W_CORNER = rpm2rads(RPM_CORNER)   # ≈ 5.76 rad/s
W_MAX    = rpm2rads(RPM_MAX)      # ≈ 8.38 rad/s

def motor_torque_limit(omega, tau_flat):
    """给定电机转速 |omega| (rad/s)，返回该工况下的最大扭矩（线性 speed-torque 模型）。"""
    omega = abs(omega)
    if omega <= W_CORNER:
        return tau_flat
    elif omega <= W_MAX:
        return tau_flat * (W_MAX - omega) / (W_MAX - W_CORNER)
    else:
        return 0.0


# ── 踝关节并联机构运动学（与原脚本一致）────────────────────────────
class CustomAnkleKinematics:
    def __init__(self):
        self.r1, self.l1, self.h1, self.d1 = 40.00, 280.00, 15.00, 37.10
        self.r2, self.l2, self.h2, self.d2 = 40.00, 208.00, 33.00, 35.00
        self.h3, self.d3 = 31.50, 47.00
        self.th1, self.th2, self.th3, self.th4 = 0.51, 1.83, 0.57, 1.92

        self.C1_w = np.array([self.d1,  (self.d3+self.d2)/4, -self.h1])
        self.C2_w = np.array([self.d1, -(self.d2+self.d3)/4, -self.h1])

        alpha1 = np.pi - self.th2
        B1_zero = self.C1_w + np.array([-self.l1*np.cos(alpha1), 0, self.l1*np.sin(alpha1)])
        self.A1 = B1_zero + np.array([self.r1*np.cos(self.th1), 0, self.r1*np.sin(self.th1)])

        alpha2 = np.pi - self.th4
        B2_zero = self.C2_w + np.array([-self.l2*np.cos(alpha2), 0, self.l2*np.sin(alpha2)])
        self.A2 = B2_zero + np.array([self.r2*np.cos(self.th3), 0, self.r2*np.sin(self.th3)])

    def calc_R(self, tx, ty):
        cx, sx = np.cos(tx), np.sin(tx)
        cy, sy = np.cos(ty), np.sin(ty)
        return np.array([[cy, sx*sy, cx*sy],
                         [0,  cx,   -sx   ],
                         [-sy, cy*sx, cx*cy]])

    def IK(self, theta_x, theta_y):
        R = self.calc_R(theta_x, theta_y)
        C1_O = R @ self.C1_w
        C2_O = R @ self.C2_w
        q_out = []
        for A, C, r, l in [(self.A1, C1_O, self.r1, self.l1),
                            (self.A2, C2_O, self.r2, self.l2)]:
            U, V, W = A[0]-C[0], A[1]-C[1], A[2]-C[2]
            K1, K2  = 2*r*W, 2*r*U
            K3      = l**2 - r**2 - U**2 - V**2 - W**2
            sv = K1**2 + K2**2 - K3**2
            if sv < 0:
                return None, None
            q = math.atan2(K3, -math.sqrt(sv)) - math.atan2(K2, K1)
            q_out.append(q)
        return q_out[0], q_out[1]

    def calc_Jacobians(self, tx, ty, q1=None, q2=None):
        if q1 is None or q2 is None:
            q1, q2 = self.IK(tx, ty)
        if q1 is None:
            return None, None

        cx, sx = np.cos(tx), np.sin(tx)
        cy, sy = np.cos(ty), np.sin(ty)
        R = self.calc_R(tx, ty)
        C1_O, C2_O = R @ self.C1_w, R @ self.C2_w

        B1 = self.A1 + np.array([self.r1 * np.cos(q1), 0, self.r1 * np.sin(q1)])
        B2 = self.A2 + np.array([self.r2 * np.cos(q2), 0, self.r2 * np.sin(q2)])
        L1, L2 = B1 - C1_O, B2 - C2_O

        v_B1 = np.array([-self.r1 * np.sin(q1), 0, self.r1 * np.cos(q1)])
        v_B2 = np.array([-self.r2 * np.sin(q2), 0, self.r2 * np.cos(q2)])

        a11 = np.dot(L1, v_B1)
        a22 = np.dot(L2, v_B2)
        if abs(a11) < 1e-6 or abs(a22) < 1e-6:
            return None, None

        dR_dtx = np.array([[0,    cx*sy, -sx*sy],
                            [0,   -sx,   -cx   ],
                            [0,    cx*cy,-sx*cy]])
        dR_dty = np.array([[-sy,  sx*cy, cx*cy ],
                            [0,   0,     0     ],
                            [-cy,-sx*sy,-cx*sy ]])

        dC1_dtx, dC1_dty = dR_dtx @ self.C1_w, dR_dty @ self.C1_w
        dC2_dtx, dC2_dty = dR_dtx @ self.C2_w, dR_dty @ self.C2_w

        B_mat = np.array([[np.dot(L1, dC1_dtx), np.dot(L1, dC1_dty)],
                           [np.dot(L2, dC2_dtx), np.dot(L2, dC2_dty)]])

        J_vel = np.array([[B_mat[0,0]/a11, B_mat[0,1]/a11],
                           [B_mat[1,0]/a22, B_mat[1,1]/a22]])
        try:
            J_tau = np.linalg.solve(J_vel.T, np.eye(2))
        except np.linalg.LinAlgError:
            return None, None
        return J_vel, J_tau


# ── 核心：将电机 speed-torque 曲线映射到关节空间 ─────────────────────
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