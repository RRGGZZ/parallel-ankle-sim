"""
脚踝并联机构：理论 vs MuJoCo 真正独立验证

独立输入：目标脚踝角 (pitch, roll)

理论侧（CustomAnkleKinematics，mm坐标系，解析几何）：
  给定 (pitch,roll) → 解析 IK → 预测电机角
  给定 (pitch,roll) → J_tau × 脚部重力矩 → 预测电机力矩

MuJoCo侧（MJCF几何，m坐标系，完全独立）：
  给定 (pitch,roll) → 写入 qpos → mj_forward
  → 从 MuJoCo 自己的 site_xpos 出发几何求解电机角
  → mj_inverse 求电机力矩

两边各用各的几何，误差 = 建模参数/坐标系差异
"""
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
from scipy.optimize import fsolve, brentq
import time

os.chdir('/home/r/Downloads/KirinArm-Design')

# ═══════════════════════════════════════════════════════
# 1. 理论模型（解析侧，单位 mm）
# ═══════════════════════════════════════════════════════
class CustomAnkleKinematics:
    def __init__(self):
        self.r1, self.l1, self.h1, self.d1 = 40.00, 280.00, 15.00, 37.10
        self.r2, self.l2, self.h2, self.d2 = 40.00, 208.00, 33.00, 35.00
        self.h3, self.d3 = 31.50, 47.00
        self.th1, self.th2, self.th3, self.th4 = 0.51, 1.83, 0.57, 1.92

        self.C1_w = np.array([ self.d1,  (self.d3+self.d2)/4, -self.h1])
        self.C2_w = np.array([ self.d1, -(self.d2+self.d3)/4, -self.h1])

        alpha1 = np.pi - self.th2
        B1z = self.C1_w + np.array([-self.l1*np.cos(alpha1), 0, self.l1*np.sin(alpha1)])
        self.A1 = B1z + np.array([self.r1*np.cos(self.th1), 0, self.r1*np.sin(self.th1)])

        alpha2 = np.pi - self.th4
        B2z = self.C2_w + np.array([-self.l2*np.cos(alpha2), 0, self.l2*np.sin(alpha2)])
        self.A2 = B2z + np.array([self.r2*np.cos(self.th3), 0, self.r2*np.sin(self.th3)])

    def calc_R(self, tx, ty):
        cx, sx = np.cos(tx), np.sin(tx)
        cy, sy = np.cos(ty), np.sin(ty)
        return np.array([[cy,  sx*sy,  cx*sy],
                         [0,   cx,    -sx   ],
                         [-sy, cy*sx,  cx*cy]])

    def IK(self, theta_x, theta_y):
        R = self.calc_R(theta_x, theta_y)
        C1_O, C2_O = R @ self.C1_w, R @ self.C2_w
        q_out = []
        for A, C, r, l in [(self.A1,C1_O,self.r1,self.l1),
                            (self.A2,C2_O,self.r2,self.l2)]:
            U,V,W = A[0]-C[0], A[1]-C[1], A[2]-C[2]
            K1,K2 = 2*r*W, 2*r*U
            K3 = l**2 - r**2 - U**2 - V**2 - W**2
            sv = K1**2 + K2**2 - K3**2
            if sv < 0: return None, None
            q_out.append(math.atan2(K3,-math.sqrt(sv)) - math.atan2(K2,K1))
        return q_out[0], q_out[1]

    def calc_Jacobians(self, tx, ty, q1=None, q2=None):
        if q1 is None or q2 is None:
            q1, q2 = self.IK(tx, ty)
        if q1 is None: return None, None
        cx, sx = np.cos(tx), np.sin(tx)
        cy, sy = np.cos(ty), np.sin(ty)
        R = self.calc_R(tx, ty)
        C1_O, C2_O = R @ self.C1_w, R @ self.C2_w
        B1 = self.A1 + np.array([self.r1*np.cos(q1), 0, self.r1*np.sin(q1)])
        B2 = self.A2 + np.array([self.r2*np.cos(q2), 0, self.r2*np.sin(q2)])
        L1, L2 = B1 - C1_O, B2 - C2_O
        v_B1 = np.array([-self.r1*np.sin(q1), 0, self.r1*np.cos(q1)])
        v_B2 = np.array([-self.r2*np.sin(q2), 0, self.r2*np.cos(q2)])
        a11, a22 = np.dot(L1,v_B1), np.dot(L2,v_B2)
        if abs(a11)<1e-6 or abs(a22)<1e-6: return None, None
        dR_dtx = np.array([[0, cx*sy,-sx*sy],[0,-sx,-cx],[0,cx*cy,-sx*cy]])
        dR_dty = np.array([[-sy,sx*cy,cx*cy],[0,0,0],[-cy,-sx*sy,-cx*sy]])
        dC1_dtx,dC1_dty = dR_dtx@self.C1_w, dR_dty@self.C1_w
        dC2_dtx,dC2_dty = dR_dtx@self.C2_w, dR_dty@self.C2_w
        B = np.array([[np.dot(L1,dC1_dtx),np.dot(L1,dC1_dty)],
                      [np.dot(L2,dC2_dtx),np.dot(L2,dC2_dty)]])
        J_vel = np.array([[B[0,0]/a11, B[0,1]/a11],
                          [B[1,0]/a22, B[1,1]/a22]])
        try:
            J_tau = np.linalg.solve(J_vel.T, np.eye(2))
        except np.linalg.LinAlgError:
            return None, None
        return J_vel, J_tau

ankle_kin = CustomAnkleKinematics()
q1_home, q2_home = ankle_kin.IK(0.0, 0.0)
print(f"理论零位: M1={np.degrees(q1_home):.3f}°  M2={np.degrees(q2_home):.3f}°")

FOOT_MASS    = 0.634796
FOOT_COM_LOC = np.array([0.0374997, -4.016e-05, -0.0419674])
G = 9.81

def theory_gravity_torque(roll, pitch):
    R = ankle_kin.calc_R(roll, pitch)
    com_w = R @ FOOT_COM_LOC
    return np.array([-FOOT_MASS*G*com_w[1],   # tau_roll
                      FOOT_MASS*G*com_w[0]])   # tau_pitch


# ═══════════════════════════════════════════════════════
# 2. MuJoCo 加载
# ═══════════════════════════════════════════════════════
mjcf_path = "adam_pro/adam_pro.xml"
model = mujoco.MjModel.from_xml_path(mjcf_path)
data  = mujoco.MjData(model)
print(f"nq={model.nq}  nv={model.nv}")

def qidx(name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0: raise ValueError(f"关节不存在: {name}")
    return model.jnt_qposadr[jid]

def vidx(name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0: raise ValueError(f"关节不存在: {name}")
    return model.jnt_dofadr[jid]

def sidx(name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0: raise ValueError(f"site不存在: {name}")
    return sid

def sensor_adr(name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0: raise ValueError(f"传感器不存在: {name}")
    return model.sensor_adr[sid]

idx_m_l1 = qidx("ankleMotor_Left_1");  idx_m_l2 = qidx("ankleMotor_Left_2")
idx_p_l  = qidx("anklePitch_Left");    idx_r_l  = qidx("ankleRoll_Left")
idx_m_r1 = qidx("ankleMotor_Right_1"); idx_m_r2 = qidx("ankleMotor_Right_2")
idx_p_r  = qidx("anklePitch_Right");   idx_r_r  = qidx("ankleRoll_Right")

dof_m_l1 = vidx("ankleMotor_Left_1");  dof_m_l2 = vidx("ankleMotor_Left_2")

sid_B1_L = sidx("ankleLeft_B1");     sid_B2_L = sidx("ankleLeft_B2")
sid_C1_L = sidx("ankleToeLeft_C1");  sid_C2_L = sidx("ankleToeLeft_C2")
sid_B1_R = sidx("ankleRight_B1");    sid_B2_R = sidx("ankleRight_B2")
sid_C1_R = sidx("ankleToeRight_C1"); sid_C2_R = sidx("ankleToeRight_C2")

# MJCF 中电机曲柄的几何参数（从 MJCF 读取，单位 m）
# ankleMotorLeft_1: capsule fromto="0 0 0  -0.038683 0 0.01018" → B1 在此
# 曲柄旋转轴 Y，A1 是电机关节原点，B1 = A1 + r*[cos(q),0,sin(q)]
# 从 MJCF site pos 可以算出 r 和 phi_0
B1_local = np.array([-0.038683, 0.0, 0.01018])   # 左脚 motor1 的 B 相对电机体坐标
B2_local = np.array([-0.038966, 0.0, 0.009037])  # 左脚 motor2 的 B 相对电机体坐标
r1_mj = np.linalg.norm(B1_local[[0,2]])  # 曲柄半径（XZ平面）
r2_mj = np.linalg.norm(B2_local[[0,2]])
phi1_0 = math.atan2(B1_local[2], B1_local[0])  # B1 零位相位角
phi2_0 = math.atan2(B2_local[2], B2_local[0])

s_pos_m_l1 = sensor_adr("pos_ankleMotor_Left_1")
s_pos_m_l2 = sensor_adr("pos_ankleMotor_Left_2")

def sread(adr): return float(data.sensordata[adr])

# MuJoCo 零位电机角（pitch=roll=0 时从 MJCF 几何反解，用于相对角度计算）
# 在所有函数定义完成后再调用，此处先占位，collect() 调用前完成初始化


# ═══════════════════════════════════════════════════════
# 3. MuJoCo 侧独立求解：
#    给定 pitch/roll → 写入 qpos → mj_forward
#    → 从 site_xpos 读取 B、C 的世界坐标
#    → 用 MuJoCo 自己的几何反解电机角 q（连杆长度约束）
#    → mj_inverse 求力矩
#    ★ 全程不使用理论 IK 结果 ★
# ═══════════════════════════════════════════════════════

# 连杆自然长度（来自 MJCF springlength）
L1_mj = 0.3097
L2_mj = 0.2386

def mujoco_set_ankle(pitch, roll):
    """只设置脚踝角，电机设为0（完全不用理论IK），调用 mj_forward。"""
    mujoco.mj_resetData(model, data)
    data.qpos[idx_p_l] = pitch;  data.qpos[idx_r_l] = roll
    data.qpos[idx_p_r] = pitch;  data.qpos[idx_r_r] = roll
    data.qpos[idx_m_l1] = 0.0;   data.qpos[idx_m_l2] = 0.0
    data.qpos[idx_m_r1] = 0.0;   data.qpos[idx_m_r2] = 0.0
    mujoco.mj_forward(model, data)

def mujoco_get_B_world(q_motor1, q_motor2):
    """
    给定电机角，读取 B1/B2 的世界坐标。
    B 点随电机关节转动，其世界坐标由 mj_forward 计算。
    """
    data.qpos[idx_m_l1] = q_motor1
    data.qpos[idx_m_l2] = q_motor2
    mujoco.mj_forward(model, data)
    return (data.site_xpos[sid_B1_L].copy(),
            data.site_xpos[sid_B2_L].copy())

def mujoco_solve_motor_angles(pitch, roll, q0_guess=(0.0, 0.0)):
    """
    MuJoCo 侧独立求电机角：
    1. 设脚踝角，读取 C1/C2 世界坐标（随 pitch/roll 变化）
    2. 以电机角 (q1,q2) 为未知量，求解 |B1(q1)-C1|=L1, |B2(q2)-C2|=L2
       B 随电机角变化，通过 mj_forward 计算（用的是 MJCF 自己的几何）
    ★ 完全独立，不依赖理论模型 ★
    """
    # 固定脚踝，读 C1/C2
    mujoco.mj_resetData(model, data)
    data.qpos[idx_p_l] = pitch;  data.qpos[idx_r_l] = roll
    data.qpos[idx_p_r] = pitch;  data.qpos[idx_r_r] = roll
    mujoco.mj_forward(model, data)
    C1_w = data.site_xpos[sid_C1_L].copy()
    C2_w = data.site_xpos[sid_C2_L].copy()

    def residual(q12):
        q1, q2 = q12
        data.qpos[idx_m_l1] = q1
        data.qpos[idx_m_l2] = q2
        mujoco.mj_forward(model, data)
        B1 = data.site_xpos[sid_B1_L].copy()
        B2 = data.site_xpos[sid_B2_L].copy()
        e1 = np.linalg.norm(B1 - C1_w) - L1_mj
        e2 = np.linalg.norm(B2 - C2_w) - L2_mj
        return [e1, e2]

    sol, _, ier, msg = fsolve(residual, q0_guess, full_output=True, xtol=1e-9)
    if ier != 1:
        print(f"  [警告] MuJoCo 电机角求解未收敛: {msg.strip()}")
    return sol[0], sol[1]   # q1_mj, q2_mj (rad)

def mujoco_inverse_torque(pitch, roll, q1_mj, q2_mj):
    """
    在 MuJoCo 解出的姿态下，用 mj_inverse 计算电机所需力矩。
    """
    mujoco.mj_resetData(model, data)
    data.qpos[idx_p_l] = pitch;   data.qpos[idx_r_l] = roll
    data.qpos[idx_p_r] = pitch;   data.qpos[idx_r_r] = roll
    data.qpos[idx_m_l1] = q1_mj;  data.qpos[idx_m_l2] = q2_mj
    data.qpos[idx_m_r1] = q1_mj;  data.qpos[idx_m_r2] = q2_mj
    mujoco.mj_forward(model, data)
    data.qacc[:] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_inverse(model, data)
    return float(data.qfrc_inverse[dof_m_l1]), float(data.qfrc_inverse[dof_m_l2])


# ═══════════════════════════════════════════════════════
# 4. 扫描采集
# ═══════════════════════════════════════════════════════

# MuJoCo 零位：pitch=roll=0 时反解出的电机绝对角，用于做相对角度
q1_mj_home, q2_mj_home = mujoco_solve_motor_angles(0.0, 0.0)
print(f'MuJoCo 零位: M1={np.degrees(q1_mj_home):.3f}  M2={np.degrees(q2_mj_home):.3f}')

ROLL_RANGE  = np.linspace(-0.30,  0.30, 40)
PITCH_RANGE = np.linspace(-0.70,  0.30, 40)

def collect(axis):
    rng = ROLL_RANGE if axis=='roll' else PITCH_RANGE
    out = dict(
        val=[],
        th_m1=[], th_m2=[],       # 理论 IK 电机角 (deg)
        mj_m1=[], mj_m2=[],       # MuJoCo 几何求解电机角 (deg)
        th_tau1=[], th_tau2=[],   # 理论力矩 (Nm)
        mj_tau1=[], mj_tau2=[],   # MuJoCo mj_inverse 力矩 (Nm)
        err_m1=[], err_m2=[],
        err_tau1=[], err_tau2=[],
    )
    q_mj_prev = np.array([0.0, 0.0])

    for v in rng:
        roll  = v   if axis=='roll'  else 0.0
        pitch = 0.0 if axis=='roll'  else v

        # ── 理论侧（解析，mm坐标系）────────────────
        q1_th, q2_th = ankle_kin.IK(roll, pitch)
        if q1_th is None: continue
        _, J_tau = ankle_kin.calc_Jacobians(roll, pitch, q1=q1_th, q2=q2_th)
        if J_tau is None: continue

        dq1_th = math.remainder(q1_th - q1_home, 2*math.pi)
        dq2_th = math.remainder(q2_th - q2_home, 2*math.pi)
        tau_th = J_tau @ theory_gravity_torque(roll, pitch)

        # ── MuJoCo侧（MJCF几何，m坐标系，完全独立）─
        # 输入只有 pitch/roll，电机角从 MuJoCo site 坐标反解
        q1_mj, q2_mj = mujoco_solve_motor_angles(
            pitch, roll, q0_guess=q_mj_prev)
        q_mj_prev = np.array([q1_mj, q2_mj])

        tau_mj1, tau_mj2 = mujoco_inverse_torque(pitch, roll, q1_mj, q2_mj)

        out['val'].append(v)
        out['th_m1'].append(np.degrees(dq1_th))
        out['th_m2'].append(np.degrees(dq2_th))
        dq1_mj = math.remainder(q1_mj - q1_mj_home, 2*math.pi)
        dq2_mj = math.remainder(q2_mj - q2_mj_home, 2*math.pi)
        out['mj_m1'].append(np.degrees(dq1_mj))
        out['mj_m2'].append(np.degrees(dq2_mj))
        out['th_tau1'].append(tau_th[0])
        out['th_tau2'].append(tau_th[1])
        out['mj_tau1'].append(tau_mj1)
        out['mj_tau2'].append(tau_mj2)
        out['err_m1'].append(np.degrees(dq1_mj) - np.degrees(dq1_th))
        out['err_m2'].append(np.degrees(dq2_mj) - np.degrees(dq2_th))
        out['err_tau1'].append(tau_mj1 - tau_th[0])
        out['err_tau2'].append(tau_mj2 - tau_th[1])

        print(f"  {axis} {v:+.3f} | "
              f"理论 M1={np.degrees(dq1_th):+6.2f}° | "
              f"MuJoCo M1={np.degrees(dq1_mj):+6.2f}° | "
              f"ΔM1={np.degrees(dq1_mj)-np.degrees(dq1_th):+.3f}°")

    return out

print("\n═══ 采集 Roll 方向（理论 vs MuJoCo 几何，完全独立）═══")
dr = collect('roll')
print(f"\nRoll: M1角误差 max={max(abs(e) for e in dr['err_m1']):.4f}°  "
      f"M1力矩误差 max={max(abs(e) for e in dr['err_tau1']):.4f} Nm")

print("\n═══ 采集 Pitch 方向 ═══")
dp = collect('pitch')
print(f"\nPitch: M1角误差 max={max(abs(e) for e in dp['err_m1']):.4f}°  "
      f"M1力矩误差 max={max(abs(e) for e in dp['err_tau1']):.4f} Nm")


# ═══════════════════════════════════════════════════════
# 5. 简单可视化演示
# ═══════════════════════════════════════════════════════
print("\n启动 MuJoCo 查看器（关闭后绘图）...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth=-140; viewer.cam.elevation=-20
    viewer.cam.distance=2.5; viewer.cam.lookat[:]=[0.,0.,0.5]

    q_prev = np.zeros(2)
    for pitch, roll in [(0,0),(0.2,0),(-0.2,0),(0,0.25),(0,-0.25),(0,0)]:
        if not viewer.is_running(): break
        q1_mj, q2_mj = mujoco_solve_motor_angles(pitch, roll, q0_guess=q_prev)
        q_prev = np.array([q1_mj, q2_mj])
        # 显示最终姿态
        mujoco.mj_resetData(model, data)
        data.qpos[idx_p_l]=pitch; data.qpos[idx_r_l]=roll
        data.qpos[idx_p_r]=pitch; data.qpos[idx_r_r]=roll
        data.qpos[idx_m_l1]=q1_mj; data.qpos[idx_m_l2]=q2_mj
        data.qpos[idx_m_r1]=q1_mj; data.qpos[idx_m_r2]=q2_mj
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.8)

    print("=== 关闭窗口进入绘图 ===")
    while viewer.is_running():
        viewer.sync(); time.sleep(0.05)


# ═══════════════════════════════════════════════════════
# 6. 对比绘图
# ═══════════════════════════════════════════════════════
plt.style.use('bmh')
fig, axs = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle(
    'Ankle Parallel Mechanism — Theory vs MuJoCo (Truly Independent)\n'
    'Theory: Analytical IK (mm coords)  |  MuJoCo: Geometric solve from site_xpos (m coords)\n'
    'Error reveals actual modeling/parameter discrepancy',
    fontsize=11, fontweight='bold')

datasets   = [dr, dp]
x_labels   = ['Ankle Roll (rad)', 'Ankle Pitch (rad)']
col_titles = ['Roll Sweep  (Pitch=0)', 'Pitch Sweep  (Roll=0)']

for col, (d, xl, ct) in enumerate(zip(datasets, x_labels, col_titles)):
    val = d['val']

    ax = axs[0, col]
    ax.plot(val, d['th_m1'], 'C0-',  lw=2.5, label='Theory M1 (analytical IK)')
    ax.plot(val, d['th_m2'], 'C1-',  lw=2.5, label='Theory M2 (analytical IK)')
    ax.plot(val, d['mj_m1'], 'C0--', lw=2,   label='MuJoCo M1 (from site_xpos)')
    ax.plot(val, d['mj_m2'], 'C1--', lw=2,   label='MuJoCo M2 (from site_xpos)')
    ax.set_title(f'{ct}\n① IK: Motor Angle (deg)')
    ax.set_ylabel('Motor Angle (deg)')
    ax.legend(fontsize=8); ax.grid(True,alpha=0.4)
    ax.axhline(0,color='gray',lw=0.7,ls='--')
    ax.axvline(0,color='gray',lw=0.7,ls='--')

    ax = axs[1, col]
    ax.plot(val, d['th_tau1'], 'C0-',  lw=2.5, label='Theory M1 τ (J_tau × gravity)')
    ax.plot(val, d['th_tau2'], 'C1-',  lw=2.5, label='Theory M2 τ')
    ax.plot(val, d['mj_tau1'], 'C0--', lw=2,   label='MuJoCo M1 τ (mj_inverse)')
    ax.plot(val, d['mj_tau2'], 'C1--', lw=2,   label='MuJoCo M2 τ')
    ax.set_title('② Dynamics: Motor Torque (Nm)')
    ax.set_ylabel('Torque (Nm)')
    ax.legend(fontsize=8); ax.grid(True,alpha=0.4)
    ax.axhline(0,color='gray',lw=0.7,ls='--')
    ax.axvline(0,color='gray',lw=0.7,ls='--')

    ax  = axs[2, col]
    ax2 = ax.twinx()
    ax.plot( val, d['err_m1'],   'C0-',  lw=2, label='ΔAngle M1 (deg)')
    ax.plot( val, d['err_m2'],   'C1-',  lw=2, label='ΔAngle M2 (deg)')
    ax2.plot(val, d['err_tau1'], 'C0--', lw=2, label='ΔTorque M1 (Nm)')
    ax2.plot(val, d['err_tau2'], 'C1--', lw=2, label='ΔTorque M2 (Nm)')
    ax.set_title('③ Error: MuJoCo − Theory\n(建模参数/坐标系偏差)')
    ax.set_ylabel('Angle Error (deg)')
    ax2.set_ylabel('Torque Error (Nm)', color='dimgray')
    ax.set_xlabel(xl)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines,[l.get_label() for l in lines], fontsize=7)
    ax.grid(True,alpha=0.3)
    ax.axhline(0,color='gray',lw=0.7,ls='--')
    ax.axvline(0,color='gray',lw=0.7,ls='--')

plt.tight_layout()
plt.savefig('ankle_theory_vs_mujoco.png', dpi=150, bbox_inches='tight')
print("图已保存为 ankle_theory_vs_mujoco.png")
plt.show()
print("程序退出")