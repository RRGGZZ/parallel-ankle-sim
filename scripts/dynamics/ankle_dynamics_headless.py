"""
脚踝并联机构动力学演示（轻量版，不需要Meshcat可视化）
用于快速测试动力学计算功能。

运动学部分统一使用 src/pndbotics_ankle.CustomAnkleKinematics，
Pinocchio 仅用于刚体动力学量计算。
"""
import sys
from pathlib import Path
import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pndbotics_ankle import CustomAnkleKinematics

# Resolve project root from this script location
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 从 MJCF 加载模型
mjcf_path = PROJECT_ROOT / "adam_pro" / "adam_pro.xml"
model, collision_model, visual_model = pin.buildModelsFromMJCF(str(mjcf_path))
data = model.createData()
ankle_kin = CustomAnkleKinematics()

print(f"模型名称: {model.name}")
print(f"自由度数量: {model.nq}")
print(f"关节数量: {model.njoints}\n")

# ─────────────────────────────────────────────
# 关节索引
# ─────────────────────────────────────────────
def get_joint_q_index(joint_id):
    return model.joints[joint_id].idx_q

ankle_motor_left_1_id  = model.getJointId("ankleMotor_Left_1")
ankle_motor_left_2_id  = model.getJointId("ankleMotor_Left_2")
ankle_pitch_left_id    = model.getJointId("anklePitch_Left")
ankle_roll_left_id     = model.getJointId("ankleRoll_Left")

ankle_motor_left_1_idx  = get_joint_q_index(ankle_motor_left_1_id)
ankle_motor_left_2_idx  = get_joint_q_index(ankle_motor_left_2_id)
ankle_pitch_left_idx    = get_joint_q_index(ankle_pitch_left_id)
ankle_roll_left_idx     = get_joint_q_index(ankle_roll_left_id)

q_neutral = pin.neutral(model)
q1_home_src, q2_home_src = ankle_kin.home_position
q1_home_raw = float(q_neutral[ankle_motor_left_1_idx])
q2_home_raw = float(q_neutral[ankle_motor_left_2_idx])


def raw_motor_to_src(raw_q: float, raw_home: float, src_home: float) -> float:
    return src_home - (raw_q - raw_home)


def src_motor_to_raw(src_q: float, src_home: float, raw_home: float) -> float:
    return raw_home - (src_q - src_home)

def solve_parallel_fk(q_base, motor1_left=None, motor2_left=None):
    q_solved = q_base.copy()
    
    if motor1_left is not None: q_solved[ankle_motor_left_1_idx] = motor1_left
    if motor2_left is not None: q_solved[ankle_motor_left_2_idx] = motor2_left

    q1_src = raw_motor_to_src(q_solved[ankle_motor_left_1_idx], q1_home_raw, q1_home_src)
    q2_src = raw_motor_to_src(q_solved[ankle_motor_left_2_idx], q2_home_raw, q2_home_src)

    roll_left, pitch_left = ankle_kin.FK(
        q1_src,
        q2_src,
        initial_guess=(
            q_solved[ankle_roll_left_idx],
            q_solved[ankle_pitch_left_idx],
        ),
    )
    q_solved[ankle_pitch_left_idx] = pitch_left
    q_solved[ankle_roll_left_idx] = roll_left

    return q_solved

# ─────────────────────────────────────────────
# 动力学计算函数
# ─────────────────────────────────────────────

def compute_ankle_jacobian_numerical(q_config, delta=1e-7):
    del delta
    roll = q_config[ankle_roll_left_idx]
    pitch = q_config[ankle_pitch_left_idx]
    q1_src = raw_motor_to_src(q_config[ankle_motor_left_1_idx], q1_home_raw, q1_home_src)
    q2_src = raw_motor_to_src(q_config[ankle_motor_left_2_idx], q2_home_raw, q2_home_src)
    J_vel, _ = ankle_kin.calc_Jacobians(roll, pitch, q1=q1_src, q2=q2_src)
    if J_vel is None:
        raise RuntimeError("Shared kinematics Jacobian unavailable at this pose")
    J_joint = np.linalg.solve(J_vel, np.eye(2))
    return J_joint[[1, 0], :]

def compute_mass_matrix_ankle(q_config):
    J_m2j = compute_ankle_jacobian_numerical(q_config)

    pin.computeAllTerms(model, data, q_config, np.zeros(model.nv))
    M_full = data.M.copy()

    pitch_v_idx = model.joints[model.getJointId("anklePitch_Left")].idx_v
    roll_v_idx = model.joints[model.getJointId("ankleRoll_Left")].idx_v
    idx = [pitch_v_idx, roll_v_idx]
    M_joint = M_full[np.ix_(idx, idx)]

    M_motor = J_m2j.T @ M_joint @ J_m2j
    return M_motor

def compute_coriolis_gravity(q_config, qd_config):
    qd_local = qd_config.copy()
    motor_v_1_idx = model.joints[model.getJointId("ankleMotor_Left_1")].idx_v
    motor_v_2_idx = model.joints[model.getJointId("ankleMotor_Left_2")].idx_v
    pitch_v_idx = model.joints[model.getJointId("anklePitch_Left")].idx_v
    roll_v_idx = model.joints[model.getJointId("ankleRoll_Left")].idx_v

    J_m2j = compute_ankle_jacobian_numerical(q_config)
    motor_qd = np.array([qd_local[motor_v_1_idx], qd_local[motor_v_2_idx]])
    joint_qd = J_m2j @ motor_qd
    qd_local[pitch_v_idx] = joint_qd[0]
    qd_local[roll_v_idx] = joint_qd[1]

    pin.computeAllTerms(model, data, q_config, qd_local)
    nle = data.nle.copy()
    tau_joint = np.array([nle[pitch_v_idx], nle[roll_v_idx]])

    J_m2j = compute_ankle_jacobian_numerical(q_config)
    tau_motor = J_m2j.T @ tau_joint
    return tau_motor

def compute_inverse_dynamics_ankle(q_config, qd_config, qdd_config):
    """计算脚踝的逆动力学"""
    M = compute_mass_matrix_ankle(q_config)
    c_g = compute_coriolis_gravity(q_config, qd_config)
    pitch_v_idx = model.joints[model.getJointId("anklePitch_Left")].idx_v
    roll_v_idx = model.joints[model.getJointId("ankleRoll_Left")].idx_v
    joint_qdd = np.array([qdd_config[pitch_v_idx], qdd_config[roll_v_idx]])

    J_m2j = compute_ankle_jacobian_numerical(q_config)
    try:
        motor_qdd = np.linalg.solve(J_m2j, joint_qdd)
    except np.linalg.LinAlgError:
        motor_qdd = np.linalg.lstsq(J_m2j, joint_qdd, rcond=None)[0]
    tau = M @ motor_qdd + c_g
    return tau

# ─────────────────────────────────────────────
# 演示
# ─────────────────────────────────────────────

# 初始化配置
q = q_neutral.copy()
q = solve_parallel_fk(q, motor1_left=0., motor2_left=0.)

# 初始化速度和加速度（nv可能与nq不同）
qd = np.zeros(model.nv)
qdd = np.zeros(model.nv)

motor_v_1_idx = model.joints[model.getJointId("ankleMotor_Left_1")].idx_v
motor_v_2_idx = model.joints[model.getJointId("ankleMotor_Left_2")].idx_v

qd[motor_v_1_idx] = 0.5
qd[motor_v_2_idx] = 0.3
qdd[motor_v_1_idx] = 0.2
qdd[motor_v_2_idx] = 0.1

print("\n" + "="*70)
print("=== 动力学计算演示 ===")
print("="*70)

J = compute_ankle_jacobian_numerical(q)
print("\n雅可比矩阵 J_m2j (d[pitch,roll]/d[motor1,motor2]):")
print(f"{J}")
print(f"条件数: {np.linalg.cond(J):.4f}")

# 计算质量矩阵
M = compute_mass_matrix_ankle(q)
print("\n电机空间质量矩阵 M (2x2):")
print(f"{M}")

# 计算科氏力+重力
c_g = compute_coriolis_gravity(q, qd)
print(f"\n科氏力+重力补偿扭矩:")
print(f"  Motor_Left_1: {c_g[0]:+.4f} N·m")
print(f"  Motor_Left_2: {c_g[1]:+.4f} N·m")

# ─────────────────────────────────────────────
# 不同电机角度下的动力学
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("=== 不同电机角度下的动力学特性 ===")
print("="*70)

motor_angles_range = np.linspace(-0.3, 0.3, 5)

print(f"\n{'电机角度':^20} | {'雅可比条件数':^20} | {'质量矩阵特征值':^30}")
print("-"*72)

for motor_angle in motor_angles_range:
    q_test = solve_parallel_fk(q, motor1_left=motor_angle, motor2_left=-motor_angle)
    J = compute_ankle_jacobian_numerical(q_test)
    cond_num = np.linalg.cond(J)
    M = compute_mass_matrix_ankle(q_test)
    eigs = np.linalg.eigvals(M)
    
    print(f"{motor_angle:+.3f} / {-motor_angle:+.3f}            | "
          f"{cond_num:18.2f} | {eigs}")

# ─────────────────────────────────────────────
# 逆动力学演示
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("=== 逆动力学演示 (期望加速度 → 电机扭矩) ===")
print("="*70)

desired_accelerations = [
    (0.2, 0.0, "快速pitch加速"),
    (0.0, 0.2, "快速roll加速"),
    (0.1, 0.1, "耦合运动加速"),
    (0.5, 0.5, "高速耦合运动"),
]

print(f"\n{'运动描述':^25} | {'期望加速度':^30} | {'所需电机扭矩':^30}")
print("-"*88)

for pitch_accel, roll_accel, description in desired_accelerations:
    qdd_test = np.zeros(model.nv)
    pitch_v_idx = model.joints[model.getJointId("anklePitch_Left")].idx_v
    roll_v_idx = model.joints[model.getJointId("ankleRoll_Left")].idx_v
    qdd_test[pitch_v_idx] = pitch_accel
    qdd_test[roll_v_idx] = roll_accel
    
    tau_motor = compute_inverse_dynamics_ankle(q, qd, qdd_test)
    
    print(f"{description:^25} | "
          f"pitch={pitch_accel:+.2f}, roll={roll_accel:+.2f}  | "
          f"τ1={tau_motor[0]:+.4f}, τ2={tau_motor[1]:+.4f} N·m")

print("\n✓ 动力学计算演示完成！")
