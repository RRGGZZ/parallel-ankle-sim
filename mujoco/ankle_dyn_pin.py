"""
使用 Pinocchio 进行脚踝并联机构的正逆运动学、动力学求解
使用 Meshcat 进行可视化（含连杆可视化 + 并联约束求解 + 动力学计算）
"""
import os
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as mg
from scipy.optimize import fsolve
import time

# 设置工作目录
os.chdir('/home/r/Downloads/KirinArm-Design')

# 从 MJCF 加载模型
mjcf_path = "adam_pro/adam_pro.xml"
model, collision_model, visual_model = pin.buildModelsFromMJCF(mjcf_path)
data = model.createData()

print(f"模型名称: {model.name}")
print(f"自由度数量: {model.nq}")
print(f"关节数量: {model.njoints}")

# ─────────────────────────────────────────────
# 关节索引
# ─────────────────────────────────────────────
def get_joint_q_index(joint_id):
    return model.joints[joint_id].idx_q

ankle_motor_left_1_id  = model.getJointId("ankleMotor_Left_1")
ankle_motor_left_2_id  = model.getJointId("ankleMotor_Left_2")
ankle_pitch_left_id    = model.getJointId("anklePitch_Left")
ankle_roll_left_id     = model.getJointId("ankleRoll_Left")

ankle_motor_right_1_id = model.getJointId("ankleMotor_Right_1")
ankle_motor_right_2_id = model.getJointId("ankleMotor_Right_2")
ankle_pitch_right_id   = model.getJointId("anklePitch_Right")
ankle_roll_right_id    = model.getJointId("ankleRoll_Right")

ankle_motor_left_1_idx  = get_joint_q_index(ankle_motor_left_1_id)
ankle_motor_left_2_idx  = get_joint_q_index(ankle_motor_left_2_id)
ankle_pitch_left_idx    = get_joint_q_index(ankle_pitch_left_id)
ankle_roll_left_idx     = get_joint_q_index(ankle_roll_left_id)

ankle_motor_right_1_idx = get_joint_q_index(ankle_motor_right_1_id)
ankle_motor_right_2_idx = get_joint_q_index(ankle_motor_right_2_id)
ankle_pitch_right_idx   = get_joint_q_index(ankle_pitch_right_id)
ankle_roll_right_idx    = get_joint_q_index(ankle_roll_right_id)

print(f"\n左脚踝关节 q-index:")
print(f"  ankleMotor_Left_1: {ankle_motor_left_1_idx}")
print(f"  ankleMotor_Left_2: {ankle_motor_left_2_idx}")
print(f"  anklePitch_Left:   {ankle_pitch_left_idx}")
print(f"  ankleRoll_Left:    {ankle_roll_left_idx}")

print(f"\n右脚踝关节 q-index:")
print(f"  ankleMotor_Right_1: {ankle_motor_right_1_idx}")
print(f"  ankleMotor_Right_2: {ankle_motor_right_2_idx}")
print(f"  anklePitch_Right:   {ankle_pitch_right_idx}")
print(f"  ankleRoll_Right:    {ankle_roll_right_idx}")

# ─────────────────────────────────────────────
# 查找连杆 site frames
# ─────────────────────────────────────────────
site_names = [
    "ankleLeft_B1",    "ankleLeft_B2",
    "ankleToeLeft_C1", "ankleToeLeft_C2",
    "ankleRight_B1",    "ankleRight_B2",
    "ankleToeRight_C1", "ankleToeRight_C2",
]

frame_ids = {}
print("\n查找连杆 site frames:")
for name in site_names:
    fid = model.getFrameId(name)
    if fid < len(model.frames):
        frame_ids[name] = fid
        print(f"  找到: {name} -> frame {fid}")
    else:
        print(f"  未找到: {name}")

# ─────────────────────────────────────────────
# 并联机构约束参数（来自 MJCF springlength）
# 弹簧刚度极大时 springlength 即等效为刚性连杆长度
# ─────────────────────────────────────────────
L_LEFT_1  = 0.3097   # ankle_link_left_1
L_LEFT_2  = 0.2386   # ankle_link_left_2
L_RIGHT_1 = 0.3097   # ankle_link_right_1
L_RIGHT_2 = 0.2386   # ankle_link_right_2

# ─────────────────────────────────────────────
# 并联机构正运动学约束求解
# ─────────────────────────────────────────────

def _constraint_error_left(ankle_angles, q_with_motors):
    """
    左脚约束残差：
    在给定电机角度（已写入 q_with_motors）的情况下，
    返回 [|B1-C1| - L1,  |B2-C2| - L2]，使其趋近于 0。
    """
    pitch, roll = ankle_angles
    q = q_with_motors.copy()
    q[ankle_pitch_left_idx] = pitch
    q[ankle_roll_left_idx]  = roll

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    p_B1 = data.oMf[frame_ids["ankleLeft_B1"]].translation
    p_C1 = data.oMf[frame_ids["ankleToeLeft_C1"]].translation
    p_B2 = data.oMf[frame_ids["ankleLeft_B2"]].translation
    p_C2 = data.oMf[frame_ids["ankleToeLeft_C2"]].translation

    return [np.linalg.norm(p_B1 - p_C1) - L_LEFT_1,
            np.linalg.norm(p_B2 - p_C2) - L_LEFT_2]


def _constraint_error_right(ankle_angles, q_with_motors):
    """右脚约束残差"""
    pitch, roll = ankle_angles
    q = q_with_motors.copy()
    q[ankle_pitch_right_idx] = pitch
    q[ankle_roll_right_idx]  = roll

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    p_B1 = data.oMf[frame_ids["ankleRight_B1"]].translation
    p_C1 = data.oMf[frame_ids["ankleToeRight_C1"]].translation
    p_B2 = data.oMf[frame_ids["ankleRight_B2"]].translation
    p_C2 = data.oMf[frame_ids["ankleToeRight_C2"]].translation

    return [np.linalg.norm(p_B1 - p_C1) - L_RIGHT_1,
            np.linalg.norm(p_B2 - p_C2) - L_RIGHT_2]


# 热启动：上一次求解结果作为下一次初始猜测，加快收敛
_last_left_solution  = np.array([0.0, 0.0])
_last_right_solution = np.array([0.0, 0.0])


def solve_parallel_fk(q_base,
                      motor1_left=None,  motor2_left=None,
                      motor1_right=None, motor2_right=None):
    """
    并联机构正运动学：给定电机角度，数值求解 ankle pitch/roll。

    参数:
        q_base:       基础配置向量（其余关节角度保持不变）
        motor1/2_left/right: 电机角度（为 None 则沿用 q_base 中的值）

    返回:
        q_solved: 包含正确 ankle pitch/roll 的完整配置向量
    """
    global _last_left_solution, _last_right_solution

    q_solved = q_base.copy()

    # 写入电机角度
    if motor1_left  is not None: q_solved[ankle_motor_left_1_idx]  = motor1_left
    if motor2_left  is not None: q_solved[ankle_motor_left_2_idx]  = motor2_left
    if motor1_right is not None: q_solved[ankle_motor_right_1_idx] = motor1_right
    if motor2_right is not None: q_solved[ankle_motor_right_2_idx] = motor2_right

    # ── 求解左脚 ──
    left_sites = ["ankleLeft_B1", "ankleLeft_B2", "ankleToeLeft_C1", "ankleToeLeft_C2"]
    if all(k in frame_ids for k in left_sites):
        sol, info, ier, msg = fsolve(
            _constraint_error_left, _last_left_solution,
            args=(q_solved,), full_output=True, xtol=1e-8
        )
        if ier == 1:
            _last_left_solution = sol
        else:
            print(f"  [警告] 左脚约束求解未收敛: {msg.strip()}")
        q_solved[ankle_pitch_left_idx] = sol[0]
        q_solved[ankle_roll_left_idx]  = sol[1]

    # ── 求解右脚 ──
    right_sites = ["ankleRight_B1", "ankleRight_B2", "ankleToeRight_C1", "ankleToeRight_C2"]
    if all(k in frame_ids for k in right_sites):
        sol, info, ier, msg = fsolve(
            _constraint_error_right, _last_right_solution,
            args=(q_solved,), full_output=True, xtol=1e-8
        )
        if ier == 1:
            _last_right_solution = sol
        else:
            print(f"  [警告] 右脚约束求解未收敛: {msg.strip()}")
        q_solved[ankle_pitch_right_idx] = sol[0]
        q_solved[ankle_roll_right_idx]  = sol[1]

    return q_solved


# ─────────────────────────────────────────────
# 逆运动学：给定 pitch/roll 求电机角度（线性近似）
# ─────────────────────────────────────────────
def inverse_kinematics_ankle(desired_pitch, desired_roll):
    """
    脚踝逆运动学（线性近似）。
    注：系数为简化近似，需根据实际并联机构几何关系标定。
    """
    motor1 = -2.0 * desired_pitch + 2.0 * desired_roll
    motor2 = -2.0 * desired_pitch - 2.0 * desired_roll
    return np.array([motor1, motor2])


# ─────────────────────────────────────────────
# 动力学计算部分
# ─────────────────────────────────────────────

def compute_ankle_jacobian_numerical(q_config, ankle_type='left', delta=1e-7):
    """
    数值计算脚踝并联机构的雅可比矩阵（电机 → 脚踝端点）。
    
    对于左脚：将电机角度的微小变化映射到脚踝末端的位置/速度变化。
    
    参数:
        q_config:     配置向量
        ankle_type:   'left' 或 'right'
        delta:        数值微分步长
    
    返回:
        J: 雅可比矩阵 (6, 2) - 将电机速度映射到脚踝末端速度
           前3行为线速度，后3行为角速度
    """
    if ankle_type == 'left':
        motor_idx = [ankle_motor_left_1_idx, ankle_motor_left_2_idx]
        pitch_idx = ankle_pitch_left_idx
        roll_idx = ankle_roll_left_idx
        frame_b = "ankleLeft_B1"
        frame_c = "ankleToeLeft_C1"
    else:
        motor_idx = [ankle_motor_right_1_idx, ankle_motor_right_2_idx]
        pitch_idx = ankle_pitch_right_idx
        roll_idx = ankle_roll_right_idx
        frame_b = "ankleRight_B1"
        frame_c = "ankleToeRight_C1"
    
    if frame_b not in frame_ids or frame_c not in frame_ids:
        return np.zeros((6, 2))
    
    J = np.zeros((6, 2))
    
    # 数值微分：计算 F, R, T 相对于各电机的微分
    for motor_i, m_idx in enumerate(motor_idx):
        # 正向扰动
        q_plus = q_config.copy()
        q_plus[m_idx] += delta
        q_plus = solve_parallel_fk(q_plus)
        pin.forwardKinematics(model, data, q_plus)
        pin.updateFramePlacements(model, data)
        p_b_plus = data.oMf[frame_ids[frame_b]].translation.copy()
        R_plus = data.oMf[frame_ids[frame_b]].rotation.copy()
        
        # 负向扰动
        q_minus = q_config.copy()
        q_minus[m_idx] -= delta
        q_minus = solve_parallel_fk(q_minus)
        pin.forwardKinematics(model, data, q_minus)
        pin.updateFramePlacements(model, data)
        p_b_minus = data.oMf[frame_ids[frame_b]].translation.copy()
        R_minus = data.oMf[frame_ids[frame_b]].rotation.copy()
        
        # 线速度微分
        dp_dm = (p_b_plus - p_b_minus) / (2.0 * delta)
        J[0:3, motor_i] = dp_dm
        
        # 角速度微分（通过旋转矩阵）
        dR_dm = (R_plus - R_minus) / (2.0 * delta)
        # 将旋转矩阵变化转换为角速度
        # omega = 0.5 * [dR^T * dR]_vee，这里用简化形式
        skew_omega = R_minus.T @ dR_dm / delta
        omega_approx = np.array([skew_omega[2, 1], skew_omega[0, 2], skew_omega[1, 0]])
        J[3:6, motor_i] = omega_approx
    
    return J


def compute_motor_torque_to_ankle_force(q_config, ankle_type='left'):
    """
    计算电机扭矩到脚踝端点力/扭矩的映射矩阵。
    τ_ankle = J^T * τ_motor （静力学对偶）
    
    参数:
        q_config:   配置向量
        ankle_type: 'left' 或 'right'
    
    返回:
        J_T: 转置的雅可比矩阵 (2, 6)
    """
    J = compute_ankle_jacobian_numerical(q_config, ankle_type)
    return J.T


def compute_mass_matrix_ankle(q_config, ankle_type='left'):
    """
    计算脚踝关节的质量矩阵（电机空间）。
    M = (J^{-T} M_end J^{-1}) 其中 M_end 为末端效应器的质量
    
    这里使用简化近似：通过数值积分关节速度到末端速度的映射。
    """
    J = compute_ankle_jacobian_numerical(q_config, ankle_type)
    
    # 假设脚掌质量为 0.5 kg （可根据实际模型修改）
    m_foot = 0.5
    # 假设惯性张量为球形（简化）
    I_foot = m_foot * 0.001 * np.eye(3)
    
    # 质量矩阵在末端坐标系
    M_end = np.eye(6)
    M_end[0:3, 0:3] = m_foot * np.eye(3)
    M_end[3:6, 3:6] = I_foot
    
    # 映射到电机空间（简化：只考虑线性项）
    J_pos = J[0:3, :]
    try:
        M_motor = np.linalg.inv(J_pos).T @ m_foot * np.eye(3) @ np.linalg.inv(J_pos)
    except np.linalg.LinAlgError:
        M_motor = np.eye(2)
    
    return M_motor


def compute_coriolis_gravity(q_config, qd_config, ankle_type='left'):
    """
    计算科氏力和重力补偿力矩（在电机空间）。
    
    参数:
        q_config:     配置向量
        qd_config:    速度向量
        ankle_type:   'left' 或 'right'
    
    返回:
        c_g: 科氏力和重力补偿扭矩 (2,)
    """
    if ankle_type == 'left':
        pitch_id = model.getJointId("anklePitch_Left")
        roll_id = model.getJointId("ankleRoll_Left")
    else:
        pitch_id = model.getJointId("anklePitch_Right")
        roll_id = model.getJointId("ankleRoll_Right")
    
    pitch_q_idx = model.joints[pitch_id].idx_q
    roll_q_idx = model.joints[roll_id].idx_q
    
    # 计算重力补偿通过数值微分重力势能
    tau_gravity = np.zeros(2)
    delta = 1e-7
    
    # pitch 方向的重力
    q_plus = q_config.copy()
    q_plus[pitch_q_idx] += delta
    pin.computeAllTerms(model, data, q_plus, np.zeros(model.nv))
    U_plus = pin.computePotentialEnergy(model, data)
    
    q_minus = q_config.copy()
    q_minus[pitch_q_idx] -= delta
    pin.computeAllTerms(model, data, q_minus, np.zeros(model.nv))
    U_minus = pin.computePotentialEnergy(model, data)
    
    tau_gravity[0] = -(U_plus - U_minus) / (2 * delta)
    
    # roll 方向的重力
    q_plus = q_config.copy()
    q_plus[roll_q_idx] += delta
    pin.computeAllTerms(model, data, q_plus, np.zeros(model.nv))
    U_plus = pin.computePotentialEnergy(model, data)
    
    q_minus = q_config.copy()
    q_minus[roll_q_idx] -= delta
    pin.computeAllTerms(model, data, q_minus, np.zeros(model.nv))
    U_minus = pin.computePotentialEnergy(model, data)
    
    tau_gravity[1] = -(U_plus - U_minus) / (2 * delta)
    
    return tau_gravity


def compute_inverse_dynamics_ankle(q_config, qd_config, qdd_config, ankle_type='left'):
    """
    计算脚踝的逆动力学（电机空间）。
    τ = M * qdd + c_g
    
    参数:
        q_config:     配置向量
        qd_config:    速度向量
        qdd_config:   加速度向量
        ankle_type:   'left' 或 'right'
    
    返回:
        tau: 电机扭矩 (2,)
    """
    M = compute_mass_matrix_ankle(q_config, ankle_type)
    c_g = compute_coriolis_gravity(q_config, qd_config, ankle_type)
    
    if ankle_type == 'left':
        motor_id_1 = model.getJointId("ankleMotor_Left_1")
        motor_id_2 = model.getJointId("ankleMotor_Left_2")
    else:
        motor_id_1 = model.getJointId("ankleMotor_Right_1")
        motor_id_2 = model.getJointId("ankleMotor_Right_2")
    
    motor_v_1_idx = model.joints[motor_id_1].idx_v
    motor_v_2_idx = model.joints[motor_id_2].idx_v
    
    motor_qdd = np.array([qdd_config[motor_v_1_idx],
                          qdd_config[motor_v_2_idx]])
    
    tau = M @ motor_qdd + c_g
    
    return tau


def print_ankle_dynamics(q_config, qd_config, ankle_type='left'):
    """
    打印脚踝的动力学信息。
    """
    print(f"\n{'='*60}")
    print(f"脚踝动力学信息 ({ankle_type.upper()}):")
    print(f"{'='*60}")
    
    # 雅可比矩阵
    J = compute_ankle_jacobian_numerical(q_config, ankle_type)
    print(f"\n雅可比矩阵 J (位置分量，6x2):")
    print(f"{J}")
    print(f"条件数 cond(J_pos) = {np.linalg.cond(J[0:3, :]):.4f}")
    
    # 质量矩阵
    M = compute_mass_matrix_ankle(q_config, ankle_type)
    print(f"\n电机空间质量矩阵 M (2x2):")
    print(f"{M}")
    
    # 科氏力+重力
    c_g = compute_coriolis_gravity(q_config, qd_config, ankle_type)
    print(f"\n科氏力+重力补偿扭矩:")
    if ankle_type == 'left':
        print(f"  Motor_Left_1: {c_g[0]:+.4f} N·m")
        print(f"  Motor_Left_2: {c_g[1]:+.4f} N·m")
    else:
        print(f"  Motor_Right_1: {c_g[0]:+.4f} N·m")
        print(f"  Motor_Right_2: {c_g[1]:+.4f} N·m")


# ─────────────────────────────────────────────
# 连杆（Tendon）可视化
# ─────────────────────────────────────────────
tendon_pairs = [
    ("ankleLeft_B1",   "ankleToeLeft_C1",  "tendon/left_1",  0xFF8800),
    ("ankleLeft_B2",   "ankleToeLeft_C2",  "tendon/left_2",  0xFF8800),
    ("ankleRight_B1",  "ankleToeRight_C1", "tendon/right_1", 0xFF8800),
    ("ankleRight_B2",  "ankleToeRight_C2", "tendon/right_2", 0xFF8800),
]
TENDON_RADIUS = 0.004


def draw_cylinder_between(viewer, node_name, p1, p2,
                           radius=TENDON_RADIUS, color=0xFF8800):
    vec    = p2 - p1
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return
    mid    = (p1 + p2) / 2.0
    y_axis = vec / length
    ref    = np.array([1., 0., 0.]) if abs(y_axis[0]) < 0.9 else np.array([0., 1., 0.])
    x_axis = np.cross(ref, y_axis);  x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    T      = np.eye(4)
    T[:3, 0] = x_axis;  T[:3, 1] = y_axis
    T[:3, 2] = z_axis;  T[:3, 3] = mid
    viewer[node_name].set_object(mg.Cylinder(length, radius),
                                  mg.MeshLambertMaterial(color=color))
    viewer[node_name].set_transform(T)


def update_tendons():
    """更新连杆可视化（须在 FK 已执行后调用）。"""
    for b_name, c_name, node_name, color in tendon_pairs:
        if b_name not in frame_ids or c_name not in frame_ids:
            continue
        p_b = data.oMf[frame_ids[b_name]].translation.copy()
        p_c = data.oMf[frame_ids[c_name]].translation.copy()
        draw_cylinder_between(viz.viewer, node_name, p_b, p_c, color=color)


def display_with_tendons(q_config):
    """显示机器人姿态并同步更新连杆。"""
    pin.forwardKinematics(model, data, q_config)
    pin.updateFramePlacements(model, data)
    viz.display(q_config)
    update_tendons()


# ─────────────────────────────────────────────
# 初始化 Meshcat
# ─────────────────────────────────────────────
viz = MeshcatVisualizer(model, collision_model, visual_model)
try:
    viz.initViewer(open=True)
except ImportError as err:
    print("错误: 无法启动 Meshcat 查看器:", err)
    viz.initViewer(loadModel=False, open=False)

viz.loadViewerModel()
print(f"\nMeshcat 可视化已启动，访问: {viz.viewer.url()}")

# ─────────────────────────────────────────────
# 初始姿态：在零电机角度下先求一次约束，
# 确保初始 ankle pitch/roll 与连杆长度吻合
# ─────────────────────────────────────────────
q = pin.neutral(model)
q = solve_parallel_fk(q,
                      motor1_left=0., motor2_left=0.,
                      motor1_right=0., motor2_right=0.)
display_with_tendons(q)
time.sleep(1.0)

# ─────────────────────────────────────────────
# 演示：正运动学（电机驱动 → 约束求解 → 脚掌跟随）
# ─────────────────────────────────────────────
print("\n=== 正运动学演示 ===")
print("电机转动 → 约束求解 → 脚掌跟随运动\n")

for i, motor_angle in enumerate(np.linspace(-0.5, 0.5, 10)):
    q_test = solve_parallel_fk(q,
                               motor1_left= motor_angle,
                               motor2_left=-motor_angle,
                               motor1_right= motor_angle,
                               motor2_right=-motor_angle)
    display_with_tendons(q_test)

    print(f"步骤 {i+1:2d}:  Motor1={motor_angle:+.3f}  Motor2={-motor_angle:+.3f}"
          f"  =>  左Pitch={q_test[ankle_pitch_left_idx]:+.4f} 左Roll={q_test[ankle_roll_left_idx]:+.4f}"
          f"  右Pitch={q_test[ankle_pitch_right_idx]:+.4f} 右Roll={q_test[ankle_roll_right_idx]:+.4f}")
    time.sleep(0.5)

# ─────────────────────────────────────────────
# 演示：逆运动学（给定期望姿态 → 求电机角 → 约束验证）
# ─────────────────────────────────────────────
print("\n=== 逆运动学演示 ===")
print("给定期望脚踝姿态 → 计算电机角度 → 约束正向验证\n")

desired_poses = [
    ( 0.0,   0.0),   # 中立
    ( 0.2,   0.0),   # 向前倾
    (-0.2,   0.0),   # 向后倾
    ( 0.0,   0.15),  # 向左倾
    ( 0.0,  -0.15),  # 向右倾
]

for i, (pitch, roll) in enumerate(desired_poses):
    motor_angles = inverse_kinematics_ankle(pitch, roll)
    q_ik = solve_parallel_fk(q,
                             motor1_left=motor_angles[0],
                             motor2_left=motor_angles[1],
                             motor1_right=motor_angles[0],
                             motor2_right=motor_angles[1])
    display_with_tendons(q_ik)

    print(f"步骤 {i+1}: 期望 Pitch={pitch:+.3f} Roll={roll:+.3f}  "
          f"=> Motor1={motor_angles[0]:+.3f} Motor2={motor_angles[1]:+.3f}  "
          f"=> 左 Pitch={q_ik[ankle_pitch_left_idx]:+.4f} Roll={q_ik[ankle_roll_left_idx]:+.4f}"
          f"   右 Pitch={q_ik[ankle_pitch_right_idx]:+.4f} Roll={q_ik[ankle_roll_right_idx]:+.4f}")
    time.sleep(1.0)

# ─────────────────────────────────────────────
# 演示：动力学计算
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("=== 动力学计算演示 ===")
print("="*70)

# 初始化速度和加速度向量
qd = np.zeros(model.nv)
qdd = np.zeros(model.nv)

# 为电机添加一些速度（示例）
motor_v_1_idx = model.joints[ankle_motor_left_1_id].idx_v
motor_v_2_idx = model.joints[ankle_motor_left_2_id].idx_v
qd[motor_v_1_idx] = 0.5   # rad/s
qd[motor_v_2_idx] = 0.3   # rad/s

# 为电机添加加速度（示例）
qdd[motor_v_1_idx] = 0.2  # rad/s^2
qdd[motor_v_2_idx] = 0.1  # rad/s^2

# 使用当前配置计算动力学
print_ankle_dynamics(q, qd, ankle_type='left')
print_ankle_dynamics(q, qd, ankle_type='right')

# ─────────────────────────────────────────────
# 演示：不同姿态下的动力学变化
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("=== 不同电机角度下的动力学特性 ===")
print("="*70)

motor_angles_range = np.linspace(-0.3, 0.3, 3)
print(f"\n测试电机角度: {motor_angles_range}")

for motor_angle in motor_angles_range:
    q_test = solve_parallel_fk(q,
                               motor1_left=motor_angle,
                               motor2_left=-motor_angle,
                               motor1_right=motor_angle,
                               motor2_right=-motor_angle)
    
    # 计算雅可比条件数（反映灵活性）
    J = compute_ankle_jacobian_numerical(q_test, ankle_type='left')
    J_pos = J[0:3, :]
    cond_num = np.linalg.cond(J_pos)
    
    # 计算质量矩阵特征值（反映加速性能）
    M = compute_mass_matrix_ankle(q_test, ankle_type='left')
    eigs = np.linalg.eigvals(M)
    
    print(f"\n电机角度: Motor1={motor_angle:+.3f}, Motor2={-motor_angle:+.3f}")
    print(f"  左脚 - 雅可比条件数: {cond_num:.2f}")
    print(f"  左脚 - 质量矩阵特征值: {eigs}")
    print(f"  左脚 - 脚踝姿态: pitch={q_test[ankle_pitch_left_idx]:+.4f}, roll={q_test[ankle_roll_left_idx]:+.4f}")

# ─────────────────────────────────────────────
# 演示：给定期望加速度计算所需电机扭矩
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("=== 逆动力学演示 (期望加速度 → 电机扭矩) ===")
print("="*70)

desired_accelerations = [
    (0.2, 0.0, "快速pitch加速"),
    (0.0, 0.2, "快速roll加速"),
    (0.1, 0.1, "耦合运动加速"),
]

for pitch_accel, roll_accel, description in desired_accelerations:
    # 设置期望的pitch和roll加速度
    qdd_test = np.zeros(model.nv)
    pitch_v_idx = model.joints[model.getJointId("anklePitch_Left")].idx_v
    roll_v_idx = model.joints[model.getJointId("ankleRoll_Left")].idx_v
    qdd_test[pitch_v_idx] = pitch_accel
    qdd_test[roll_v_idx] = roll_accel
    
    # 计算所需的电机扭矩
    tau_motor = compute_inverse_dynamics_ankle(q, qd, qdd_test, ankle_type='left')
    
    print(f"\n{description}:")
    print(f"  期望加速度: pitch={pitch_accel:+.3f}, roll={roll_accel:+.3f} rad/s^2")
    print(f"  所需电机扭矩: Motor_1={tau_motor[0]:+.4f}, Motor_2={tau_motor[1]:+.4f} N·m")

# ─────────────────────────────────────────────
# 保持可视化运行
# ─────────────────────────────────────────────
print("\n=== 可视化运行中，按 Ctrl+C 退出 ===")
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n程序退出")