"""
Pinocchio-based dynamics and visualization demo for the parallel ankle.

Kinematics are provided by `src/pndbotics_ankle.CustomAnkleKinematics`.
Pinocchio and Meshcat are used for rigid-body dynamics terms and rendering.
"""
import sys
from pathlib import Path
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as mg
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pndbotics_ankle import CustomAnkleKinematics

# Resolve project root from this script location
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load model from MJCF.
mjcf_path = PROJECT_ROOT / "adam_pro" / "adam_pro.xml"
model, collision_model, visual_model = pin.buildModelsFromMJCF(str(mjcf_path))
data = model.createData()
ankle_kin = CustomAnkleKinematics()

print(f"Model name: {model.name}")
print(f"DoF count: {model.nq}")
print(f"Joint count: {model.njoints}")

# ─────────────────────────────────────────────
# Joint index helpers.
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

q_neutral = pin.neutral(model)
q1_home_src, q2_home_src = ankle_kin.home_position
q1_home_raw_left = float(q_neutral[ankle_motor_left_1_idx])
q2_home_raw_left = float(q_neutral[ankle_motor_left_2_idx])
q1_home_raw_right = float(q_neutral[ankle_motor_right_1_idx])
q2_home_raw_right = float(q_neutral[ankle_motor_right_2_idx])


def raw_motor_to_src(raw_q: float, raw_home: float, src_home: float) -> float:
    return src_home - (raw_q - raw_home)


def src_motor_to_raw(src_q: float, src_home: float, raw_home: float) -> float:
    return raw_home - (src_q - src_home)

print("\nLeft ankle joint q-index:")
print(f"  ankleMotor_Left_1: {ankle_motor_left_1_idx}")
print(f"  ankleMotor_Left_2: {ankle_motor_left_2_idx}")
print(f"  anklePitch_Left:   {ankle_pitch_left_idx}")
print(f"  ankleRoll_Left:    {ankle_roll_left_idx}")

print("\nRight ankle joint q-index:")
print(f"  ankleMotor_Right_1: {ankle_motor_right_1_idx}")
print(f"  ankleMotor_Right_2: {ankle_motor_right_2_idx}")
print(f"  anklePitch_Right:   {ankle_pitch_right_idx}")
print(f"  ankleRoll_Right:    {ankle_roll_right_idx}")

# ─────────────────────────────────────────────
# Resolve tendon site frames.
# ─────────────────────────────────────────────
site_names = [
    "ankleLeft_B1",    "ankleLeft_B2",
    "ankleToeLeft_C1", "ankleToeLeft_C2",
    "ankleRight_B1",    "ankleRight_B2",
    "ankleToeRight_C1", "ankleToeRight_C2",
]

frame_ids = {}
print("\nResolving tendon site frames:")
for name in site_names:
    fid = model.getFrameId(name)
    if fid < len(model.frames):
        frame_ids[name] = fid
        print(f"  Found: {name} -> frame {fid}")
    else:
        print(f"  Missing: {name}")

def solve_parallel_fk(q_base,
                      motor1_left=None,  motor2_left=None,
                      motor1_right=None, motor2_right=None):
    q_solved = q_base.copy()

    if motor1_left  is not None: q_solved[ankle_motor_left_1_idx]  = motor1_left
    if motor2_left  is not None: q_solved[ankle_motor_left_2_idx]  = motor2_left
    if motor1_right is not None: q_solved[ankle_motor_right_1_idx] = motor1_right
    if motor2_right is not None: q_solved[ankle_motor_right_2_idx] = motor2_right

    q1_left_src = raw_motor_to_src(q_solved[ankle_motor_left_1_idx], q1_home_raw_left, q1_home_src)
    q2_left_src = raw_motor_to_src(q_solved[ankle_motor_left_2_idx], q2_home_raw_left, q2_home_src)
    q1_right_src = raw_motor_to_src(q_solved[ankle_motor_right_1_idx], q1_home_raw_right, q1_home_src)
    q2_right_src = raw_motor_to_src(q_solved[ankle_motor_right_2_idx], q2_home_raw_right, q2_home_src)

    roll_left, pitch_left = ankle_kin.FK(
        q1_left_src,
        q2_left_src,
        initial_guess=(
            q_solved[ankle_roll_left_idx],
            q_solved[ankle_pitch_left_idx],
        ),
    )
    roll_right, pitch_right = ankle_kin.FK(
        q1_right_src,
        q2_right_src,
        initial_guess=(
            q_solved[ankle_roll_right_idx],
            q_solved[ankle_pitch_right_idx],
        ),
    )

    q_solved[ankle_pitch_left_idx] = pitch_left
    q_solved[ankle_roll_left_idx] = roll_left
    q_solved[ankle_pitch_right_idx] = pitch_right
    q_solved[ankle_roll_right_idx] = roll_right

    return q_solved


# ─────────────────────────────────────────────
# Inverse kinematics: map desired pitch/roll to motor angles.
# ─────────────────────────────────────────────
def inverse_kinematics_ankle(desired_pitch, desired_roll, ankle_type='left', q_seed=None):
    if q_seed is None:
        q_seed = pin.neutral(model)
    del q_seed
    q1_src, q2_src = ankle_kin.IK(desired_roll, desired_pitch)
    if ankle_type == 'left':
        return np.array([
            src_motor_to_raw(q1_src, q1_home_src, q1_home_raw_left),
            src_motor_to_raw(q2_src, q2_home_src, q2_home_raw_left),
        ])
    return np.array([
        src_motor_to_raw(q1_src, q1_home_src, q1_home_raw_right),
        src_motor_to_raw(q2_src, q2_home_src, q2_home_raw_right),
    ])


# ─────────────────────────────────────────────
# Dynamics helper functions.
# ─────────────────────────────────────────────

def compute_ankle_jacobian_numerical(q_config, ankle_type='left', delta=1e-7):
    del delta
    if ankle_type == 'left':
        roll = q_config[ankle_roll_left_idx]
        pitch = q_config[ankle_pitch_left_idx]
        q1_src = raw_motor_to_src(q_config[ankle_motor_left_1_idx], q1_home_raw_left, q1_home_src)
        q2_src = raw_motor_to_src(q_config[ankle_motor_left_2_idx], q2_home_raw_left, q2_home_src)
    else:
        roll = q_config[ankle_roll_right_idx]
        pitch = q_config[ankle_pitch_right_idx]
        q1_src = raw_motor_to_src(q_config[ankle_motor_right_1_idx], q1_home_raw_right, q1_home_src)
        q2_src = raw_motor_to_src(q_config[ankle_motor_right_2_idx], q2_home_raw_right, q2_home_src)

    J_vel, _ = ankle_kin.calc_Jacobians(roll, pitch, q1=q1_src, q2=q2_src)
    if J_vel is None:
        raise RuntimeError("Shared kinematics Jacobian unavailable at this pose")
    J_joint = np.linalg.solve(J_vel, np.eye(2))
    return J_joint[[1, 0], :]


def compute_motor_torque_to_ankle_force(q_config, ankle_type='left'):
    J_m2j = compute_ankle_jacobian_numerical(q_config, ankle_type)
    try:
        return np.linalg.inv(J_m2j.T)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(J_m2j.T)


def compute_mass_matrix_ankle(q_config, ankle_type='left'):
    J_m2j = compute_ankle_jacobian_numerical(q_config, ankle_type)
    pin.computeAllTerms(model, data, q_config, np.zeros(model.nv))
    M_full = data.M.copy()

    if ankle_type == 'left':
        pitch_v_idx = model.joints[model.getJointId("anklePitch_Left")].idx_v
        roll_v_idx = model.joints[model.getJointId("ankleRoll_Left")].idx_v
    else:
        pitch_v_idx = model.joints[model.getJointId("anklePitch_Right")].idx_v
        roll_v_idx = model.joints[model.getJointId("ankleRoll_Right")].idx_v

    idx = [pitch_v_idx, roll_v_idx]
    M_joint = M_full[np.ix_(idx, idx)]
    return J_m2j.T @ M_joint @ J_m2j


def compute_coriolis_gravity(q_config, qd_config, ankle_type='left'):
    if ankle_type == 'left':
        pitch_id = model.getJointId("anklePitch_Left")
        roll_id = model.getJointId("ankleRoll_Left")
        motor_id_1 = model.getJointId("ankleMotor_Left_1")
        motor_id_2 = model.getJointId("ankleMotor_Left_2")
    else:
        pitch_id = model.getJointId("anklePitch_Right")
        roll_id = model.getJointId("ankleRoll_Right")
        motor_id_1 = model.getJointId("ankleMotor_Right_1")
        motor_id_2 = model.getJointId("ankleMotor_Right_2")

    qd_local = qd_config.copy()
    motor_v_1_idx = model.joints[motor_id_1].idx_v
    motor_v_2_idx = model.joints[motor_id_2].idx_v
    pitch_v_idx = model.joints[pitch_id].idx_v
    roll_v_idx = model.joints[roll_id].idx_v

    J_m2j = compute_ankle_jacobian_numerical(q_config, ankle_type)
    motor_qd = np.array([qd_local[motor_v_1_idx], qd_local[motor_v_2_idx]])
    joint_qd = J_m2j @ motor_qd
    qd_local[pitch_v_idx] = joint_qd[0]
    qd_local[roll_v_idx] = joint_qd[1]

    pin.computeAllTerms(model, data, q_config, qd_local)
    tau_joint = np.array([data.nle[pitch_v_idx], data.nle[roll_v_idx]])
    return J_m2j.T @ tau_joint


def compute_inverse_dynamics_ankle(q_config, qd_config, qdd_config, ankle_type='left'):
    M = compute_mass_matrix_ankle(q_config, ankle_type)
    c_g = compute_coriolis_gravity(q_config, qd_config, ankle_type)

    if ankle_type == 'left':
        pitch_v_idx = model.joints[model.getJointId("anklePitch_Left")].idx_v
        roll_v_idx = model.joints[model.getJointId("ankleRoll_Left")].idx_v
    else:
        pitch_v_idx = model.joints[model.getJointId("anklePitch_Right")].idx_v
        roll_v_idx = model.joints[model.getJointId("ankleRoll_Right")].idx_v

    joint_qdd = np.array([qdd_config[pitch_v_idx], qdd_config[roll_v_idx]])
    J_m2j = compute_ankle_jacobian_numerical(q_config, ankle_type)
    try:
        motor_qdd = np.linalg.solve(J_m2j, joint_qdd)
    except np.linalg.LinAlgError:
        motor_qdd = np.linalg.lstsq(J_m2j, joint_qdd, rcond=None)[0]

    return M @ motor_qdd + c_g


def print_ankle_dynamics(q_config, qd_config, ankle_type='left'):
    """
    Print ankle dynamics diagnostics for one side.
    """
    print(f"\n{'='*60}")
    print(f"Ankle dynamics info ({ankle_type.upper()}):")
    print(f"{'='*60}")
    
    # Jacobian matrix.
    J = compute_ankle_jacobian_numerical(q_config, ankle_type)
    print("\nJacobian J_m2j (d[pitch,roll]/d[motor1,motor2]):")
    print(f"{J}")
    print(f"Condition number cond(J_m2j) = {np.linalg.cond(J):.4f}")
    
    # Mass matrix.
    M = compute_mass_matrix_ankle(q_config, ankle_type)
    print("\nMotor-space mass matrix M (2x2):")
    print(f"{M}")
    
    # Coriolis + gravity term.
    c_g = compute_coriolis_gravity(q_config, qd_config, ankle_type)
    print("\nCoriolis + gravity compensation torque:")
    if ankle_type == 'left':
        print(f"  Motor_Left_1: {c_g[0]:+.4f} N*m")
        print(f"  Motor_Left_2: {c_g[1]:+.4f} N*m")
    else:
        print(f"  Motor_Right_1: {c_g[0]:+.4f} N*m")
        print(f"  Motor_Right_2: {c_g[1]:+.4f} N*m")


# ─────────────────────────────────────────────
# Tendon visualization.
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
    """Update tendon visuals after forward kinematics has run."""
    for b_name, c_name, node_name, color in tendon_pairs:
        if b_name not in frame_ids or c_name not in frame_ids:
            continue
        p_b = data.oMf[frame_ids[b_name]].translation.copy()
        p_c = data.oMf[frame_ids[c_name]].translation.copy()
        draw_cylinder_between(viz.viewer, node_name, p_b, p_c, color=color)


def display_with_tendons(q_config):
    """Display pose and keep tendon overlays synchronized."""
    pin.forwardKinematics(model, data, q_config)
    pin.updateFramePlacements(model, data)
    viz.display(q_config)
    update_tendons()


# ─────────────────────────────────────────────
# Initialize Meshcat.
# ─────────────────────────────────────────────
viz = MeshcatVisualizer(model, collision_model, visual_model)
try:
    viz.initViewer(open=True)
except ImportError as err:
    print("Error: failed to start Meshcat viewer:", err)
    viz.initViewer(loadModel=False, open=False)

viz.loadViewerModel()
print(f"\nMeshcat visualization started. Open: {viz.viewer.url()}")

# ─────────────────────────────────────────────
# Initial pose: solve the linkage constraints at zero motor angles
# so ankle pitch/roll are consistent with tendon lengths.
# ─────────────────────────────────────────────
q = q_neutral.copy()
q = solve_parallel_fk(q,
                      motor1_left=0., motor2_left=0.,
                      motor1_right=0., motor2_right=0.)
display_with_tendons(q)
time.sleep(1.0)

# ─────────────────────────────────────────────
# Demo: forward kinematics (motor input -> constraint solve -> foot motion).
# ─────────────────────────────────────────────
print("\n=== Forward Kinematics Demo ===")
print("Motor motion -> constraint solve -> foot motion\n")

q_demo = q.copy()
for i, motor_angle in enumerate(np.linspace(-0.3, 0.3, 10)):
    q_test = solve_parallel_fk(q_demo,
                               motor1_left= motor_angle,
                               motor2_left=-motor_angle,
                               motor1_right= motor_angle,
                               motor2_right=-motor_angle)
    q_demo = q_test.copy()
    display_with_tendons(q_test)

    print(f"Step {i+1:2d}:  Motor1={motor_angle:+.3f}  Motor2={-motor_angle:+.3f}"
          f"  ->  LeftPitch={q_test[ankle_pitch_left_idx]:+.4f} LeftRoll={q_test[ankle_roll_left_idx]:+.4f}"
          f"  RightPitch={q_test[ankle_pitch_right_idx]:+.4f} RightRoll={q_test[ankle_roll_right_idx]:+.4f}")
    time.sleep(0.5)

# ─────────────────────────────────────────────
# Demo: inverse kinematics (target pose -> motor angles -> constraint check).
# ─────────────────────────────────────────────
print("\n=== Inverse Kinematics Demo ===")
print("Target ankle pose -> motor angles -> forward constraint check\n")

desired_poses = [
    (0.0, 0.0),    # Neutral.
    (0.2, 0.0),    # Forward lean.
    (-0.2, 0.0),   # Backward lean.
    (0.0, 0.15),   # Left lean.
    (0.0, -0.15),  # Right lean.
]

q_ik_seed = q.copy()
for i, (pitch, roll) in enumerate(desired_poses):
    motor_angles = inverse_kinematics_ankle(pitch, roll, ankle_type='left', q_seed=q_ik_seed)
    q_ik = solve_parallel_fk(q_ik_seed,
                             motor1_left=motor_angles[0],
                             motor2_left=motor_angles[1],
                             motor1_right=motor_angles[0],
                             motor2_right=motor_angles[1])
    q_ik_seed = q_ik.copy()
    display_with_tendons(q_ik)

    print(f"Step {i+1}: Target Pitch={pitch:+.3f} Roll={roll:+.3f}  "
          f"-> Motor1={motor_angles[0]:+.3f} Motor2={motor_angles[1]:+.3f}  "
          f"-> Left Pitch={q_ik[ankle_pitch_left_idx]:+.4f} Roll={q_ik[ankle_roll_left_idx]:+.4f}"
          f"   Right Pitch={q_ik[ankle_pitch_right_idx]:+.4f} Roll={q_ik[ankle_roll_right_idx]:+.4f}")
    time.sleep(1.0)

# ─────────────────────────────────────────────
# Demo: dynamics calculations.
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("=== Dynamics Demo ===")
print("="*70)

# Initialize velocity and acceleration vectors.
qd = np.zeros(model.nv)
qdd = np.zeros(model.nv)

# Set sample motor velocities.
motor_v_1_idx = model.joints[ankle_motor_left_1_id].idx_v
motor_v_2_idx = model.joints[ankle_motor_left_2_id].idx_v
qd[motor_v_1_idx] = 0.5   # rad/s
qd[motor_v_2_idx] = 0.3   # rad/s

# Set sample motor accelerations.
qdd[motor_v_1_idx] = 0.2  # rad/s^2
qdd[motor_v_2_idx] = 0.1  # rad/s^2

# Compute dynamics at the current configuration.
print_ankle_dynamics(q, qd, ankle_type='left')
print_ankle_dynamics(q, qd, ankle_type='right')

# ─────────────────────────────────────────────
# Demo: dynamics variation across poses.
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("=== Dynamics Across Motor Angles ===")
print("="*70)

motor_angles_range = np.linspace(-0.3, 0.3, 3)
print(f"\nTest motor angles: {motor_angles_range}")

for motor_angle in motor_angles_range:
    q_test = solve_parallel_fk(q,
                               motor1_left=motor_angle,
                               motor2_left=-motor_angle,
                               motor1_right=motor_angle,
                               motor2_right=-motor_angle)
    
    # Jacobian condition number (dexterity proxy).
    J = compute_ankle_jacobian_numerical(q_test, ankle_type='left')
    cond_num = np.linalg.cond(J)
    
    # Mass-matrix eigenvalues (acceleration response proxy).
    M = compute_mass_matrix_ankle(q_test, ankle_type='left')
    eigs = np.linalg.eigvals(M)
    
    print(f"\nMotor angles: Motor1={motor_angle:+.3f}, Motor2={-motor_angle:+.3f}")
    print(f"  Left - Jacobian condition number: {cond_num:.2f}")
    print(f"  Left - Mass-matrix eigenvalues: {eigs}")
    print(f"  Left - Ankle pose: pitch={q_test[ankle_pitch_left_idx]:+.4f}, roll={q_test[ankle_roll_left_idx]:+.4f}")

# ─────────────────────────────────────────────
# Demo: required motor torque for target accelerations.
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("=== Inverse Dynamics Demo (Target Acceleration -> Motor Torque) ===")
print("="*70)

desired_accelerations = [
    (0.2, 0.0, "Fast pitch acceleration"),
    (0.0, 0.2, "Fast roll acceleration"),
    (0.1, 0.1, "Coupled acceleration"),
]

for pitch_accel, roll_accel, description in desired_accelerations:
    # Set target pitch and roll accelerations.
    qdd_test = np.zeros(model.nv)
    pitch_v_idx = model.joints[model.getJointId("anklePitch_Left")].idx_v
    roll_v_idx = model.joints[model.getJointId("ankleRoll_Left")].idx_v
    qdd_test[pitch_v_idx] = pitch_accel
    qdd_test[roll_v_idx] = roll_accel
    
    # Compute the required motor torques.
    tau_motor = compute_inverse_dynamics_ankle(q, qd, qdd_test, ankle_type='left')
    
    print(f"\n{description}:")
    print(f"  Target acceleration: pitch={pitch_accel:+.3f}, roll={roll_accel:+.3f} rad/s^2")
    print(f"  Required motor torque: Motor_1={tau_motor[0]:+.4f}, Motor_2={tau_motor[1]:+.4f} N*m")

# ─────────────────────────────────────────────
# Keep visualization alive briefly, then exit for scripted runs.
# ─────────────────────────────────────────────
print("\n=== Keeping visualization for 3 seconds, then exit ===")
time.sleep(3.0)
print("Program exited.")
