"""
Ankle Dynamics Computation using CustomAnkleKinematics
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from pndbotics_ankle import CustomAnkleKinematics


def compute_jacobian(ankle, roll, pitch):
    """Compute velocity and torque Jacobians.

    Note: the kinematics API uses the order (tx, ty) = (roll, pitch).
    This function accepts (roll, pitch) to match ankle.calc_Jacobians.
    """
    J_vel, J_tau = ankle.calc_Jacobians(roll, pitch)
    return J_vel, J_tau


def compute_mass_matrix(J_vel, m_foot=0.635):
    """Compute motor-space mass matrix for the foot COM.

    The previous implementation attempted to use a 3x3 identity with a 2x2
    Jacobian which caused shape mismatches. Correct approach:

    1. Compute the 3x2 Jacobian J_com mapping joint velocities (tx,ty)
       to the linear velocity of the foot COM in world frame.
    2. Joint-space inertia M_joint = m * (J_com.T @ J_com)
    3. Motor-space inertia M_motor = J_tau @ M_joint @ J_tau.T
       where J_tau is the torque Jacobian (maps joint torque -> motor torque)
       and is provided alongside J_vel by calc_Jacobians.

    Note: This function expects J_vel to be the first return of
    ankle.calc_Jacobians; the caller should supply J_tau when available.
    """
    # This helper expects the caller to compute J_tau separately; we can't
    # invert J_vel here reliably if caller didn't get J_tau. To keep the
    # signature minimal, we'll compute a joint-space mass using the
    # foot COM Jacobian derived from the rotation derivatives and return it.
    # Callers can transform to motor space using J_tau.
    raise RuntimeError("compute_mass_matrix requires ankle, roll, pitch -> use compute_mass_matrix_full")


def compute_mass_matrix_full(ankle, roll, pitch, m_foot=0.635):
    """Compute motor-space mass matrix given ankle pose.

    Returns a 2x2 motor-space inertia matrix.
    """
    # Foot COM relative to ankle joint (from MJCF)
    foot_com_local = np.array([0.0375, 0.0, -0.0420])

    tx, ty = roll, pitch
    cx, sx = np.cos(tx), np.sin(tx)
    cy, sy = np.cos(ty), np.sin(ty)

    # Derivatives of rotation matrix (copied from kinematics implementation)
    dR_dtx = np.array([
        [0,     cx * sy,  -sx * sy],
        [0,     -sx,      -cx     ],
        [0,     cx * cy,  -sx * cy]
    ])
    dR_dty = np.array([
        [-sy,  sx * cy,  cx * cy],
        [0,    0,        0      ],
        [-cy, -sx * sy, -cx * sy]
    ])

    # 3x2 Jacobian mapping (tx,ty) -> linear velocity of COM
    J_com = np.column_stack((dR_dtx @ foot_com_local, dR_dty @ foot_com_local))

    # Joint-space inertia (2x2)
    M_joint = m_foot * (J_com.T @ J_com)

    # Get torque Jacobian (maps joint torque -> motor torque)
    J_vel, J_tau = ankle.calc_Jacobians(roll, pitch)
    if J_tau is None:
        raise RuntimeError("J_tau is singular or unavailable at this pose")

    # Motor-space inertia
    M_motor = J_tau @ M_joint @ J_tau.T
    return M_motor


def compute_gravity_torque(ankle, roll, pitch, m_foot=0.635, g=9.81):
    """Compute gravity compensation torque"""
    R = ankle.calc_R(roll, pitch)
    
    # Foot COM relative to ankle joint (from MJCF)
    foot_com_local = np.array([0.0375, 0.0, -0.0420])
    foot_com_world = R @ foot_com_local
    
    # Gravity torque on ankle joint
    tau_gravity_ankle = m_foot * g * np.array([
        foot_com_world[1],  # Roll torque
        -foot_com_world[0]  # Pitch torque (sign convention)
    ])
    
    # Transform to motor space using J_tau (use roll, pitch ordering)
    _, J_tau = ankle.calc_Jacobians(roll, pitch)
    tau_gravity_motor = J_tau @ tau_gravity_ankle
    
    return tau_gravity_motor


def test_dynamics():
    """Test dynamics computation"""
    ankle = CustomAnkleKinematics()
    
    print("="*60)
    print("ANKLE DYNAMICS TEST (using CustomAnkleKinematics)")
    print("="*60)
    
    test_poses = [
        (0.0, 0.0, "Home"),
        (0.1, 0.0, "Pitch 0.1"),
        (0.0, 0.1, "Roll 0.1"),
        (0.2, 0.15, "Combined"),
    ]
    
    for pitch, roll, name in test_poses:
        try:
            q1, q2 = ankle.IK(roll, pitch)  # Note: order is (theta_x, theta_y) = (roll, pitch)
            J_vel, J_tau = compute_jacobian(ankle, roll, pitch)  # Same here
            M = compute_mass_matrix_full(ankle, roll, pitch)
            tau_g = compute_gravity_torque(ankle, roll, pitch)
            
            print(f"\n--- {name} (pitch={pitch:+.4f} rad, roll={roll:+.4f} rad) ---")
            print(f"  Motor angles: q1={q1:+.4f} rad, q2={q2:+.4f} rad")
            print(f"  J_vel:\\n    {J_vel}")
            print(f"  Mass matrix M:\\n    {M}")
            print(f"  Gravity torque: {tau_g}")
            
        except Exception as e:
            # Catch all exceptions during the test to surface detailed diagnostics
            import traceback
            print(f"\n--- {name}: Exception occurred ({type(e).__name__}): {e}")
            traceback.print_exc()

            # If it was a ValueError from IK (outside workspace), print discriminant
            if isinstance(e, ValueError):
                print("Diagnostic: computing discriminant for each link...")
                try:
                    R = ankle.calc_R(roll, pitch)
                    C1_O = R @ ankle.C1_w
                    C2_O = R @ ankle.C2_w
                    for A, C, r, l, idx in [(ankle.A1, C1_O, ankle.r1, ankle.l1, 1),
                                              (ankle.A2, C2_O, ankle.r2, ankle.l2, 2)]:
                        U = A[0] - C[0]
                        V = A[1] - C[1]
                        W = A[2] - C[2]
                        K1 = 2 * r * W
                        K2 = 2 * r * U
                        K3 = l**2 - r**2 - U**2 - V**2 - W**2
                        sv = K1**2 + K2**2 - K3**2
                        print(f"    Link {idx} discriminant sv={sv:.6e}")
                except Exception as dbg_e:
                    print(f"    Diagnostic computation failed: {dbg_e}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    test_dynamics()
