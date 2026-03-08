"""
PnDBotics Ankle Simulation - Shared Kinematics Module
=====================================================

This module provides the core kinematics model for the parallel ankle mechanism.
It is shared across all analysis scripts to avoid code duplication.

Author: PnDBotics Research Team
License: Academic/Research Use
"""

import math
import numpy as np

__version__ = "1.0.0"
__author__ = "PnDBotics Research Team"


class CustomAnkleKinematics:
    """
    Parallel Ankle Mechanism Kinematics Model.
    
    This class implements the analytical inverse kinematics (IK), forward kinematics (FK),
    and Jacobian calculations for a dual-motor parallel ankle mechanism.
    
    Geometric Parameters (all in mm):
        - r1, r2: Crank radii (motor shaft to B-point)
        - l1, l2: Coupling link lengths (B-point to C-point)
        - h1, h2, h3, d1, d2, d3: Frame mounting parameters
        - th1-th4: Installation angles
    
    Coordinate System:
        - X: Forward (front of robot)
        - Y: Left (robot's left side)
        - Z: Up (vertical)
    
    Example:
        >>> robot = CustomAnkleKinematics()
        >>> q1, q2 = robot.IK(roll=0.1, pitch=0.05)
        >>> print(f"Motor angles: {q1:.3f}, {q2:.3f} rad")
    """
    
    def __init__(self):
        # ─────────────────────────────────────────────────────────────────────
        # Geometric parameters (unit: mm)
        # ─────────────────────────────────────────────────────────────────────
        self.r1, self.l1, self.h1, self.d1 = 40.00, 280.00, 15.00, 37.10
        self.r2, self.l2, self.h2, self.d2 = 40.00, 208.00, 33.00, 35.00
        self.h3, self.d3 = 31.50, 47.00
        
        # Installation angles (unit: rad)
        self.th1, self.th2, self.th3, self.th4 = 0.51, 1.83, 0.57, 1.92
        
        # ─────────────────────────────────────────────────────────────────────
        # Compute frame attachment points in world coordinates
        # C1, C2: Foot pedal attachment points (toe side)
        # A1, A2: Motor shaft centers
        # ─────────────────────────────────────────────────────────────────────
        self.C1_w = np.array([self.d1, (self.d3 + self.d2) / 4, -self.h1])
        self.C2_w = np.array([self.d1, -(self.d2 + self.d3) / 4, -self.h1])
        
        alpha1 = np.pi - self.th2
        B1_zero = self.C1_w + np.array([
            -self.l1 * np.cos(alpha1), 
            0, 
            self.l1 * np.sin(alpha1)
        ])
        self.A1 = B1_zero + np.array([
            self.r1 * np.cos(self.th1), 
            0, 
            self.r1 * np.sin(self.th1)
        ])
        
        alpha2 = np.pi - self.th4
        B2_zero = self.C2_w + np.array([
            -self.l2 * np.cos(alpha2), 
            0, 
            self.l2 * np.sin(alpha2)
        ])
        self.A2 = B2_zero + np.array([
            self.r2 * np.cos(self.th3), 
            0, 
            self.r2 * np.sin(self.th3)
        ])
    
    def calc_R(self, tx: float, ty: float) -> np.ndarray:
        """
        Compute rotation matrix for ankle orientation.
        
        Parameters
        ----------
        tx : float
            Roll angle about X-axis (radians)
        ty : float
            Pitch angle about Y-axis (radians)
        
        Returns
        -------
        R : ndarray (3, 3)
            Rotation matrix transforming from body frame to world frame
        
        Note
        ----
        The rotation order is: first pitch (Y), then roll (X).
        This corresponds to R = R_x(tx) @ R_y(ty)
        """
        cx, sx = np.cos(tx), np.sin(tx)
        cy, sy = np.cos(ty), np.sin(ty)
        
        return np.array([
            [cy,  sx * sy,  cx * sy],
            [0,   cx,       -sx     ],
            [-sy, cy * sx,  cx * cy]
        ])
    
    def IK(self, theta_x: float, theta_y: float):
        """
        Inverse Kinematics: Compute motor angles from desired ankle orientation.
        
        Parameters
        ----------
        theta_x : float
            Roll angle (radians)
        theta_y : float
            Pitch angle (radians)
        
        Returns
        -------
        q1, q2 : tuple of floats or (None, None)
            Motor angles in radians. Returns (None, None) if target is
            outside the reachable workspace.
        
        Algorithm
        ---------
        Uses geometric constraints: the distance between B-points (crank tips)
        and C-points (pedal attachments) must equal the link lengths l1, l2.
        Solves using the law of cosines in the crank rotation plane.
        """
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
                raise ValueError(
                    f"Pose outside workspace (discriminant sv={sv:.2f} < 0), "
                    f"Roll={np.degrees(theta_x):.1f}°, Pitch={np.degrees(theta_y):.1f}°"
                )
            
            q = math.atan2(K3, -math.sqrt(sv)) - math.atan2(K2, K1)
            q_out.append(q)
        
        return q_out[0], q_out[1]
    
    def FK(self, q1: float, q2: float, tol: float = 1e-8, max_iter: int = 30):
        """
        Forward Kinematics: Compute ankle orientation from motor angles.
        
        Parameters
        ----------
        q1, q2 : float
            Motor angles in radians
        tol : float, optional
            Convergence tolerance (default: 1e-8)
        max_iter : int, optional
            Maximum Newton iterations (default: 30)
        
        Returns
        -------
        tx, ty : tuple of floats
            Roll and pitch angles in radians
        
        Raises
        ------
        RuntimeError
            If FK fails to converge (singular configuration or unreachable)
        """
        # Compute crank tip positions in world frame
        B1 = self.A1 + np.array([
            self.r1 * np.cos(q1), 
            0, 
            self.r1 * np.sin(q1)
        ])
        B2 = self.A2 + np.array([
            self.r2 * np.cos(q2), 
            0, 
            self.r2 * np.sin(q2)
        ])
        
        # Initial guess: neutral position (converges well near home)
        tx, ty = 0.0, 0.0
        
        for it in range(max_iter):
            R = self.calc_R(tx, ty)
            C1_O, C2_O = R @ self.C1_w, R @ self.C2_w
            
            # Geometric constraints (should be zero when solved)
            g = np.array([
                np.sum((B1 - C1_O)**2) - self.l1**2,
                np.sum((B2 - C2_O)**2) - self.l2**2
            ])
            
            if np.linalg.norm(g) < tol:
                return tx, ty
            
            # Analytical Jacobian of the constraint equations
            cx, sx = np.cos(tx), np.sin(tx)
            cy, sy = np.cos(ty), np.sin(ty)
            
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
            
            J = np.zeros((2, 2))
            J[0, 0] = -2 * np.dot(B1 - C1_O, dR_dtx @ self.C1_w)
            J[0, 1] = -2 * np.dot(B1 - C1_O, dR_dty @ self.C1_w)
            J[1, 0] = -2 * np.dot(B2 - C2_O, dR_dtx @ self.C2_w)
            J[1, 1] = -2 * np.dot(B2 - C2_O, dR_dty @ self.C2_w)
            
            try:
                delta = np.linalg.solve(J, g)
            except np.linalg.LinAlgError:
                raise RuntimeError(f"FK singularity: Jacobian not invertible (iter={it})")
            
            tx -= delta[0]
            ty -= delta[1]
        
        raise RuntimeError(f"FK did not converge: final residual = {np.linalg.norm(g):.4e}")
    
    def calc_Jacobians(self, tx: float, ty: float, q1: float = None, q2: float = None):
        """
        Compute velocity and torque Jacobian matrices.
        
        Parameters
        ----------
        tx, ty : float
            Current roll and pitch angles (radians)
        q1, q2 : float, optional
            Motor angles. If None, will be computed via IK.
        
        Returns
        -------
        J_vel : ndarray (2, 2)
            Velocity Jacobian: motor_velocity = J_vel @ joint_velocity
        J_tau : ndarray (2, 2)
            Torque Jacobian: motor_torque = J_tau @ joint_torque
            (inverse of J_vel.T for force mapping)
        
        Note
        ----
        J_vel maps joint space velocity to motor space velocity.
        J_tau maps joint torque to motor torque (based on virtual work principle).
        """
        if q1 is None or q2 is None:
            q1, q2 = self.IK(tx, ty)
        
        if q1 is None:
            return None, None
        
        R = self.calc_R(tx, ty)
        C1_O, C2_O = R @ self.C1_w, R @ self.C2_w
        
        # Compute crank tip positions
        B1 = self.A1 + np.array([
            self.r1 * np.cos(q1), 
            0, 
            self.r1 * np.sin(q1)
        ])
        B2 = self.A2 + np.array([
            self.r2 * np.cos(q2), 
            0, 
            self.r2 * np.sin(q2)
        ])
        
        # Link vectors (B to C)
        L1 = B1 - C1_O
        L2 = B2 - C2_O
        
        # Velocity of crank tips due to motor rotation
        v_B1 = np.array([
            -self.r1 * np.sin(q1), 
            0, 
            self.r1 * np.cos(q1)
        ])
        v_B2 = np.array([
            -self.r2 * np.sin(q2), 
            0, 
            self.r2 * np.cos(q2)
        ])
        
        # Coefficients for velocity relationship
        a11 = np.dot(L1, v_B1)
        a22 = np.dot(L2, v_B2)
        
        if abs(a11) < 1e-6 or abs(a22) < 1e-6:
            return None, None  # Near singularity
        
        # Derivatives of rotation matrix
        cx, sx = np.cos(tx), np.sin(tx)
        cy, sy = np.cos(ty), np.sin(ty)
        
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
        
        dC1_dtx = dR_dtx @ self.C1_w
        dC1_dty = dR_dty @ self.C1_w
        dC2_dtx = dR_dtx @ self.C2_w
        dC2_dty = dR_dty @ self.C2_w
        
        # Build velocity Jacobian
        B_mat = np.array([
            [np.dot(L1, dC1_dtx), np.dot(L1, dC1_dty)],
            [np.dot(L2, dC2_dtx), np.dot(L2, dC2_dty)]
        ])
        
        J_vel = np.array([
            [B_mat[0, 0] / a11, B_mat[0, 1] / a11],
            [B_mat[1, 0] / a22, B_mat[1, 1] / a22]
        ])
        
        # Torque Jacobian (virtual work principle)
        try:
            J_tau = np.linalg.solve(J_vel.T, np.eye(2))
        except np.linalg.LinAlgError:
            return None, None
        
        return J_vel, J_tau
    
    def calc_Jvel(self, tx: float, ty: float, q1: float = None, q2: float = None):
        """
        Compute only the velocity Jacobian (alias for compatibility).
        
        See calc_Jacobians for full documentation.
        """
        result = self.calc_Jacobians(tx, ty, q1, q2)
        if result[0] is None:
            return None
        return result[0]
    
    @property
    def home_position(self):
        """
        Get the home (zero) motor positions.
        
        Returns
        -------
        q1_home, q2_home : tuple of floats
            Motor angles at neutral ankle orientation (roll=0, pitch=0)
        """
        return self.IK(0.0, 0.0)
