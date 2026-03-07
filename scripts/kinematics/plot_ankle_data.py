import numpy as np
import math
import matplotlib.pyplot as plt


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
        A_mat = np.diag([a11, a22])

        dR_dtx = np.array([[0,     cx*sy,  -sx*sy],
                            [0,     -sx,    -cx   ],
                            [0,     cx*cy,  -sx*cy]])
        dR_dty = np.array([[-sy,   sx*cy,   cx*cy ],
                            [0,    0,       0     ],
                            [-cy, -sx*sy,  -cx*sy ]])

        dC1_dtx, dC1_dty = dR_dtx @ self.C1_w, dR_dty @ self.C1_w
        dC2_dtx, dC2_dty = dR_dtx @ self.C2_w, dR_dty @ self.C2_w

        B_mat = np.array([[np.dot(L1, dC1_dtx), np.dot(L1, dC1_dty)],
                           [np.dot(L2, dC2_dtx), np.dot(L2, dC2_dty)]])

        J_vel = np.array([[B_mat[0, 0] / a11, B_mat[0, 1] / a11],
                           [B_mat[1, 0] / a22, B_mat[1, 1] / a22]])

        try:
            J_tau = np.linalg.solve(J_vel.T, np.eye(2))
        except np.linalg.LinAlgError:
            return None, None

        return J_vel, J_tau


def plot_analysis():
    robot = CustomAnkleKinematics()

    roll_range  = np.linspace(-0.35, 0.35, 50)
    pitch_range = np.linspace(-1, 0.35, 50)

    q1_home, q2_home = robot.IK(0.0, 0.0)

    def generate_data(sweep_type='roll'):
        data = {'task': [], 'q1': [], 'q2': [],
                'dq1': [], 'dq2': [], 'tau1': [], 'tau2': []}

        for val in (roll_range if sweep_type == 'roll' else pitch_range):
            tx = val if sweep_type == 'roll' else 0.0
            ty = 0.0 if sweep_type == 'roll' else val

            q1, q2 = robot.IK(tx, ty)
            if q1 is None:
                continue
            J_vel, J_tau = robot.calc_Jacobians(tx, ty, q1=q1, q2=q2)
            if J_vel is None:
                continue

            data['task'].append(val)
            data['q1'].append(np.degrees(math.remainder(q1 - q1_home, 2*math.pi)))
            data['q2'].append(np.degrees(math.remainder(q2 - q2_home, 2*math.pi)))

            vel_target = np.array([1.0, 0.0]) if sweep_type == 'roll' else np.array([0.0, 1.0])
            dq = J_vel @ vel_target
            data['dq1'].append(dq[0])
            data['dq2'].append(dq[1])

            tau_target = np.array([10.0, 0.0]) if sweep_type == 'roll' else np.array([0.0, 10.0])
            tau_q = J_tau @ tau_target
            data['tau1'].append(tau_q[0])
            data['tau2'].append(tau_q[1])

        return data

    data_roll  = generate_data('roll')
    data_pitch = generate_data('pitch')

    plt.style.use('bmh')
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    fig.canvas.manager.set_window_title('Ankle Parallel Mechanism — Kinematics & Statics')

    axs[0, 0].plot(data_roll['task'], data_roll['q1'], label='Motor 1 (Left)')
    axs[0, 0].plot(data_roll['task'], data_roll['q2'], label='Motor 2 (Right)')
    axs[0, 0].set_title('Angle Mapping  (Pitch = 0)')
    axs[0, 0].set_ylabel('Motor Angle Change (deg)')
    axs[0, 0].legend()

    axs[1, 0].plot(data_roll['task'], data_roll['dq1'], label='Motor 1')
    axs[1, 0].plot(data_roll['task'], data_roll['dq2'], label='Motor 2')
    axs[1, 0].set_title('Speed Mapping  (Target Roll vel = 1 rad/s)')
    axs[1, 0].set_ylabel('Motor Speed (rad/s)')
    axs[1, 0].legend()

    axs[2, 0].plot(data_roll['task'], data_roll['tau1'], label='Motor 1')
    axs[2, 0].plot(data_roll['task'], data_roll['tau2'], label='Motor 2')
    axs[2, 0].set_title('Torque Mapping  (Target Roll torque = 10 Nm)')
    axs[2, 0].set_ylabel('Motor Torque (Nm)')
    axs[2, 0].set_xlabel('Ankle Roll Angle (rad)')
    axs[2, 0].legend()

    axs[0, 1].plot(data_pitch['task'], data_pitch['q1'], label='Motor 1')
    axs[0, 1].plot(data_pitch['task'], data_pitch['q2'], label='Motor 2')
    axs[0, 1].set_title('Angle Mapping  (Roll = 0)')
    axs[0, 1].set_ylabel('Motor Angle Change (deg)')
    axs[0, 1].legend()

    axs[1, 1].plot(data_pitch['task'], data_pitch['dq1'], label='Motor 1')
    axs[1, 1].plot(data_pitch['task'], data_pitch['dq2'], label='Motor 2')
    axs[1, 1].set_title('Speed Mapping  (Target Pitch vel = 1 rad/s)')
    axs[1, 1].set_ylabel('Motor Speed (rad/s)')
    axs[1, 1].legend()

    axs[2, 1].plot(data_pitch['task'], data_pitch['tau1'], label='Motor 1')
    axs[2, 1].plot(data_pitch['task'], data_pitch['tau2'], label='Motor 2')
    axs[2, 1].set_title('Torque Mapping  (Target Pitch torque = 10 Nm)')
    axs[2, 1].set_ylabel('Motor Torque (Nm)')
    axs[2, 1].set_xlabel('Ankle Pitch Angle (rad)')
    axs[2, 1].legend()

    for ax in axs.flat:
        ax.axhline(0, color='gray', lw=0.6, ls='--')
        ax.axvline(0, color='gray', lw=0.6, ls='--')
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_analysis()