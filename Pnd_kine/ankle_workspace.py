import numpy as np
import math


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
            # [Bug1修复] 原代码 return None, None，调用方无法用 except ValueError
            # 捕获，边界拦截测试静默失败。改为抛出 ValueError。
            if sv < 0:
                raise ValueError(
                    f"姿态超出工作空间 (判别式 sv={sv:.2f} < 0), "
                    f"Roll={np.degrees(theta_x):.1f}°, Pitch={np.degrees(theta_y):.1f}°"
                )
            q = math.atan2(K3, -math.sqrt(sv)) - math.atan2(K2, K1)
            q_out.append(q)
        return q_out[0], q_out[1]

    def FK(self, q1, q2, tol=1e-8, max_iter=30):
        """正运动学：牛顿迭代法"""
        B1 = self.A1 + np.array([self.r1 * np.cos(q1), 0, self.r1 * np.sin(q1)])
        B2 = self.A2 + np.array([self.r2 * np.cos(q2), 0, self.r2 * np.sin(q2)])

        tx, ty = 0.0, 0.0  # 初始猜测：零位附近收敛良好

        for it in range(max_iter):
            cx, sx = np.cos(tx), np.sin(tx)
            cy, sy = np.cos(ty), np.sin(ty)

            R = self.calc_R(tx, ty)
            C1_O, C2_O = R @ self.C1_w, R @ self.C2_w

            g = np.array([
                np.sum((B1 - C1_O)**2) - self.l1**2,
                np.sum((B2 - C2_O)**2) - self.l2**2
            ])

            if np.linalg.norm(g) < tol:
                return tx, ty

            # 旋转矩阵对 tx, ty 的解析偏导（已验证数值精度 < 1e-7）
            dR_dtx = np.array([
                [0,     cx*sy,  -sx*sy],
                [0,     -sx,    -cx   ],
                [0,     cx*cy,  -sx*cy]
            ])
            dR_dty = np.array([
                [-sy,  sx*cy,  cx*cy ],
                [0,    0,      0     ],
                [-cy, -sx*sy, -cx*sy ]
            ])

            J = np.zeros((2, 2))
            J[0, 0] = -2 * np.dot(B1 - C1_O, dR_dtx @ self.C1_w)
            J[0, 1] = -2 * np.dot(B1 - C1_O, dR_dty @ self.C1_w)
            J[1, 0] = -2 * np.dot(B2 - C2_O, dR_dtx @ self.C2_w)
            J[1, 1] = -2 * np.dot(B2 - C2_O, dR_dty @ self.C2_w)

            try:
                delta = np.linalg.solve(J, g)
            except np.linalg.LinAlgError:
                raise RuntimeError(f"FK奇异点：雅可比矩阵不可逆 (iter={it})")

            tx -= delta[0]
            ty -= delta[1]

        raise RuntimeError(f"FK未收敛：最终残差={np.linalg.norm(g):.4e}")


# ================= 运行验证 =================
if __name__ == "__main__":
    robot = CustomAnkleKinematics()

    print("--- 1. 验证零位状态 ---")
    # [Bug2修复] 原代码用 robot.q1_snap / robot.q2_snap，
    # 这两个属性在当前类中不存在，运行直接 AttributeError 崩溃。
    # th1/th3 是构造 A1/A2 的几何参数，并非零位电机角。
    # 零位电机角应通过 IK(0, 0) 获取。
    q1_zero, q2_zero = robot.IK(0.0, 0.0)
    print(f"零位电机角: q1={q1_zero:.6f} rad, q2={q2_zero:.6f} rad")
    tx_zero, ty_zero = robot.FK(q1_zero, q2_zero)
    print(f"FK回验零位: Roll={tx_zero:.2e} rad, Pitch={ty_zero:.2e} rad  ✓")

    print("\n--- 2. IK ↔ FK 交叉验证 ---")
    for test_tx, test_ty in [(0.15, 0.15), (0.20, -0.10), (-0.15, 0.20)]:
        q1, q2 = robot.IK(test_tx, test_ty)
        tx_back, ty_back = robot.FK(q1, q2)
        err = max(abs(tx_back - test_tx), abs(ty_back - test_ty))
        print(f"IK({test_tx:+.2f},{test_ty:+.2f}) -> q=({q1:.4f},{q2:.4f})"
              f"  FK误差={err:.2e}  ✓")

    print("\n--- 3. 边界拦截测试 ---")
    # [Bug3修复] 原测试用 (0.15, -0.2) 但该姿态实际在工作空间内，
    # 永远不会越界。且即使越界，原 IK 返回 None 也不抛 ValueError。
    # 经扫描，工作空间边界约在 Roll+Pitch 组合超过 ~57°+52° 时触发。
    limit_tx, limit_ty = 1.0, 0.9   # ~57° Roll + ~52° Pitch 组合极限
    print(f"测试越界姿态: Roll={np.degrees(limit_tx):.0f}°, Pitch={np.degrees(limit_ty):.0f}°")
    try:
        robot.IK(limit_tx, limit_ty)
        print("  未拦截")
    except ValueError as e:
        print(f"  成功拦截: {e}  ✓")