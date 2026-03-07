import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # 导入3D多边形集合

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
                         [0,  cx,   -sx   ],[-sy, cy*sx, cx*cy]])

    def IK(self, theta_x, theta_y):
        R = self.calc_R(theta_x, theta_y)
        C1_O = R @ self.C1_w
        C2_O = R @ self.C2_w
        q_out = []
        for A, C, r, l in[(self.A1, C1_O, self.r1, self.l1),
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


def main():
    robot = CustomAnkleKinematics()

    fig = plt.figure(figsize=(10, 8))
    fig.canvas.manager.set_window_title('Ankle Parallel Mechanism 3D Viewer')
    plt.subplots_adjust(bottom=0.25)
    ax = fig.add_subplot(111, projection='3d')

    # 设置更利于观察的初始视角
    ax.view_init(elev=25, azim=-50)

    ax.set_xlim([-100, 150])
    ax.set_ylim([-100, 100])
    ax.set_zlim([-50, 300])
    ax.set_xlabel('X (Front)')
    ax.set_ylabel('Y (Left)')
    ax.set_zlabel('Z (Up)')
    ax.set_title("Drag sliders to move the foot pedal!", fontsize=14, pad=10)

    # 绘制固定电机和原点
    ax.scatter(*robot.A1, color='b', s=60, marker='s', label=f'Motor 1')
    ax.scatter(*robot.A2, color='r', s=60, marker='s', label=f'Motor 2')
    ax.scatter(0, 0, 0,   color='k', s=80, marker='*', label='Ankle Joint (Origin)')
    ax.legend(fontsize=9, loc='upper left')

    # 绘制连杆，增加了小圆点(marker='o')以凸显机械关节
    line_crank1, = ax.plot([], [],[], color='b', linestyle='-', linewidth=3, marker='o', markersize=5)
    line_crank2, = ax.plot([], [],[], color='r', linestyle='-', linewidth=3, marker='o', markersize=5)
    line_link1,  = ax.plot([], [],[], color='#4da6ff', linestyle='--', linewidth=2, marker='o', markersize=4)
    line_link2,  = ax.plot([], [],[], color='#ff4d4d', linestyle='--', linewidth=2, marker='o', markersize=4)

    # 脚踏板几何定义 (在局部坐标系下)
    arm_len = robot.h2 + robot.h3   # 前后延伸长度
    mid_w = (robot.C1_w + robot.C2_w) / 2
    fwd_w = mid_w + np.array([arm_len, 0.0, 0.0])  
    bwd_w = mid_w + np.array([-arm_len, 0.0, 0.0]) 
    
    # 提取左右宽度作为踏板边界
    y_left = robot.C1_w[1] + 10  # 稍微加宽一点让连杆末端包在踏板内侧更美观
    y_right = robot.C2_w[1] - 10
    z_pedal = mid_w[2]

    # 定义踏板的四个角点 (按逆时针顺序)
    corners_w = [
        np.array([fwd_w[0], y_left, z_pedal]),  # 左前
        np.array([bwd_w[0], y_left, z_pedal]),  # 左后
        np.array([bwd_w[0], y_right, z_pedal]), # 右后
        np.array([fwd_w[0], y_right, z_pedal])  # 右前
    ]

    # 初始化3D多边形 (半透明绿色填充，深绿色边框)
    pedal_poly = Poly3DCollection([], facecolors='#32CD32', edgecolors='#006400', alpha=0.35, linewidths=1.5)
    ax.add_collection3d(pedal_poly)

    # 踏板内部的十字结构骨架 (改为细线，作为金属骨架示意)
    line_lat, = ax.plot([], [],[], color='gray', linestyle='-', linewidth=2)
    line_fwd, = ax.plot([], [],[], color='gray', linestyle='-', linewidth=2)

    status_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=10, family='monospace')

    def update(tx, ty):
        q1, q2 = robot.IK(tx, ty)
        if q1 is None:
            status_text.set_text(f"[!] UNREACHABLE POS\nRoll={np.degrees(tx):.1f}°\nPitch={np.degrees(ty):.1f}°")
            status_text.set_color('red')
            # 如果不可达，将面变成红色警告
            pedal_poly.set_facecolors('#FF0000') 
            return

        R = robot.calc_R(tx, ty)
        C1_O = R @ robot.C1_w
        C2_O = R @ robot.C2_w

        B1_O = robot.A1 + np.array([robot.r1*np.cos(q1), 0, robot.r1*np.sin(q1)])
        B2_O = robot.A2 + np.array([robot.r2*np.cos(q2), 0, robot.r2*np.sin(q2)])

        line_crank1.set_data_3d([robot.A1[0],B1_O[0]], [robot.A1[1],B1_O[1]], [robot.A1[2],B1_O[2]])
        line_crank2.set_data_3d([robot.A2[0],B2_O[0]],[robot.A2[1],B2_O[1]], [robot.A2[2],B2_O[2]])
        line_link1.set_data_3d([B1_O[0],C1_O[0]], [B1_O[1],C1_O[1]],[B1_O[2],C1_O[2]])
        line_link2.set_data_3d([B2_O[0],C2_O[0]], [B2_O[1],C2_O[1]], [B2_O[2],C2_O[2]])

        # 旋转并更新踏板四个角点
        corners_O =[R @ pt for pt in corners_w]
        pedal_poly.set_verts([corners_O])
        pedal_poly.set_facecolors('#32CD32') # 恢复绿色

        # 更新十字骨架
        line_lat.set_data_3d([C1_O[0], C2_O[0]], [C1_O[1], C2_O[1]], [C1_O[2], C2_O[2]])
        bwd_O, fwd_O = R @ bwd_w, R @ fwd_w
        line_fwd.set_data_3d([bwd_O[0], fwd_O[0]], [bwd_O[1], fwd_O[1]], [bwd_O[2], fwd_O[2]])

        status_text.set_color('black')
        status_text.set_text(
            f"Roll = {np.degrees(tx):+05.1f}° | Pitch = {np.degrees(ty):+05.1f}°\n"
            f"q1   = {np.degrees(q1):+06.2f}° | q2    = {np.degrees(q2):+06.2f}°\n"
        )
        fig.canvas.draw_idle()

    update(0.0, 0.0)

    # UI 控件美化
    axcolor  = '#f0f0f0'
    ax_roll  = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)
    ax_pitch = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)

    s_roll  = Slider(ax_roll,  'Roll (X)', -0.35, 0.35,  valinit=0.0, valstep=0.01, color='#32CD32')
    s_pitch = Slider(ax_pitch, 'Pitch (Y)', -1, 0.35, valinit=0.0, valstep=0.01, color='#32CD32')

    s_roll.on_changed(lambda v: update(s_roll.val, s_pitch.val))
    s_pitch.on_changed(lambda v: update(s_roll.val, s_pitch.val))

    plt.show()

if __name__ == "__main__":
    main()