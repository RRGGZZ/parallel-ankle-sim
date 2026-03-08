"""
3D Viewer for Parallel Ankle Mechanism
======================================

Interactive 3D visualization of the parallel ankle mechanism with sliders
to control roll and pitch angles.

Usage:
    python scripts/kinematics/ankle_3d_viewer.py
"""

import sys
from pathlib import Path

# Add src to path for shared module
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pndbotics_ankle import CustomAnkleKinematics


def main():
    robot = CustomAnkleKinematics()

    fig = plt.figure(figsize=(10, 8))
    fig.canvas.manager.set_window_title('Ankle Parallel Mechanism 3D Viewer')
    plt.subplots_adjust(bottom=0.25)
    ax = fig.add_subplot(111, projection='3d')

    # Set an initial view angle that better exposes mechanism depth.
    ax.view_init(elev=25, azim=-50)

    ax.set_xlim([-100, 150])
    ax.set_ylim([-100, 100])
    ax.set_zlim([-50, 300])
    ax.set_xlabel('X (Front)')
    ax.set_ylabel('Y (Left)')
    ax.set_zlabel('Z (Up)')
    ax.set_title("Drag sliders to move the foot pedal!", fontsize=14, pad=10)

    # Draw fixed motor locations and the ankle origin.
    ax.scatter(*robot.A1, color='b', s=60, marker='s', label=f'Motor 1')
    ax.scatter(*robot.A2, color='r', s=60, marker='s', label=f'Motor 2')
    ax.scatter(0, 0, 0,   color='k', s=80, marker='*', label='Ankle Joint (Origin)')
    ax.legend(fontsize=9, loc='upper left')

    # Draw crank/link segments with markers to highlight joints.
    line_crank1, = ax.plot([], [],[], color='b', linestyle='-', linewidth=3, marker='o', markersize=5)
    line_crank2, = ax.plot([], [],[], color='r', linestyle='-', linewidth=3, marker='o', markersize=5)
    line_link1,  = ax.plot([], [],[], color='#4da6ff', linestyle='--', linewidth=2, marker='o', markersize=4)
    line_link2,  = ax.plot([], [],[], color='#ff4d4d', linestyle='--', linewidth=2, marker='o', markersize=4)

    # Define pedal geometry in the local frame.
    arm_len = robot.h2 + robot.h3   # Forward/backward extension length.
    mid_w = (robot.C1_w + robot.C2_w) / 2
    fwd_w = mid_w + np.array([arm_len, 0.0, 0.0])  
    bwd_w = mid_w + np.array([-arm_len, 0.0, 0.0]) 
    
    # Derive pedal side boundaries from lateral anchor positions.
    y_left = robot.C1_w[1] + 10  # Slightly wider for cleaner visual containment.
    y_right = robot.C2_w[1] - 10
    z_pedal = mid_w[2]

    # Define pedal corners in counter-clockwise order.
    corners_w = [
        np.array([fwd_w[0], y_left, z_pedal]),   # Front-left.
        np.array([bwd_w[0], y_left, z_pedal]),   # Rear-left.
        np.array([bwd_w[0], y_right, z_pedal]),  # Rear-right.
        np.array([fwd_w[0], y_right, z_pedal]),  # Front-right.
    ]

    # Initialize pedal polygon (semi-transparent green with darker edges).
    pedal_poly = Poly3DCollection([], facecolors='#32CD32', edgecolors='#006400', alpha=0.35, linewidths=1.5)
    ax.add_collection3d(pedal_poly)

    # Internal crossbars for visual structure cues.
    line_lat, = ax.plot([], [],[], color='gray', linestyle='-', linewidth=2)
    line_fwd, = ax.plot([], [],[], color='gray', linestyle='-', linewidth=2)

    status_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=10, family='monospace')

    def update(tx, ty):
        q1, q2 = robot.IK(tx, ty)
        if q1 is None:
            status_text.set_text(f"[!] UNREACHABLE POS\nRoll={tx:+.3f} rad\nPitch={ty:+.3f} rad")
            status_text.set_color('red')
            # Mark unreachable states by coloring the pedal red.
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

        # Rotate and update pedal corners.
        corners_O =[R @ pt for pt in corners_w]
        pedal_poly.set_verts([corners_O])
        pedal_poly.set_facecolors('#32CD32')  # Restore normal pedal color.

        # Update internal crossbars.
        line_lat.set_data_3d([C1_O[0], C2_O[0]], [C1_O[1], C2_O[1]], [C1_O[2], C2_O[2]])
        bwd_O, fwd_O = R @ bwd_w, R @ fwd_w
        line_fwd.set_data_3d([bwd_O[0], fwd_O[0]], [bwd_O[1], fwd_O[1]], [bwd_O[2], fwd_O[2]])

        status_text.set_color('black')
        status_text.set_text(
            f"Roll = {tx:+.3f} rad | Pitch = {ty:+.3f} rad\n"
            f"q1   = {q1:+.4f} rad | q2    = {q2:+.4f} rad\n"
        )
        fig.canvas.draw_idle()

    update(0.0, 0.0)

    # Style slider controls.
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
