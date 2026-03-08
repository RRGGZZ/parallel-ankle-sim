"""
Ankle Kinematics Analysis Plotter
================================

Generates plots for angle mapping, speed mapping, and torque mapping
of the parallel ankle mechanism.

Usage:
    python scripts/kinematics/plot_ankle_data.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# Use non-interactive backend for headless environments
import matplotlib
matplotlib.use('Agg')

import numpy as np
import math
import matplotlib.pyplot as plt

from pndbotics_ankle import CustomAnkleKinematics


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
            data['q1'].append(math.remainder(q1 - q1_home, 2 * math.pi))
            data['q2'].append(math.remainder(q2 - q2_home, 2 * math.pi))

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
    axs[0, 0].set_ylabel('Motor Angle Change (rad)')
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
    axs[0, 1].set_ylabel('Motor Angle Change (rad)')
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
