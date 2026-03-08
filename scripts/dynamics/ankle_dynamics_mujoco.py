"""
Ankle Kinematics Verification: Theory vs MuJoCo
=============================================

This script verifies CustomAnkleKinematics against MuJoCo (ground truth).

Issue Found:
  - Theory (CustomAnkleKinematics): l1=280mm, l2=208mm  
  - MuJoCo (ground truth): L1=309.7mm, L2=238.6mm
  
The theory model uses different parameters than MuJoCo, causing large errors.

Usage:
    python scripts/dynamics/ankle_dynamics_mujoco.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib
matplotlib.use('Agg')

import math
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
from scipy.optimize import fsolve
import time

from pndbotics_ankle import CustomAnkleKinematics


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ═══════════════════════════════════════════════════════════════════════
# Load MuJoCo
# ═══════════════════════════════════════════════════════════════════════
mjcf_path = PROJECT_ROOT / "adam_pro" / "adam_pro.xml"
model = mujoco.MjModel.from_xml_path(str(mjcf_path))
data = mujoco.MjData(model)

def qidx(name):
    return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]

def sidx(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)

# Joint indices
idx_m_l1 = qidx("ankleMotor_Left_1")
idx_m_l2 = qidx("ankleMotor_Left_2")
idx_p_l = qidx("anklePitch_Left")
idx_r_l = qidx("ankleRoll_Left")
idx_p_r = qidx("anklePitch_Right")
idx_r_r = qidx("ankleRoll_Right")

# Sites
sid_B1 = sidx("ankleLeft_B1")
sid_B2 = sidx("ankleLeft_B2")
sid_C1 = sidx("ankleToeLeft_C1")
sid_C2 = sidx("ankleToeLeft_C2")

# Get link lengths from MuJoCo
mujoco.mj_forward(model, data)
L1 = np.linalg.norm(data.site_xpos[sid_B1] - data.site_xpos[sid_C1])
L2 = np.linalg.norm(data.site_xpos[sid_B2] - data.site_xpos[sid_C2])

print("="*60)
print("ANKLE KINEMATICS VERIFICATION")
print("="*60)
print(f"\nMuJoCo link lengths: L1={L1*1000:.2f}mm, L2={L2*1000:.2f}mm")


# ═══════════════════════════════════════════════════════════════════════
# Theory Model
# ═══════════════════════════════════════════════════════════════════════
ankle_kin = CustomAnkleKinematics()
print(f"Theory link lengths: l1={ankle_kin.l1}mm, l2={ankle_kin.l2}mm")
print(f"Parameter difference: d_l1={ankle_kin.l1-L1*1000:.2f}mm, d_l2={ankle_kin.l2-L2*1000:.2f}mm")


# ═══════════════════════════════════════════════════════════════════════
# MuJoCo IK Solver (Ground Truth)
# ═══════════════════════════════════════════════════════════════════════

def mujoco_ik(pitch, roll, q0=(0.0, 0.0)):
    """Solve IK numerically in MuJoCo."""
    mujoco.mj_resetData(model, data)
    data.qpos[idx_p_l] = pitch
    data.qpos[idx_r_l] = roll
    data.qpos[idx_p_r] = pitch
    data.qpos[idx_r_r] = roll
    mujoco.mj_forward(model, data)
    
    C1 = data.site_xpos[sid_C1].copy()
    C2 = data.site_xpos[sid_C2].copy()
    
    def res(q):
        data.qpos[idx_m_l1] = q[0]
        data.qpos[idx_m_l2] = q[1]
        mujoco.mj_forward(model, data)
        B1 = data.site_xpos[sid_B1].copy()
        B2 = data.site_xpos[sid_B2].copy()
        return [np.linalg.norm(B1-C1)-L1, np.linalg.norm(B2-C2)-L2]
    
    sol, _, ier, _ = fsolve(res, q0, full_output=True, xtol=1e-9)
    return (sol[0], sol[1]) if ier == 1 else (None, None)


# ═══════════════════════════════════════════════════════════════════════
# Find Home Position
# ═══════════════════════════════════════════════════════════════════════
q1_home_mj, q2_home_mj = mujoco_ik(0.0, 0.0)
q1_home_th, q2_home_th = ankle_kin.IK(0.0, 0.0)

print(f"\nHome position (ankle angle = 0):")
print(f"  MuJoCo: M1={math.degrees(q1_home_mj):.2f}deg, M2={math.degrees(q2_home_mj):.2f}deg")
print(f"  Theory:  M1={math.degrees(q1_home_th):.2f}deg, M2={math.degrees(q2_home_th):.2f}deg")


# ═══════════════════════════════════════════════════════════════════════
# Scan and Compare
# ═══════════════════════════════════════════════════════════════════════
ROLL_RANGE = np.linspace(-0.30, 0.30, 20)
PITCH_RANGE = np.linspace(-0.70, 0.25, 20)

def scan(axis):
    rng = ROLL_RANGE if axis == 'roll' else PITCH_RANGE
    
    results = {
        'angle': [], 'mj_m1': [], 'th_m1': [], 'err_m1': [],
        'mj_m2': [], 'th_m2': [], 'err_m2': []
    }
    
    q_prev = [q1_home_mj, q2_home_mj]
    
    for v in rng:
        roll = v if axis == 'roll' else 0.0
        pitch = 0.0 if axis == 'roll' else v
        
        # MuJoCo (ground truth)
        q1_mj, q2_mj = mujoco_ik(pitch, roll, q0=q_prev)
        if q1_mj is None:
            continue
        q_prev = [q1_mj, q2_mj]
        
        # Theory
        try:
            q1_th, q2_th = ankle_kin.IK(roll, pitch)
        except ValueError:
            continue
        
        # Relative angles from home
        dq1_mj = math.degrees(q1_mj - q1_home_mj)
        dq2_mj = math.degrees(q2_mj - q2_home_mj)
        dq1_th = math.degrees(q1_th - q1_home_th)
        dq2_th = math.degrees(q2_th - q2_home_th)
        
        results['angle'].append(v)
        results['mj_m1'].append(dq1_mj)
        results['mj_m2'].append(dq2_mj)
        results['th_m1'].append(dq1_th)
        results['th_m2'].append(dq2_th)
        results['err_m1'].append(dq1_mj - dq1_th)
        results['err_m2'].append(dq2_mj - dq2_th)
    
    return results

print("\n=== Scanning ===")
print("Roll sweep...")
roll_data = scan('roll')
print("Pitch sweep...")
pitch_data = scan('pitch')

print(f"\nRoll error:  M1 max={max(abs(e) for e in roll_data['err_m1']):.2f}deg, "
      f"M2 max={max(abs(e) for e in roll_data['err_m2']):.2f}deg")
print(f"Pitch error: M1 max={max(abs(e) for e in pitch_data['err_m1']):.2f}deg, "
      f"M2 max={max(abs(e) for e in pitch_data['err_m2']):.2f}deg")


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Roll - Motor angles
ax = axes[0, 0]
ax.plot(roll_data['angle'], roll_data['th_m1'], 'b-', lw=2, label='Theory M1')
ax.plot(roll_data['angle'], roll_data['mj_m1'], 'b--', lw=2, label='MuJoCo M1')
ax.plot(roll_data['angle'], roll_data['th_m2'], 'r-', lw=2, label='Theory M2')
ax.plot(roll_data['angle'], roll_data['mj_m2'], 'r--', lw=2, label='MuJoCo M2')
ax.set_xlabel('Roll (rad)')
ax.set_ylabel('Motor angle (deg, relative)')
ax.set_title('Roll Sweep: Motor Angles')
ax.legend()
ax.grid(True, alpha=0.3)

# Pitch - Motor angles
ax = axes[0, 1]
ax.plot(pitch_data['angle'], pitch_data['th_m1'], 'b-', lw=2, label='Theory M1')
ax.plot(pitch_data['angle'], pitch_data['mj_m1'], 'b--', lw=2, label='MuJoCo M1')
ax.plot(pitch_data['angle'], pitch_data['th_m2'], 'r-', lw=2, label='Theory M2')
ax.plot(pitch_data['angle'], pitch_data['mj_m2'], 'r--', lw=2, label='MuJoCo M2')
ax.set_xlabel('Pitch (rad)')
ax.set_ylabel('Motor angle (deg, relative)')
ax.set_title('Pitch Sweep: Motor Angles')
ax.legend()
ax.grid(True, alpha=0.3)

# Roll - Error
ax = axes[1, 0]
ax.plot(roll_data['angle'], roll_data['err_m1'], 'b-', lw=2, label='M1 error')
ax.plot(roll_data['angle'], roll_data['err_m2'], 'r-', lw=2, label='M2 error')
ax.axhline(0, color='k', ls='--', lw=1)
ax.fill_between(roll_data['angle'], roll_data['err_m1'], 0, alpha=0.2, color='blue')
ax.fill_between(roll_data['angle'], roll_data['err_m2'], 0, alpha=0.2, color='red')
ax.set_xlabel('Roll (rad)')
ax.set_ylabel('Error (deg)')
max_err = max(max(abs(e) for e in roll_data['err_m1']), max(abs(e) for e in roll_data['err_m2']))
ax.set_title(f'Roll Error: MuJoCo - Theory\nMax = {max_err:.2f}deg')
ax.legend()
ax.grid(True, alpha=0.3)

# Pitch - Error
ax = axes[1, 1]
ax.plot(pitch_data['angle'], pitch_data['err_m1'], 'b-', lw=2, label='M1 error')
ax.plot(pitch_data['angle'], pitch_data['err_m2'], 'r-', lw=2, label='M2 error')
ax.axhline(0, color='k', ls='--', lw=1)
ax.fill_between(pitch_data['angle'], pitch_data['err_m1'], 0, alpha=0.2, color='blue')
ax.fill_between(pitch_data['angle'], pitch_data['err_m2'], 0, alpha=0.2, color='red')
ax.set_xlabel('Pitch (rad)')
ax.set_ylabel('Error (deg)')
max_err = max(max(abs(e) for e in pitch_data['err_m1']), max(abs(e) for e in pitch_data['err_m2']))
ax.set_title(f'Pitch Error: MuJoCo - Theory\nMax = {max_err:.2f}deg')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Ankle Kinematics Verification: Theory vs MuJoCo (Ground Truth)', fontsize=12)
plt.tight_layout()
plt.savefig('ankle_verification_vs_mujoco.png', dpi=150, bbox_inches='tight')
print(f"\nFigure saved: ankle_verification_vs_mujoco.png")


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("Parameter mismatch detected:")
print(f"  Theory:  l1={ankle_kin.l1}mm, l2={ankle_kin.l2}mm")
print(f"  MuJoCo:  L1={L1*1000:.1f}mm, L2={L2*1000:.1f}mm")
print("\nTo align: Update src/pndbotics_ankle/__init__.py with correct l1, l2 values")
print("="*60)
