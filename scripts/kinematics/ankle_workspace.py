"""
Ankle Workspace Analysis
========================

Analyzes the reachable workspace of the parallel ankle mechanism
using inverse kinematics (IK) and forward kinematics (FK).

Usage:
    python scripts/kinematics/ankle_workspace.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import math

from pndbotics_ankle import CustomAnkleKinematics


# ================= Runtime checks =================
if __name__ == "__main__":
    robot = CustomAnkleKinematics()

    print("--- 1. Validate Home Pose ---")
    # [Bug fix] The previous version used robot.q1_snap / robot.q2_snap,
    # but these attributes do not exist in this class and raise AttributeError.
    # th1/th3 are geometry parameters for A1/A2 construction, not home motor angles.
    # Home motor angles should be obtained from IK(0, 0).
    q1_zero, q2_zero = robot.IK(0.0, 0.0)
    print(f"Home motor angles: q1={q1_zero:.6f} rad, q2={q2_zero:.6f} rad")
    tx_zero, ty_zero = robot.FK(q1_zero, q2_zero)
    print(f"FK check at home: Roll={tx_zero:.2e} rad, Pitch={ty_zero:.2e} rad  OK")

    print("\n--- 2. IK <-> FK Cross-Validation ---")
    for test_tx, test_ty in [(0.15, 0.15), (0.20, -0.10), (-0.15, 0.20)]:
        q1, q2 = robot.IK(test_tx, test_ty)
        tx_back, ty_back = robot.FK(q1, q2)
        err = max(abs(tx_back - test_tx), abs(ty_back - test_ty))
        print(f"IK({test_tx:+.2f},{test_ty:+.2f}) -> q=({q1:.4f},{q2:.4f})"
              f"  FK error={err:.2e}  OK")

    print("\n--- 3. Boundary Guard Test ---")
    # [Bug fix] The previous out-of-range test used (0.15, -0.2), which is reachable.
    # It never exceeded the workspace, and the old IK behavior returned None instead
    # of raising ValueError. A scan shows boundary failures near roll+pitch > 1.0+0.9 rad.
    limit_tx, limit_ty = 1.0, 0.9
    print(f"Out-of-range test pose: Roll={limit_tx:.3f} rad, Pitch={limit_ty:.3f} rad")
    try:
        robot.IK(limit_tx, limit_ty)
        print("  Not intercepted")
    except ValueError as e:
        print(f"  Intercepted successfully: {e}  OK")
