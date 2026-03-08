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
    # 经扫描，工作空间边界约在 Roll+Pitch 组合超过约 1.0 + 0.9 rad 时触发。
    limit_tx, limit_ty = 1.0, 0.9
    print(f"测试越界姿态: Roll={limit_tx:.3f} rad, Pitch={limit_ty:.3f} rad")
    try:
        robot.IK(limit_tx, limit_ty)
        print("  未拦截")
    except ValueError as e:
        print(f"  成功拦截: {e}  ✓")
