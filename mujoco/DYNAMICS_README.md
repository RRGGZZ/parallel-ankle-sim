# 脚踝并联机构动力学计算说明

## 功能概述

已为 `ankle_dyn_pin.py` 添加完整的动力学计算模块，包括：

### 1. **雅可比矩阵计算** (`compute_ankle_jacobian_numerical`)
- 数值计算电机角度与脚踝末端位置/速度的映射关系
- 返回 6×2 矩阵（前3行为线速度，后3行为角速度）
- 支持检测运动学奇异位置（通过条件数）

### 2. **力/扭矩映射** (`compute_motor_torque_to_ankle_force`)
- 计算电机扭矩到脚踝末端力/力矩的映射（静力学对偶）
- 基于转置雅可比矩阵：τ_ankle = J^T × τ_motor

### 3. **质量矩阵** (`compute_mass_matrix_ankle`)
- 计算电机空间的质量矩阵 M (2×2)
- 反映系统的加速特性
- 假设脚掌质量 0.5 kg（可根据实际模型调整）

### 4. **科氏力与重力补偿** (`compute_coriolis_gravity`)
- 通过数值微分重力势能计算重力补偿扭矩
- 返回2维向量，分别对应两个电机

### 5. **逆动力学** (`compute_inverse_dynamics_ankle`)
- 计算给定期望加速度所需的电机扭矩
- 公式：τ = M × qdd + c_g
- 用于控制系统设计

### 6. **动力学信息打印** (`print_ankle_dynamics`)
- 汇总显示脚踝的各项动力学参数
- 便于快速调试和分析

## 使用示例

```python
# 初始化配置、速度、加速度
q = pin.neutral(model)
qd = np.zeros(model.nv)
qdd = np.zeros(model.nv)

# 计算某个姿态下的雅可比矩阵
J = compute_ankle_jacobian_numerical(q, ankle_type='left')
print(f"雅可比条件数: {np.linalg.cond(J[0:3, :]):.2f}")

# 计算质量矩阵
M = compute_mass_matrix_ankle(q, ankle_type='left')
print(f"质量矩阵:\n{M}")

# 计算重力补偿扭矩
c_g = compute_coriolis_gravity(q, qd, ankle_type='left')
print(f"重力补偿: τ1={c_g[0]:+.4f}, τ2={c_g[1]:+.4f} N·m")

# 给定期望加速度，计算所需扭矩
qdd_desired = np.zeros(model.nv)
qdd_desired[pitch_v_idx] = 0.2  # rad/s^2
tau = compute_inverse_dynamics_ankle(q, qd, qdd_desired, ankle_type='left')
print(f"所需电机扭矩: τ1={tau[0]:+.4f}, τ2={tau[1]:+.4f} N·m")
```

## 演示脚本

### 主脚本：`ankle_dyn_pin.py`
- 包含完整的正逆运动学 + 动力学 + Meshcat 可视化
- 运行时需要Meshcat浏览器支持
- 演示内容包括：
  - 正运动学演示
  - 逆运动学演示
  - 不同姿态下的动力学特性变化
  - 逆动力学演示

### 轻量演示：`ankle_dynamics_demo.py`
- 纯数值计算，无需Meshcat可视化
- 快速测试动力学功能
- 运行命令：
  ```bash
  python mujoco/ankle_dynamics_demo.py
  ```

## 重要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 脚掌质量 (`m_foot`) | 0.5 kg | 可根据实际模型修改 |
| 脚掌惯性张量 | 0.001 kg·m² | 简化为球形，可优化 |
| 数值微分步长 (`delta`) | 1e-7 | 用于梯度计算的精度控制 |
| 约束求解精度 (`xtol`) | 1e-8 | IK约束满足精度 |

## 关键限制与改进方向

1. **简化假设**：
   - 脚掌质量与惯性为简化模型
   - 只考虑重力，忽略摩擦和空气阻力

2. **改进方向**：
   - 从URDF/MJCF中自动提取脚掌质量
   - 添加传动系统（减速器）的刚度与阻尼建模
   - 集成反演动力学控制器（PD/ADAPTIVE等）
   - 优化数值计算效率（缓存雅可比、M矩阵等）

## 相关文件位置

- [ankle_dyn_pin.py](anchor_dyn_pin.py) - 主文件（含完整可视化）
- [ankle_dynamics_demo.py](anchor_dynamics_demo.py) - 轻量演示脚本
- [ForwardKinematics.py](../../IK/ForwardKinematics.py) - IK模块参考
- [InverseKinematics.py](../../IK/InverseKinematics.py) - IK模块参考

## 参考资源

- Pinocchio官方文档：https://stack-of-tasks.github.io/pinocchio/
- 并联机构动力学理论：Craig, Introduction to Robotics

---

**最后更新**: 2026-03-06
