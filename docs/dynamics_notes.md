# 脚踝并联机构动力学说明

本文档对应 `scripts/dynamics/` 下的脚本整理说明。

## 功能概述

当前动力学相关脚本包含：

- `ankle_dynamics_pinocchio.py`：Pinocchio + Meshcat 可视化版
- `ankle_dynamics_headless.py`：Pinocchio 纯数值/无可视化版
- `ankle_dynamics_mujoco.py`：理论模型 vs MuJoCo 独立对比验证
- `mujoco_model_viewer.py`：快速模型查看器

## 已包含的核心能力

1. 数值雅可比估计（用于电机-脚踝映射分析）
2. 电机扭矩到末端广义力的映射分析
3. 质量矩阵、重力补偿与逆动力学计算
4. 基于 MuJoCo 的几何独立求解与误差对比

## 运行示例

```bash
python scripts/dynamics/ankle_dynamics_headless.py
```

## 备注

- 脚本已统一改为相对仓库根目录定位模型文件，不再依赖绝对路径。
- 若要复现实验，请确认本地已安装 MuJoCo / Pinocchio 相关依赖。
