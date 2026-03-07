# PnDBotics Ankle Simulation

一个面向学术研究的并联踝关节仿真仓库，包含：
- 理论几何/雅可比模型分析（workspace、速度映射、力矩映射、armature 估计）
- MuJoCo 与 Pinocchio 的动力学验证脚本
- ADAM Pro 机器人模型（MJCF/URDF + 多格式网格）

## 1. 项目目标

本项目用于研究双电机并联踝关节在 **运动学、静力学、动力学** 方面的特性，并支持在 MuJoCo / Pinocchio 环境下进行一致性对比与可视化验证。

## 2. 仓库结构

```text
pndbotics-ankle-sim/
├── adam_pro/                     # 机器人模型与网格资源（MJCF/URDF/STL/DAE/OBJ）
├── scripts/
│   ├── kinematics/               # 理论模型与可视化分析脚本
│   │   ├── ankle_3d_viewer.py
│   │   ├── ankle_armature.py
│   │   ├── ankle_speed_torque.py
│   │   ├── ankle_workspace.py
│   │   └── plot_ankle_data.py
│   └── dynamics/                 # MuJoCo / Pinocchio 动力学与验证脚本
│       ├── ankle_dynamics_headless.py
│       ├── ankle_dynamics_mujoco.py
│       ├── ankle_dynamics_pinocchio.py
│       └── mujoco_model_viewer.py
├── docs/
│   ├── ankle_joint_speed_torque.png
│   └── dynamics_notes.md
└── README.md
```

## 3. 环境依赖

建议使用 Python 3.10+。

核心依赖（按需安装）：
- `numpy`
- `matplotlib`
- `scipy`
- `mujoco`
- `pin`
- `meshcat`（仅 `ankle_dynamics_pinocchio.py` 可视化需要）

示例：

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib scipy mujoco pin meshcat
```

## 4. 快速开始

### 4.1 理论分析

```bash
python scripts/kinematics/plot_ankle_data.py
python scripts/kinematics/ankle_workspace.py
python scripts/kinematics/ankle_armature.py
```

### 4.2 MuJoCo / Pinocchio 动力学

```bash
python scripts/dynamics/mujoco_model_viewer.py
python scripts/dynamics/ankle_dynamics_mujoco.py
python scripts/dynamics/ankle_dynamics_headless.py
python scripts/dynamics/ankle_dynamics_pinocchio.py
```

> 说明：动力学脚本已改为自动定位仓库根目录，不再依赖本地硬编码 `os.chdir(...)` 路径。

## 5. 命名与组织规范（本次整理后）

- 目录采用语义分类：`scripts/kinematics` 与 `scripts/dynamics`
- 脚本文件统一使用 `snake_case` 英文命名
- 文档资源集中于 `docs/`
- 模型资产集中于 `adam_pro/`

## 6. 学术使用建议

若用于论文/报告，建议在实验章节明确：
- 理论模型参数来源（连杆长度、安装角）
- MuJoCo 与理论模型的误差来源（坐标系、参数识别误差、数值解精度）
- 关节工作空间与扭矩边界的标定流程

## 7. 后续可扩展方向

- 将 `CustomAnkleKinematics` 抽取为可复用 Python 包（减少多脚本重复）
- 增加统一配置文件（如 `configs/ankle_geometry.yaml`）
- 引入自动化单元测试（IK 一致性、雅可比数值校验、奇异点检测）
- 增加实验数据导入/导出规范（CSV + plots）
