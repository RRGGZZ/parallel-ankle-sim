# PNDbotics Ankle Simulation

A parallel ankle mechanism simulation repository for academic research, featuring:
- Theoretical geometric/Jacobian model analysis (workspace, velocity mapping, torque mapping, armature estimation)
- MuJoCo and Pinocchio dynamics verification scripts
- ADAM Pro robot model (MJCF/URDF + multi-format meshes)

## 1. Project Overview

This project studies the **kinematics, statics, and dynamics** characteristics of a dual-motor parallel ankle mechanism, with support for comparative validation and visualization in MuJoCo / Pinocchio environments.

## 2. Repository Structure

```
pndbotics-ankle-sim/
├── adam_pro/                     # Robot model and mesh assets (MJCF/URDF/STL/DAE/OBJ)
├── src/
│   └── pndbotics_ankle/          # Shared kinematics module
│       └── __init__.py           # CustomAnkleKinematics class
├── scripts/
│   ├── kinematics/               # Theoretical model and visualization scripts
│   │   ├── ankle_3d_viewer.py    # Interactive 3D visualization
│   │   ├── ankle_armature.py     # Armature (reflected inertia) calculation
│   │   ├── ankle_speed_torque.py  # Speed-torque curve mapping
│   │   ├── ankle_workspace.py    # Workspace analysis (IK/FK)
│   │   └── plot_ankle_data.py    # Kinematics data plotting
│   └── dynamics/                 # MuJoCo / Pinocchio dynamics scripts
│       ├── ankle_dynamics_headless.py    # Pinocchio without visualization
│       ├── ankle_dynamics_mujoco.py      # Theory vs MuJoCo comparison
│       ├── ankle_dynamics_pinocchio.py   # Pinocchio with Meshcat visualization
│       └── mujoco_model_viewer.py        # MuJoCo model viewer
├── docs/
│   ├── ankle_joint_speed_torque.png
│   └── dynamics_notes.md
└── README.md
```

## 3. Dependencies

Recommended: Python 3.10+

Core dependencies (install as needed):
- `numpy`
- `matplotlib`
- `scipy`
- `mujoco`
- `pin`
- `meshcat` (only for `ankle_dynamics_pinocchio.py` visualization)

Installation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib scipy mujoco pin meshcat
```

## 4. Quick Start

### 4.1 Theoretical Analysis

```bash
python scripts/kinematics/plot_ankle_data.py
python scripts/kinematics/ankle_workspace.py
python scripts/kinematics/ankle_armature.py
```

### 4.2 MuJoCo / Pinocchio Dynamics

```bash
python scripts/dynamics/mujoco_model_viewer.py
python scripts/dynamics/ankle_dynamics_mujoco.py
python scripts/dynamics/ankle_dynamics_headless.py
python scripts/dynamics/ankle_dynamics_pinocchio.py
```

> Note: All scripts automatically locate the repository root directory, no longer requiring local hardcoded `os.chdir(...)` paths.

## 5. Shared Module

The `CustomAnkleKinematics` class has been extracted to a shared module at `src/pndbotics_ankle/`. This eliminates code duplication across multiple scripts and provides a single source of truth for the kinematics model.

### Using the Shared Module

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pndbotics_ankle import CustomAnkleKinematics

robot = CustomAnkleKinematics()
q1, q2 = robot.IK(roll=0.1, pitch=0.05)  # Inverse kinematics
J_vel, J_tau = robot.calc_Jacobians(0.1, 0.05)  # Jacobians
```

### Key Methods

| Method | Description |
|--------|-------------|
| `IK(theta_x, theta_y)` | Inverse kinematics: motor angles from ankle pose |
| `FK(q1, q2)` | Forward kinematics: ankle pose from motor angles |
| `calc_Jacobians(tx, ty)` | Compute velocity and torque Jacobian matrices |
| `calc_R(tx, ty)` | Compute rotation matrix for ankle orientation |

## 6. Naming Conventions

- Directories: Semantic classification (`scripts/kinematics`, `scripts/dynamics`)
- Scripts: `snake_case` English naming
- Documentation: Centralized in `docs/`
- Models: Centralized in `adam_pro/`

## 7. Recommendations for Academic Use

If used for papers/reports, clearly state in the methods section:
- Source of theoretical model parameters (link lengths, installation angles)
- Error sources between MuJoCo and theoretical models (coordinate systems, parameter identification errors, numerical solution accuracy)
- Calibration process for joint workspace and torque boundaries

## 8. Future Extensions

- Add unified configuration file (e.g., `configs/ankle_geometry.yaml`)
- Introduce automated unit tests (IK consistency, Jacobian numerical verification, singularity detection)
- Add experimental data import/export specifications (CSV + plots)
