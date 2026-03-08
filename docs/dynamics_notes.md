# Dynamics Documentation

This document describes the dynamics-related scripts in `scripts/dynamics/`.

## Overview

The dynamics scripts provide comprehensive analysis of the parallel ankle mechanism using both theoretical models and physics engine simulations.

## Scripts

### 1. ankle_dynamics_mujoco.py

Compares theoretical kinematics/dynamics against MuJoCo physics engine.

**Features:**
- Analytical IK (theory side)
- Geometric solver from MuJoCo site positions (independent)
- Inverse dynamics comparison via mj_inverse
- Error analysis between theory and simulation

**Usage:**
```bash
python scripts/dynamics/ankle_dynamics_mujoco.py
```

### 2. ankle_dynamics_pinocchio.py

Pinocchio-based forward/inverse dynamics with Meshcat visualization.

**Features:**
- Forward kinematics via constraint solving
- Numerical Jacobian computation
- Mass matrix calculation
- Coriolis and gravity compensation
- Real-time tendon visualization

**Usage:**
```bash
python scripts/dynamics/ankle_dynamics_pinocchio.py
```

### 3. ankle_dynamics_headless.py

Lightweight Pinocchio dynamics without visualization.

**Features:**
- Same dynamics computation as pinocchio version
- No external visualization dependencies
- Faster execution for batch processing

**Usage:**
```bash
python scripts/dynamics/ankle_dynamics_headless.py
```

### 4. mujoco_model_viewer.py

Quick MuJoCo model viewer.

**Usage:**
```bash
python scripts/dynamics/mujoco_model_viewer.py
```

## Core Capabilities

1. **Numerical Jacobian Estimation**: Motor-to-ankle endpoint mapping analysis
2. **Torque Mapping**: Motor torque to end-effector force/torque transformation
3. **Mass Matrix Computation**: Motor space inertia calculation
4. **Gravity Compensation**: Inverse dynamics with gravity
5. **MuJoCo Geometric Solver**: Independent verification using MJCF geometry

## Installation

All scripts automatically locate the repository root. Ensure MuJoCo/Pinocchio dependencies are installed:

```bash
pip install mujoco pin meshcat
```

## Notes

- Scripts use relative paths to locate model files
- Theory side uses mm coordinates
- MuJoCo side uses m coordinates
- Error reveals modeling parameter differences
