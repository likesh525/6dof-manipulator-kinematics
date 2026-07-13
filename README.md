# YARO-1105 6-DOF Manipulator Kinematics, Trajectory, Visualization, and Dynamics


![Trajectory Animation](videos/trajectory.gif)

A from-scratch Python pipeline for a 6-DOF manipulator, built using **Peter Corke's Robotics Toolbox**, covering:

- MDH-based robot modeling
- Forward and inverse kinematics
- Jacobian derivation and verification
- Cartesian-to-joint trajectory generation
- Robot visualization
- Inverse dynamics (Recursive Newton-Euler) torque computation, with and without payload




---

# Overview

The project follows a reproducible robotics pipeline:

1. Build the robot model from the URDF
2. Derive and verify forward kinematics
3. Solve analytical inverse kinematics
4. Derive and verify the Jacobian
5. Generate a feasible joint-space trajectory through the required Cartesian waypoints
6. Visualize the robot motion
7. Compute joint torques with and without a **5 kg payload**

Each module includes its own verification checks (see the summary table below).

---

# Repository Structure

```text
project/
├── README.md
├── requirements.txt
├── requirements-lock.txt
├── report.pdf
├── figures/
│   ├── joint_trajectory.png
│   ├── joint_torques.png
│   └── waypoint_01.png ...
├── models/
│   └── robot.urdf
├── results/
│   ├── torques.npz
│   └── trajectory.npz
├── scripts/
│   └── visualize_robot_trace.py
├── src/
│   ├── __init__.py
│   ├── robot.py
│   ├── kinematics.py
│   ├── inverse_kinematics.py
│   ├── jacobian.py
│   ├── trajectory.py
│   ├── dynamics.py
│   ├── visualization.py
│   └── plot_torque.py
├── tests/
│   └── test_dynamics.py
└── videos/
    └── trajectory.gif
```

---

# Setup

Use **Python 3.10–3.12**.

## 1. Create a virtual environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

### Important dependency versions

- **matplotlib 3.8.4**
  - Kept below 3.9 for Robotics Toolbox PyPlot compatibility.

- **scipy 1.15.3**
  - Kept below 1.16 for stable Python 3.10 compatibility.

### Exact reproducibility (optional)

`requirements.txt` lists direct dependencies; `requirements-lock.txt` pins every package (including transitive ones) to the exact versions this project was tested against. If `requirements.txt` doesn't reproduce the pipeline cleanly on your system, install from the lock file instead:

```bash
pip install -r requirements-lock.txt
```
---

# Quickstart

Run the complete pipeline with:

```bash
python visualize_robot_trace.py
```

The pipeline performs:

1. Robot model creation
2. Forward kinematics
3. Inverse kinematics verification
4. Jacobian verification
5. Workspace trajectory generation
6. Joint torque computation
7. Waypoint visualization
8. Optional trajectory animation

Generated outputs are written to:

- `results/`
- `figures/`
- `videos/`

The animation step is the slowest part of the pipeline.

## Fast mode

Skip only GIF generation:

```bash
python visualize_robot_trace.py --fast
```

---

# Running Individual Modules

Each stage can also be run independently.

```bash
python -m src.robot
python -m src.kinematics
python -m src.inverse_kinematics
python -m src.jacobian
python -m src.trajectory
python -m src.visualization
python -m src.dynamics
python -m src.plot_torque
```

---

# Pipeline Summary

| Stage | Description | File | Verification |
|------|-------------|------|--------------|
| **Robot modeling** | MDH derivation from the URDF | `src/robot.py` | Hand-derived MDH parameters are constructed from the URDF. Prints robot model and joint limits. |
| **Forward kinematics** | | `src/kinematics.py` | Uses `DHRobot.fkine()` and evaluates FK at home and sample configurations. |
| **Inverse kinematics** | Closed-form, spherical wrist | `src/inverse_kinematics.py` | Returns up to four solutions. FK/IK round-trip position error < `1e-6 m`. |
| **Jacobian** | | `src/jacobian.py` | Hand-derived Jacobian compared with `robot.jacob0()`, `atol = 1e-8`. |
| **Workspace trajectory** | | `src/trajectory.py` | Waypoints revalidated via FK; position/orientation error < `1e-6`; joint velocities within URDF limits. |
| **Visualization** | | `src/visualization.py` | Waypoint snapshots and stick-figure animation. |
| **Joint torques** | Custom RNEA | `src/dynamics.py`, `tests/test_dynamics.py` | Verified via static gravity check and energy/power consistency (< `1e-8`, < `1e-9`). |

--

# Outputs

Running the full pipeline generates:

```
results/
├── trajectory.npz
└── torques.npz

figures/
├── joint_trajectory.png
├── joint_torques.png
├── waypoint_01.png
├── waypoint_02.png
├── waypoint_03.png
├── waypoint_04.png
└── waypoint_05.png

videos/
└── trajectory.gif
```

## Output Description

### `results/trajectory.npz`

Contains:

- Joint positions
- Joint velocities
- Joint accelerations
- Time vector
- Waypoint indices

### `results/torques.npz`

Contains:

- Joint torque profiles without payload
- Joint torque profiles with a **5 kg payload**

### `figures/joint_trajectory.png`

Plots of

- Joint positions
- Joint velocities
- Joint accelerations

versus time.

### `figures/joint_torques.png`

Comparison of joint torque profiles for both payload cases.

### `figures/waypoint_01.png` → `waypoint_05.png`

Robot snapshots at each Cartesian waypoint.

### `videos/trajectory.gif`

Animated stick-figure visualization of the complete motion.

---

# Key Assumptions

- Home configuration is **q = 0** for all six joints.
- Robot geometry uses a hand-derived **Modified Denavit–Hartenberg (MDH)** model.
- The tool transform from Joint 6 to the end-effector is a fixed translation.
- Dynamic parameters are modeled separately from the kinematic chain.
- The trajectory follows five predefined Cartesian waypoints.
- Each trajectory segment lasts **2.0 seconds**.
- Each segment contains **50 samples**.
- Velocities and accelerations are computed using an explicit time vector.
- The IK solver selects the solution closest to the previous configuration to maintain trajectory continuity.
- URDF joint velocity limits are enforced as the trajectory feasibility criterion.

---

# Dependencies

- Python 3.10–3.12
- roboticstoolbox-python
- spatialmath-python
- numpy
- scipy
- matplotlib

Install all required packages with:

```bash
pip install -r requirements.txt
```

---
