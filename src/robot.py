"""
Single source of truth for the YARO-1105 kinematic model [defining robot geometry].

Builds a roboticstoolbox `DHRobot` from the Modified Denavit-Hartenberg
(MDH / Craig convention) parameters derived by hand from the  URDF (models/robotics.urdf).

URDF file can be found here:https://github.com/Yardstick-Robotics/yaro_1105_description/tree/main/urdf

Every other module (kinematics, jacobian, dynamics, trajectory,
visualization) imports `robot` from here

MDH convention used:
    T_{i-1}^{i} = Rx(alpha_{i-1}) . Tx(a_{i-1}) . Rz(theta_i) . Tz(d_i)

See report/ for the full derivation, frame diagrams, and assumptions
(home position, X-axis convention, spherical-wrist reasoning).
"""

import numpy as np
from roboticstoolbox import DHRobot, RevoluteMDH
from spatialmath import SE3


# MDH parameter table (derived by hand, cross-checked against
# models/robot.urdf joint origins/axes)
#
#   i | alpha_{i-1} | a_{i-1} (m) | d_i (m) | theta_i offset
#   1 |     0       |    0        | 0.3035  |    180 deg
#   2 |    90       |    0        | 0.082   |     90 deg
#   3 |   180       |   0.49      | 0.082   |     90 deg
#   4 |    90       |    0        | 0.4703  |      0 deg
#   5 |    90       |    0        |  0      |    180 deg
#   6 |    90       |    0        |  0      |      0 deg


DEG = np.pi / 180.0

MDH_PARAMS = [
    # alpha_{i-1},   a_{i-1},   d_i,      theta offset
    dict(alpha=0.0,        a=0.0,   d=0.3035, offset=180 * DEG),  # joint 1
    dict(alpha=90 * DEG,   a=0.0,   d=0.082,  offset=90 * DEG),   # joint 2
    dict(alpha=180 * DEG,  a=0.49,  d=0.082,  offset=90 * DEG),   # joint 3
    dict(alpha=90 * DEG,   a=0.0,   d=0.4703, offset=0.0),        # joint 4
    dict(alpha=90 * DEG,   a=0.0,   d=0.0,    offset=180 * DEG),  # joint 5
    dict(alpha=90 * DEG,   a=0.0,   d=0.0,    offset=0.0),        # joint 6
]

# Joint limits (rad), pulled directly from models/robot.urdf <limit>
# tags, in the same joint_1 joint_6 order.
JOINT_LIMITS = [
    (-6.28318531, 6.28318531),   # joint 1
    (-6.28318531, 6.28318531),   # joint 2
    (-2.79252680, 2.79252680),   # joint 3
    (-6.28318531, 6.28318531),   # joint 4
    (-2.26892798, 2.26892798),   # joint 5
    (-6.28318531, 6.28318531),   # joint 6
]

# Fixed tool transform: joint-6 frame - end-effector
# Pure translation, identity rotation
TOOL_TRANSFORM = SE3(0.0, 0.000532, 0.1447)

def build_robot() -> DHRobot:
    """
    Construct the YARO-1105 DHRobot from the MDH table.
    Note: Dynamics/inertial parameters are omitted here by design and 
    are instead handled natively in src/dynamics.py.
    """
    links = []
    for p, lim in zip(MDH_PARAMS, JOINT_LIMITS):
        links.append(
            RevoluteMDH(
                alpha=p["alpha"],
                a=p["a"],
                d=p["d"],
                offset=p["offset"],
                qlim=lim,
            )
        )

    robot = DHRobot(
        links,
        name="YARO-1105",
        manufacturer="Yardstick Robotics",
        tool=TOOL_TRANSFORM,
        gravity=[0, 0, -9.81],
    )
    return robot

# Build once at import time
robot = build_robot()

# Home configuration
QZ = np.zeros(6)

if __name__ == "__main__":
    print(robot)
    print("\nJoint limits (rad):")
    for i, lim in enumerate(JOINT_LIMITS, start=1):
        print(f"  joint_{i}: {lim}")
    print("\nFK at home (q = 0):")
    print(robot.fkine(QZ))