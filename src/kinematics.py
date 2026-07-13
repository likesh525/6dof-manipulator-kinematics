"""

Forward kinematics for the YARO-1105.

Forward kinematics is constructed by chaining the per-link homogeneous
transforms from the base frame to the end-effector:

    T_6^0 = T_1^0 . T_2^1 . T_3^2 . T_4^3 . T_5^4 . T_6^5

calculated via roboticstoolbox's `DHRobot.fkine()` 

Each T_i^{i-1} is built from that link's MDH parameters
(alpha_{i-1}, a_{i-1}, d_i, theta_i):

    T_i^{i-1} = Rx(alpha_{i-1}) . Tx(a_{i-1}) . Rz(theta_i) . Tz(d_i)

The resulting T_6^0 fully defines the end-effector pose in the base
frame:
    - top-right 3x1 block  -> Cartesian position (x, y, z)
    - top-left  3x3 block  -> orientation (rotation matrix / direction
                               cosines; convertible to Euler angles or a
                               quaternion)
"""

import numpy as np
from spatialmath import SE3

from src.robot import robot, QZ


def forward_kinematics(q) -> SE3:
    """
    Compute the end-effector pose for a given joint configuration.

    Parameters
    q : array-like, shape (6,)
        Joint angles in radians
        
    Returns
    SE3
        End-effector pose (tool tip) in the base frame, including the
        fixed tool transform defined in robot.py.
    """
    q = np.asarray(q, dtype=float)
    return robot.fkine(q)


def position(q) -> np.ndarray:
    """Cartesian position (x, y, z) of the tool tip, shape (3,)."""
    return forward_kinematics(q).t


def orientation(q) -> np.ndarray:
    """3x3 rotation matrix of the tool tip orientation."""
    return forward_kinematics(q).R


def link_frames(q) -> list:
    """
    All intermediate frames T_0^0 ... T_6^0 (base to each joint, in
    order), useful for visualization (stick-figure plotting) and for
    Jacobian derivations. Does NOT include the final fixed tool offset.

    Returns
    list of SE3, length 7 (base frame + 6 joint frames)
    """
    q = np.asarray(q, dtype=float)
    frames = [SE3()]  # T_0^0 = identity (base frame)
    T = SE3()
    for link, qi in zip(robot.links, q):
        T = T * link.A(qi)
        frames.append(T)
    return frames


if __name__ == "__main__":
    print("FK at home (q = 0):")
    print(forward_kinematics(QZ))

    print("\nFK at a sample configuration:")
    q_sample = np.deg2rad([10, -20, 30, 0, 45, 0])
    T = forward_kinematics(q_sample)
    print(T)
    print("position:", np.round(position(q_sample), 4))