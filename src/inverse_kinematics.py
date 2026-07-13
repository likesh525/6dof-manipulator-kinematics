"""
inverse kinematics for the YARO-1105.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from spatialmath import SE3

from src.robot import robot, MDH_PARAMS, JOINT_LIMITS


# Geometry constants pulled directly from the MDH table in robot.py

D1 = MDH_PARAMS[0]["d"]   # 0.3035  (base height)
A3 = MDH_PARAMS[2]["a"]   # 0.49    (upper-arm length)
D4 = MDH_PARAMS[3]["d"]   # 0.4703  (forearm length)

# Fixed tool offset (frame_6 - end-effector), pure translation
TOOL_OFFSET = np.array([0.0, 0.000532, 0.1447])


@dataclass
class IKSolution:
    q: np.ndarray            # (6,) joint angles, radians
    position_branch: int     # 1 or 2 ("elbow" branch)
    wrist_flip: bool         # False = no-flip, True = flip
    singular: bool = False   # True if q5 ~ 0/pi (gimbal lock) was hit
    within_limits: bool = False


def wrist_center(T_target: SE3) -> np.ndarray:
    """Strip the fixed tool offset from the target pose to get the
    spherical-wrist center [xc, yc, zc]."""
    R = T_target.R
    p = T_target.t
    return p - R @ TOOL_OFFSET


def _solve_position(pc: np.ndarray, branch: int) -> Optional[np.ndarray]:
    """Solve q1, q2, q3 from the wrist center. Returns None if unreachable."""
    xc, yc, zc = pc

    q1 = np.arctan2(yc, xc)

    r = np.hypot(xc, yc)
    Z = zc - D1
    L = np.hypot(r, Z)
    theta_dashed = np.arctan2(r, Z)

    cos_E = (A3 ** 2 + D4 ** 2 - L ** 2) / (2 * A3 * D4)
    cos_S = (A3 ** 2 + L ** 2 - D4 ** 2) / (2 * A3 * L)

    if abs(cos_E) > 1.0 + 1e-9 or abs(cos_S) > 1.0 + 1e-9:
        return None  # target outside reachable workspace

    angle_E = np.arccos(np.clip(cos_E, -1.0, 1.0))
    angle_S = np.arccos(np.clip(cos_S, -1.0, 1.0))

    if branch == 1:
        q3 = (np.pi - angle_E)
        q2 = theta_dashed + angle_S
    else:
        q3 = -(np.pi - angle_E)
        q2 = theta_dashed - angle_S

    return np.array([q1, q2, q3])


def _R_0_3(q1: float, q2: float, q3: float) -> np.ndarray:
    """Build R_0^3 using the robot's own per-link MDH transforms
    (robot.links[i].A(q)), so offsets/convention exactly match robot.py."""
    T1 = robot.links[0].A(q1)
    T2 = robot.links[1].A(q2)
    T3 = robot.links[2].A(q3)
    T03 = T1 * T2 * T3
    return T03.R


def _solve_orientation(R03: np.ndarray, R_target: np.ndarray, flip: bool):
    """Decompose R_3^6 into q4, q5, q6. Returns (q4, q5, q6, singular)."""
    R36 = R03.T @ R_target

    c5 = -R36[1, 2]
    s5_mag = np.sqrt(max(R36[1, 0] ** 2 + R36[1, 1] ** 2, 0.0))
    s5 = -s5_mag if flip else s5_mag
    q5 = np.arctan2(s5, c5)

    singular = np.isclose(s5, 0.0, atol=1e-8)

    if singular:
        # Gimbal lock: Z3 and Z6 aligned, only q4 + q6 (or q4 - q6) is
        # determined. Lock q4 to 0 and solve q6 accordingly.
        q4 = 0.0
        q6 = np.arctan2(R36[0, 1], R36[0, 0]) - q4
        return q4, q5, q6, True

    if not flip:  # s5 > 0, "no-flip"
        q6 = np.arctan2(-R36[1, 1], R36[1, 0])
        q4 = np.arctan2(-R36[2, 2], -R36[0, 2])
    else:  # s5 < 0, "flip"
        q6 = np.arctan2(R36[1, 1], -R36[1, 0])
        q4 = np.arctan2(R36[2, 2], R36[0, 2])

    return q4, q5, q6, False


def _within_limits(q: np.ndarray) -> bool:
    return all(lo <= qi <= hi for qi, (lo, hi) in zip(q, JOINT_LIMITS))


def inverse_kinematics(T_target: SE3) -> List[IKSolution]:
    """
    Compute all closed-form IK solutions for a target end-effector pose.

    Parameters
    T_target : SE3
        Desired end-effector pose (position + orientation) in the base
        frame, INCLUDING the tool offset (i.e. the pose the tool tip
        should reach -- same convention as the given Cartesian waypoints).

    Returns
    list of IKSolution
        Up to 4 solutions (2 position branches x 2 orientation branches).
        Empty list if the target is outside the reachable workspace.
    """
    pc = wrist_center(T_target)
    R_target = T_target.R

    solutions: List[IKSolution] = []

    for pos_branch in (1, 2):
        pos = _solve_position(pc, pos_branch)
        if pos is None:
            continue
        q1, q2, q3 = pos
        R03 = _R_0_3(q1, q2, q3)

        for flip in (False, True):
            q4, q5, q6, singular = _solve_orientation(R03, R_target, flip)
            q = np.array([q1, q2, q3, q4, q5, q6])
            sol = IKSolution(
                q=q,
                position_branch=pos_branch,
                wrist_flip=flip,
                singular=singular,
                within_limits=_within_limits(q),
            )
            solutions.append(sol)
            if singular:
                # flip=True/False collapse to the same solution at a
                # singularity -- no point duplicating it
                break

    return solutions


def select_solution(
    solutions: List[IKSolution],
    reference_q: Optional[np.ndarray] = None,
    require_within_limits: bool = True,
) -> Optional[IKSolution]:
    """
    Pick one IK solution from the candidates.
    Prefers solutions within joint limits, then closest to `reference_q
    """
    candidates = solutions
    if require_within_limits:
        limited = [s for s in candidates if s.within_limits]
        if limited:
            candidates = limited
        # if none are within limits, fall back to all candidates rather than returning nothing -- caller can inspect .within_limits

    if not candidates:
        return None

    if reference_q is None:
        return candidates[0]

    def dist(sol: IKSolution) -> float:
        diff = np.arctan2(np.sin(sol.q - reference_q), np.cos(sol.q - reference_q))
        return float(np.linalg.norm(diff))

    return min(candidates, key=dist)


if __name__ == "__main__":
    # Quick manual check: round-trip FK -> IK -> FK on a non-trivial pose.
    from src.robot import QZ

    q_test = np.array([0.3, -0.4, 0.5, 0.2, -0.6, 0.1])
    T_test = robot.fkine(q_test)
    print("Target q:      ", np.round(q_test, 4))
    print("Target pose:\n", T_test)

    sols = inverse_kinematics(T_test)
    print(f"\nFound {len(sols)} IK solution(s):")
    for s in sols:
        T_check = robot.fkine(s.q)
        err = np.linalg.norm(T_check.t - T_test.t)
        print(
            f"  branch={s.position_branch} flip={s.wrist_flip} "
            f"singular={s.singular} within_limits={s.within_limits} "
            f"pos_err={err:.2e}  q={np.round(s.q, 4)}"
        )