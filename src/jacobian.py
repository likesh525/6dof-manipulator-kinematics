"""
Robot Jacobian
"""

import numpy as np

from src.robot import robot
from src.kinematics import forward_kinematics, link_frames


def geometric_jacobian(q) -> np.ndarray:
    """
    Compute the 6x6 geometric Jacobian (base frame) at configuration q.

    Returns
    J : ndarray, shape (6, 6)
        Rows 0-2: linear velocity Jacobian (J_v)
        Rows 3-5: angular velocity Jacobian (J_w)
        Columns : one per joint, 1..6
    """
    q = np.asarray(q, dtype=float)

    frames = link_frames(q)          # T_0^0 ... T_0^6 (7 frames, no tool offset)
    o_ee = forward_kinematics(q).t   # end-effector origin INCLUDING tool offset

    J = np.zeros((6, 6))
    for i in range(6):
        T_i = frames[i + 1]          # T_0^{i+1} -- frame i+1 itself (1-indexed joint i+1)
        z_i = T_i.R[:, 2]            # joint (i+1)'s rotation axis, expressed in base frame
        o_i = T_i.t

        Jv_i = np.cross(z_i, o_ee - o_i)
        Jw_i = z_i

        J[0:3, i] = Jv_i
        J[3:6, i] = Jw_i

    return J


def manipulability(q) -> float:
    """Yoshikawa manipulability index: sqrt(det(J . J^T)). Drops to 0 at
    a singularity; a convenient scalar proxy for "how close to singular"."""
    J = geometric_jacobian(q)
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def verify_against_toolbox(q, atol: float = 1e-8) -> bool:
    """
    Cross-check the hand-derived Jacobian against roboticstoolbox's
    own robot.jacob0(q) as a numerical verification
    """
    q = np.asarray(q, dtype=float)
    J_hand = geometric_jacobian(q)
    J_lib = robot.jacob0(q)
    diff = np.max(np.abs(J_hand - J_lib))
    ok = diff < atol
    print(f"  max|J_hand - J_lib| = {diff:.3e}  ->  {'MATCH' if ok else 'MISMATCH'}")
    return ok


if __name__ == "__main__":
    print("Verifying hand-derived Jacobian against roboticstoolbox.jacob0()\n")

    test_configs = {
        "home (q=0)": np.zeros(6),
        "random pose": np.array([0.3, -0.4, 0.5, 0.2, -0.6, 0.1]),
        "near wrist singularity (q5=0)": np.array([0.1, 0.2, 0.3, 0.4, 0.0, 0.6]),
    }

    all_ok = True
    for name, q in test_configs.items():
        print(f"{name}: q = {np.round(q, 3)}")
        ok = verify_against_toolbox(q)
        m = manipulability(q)
        print(f"  manipulability index = {m:.6f}"
              f"{'  <-- near-singular!' if m < 1e-4 else ''}")
        all_ok &= ok
        print()

    print("ALL CONFIGURATIONS MATCH" if all_ok else "MISMATCH DETECTED -- check derivation")