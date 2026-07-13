"""
joint torque computation via inverse dynamics.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

PAYLOAD_MASS = 5.0  # kg, per the task-7 brief


# URDF-derived kinematic + inertial parameters (per-link fixed transform
# xyz/rpy from parent joint frame, then Rz(q) about the joint's own axis).


LINKS = [
    dict(xyz=[0, 0, 0.194],        rpy=[0, 0, 0],           m=6.27832178, com=[0.00016873, 0.01322259, 0.09403804],
         I=[0.02522749, 0.01878474, 0.01919394, 0.00006762, -0.00002347, -0.00120463]),
    dict(xyz=[0, 0.082, 0.1095],   rpy=[-1.57079633, 0, 0], m=3.44377444, com=[0.00000032, -0.21005149, 0.05064607],
         I=[0.14692608, 0.00914487, 0.14712550, -0.00000411, 0.00000003, 0.00264987]),
    dict(xyz=[0, -0.49, 0],        rpy=[3.14159265, 0, 0],  m=5.08297766, com=[-0.00017574, 0.05747560, 0.06762882],
         I=[0.03908001, 0.01149368, 0.03378216, -0.00001075, 0.00002936, -0.00417208]),
    dict(xyz=[0, 0.1985, 0.082],   rpy=[-1.57079633, 0, 0], m=3.04194742, com=[0.00015525, -0.10080603, 0.19208715],
         I=[0.04745501, 0.03727700, 0.01438207, 0.00003061, -0.00003310, 0.01525868]),
    dict(xyz=[0, -0.0788, 0.2718], rpy=[-1.57079633, 0, 0], m=2.19975282, com=[-0.00019752, -0.00340357, 0.06611734],
         I=[0.00560217, 0.00335805, 0.00442283, 0.00001789, 0.00000797, 0.00006735]),
    dict(xyz=[0, -0.0642, 0.0788], rpy=[1.57079633, 0, 0],  m=0.71652484, com=[0.00026599, 0.00033585, 0.03727920],
         I=[0.00085473, 0.00085895, 0.00081969, -0.00001528, 0.00000342, -0.00001025]),
]

# Fixed tool point, link_6 - tool point (ee_frame). Also where the 5 kg
# payload is taken to be attached ("at the tool flange" == the tool point
# used to plan the Cartesian waypoints in trajectory.py).
TOOL_XYZ = np.array([-0.00000042, 0.00053249, 0.0805])
TOOL_RPY = [0.0, 0.0, 0.0]

G = 9.81
ZHAT = np.array([0.0, 0.0, 1.0])


def rpy_to_R(rpy):
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz_ = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz_ @ Ry @ Rx


def Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def Imat(I):
    ixx, iyy, izz, ixy, ixz, iyz = I
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])


def _build_chain(q, qd, qdd, payload_mass):
    """
    Construct the 7-frame kinematic and inertial chain (6 joints + 1 tool/payload frame).
    """
    xyzs = [np.array(L["xyz"], dtype=float) for L in LINKS] + [TOOL_XYZ]
    Rfix = [rpy_to_R(L["rpy"]) for L in LINKS] + [rpy_to_R(TOOL_RPY)]
    qs = list(q) + [0.0]
    qds = list(qd) + [0.0]
    qdds = list(qdd) + [0.0]
    masses = [L["m"] for L in LINKS] + [payload_mass]
    coms = [np.array(L["com"], dtype=float) for L in LINKS] + [np.zeros(3)]
    Imats = [Imat(L["I"]) for L in LINKS] + [np.zeros((3, 3))]
    R = [Rfix[k] @ Rz(qs[k]) for k in range(7)]
    return R, xyzs, qds, qdds, masses, coms, Imats


def rnea(q, qd, qdd, payload_mass: float = 0.0) -> np.ndarray:
    """Recursive Newton-Euler inverse dynamics for one configuration.
    Returns the 6 actuated-joint torques, N.m."""
    R, p, qds, qdds, masses, coms, Imats = _build_chain(q, qd, qdd, payload_mass)
    N = 7  # 6 real joints + 1 fixed tool/payload frame

    omega = [np.zeros(3)]
    omega_dot = [np.zeros(3)]
    v_dot = [np.array([0.0, 0.0, G])]  # base pseudo-acceleration encodes gravity

    for k in range(N):
        Rk = R[k]
        om_prev, omd_prev, vd_prev = omega[-1], omega_dot[-1], v_dot[-1]
        om_new = Rk.T @ om_prev + qds[k] * ZHAT
        omd_new = Rk.T @ omd_prev + np.cross(Rk.T @ om_prev, qds[k] * ZHAT) + qdds[k] * ZHAT
        vd_new = Rk.T @ (vd_prev + np.cross(omd_prev, p[k]) + np.cross(om_prev, np.cross(om_prev, p[k])))
        omega.append(om_new)
        omega_dot.append(omd_new)
        v_dot.append(vd_new)

    F = [None] * (N + 1)
    Nm = [None] * (N + 1)
    for k in range(1, N + 1):
        m_k, c_k, Ic_k = masses[k - 1], coms[k - 1], Imats[k - 1]
        om_k, omd_k, vd_k = omega[k], omega_dot[k], v_dot[k]
        vdc = vd_k + np.cross(omd_k, c_k) + np.cross(om_k, np.cross(om_k, c_k))
        F[k] = m_k * vdc
        Nm[k] = Ic_k @ omd_k + np.cross(om_k, Ic_k @ om_k)

    f_next, n_next = np.zeros(3), np.zeros(3)
    tau = np.zeros(N)
    for k in range(N, 0, -1):
        if k < N:
            f_beyond = R[k] @ f_next
            n_beyond = R[k] @ n_next
            p_offset = p[k]
        else:
            f_beyond = np.zeros(3)
            n_beyond = np.zeros(3)
            p_offset = np.zeros(3)

        c_k = coms[k - 1]
        f_k = F[k] + f_beyond
        n_k = Nm[k] + n_beyond + np.cross(c_k, F[k]) + np.cross(p_offset, f_beyond)
        tau[k - 1] = n_k @ ZHAT

        f_next, n_next = f_k, n_k

    return tau[:6]


def forward_link_frames(q):
    """World-frame (R_i, O_i) for i=1..6 (Used for energy-based verification)"""
    R_world = [np.eye(3)]
    O_world = [np.zeros(3)]
    for k in range(6):
        L = LINKS[k]
        Rk = rpy_to_R(L["rpy"]) @ Rz(q[k])
        R_prev, O_prev = R_world[-1], O_world[-1]
        R_world.append(R_prev @ Rk)
        O_world.append(O_prev + R_prev @ np.array(L["xyz"]))
    return R_world, O_world


def system_energy(q, qd, payload_mass: float = 0.0):
    """Total mechanical energy (KE + PE) of the arm (+ payload) for a given state, used for testing"""
    R_world, O_world = forward_link_frames(q)

    omega_w = [np.zeros(3)]
    for k in range(6):
        z_axis_world = R_world[k + 1] @ ZHAT
        omega_w.append(omega_w[-1] + qd[k] * z_axis_world)

    KE = 0.0
    PE = 0.0
    for k in range(1, 7):
        L = LINKS[k - 1]
        m = L["m"]
        c_local = np.array(L["com"])
        Ic_local = Imat(L["I"])
        R_k, O_k = R_world[k], O_world[k]
        c_world = O_k + R_k @ c_local

        v_com = np.zeros(3)
        for j in range(1, k + 1):
            z_j = R_world[j] @ ZHAT
            O_j = O_world[j]
            v_com += qd[j - 1] * np.cross(z_j, c_world - O_j)

        om = omega_w[k]
        Ic_world = R_k @ Ic_local @ R_k.T
        KE += 0.5 * m * (v_com @ v_com) + 0.5 * (om @ (Ic_world @ om))
        PE += m * G * c_world[2]

    if payload_mass != 0.0:
        Rk = R_world[6]
        c_world = O_world[6] + Rk @ TOOL_XYZ
        v_com = np.zeros(3)
        for j in range(1, 7):
            z_j = R_world[j] @ ZHAT
            O_j = O_world[j]
            v_com += qd[j - 1] * np.cross(z_j, c_world - O_j)
        KE += 0.5 * payload_mass * (v_com @ v_com)
        PE += payload_mass * G * c_world[2]

    return KE, PE


@dataclass
class TorqueProfile:
    """Joint torque time history for one trajectory / payload case."""
    t: np.ndarray          # (N,)   time stamps, s
    tau: np.ndarray         # (N, 6) joint torques, N.m
    payload_mass: float     # kg
    label: str              # human-readable case name


def compute_joint_torques(
    q: np.ndarray, qd: np.ndarray, qdd: np.ndarray,
    payload_mass: float = 0.0,
) -> np.ndarray:
    """
    Joint torques tau(t) for a whole (N, 6) trajectory
    """
    q, qd, qdd = np.asarray(q), np.asarray(qd), np.asarray(qdd)
    N = q.shape[0]
    tau = np.zeros((N, 6))
    for i in range(N):
        tau[i] = rnea(q[i], qd[i], qdd[i], payload_mass=payload_mass)
    return tau


def compute_both_cases(traj) -> "tuple[TorqueProfile, TorqueProfile]":
    """
    Compute torque profilesfor Case 1 (no payload) and Case 2 (5 kg payload at the tool point).
    """
    tau_no_payload = compute_joint_torques(traj.q, traj.qd, traj.qdd, payload_mass=0.0)
    case1 = TorqueProfile(t=traj.t, tau=tau_no_payload, payload_mass=0.0, label="Case 1: no payload")

    tau_with_payload = compute_joint_torques(traj.q, traj.qd, traj.qdd, payload_mass=PAYLOAD_MASS)
    case2 = TorqueProfile(
        t=traj.t, tau=tau_with_payload, payload_mass=PAYLOAD_MASS,
        label=f"Case 2: {PAYLOAD_MASS:g} kg payload",
    )
    return case1, case2


def summarize(profile: TorqueProfile) -> np.ndarray:
    """Peak absolute torque per joint, shape (6,), N.m."""
    return np.max(np.abs(profile.tau), axis=0)


if __name__ == "__main__":
    from src.trajectory import build_trajectory, verify_trajectory

    traj = build_trajectory()
    verify_trajectory(traj)

    case1, case2 = compute_both_cases(traj)

    print(f"\n{case1.label}")
    print("  peak |tau| per joint (N.m):", np.round(summarize(case1), 3))

    print(f"\n{case2.label}")
    print("  peak |tau| per joint (N.m):", np.round(summarize(case2), 3))

    delta = summarize(case2) - summarize(case1)
    print("\nPeak torque increase due to payload (N.m):", np.round(delta, 3))

    import os
    os.makedirs("results", exist_ok=True)
    np.savez(
        "results/torques.npz",
        t=traj.t,
        tau_no_payload=case1.tau,
        tau_with_payload=case2.tau,
        payload_mass=PAYLOAD_MASS,
    )
    print("\nSaved: results/torques.npz")