"""
Way point trajectory generation and verification.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
from spatialmath import SE3, UnitQuaternion
from roboticstoolbox import jtraj

from src.robot import robot, QZ
from src.inverse_kinematics import inverse_kinematics, select_solution


# The 5 Cartesian waypoints given 
# Quaternion convention: [x, y, z, w]

WAYPOINTS_RAW = [
    dict(t=[0.483491, 0.0, 0.569954], q_xyzw=[0, 1, 0, 0]),
    dict(t=[0.724633, 0.207038, 0.739836], q_xyzw=[0, 1, 0, 0]),
    dict(t=[0.724633, -0.207038, 0.739836], q_xyzw=[0, 1, 0, 0]),
    dict(t=[0.282146, -0.188098, 0.377512], q_xyzw=[0, 1, 0, 0]),
    dict(t=[0.282146, 0.188098, 0.377512], q_xyzw=[0, 1, 0, 0]),
]

SAMPLES_PER_SEGMENT = 50   # trajectory points between each pair of waypoints
SEGMENT_DURATION = 2.0     # seconds allotted to each inter-waypoint segment


def waypoint_to_SE3(wp: dict) -> SE3:
    x, y, z, w = wp["q_xyzw"]
    return SE3(wp["t"]) * UnitQuaternion(w, [x, y, z]).SE3()


@dataclass
class JointTrajectory:
    q: np.ndarray        # (N, 6) joint positions
    qd: np.ndarray        # (N, 6) joint velocities
    qdd: np.ndarray       # (N, 6) joint accelerations
    t: np.ndarray          # (N,) time stamps, seconds
    waypoint_q: List[np.ndarray] = field(default_factory=list)   # the 5 IK solutions
    waypoint_indices: List[int] = field(default_factory=list)    # index into t/q of each waypoint


def solve_waypoint_joint_configs(
    waypoints=WAYPOINTS_RAW, reference_q=QZ
) -> List[np.ndarray]:
    """Run IK on each waypoint, choosing the branch closest to the
    previous solution so the path is continuous in joint space."""
    q_list = []
    ref = reference_q
    for i, wp in enumerate(waypoints, start=1):
        T = waypoint_to_SE3(wp)
        sols = inverse_kinematics(T)
        if not sols:
            raise RuntimeError(f"Waypoint {i} is unreachable (no IK solution).")
        chosen = select_solution(sols, reference_q=ref, require_within_limits=True)
        if chosen is None:
            raise RuntimeError(f"Waypoint {i} has no solution within joint limits.")
        if not chosen.within_limits:
            raise RuntimeError(f"Waypoint {i}: best solution still violates joint limits.")
        q_list.append(chosen.q)
        ref = chosen.q
    return q_list


def build_trajectory(
    waypoints=WAYPOINTS_RAW,
    samples_per_segment: int = SAMPLES_PER_SEGMENT,
    segment_duration: float = SEGMENT_DURATION,
) -> JointTrajectory:
    """Build the full joint-space trajectory through all waypoints."""
    waypoint_q = solve_waypoint_joint_configs(waypoints)

    q_segments, qd_segments, qdd_segments, t_segments = [], [], [], []
    waypoint_indices = [0]
    t_offset = 0.0
    sample_count = 0

    for i in range(len(waypoint_q) - 1):
        q0, q1 = waypoint_q[i], waypoint_q[i + 1]

        # Pass an explicit real-time vector (not just a sample count) so
        # jtraj's returned qd/qdd are already correctly scaled to actual
        # seconds -- no manual rescaling needed (jtraj normalizes to
        # [0, N-1] if given a bare integer, which is NOT seconds).
        local_t = np.linspace(0, segment_duration, samples_per_segment)
        tg = jtraj(q0, q1, local_t)  # quintic, zero vel/acc at both ends

        seg_t = local_t + t_offset

        # avoid duplicating the shared boundary sample between segments
        if i > 0:
            q_segments.append(tg.q[1:])
            qd_segments.append(tg.qd[1:])
            qdd_segments.append(tg.qdd[1:])
            t_segments.append(seg_t[1:])
            sample_count += samples_per_segment - 1
        else:
            q_segments.append(tg.q)
            qd_segments.append(tg.qd)
            qdd_segments.append(tg.qdd)
            t_segments.append(seg_t)
            sample_count += samples_per_segment

        waypoint_indices.append(sample_count - 1)
        t_offset += segment_duration

    q = np.vstack(q_segments)
    qd = np.vstack(qd_segments)
    qdd = np.vstack(qdd_segments)
    t = np.concatenate(t_segments)

    return JointTrajectory(
        q=q, qd=qd, qdd=qdd, t=t,
        waypoint_q=waypoint_q, waypoint_indices=waypoint_indices,
    )


def verify_trajectory(traj: JointTrajectory, tol: float = 1e-6) -> None:
    """Round-trip check: FK at each recorded waypoint index should match
    the original Cartesian waypoint."""
    for wp_raw, idx in zip(WAYPOINTS_RAW, traj.waypoint_indices):
        T_target = waypoint_to_SE3(wp_raw)
        T_actual = robot.fkine(traj.q[idx])
        pos_err = np.linalg.norm(T_actual.t - T_target.t)
        rot_err = np.linalg.norm(T_actual.R - T_target.R)
        assert pos_err < tol, f"position error too large at waypoint idx {idx}: {pos_err}"
        assert rot_err < tol, f"orientation error too large at waypoint idx {idx}: {rot_err}"
    print(f"All {len(traj.waypoint_indices)} waypoints verified "
          f"(position & orientation error < {tol}).")


# Per-joint velocity limits from models/yaro_1105.urdf <limit velocity="..."/>
# tags (rad/s), joint_1..joint_6. Used to confirm the trajectory is
# dynamically feasible, not just within position limits.
VELOCITY_LIMITS = [4.18879020, 4.18879020, 8.37758041, 7.33038286, 7.33038286, 7.33038286]


def verify_velocity_feasibility(traj: JointTrajectory) -> None:
    """Confirm peak joint velocities along the trajectory stay within the
    URDF's own per-joint velocity limits"""
    peak_vel = np.max(np.abs(traj.qd), axis=0)
    print("\nPeak joint velocity vs. URDF limit:")
    for i, (pv, lim) in enumerate(zip(peak_vel, VELOCITY_LIMITS), start=1):
        pct = 100 * pv / lim
        flag = "  <-- EXCEEDS LIMIT" if pv > lim else ""
        print(f"  joint_{i}: {pv:.3f} rad/s  ({pct:.1f}% of {lim:.3f} rad/s limit){flag}")
    assert np.all(peak_vel <= VELOCITY_LIMITS), (
        "Trajectory exceeds URDF velocity limits -- increase SEGMENT_DURATION."
    )


def plot_joint_trajectory(
    traj: JointTrajectory,
    out_path: str = "figures/joint_trajectory.png",
) -> str:
    """
    visual presentation of the joint-space trajectory
    (position, velocity, acceleration per joint vs. time)
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    labels = [f"q{i}" for i in range(1, 7)]
    waypoint_times = traj.t[traj.waypoint_indices]

    data = [
        (np.rad2deg(traj.q), "Joint position (deg)"),
        (np.rad2deg(traj.qd), "Joint velocity (deg/s)"),
        (np.rad2deg(traj.qdd), "Joint acceleration (deg/s^2)"),
    ]

    for ax, (values, ylabel) in zip(axes, data):
        for j in range(6):
            ax.plot(traj.t, values[:, j], label=labels[j])
        for wt in waypoint_times:
            ax.axvline(wt, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    axes[0].legend(loc="upper right", ncol=6, fontsize=8)
    axes[0].set_title("Joint-space trajectory through the 5 Cartesian waypoints\n"
                       "(dashed lines mark waypoint times)")
    axes[-1].set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    traj = build_trajectory()
    verify_trajectory(traj)
    verify_velocity_feasibility(traj)

    print(f"\nTrajectory: {traj.q.shape[0]} samples, "
          f"{traj.t[-1]:.2f}s total duration")
    print(f"Waypoint sample indices: {traj.waypoint_indices}")

    print("\nJoint configs at each waypoint (deg):")
    for i, q in enumerate(traj.waypoint_q, start=1):
        print(f"  WP{i}: {np.round(np.rad2deg(q), 2)}")

  
    import os
    os.makedirs("results", exist_ok=True)
    np.savez(
        "results/trajectory.npz",
        q=traj.q, qd=traj.qd, qdd=traj.qdd, t=traj.t,
        waypoint_indices=traj.waypoint_indices,
    )
    print("\nSaved: results/trajectory.npz")

    fig_path = plot_joint_trajectory(traj)
    print(f"Saved: {fig_path}")