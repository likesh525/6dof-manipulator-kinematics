"""
Single command entry point for the YARO-1105 project. This script runs the complete pipline

Clone the repo, install dependencies and run this script to generate all results and figures for the project report.

usage
    
    python3 scripts/visualize_robot_trace.py          # Full run, including animation 
    
    python scripts/visualize_robot_trace.py [--fast]  # Skip the slow trajectory animation GIF, everything else runs

"""

import argparse
import os
import sys
import time

from src.visualization import snapshot_waypoints


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)  

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full YARO-1105 pipeline (Tasks 1-7).")
    parser.add_argument("--fast", action="store_true",
                         help="skip the slow trajectory animation GIF (Task 6 waypoint "
                              "snapshots still run)")
    args = parser.parse_args()

    t_start = time.time()
    generated_files = []

    # Task 1: MDH model 
    section("TASK 1-2: Robot model + forward kinematics")
    from src.robot import robot, QZ
    print(robot)
    T_home = robot.fkine(QZ)
    print("\nFK at home (q = 0):")
    print(T_home)

    # Task 3: Inverse kinematics
    section("TASK 3: Inverse kinematics (round-trip verification)")
    from src.inverse_kinematics import inverse_kinematics, select_solution

    q_test = np.array([0.3, -0.4, 0.5, 0.2, -0.6, 0.1])
    T_test = robot.fkine(q_test)
    sols = inverse_kinematics(T_test)
    chosen = select_solution(sols, reference_q=q_test)
    err = np.linalg.norm(robot.fkine(chosen.q).t - T_test.t)
    print(f"Random-pose round trip: found {len(sols)} solutions, "
          f"chosen solution position error = {err:.2e} m")
    assert err < 1e-6, "IK round-trip check failed"

    # Task 4: Jacobian 
    section("TASK 4: Jacobian (cross-checked against roboticstoolbox)")
    from src.jacobian import geometric_jacobian, verify_against_toolbox, manipulability

    ok = verify_against_toolbox(q_test)
    assert ok, "Jacobian verification against roboticstoolbox failed"
    print(f"Manipulability at home (q=0): {manipulability(QZ):.6f} "
          f"(near-zero is expected -- home is a fully-outstretched pose)")

    # Task 5: Trajectory 
    section("TASK 5: Workspace trajectory (5 waypoints -> joint-space path)")
    from src.trajectory import (
        build_trajectory, verify_trajectory, verify_velocity_feasibility,
        plot_joint_trajectory,
    )

    traj = build_trajectory()
    verify_trajectory(traj)
    verify_velocity_feasibility(traj)

    os.makedirs("results", exist_ok=True)
    np.savez("results/trajectory.npz", q=traj.q, qd=traj.qd, qdd=traj.qdd,
              t=traj.t, waypoint_indices=traj.waypoint_indices)
    generated_files.append("results/trajectory.npz")

    fig_path = plot_joint_trajectory(traj)
    generated_files.append(fig_path)

    # Task 7: Dynamics / torques 
    section("TASK 7: Joint torques (no payload vs. 5 kg payload)")
    from src.dynamics import compute_both_cases, summarize

    case1, case2 = compute_both_cases(traj)
    print(f"{case1.label}: peak |tau| per joint (N.m) = {np.round(summarize(case1), 3)}")
    print(f"{case2.label}: peak |tau| per joint (N.m) = {np.round(summarize(case2), 3)}")

    np.savez("results/torques.npz", t=traj.t, tau_no_payload=case1.tau,
              tau_with_payload=case2.tau, payload_mass=case2.payload_mass)
    generated_files.append("results/torques.npz")

    #  Task 6: Visualization 
    section("TASK 6: Visualization")
    from src.visualization import animate_trajectory, snapshot_waypoints, compute_axis_limits

    

    print("Rendering per-waypoint snapshots...")
    limits = compute_axis_limits(traj.q)
    snap_paths = snapshot_waypoints(traj.waypoint_q, limits=limits)
    generated_files.extend(snap_paths)

    if args.fast:
        print("(--fast: skipping trajectory animation GIF)")
    else:
        print(f"Rendering {traj.q.shape[0]}-frame trajectory animation "
              f"(this takes a little while)...")
        gif_path = animate_trajectory(traj.q, dt=0.03, limits=limits)
        generated_files.append(gif_path)

    # Summary 
    section("DONE")
    elapsed = time.time() - t_start
    print(f"Full pipeline completed in {elapsed:.1f}s. Files generated:")
    for f in generated_files:
        exists = "OK" if os.path.exists(f) else "MISSING"
        print(f"  [{exists}] {f}")


if __name__ == "__main__":
    main()