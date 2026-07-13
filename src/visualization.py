"""
Robot visualization : (stick-figure) via roboticstoolbox

"""

import os

import matplotlib
matplotlib.use("Agg")  # headless-safe; no display required to render/save

import numpy as np

from src.robot import robot
from src.trajectory import build_trajectory, WAYPOINTS_RAW

VIDEOS_DIR = "videos"
FIGURES_DIR = "figures"


def compute_axis_limits(q_all: np.ndarray, margin: float = 0.15) -> list:
    """
    Compute a fixed [xmin,xmax,ymin,ymax,zmin,zmax] box covering every
    link origin across the given set of joint configurations 
    """
    q_all = np.atleast_2d(q_all)
    all_pts = []
    for q in q_all:
        Ts = robot.fkine_all(q)      # SE3 for every link frame at this q
        pts = np.array([T.t for T in Ts])
        all_pts.append(pts)
    all_pts = np.vstack(all_pts)     # (n_frames * n_links, 3)

    mins = all_pts.min(axis=0) - margin
    maxs = all_pts.max(axis=0) + margin
    return [mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]]


def animate_trajectory(
    q_traj: np.ndarray,
    dt: float = 0.05,
    out_path: str = os.path.join(VIDEOS_DIR, "trajectory.gif"),
    limits: list = None,
) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    env = robot.plot(
        q_traj,
        backend="pyplot",
        dt=dt,
        movie=out_path,
        block=False,
        eeframe=True,
        jointaxes=True,
        shadow=True,
        limits=limits,
        name="YARO-1105 trajectory animation",
    )
    import matplotlib.pyplot as plt
    plt.close("all")  # don't leave this figure open for later calls (e.g. snapshot_waypoints)
    return out_path


def snapshot_waypoints(
    waypoint_q: list,
    out_dir: str = FIGURES_DIR,
    limits: list = None,
) -> list:
    """
    Render and save a snapshot of the robot at each waypoint configuration.
    """
    import matplotlib.pyplot as plt

    plt.close("all")

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, q in enumerate(waypoint_q, start=1):
        fig_path = os.path.join(out_dir, f"waypoint_{i:02d}.png")
        env = robot.plot(
            np.asarray(q), backend="pyplot", block=False,
            eeframe=True, jointaxes=False, shadow=True,
            limits=limits,
            name=f"YARO-1105 waypoint {i}",
        )
        fig = plt.gcf()
        assert len(fig.axes) == 1, (
            f"waypoint {i}: expected exactly 1 axes, found {len(fig.axes)} "
            "-- figure state was not clean."
        )
        fig.suptitle(f"Waypoint {i}")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        paths.append(fig_path)

    plt.close("all")
    return paths


if __name__ == "__main__":
    traj = build_trajectory()

    # One shared bounding box, computed from the FULL trajectory, so the
    # animation and every waypoint snapshot use the same (uncropped) view.
    limits = compute_axis_limits(traj.q)
    print(f"Shared axis limits: {np.round(limits, 3)}")

    print(f"\nRendering {traj.q.shape[0]}-frame trajectory animation "
          f"(this can take a little while)...")
    gif_path = animate_trajectory(traj.q, dt=0.03, limits=limits)
    print(f"Saved animation: {gif_path}")

    print("\nRendering per-waypoint snapshots...")
    fig_paths = snapshot_waypoints(traj.waypoint_q, limits=limits)
    for p in fig_paths:
        print(f"  {p}")

    print(f"\nDone. {len(WAYPOINTS_RAW)} waypoints, "
          f"{traj.q.shape[0]} trajectory frames.")