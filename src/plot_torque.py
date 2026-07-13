"""
plot_torques.py

"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_joint_torques(
    npz_path: str = "results/torques.npz",
    out_path: str = "figures/joint_torques.png",
) -> str:
    """
    Load t, tau_no_payload, tau_with_payload, payload_mass from npz_path
    (as saved by dynamics.py) and produce a 3x2 grid, one subplot per
    joint, both cases overlaid.
    """
    d = np.load(npz_path)
    t = d["t"]
    tau0 = d["tau_no_payload"]
    tau5 = d["tau_with_payload"]
    payload_mass = float(d["payload_mass"])

    label0 = "Case 1: no payload"
    label5 = f"Case 2: {payload_mass:g} kg payload"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for j in range(6):
        ax = axes[j // 2, j % 2]
        ax.plot(t, tau0[:, j], label=label0, color="#1f77b4", lw=1.6)
        ax.plot(t, tau5[:, j], label=label5, color="#d62728", lw=1.6, ls="--")
        ax.set_title(f"Joint {j + 1}")
        ax.set_ylabel("Torque (N·m)")
        ax.grid(alpha=0.3)
        if j >= 4:
            ax.set_xlabel("Time (s)")

    axes[0, 0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Joint Torque Profiles: No Payload vs 5 kg Payload", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    fig_path = plot_joint_torques()
    print(f"Saved: {fig_path}")