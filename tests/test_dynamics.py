"""
tests/test_dynamics_verification.py
"""

import numpy as np

from src.dynamics import rnea, system_energy


def _numerical_dPE_dq(q, payload_mass=0.0, h=1e-6):
    """Central-difference gradient of potential energy w.r.t. each joint."""
    grad = np.zeros(6)
    for i in range(6):
        q_plus, q_minus = q.copy(), q.copy()
        q_plus[i] += h
        q_minus[i] -= h
        _, pe_plus = system_energy(q_plus, np.zeros(6), payload_mass)
        _, pe_minus = system_energy(q_minus, np.zeros(6), payload_mass)
        grad[i] = (pe_plus - pe_minus) / (2 * h)
    return grad


def test_static_gravity_torque_matches_energy_gradient():
    rng = np.random.default_rng(0)
    for payload_mass in (0.0, 5.0):
        for _ in range(5):
            q = rng.uniform(-1.5, 1.5, size=6)
            tau = rnea(q, np.zeros(6), np.zeros(6), payload_mass=payload_mass)
            dpe_dq = _numerical_dPE_dq(q, payload_mass=payload_mass)
            err = np.max(np.abs(tau - dpe_dq))
            assert err < 1e-6, f"static check failed: max err {err:.2e} (payload={payload_mass})"


def test_power_matches_rate_of_mechanical_energy():
    """
    Pick a random smooth-ish trajectory q(t) = q0 + A sin(w t + phi),
    and check tau(t).qd(t) == d(KE+PE)/dt at several interior times, via
    central-difference on system_energy's KE+PE sum.
    """
    rng = np.random.default_rng(1)

    for payload_mass in (0.0, 5.0):
        q0 = rng.uniform(-1.0, 1.0, size=6)
        A = rng.uniform(0.2, 0.6, size=6)
        w = rng.uniform(0.5, 1.5, size=6)
        phi = rng.uniform(0, 2 * np.pi, size=6)

        def q_of_t(t):
            return q0 + A * np.sin(w * t + phi)

        def qd_of_t(t):
            return A * w * np.cos(w * t + phi)

        def qdd_of_t(t):
            return -A * w**2 * np.sin(w * t + phi)

        def energy_of_t(t):
            ke, pe = system_energy(q_of_t(t), qd_of_t(t), payload_mass)
            return ke + pe

        h = 1e-6
        for t in (0.3, 1.1, 2.0, 3.7):
            q, qd, qdd = q_of_t(t), qd_of_t(t), qdd_of_t(t)
            tau = rnea(q, qd, qdd, payload_mass=payload_mass)
            power = tau @ qd

            dE_dt = (energy_of_t(t + h) - energy_of_t(t - h)) / (2 * h)

            rel_err = abs(power - dE_dt) / max(abs(dE_dt), 1e-6)
            assert rel_err < 1e-6, (
                f"dynamic check failed at t={t}, payload={payload_mass}: "
                f"power={power:.6f}, dE/dt={dE_dt:.6f}, rel_err={rel_err:.2e}"
            )


if __name__ == "__main__":
    test_static_gravity_torque_matches_energy_gradient()
    print("Static check (gravity torque == dPE/dq): PASSED")
    test_power_matches_rate_of_mechanical_energy()
    print("Dynamic check (power == dE/dt): PASSED")