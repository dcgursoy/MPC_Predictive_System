"""Phase 1 validation figure: sanity-check the dynamics model visually.

Generates results/figures/phase1_dynamics_validation.png with three panels:
  1. Free fall vs. the analytic -g t^2 / 2 solution
  2. Hover hold (position drift over 5 s)
  3. Torque-free tumble invariants (rotational KE, |L_world|)

Run from the repo root:  python scripts/validate_dynamics.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamics import (  # noqa: E402
    IDX_OMEGA,
    IDX_POS,
    IDX_QUAT,
    QuadrotorDynamics,
    QuadrotorParams,
    quat_to_rotmat,
)

OUT = Path(__file__).resolve().parents[1] / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def rollout(model, x0, u, t_final, dt):
    n = int(round(t_final / dt))
    xs = np.empty((n + 1, x0.size))
    xs[0] = x0
    for k in range(n):
        xs[k + 1] = model.step(xs[k], u, dt, enforce_limits=False)
    return np.linspace(0.0, t_final, n + 1), xs


def main():
    model = QuadrotorDynamics(QuadrotorParams(drag_coeff=0.0))
    g = model.params.gravity
    dt = 0.002

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1) Free fall
    t, xs = rollout(model, model.hover_state((0, 0, 50.0)), np.zeros(4), 3.0, dt)
    axes[0].plot(t, xs[:, 2], lw=2, label="simulated $z(t)$")
    axes[0].plot(t, 50.0 - 0.5 * g * t**2, "k--", lw=1, label=r"analytic $z_0 - \frac{1}{2}gt^2$")
    axes[0].set(title="Free fall", xlabel="t [s]", ylabel="altitude [m]")
    axes[0].legend()
    err = np.max(np.abs(xs[:, 2] - (50.0 - 0.5 * g * t**2)))
    print(f"[free fall] max |z - analytic| = {err:.2e} m")

    # 2) Hover hold
    t, xs = rollout(model, model.hover_state((0, 0, 3.0)), model.hover_input(), 5.0, dt)
    drift = np.linalg.norm(xs[:, IDX_POS] - np.array([0, 0, 3.0]), axis=1)
    axes[1].plot(t, drift * 1e9, lw=2)
    axes[1].set(title="Hover at exact trim thrust", xlabel="t [s]",
                ylabel="position drift [nm]")
    print(f"[hover] max position drift over 5 s = {drift.max():.2e} m")

    # 3) Torque-free tumble invariants
    x0 = model.hover_state()
    x0[IDX_OMEGA] = [2.0, -1.5, 3.0]
    t, xs = rollout(model, x0, np.zeros(4), 5.0, dt)
    J = model.params.inertia
    E = 0.5 * np.einsum("ni,ij,nj->n", xs[:, IDX_OMEGA], J, xs[:, IDX_OMEGA])
    L = np.array([
        np.linalg.norm(quat_to_rotmat(x[IDX_QUAT]) @ (J @ x[IDX_OMEGA]))
        for x in xs
    ])
    axes[2].plot(t, E / E[0], lw=2, label="rot. KE / KE$_0$")
    axes[2].plot(t, L / L[0], lw=2, ls="--", label=r"$|L_{world}| / |L_0|$")
    axes[2].set(title="Torque-free tumble invariants", xlabel="t [s]",
                ylabel="normalized invariant")
    axes[2].set_ylim(1 - 1e-6, 1 + 1e-6)
    axes[2].legend()
    print(f"[tumble] KE drift = {abs(E[-1]/E[0]-1):.2e}, "
          f"|L| drift = {abs(L[-1]/L[0]-1):.2e}")

    fig.suptitle("Phase 1 — quadrotor dynamics validation")
    fig.tight_layout()
    out = OUT / "phase1_dynamics_validation.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
