"""Phase 2 validation figure for the cascaded PID baseline.

Panels: lateral step response, circular trajectory tracking (top view +
error trace), and potential-field avoidance of a static obstacle.
Run from the repo root:  python scripts/validate_pid.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.pid import CascadedPIDController, PotentialFieldAvoidance  # noqa: E402
from dynamics import QuadrotorDynamics  # noqa: E402
from sim import (  # noqa: E402
    CircleReference,
    LineReference,
    SphereObstacle,
    StepReference,
    run_closed_loop,
)

OUT = Path(__file__).resolve().parents[1] / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = QuadrotorDynamics()


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # 1) Step response
    ref = StepReference([0, 0, 2.0], [2.0, 0, 2.0], t_step=1.0)
    log = run_closed_loop(
        CascadedPIDController(), ref, MODEL.hover_state((0, 0, 2.0)), 8.0
    )
    axes[0].plot(log.t, log.x[:, 0], lw=2, label="x(t)")
    axes[0].plot(log.t, log.ref_pos[:, 0], "k--", lw=1, label="reference")
    axes[0].set(title="2 m lateral step", xlabel="t [s]", ylabel="x [m]")
    axes[0].legend()
    xs = log.x[:, 0]
    ovs = (np.max(xs) - 2.0) / 2.0 * 100
    settled = log.t[np.abs(xs - 2.0) > 0.05]
    ts = settled[-1] - 1.0 if settled.size else 0.0
    print(f"[step] overshoot = {ovs:.1f}%, settle(5cm) = {ts:.2f} s")

    # 2) Circle tracking
    ref = CircleReference(center=[0, 0, 2.0], radius=2.0, omega=1.0)
    log = run_closed_loop(
        CascadedPIDController(), ref, MODEL.hover_state((2.0, 0, 2.0)), 15.0
    )
    axes[1].plot(log.x[:, 0], log.x[:, 1], lw=2, label="flown")
    axes[1].plot(log.ref_pos[:, 0], log.ref_pos[:, 1], "k--", lw=1, label="reference")
    axes[1].set(title="Circle @ 2 m/s (top view)", xlabel="x [m]", ylabel="y [m]")
    axes[1].axis("equal")
    axes[1].legend()
    steady = log.pos_err[log.t > 5.0]
    print(f"[circle] steady-state err: mean {np.mean(steady)*100:.1f} cm, "
          f"max {np.max(steady)*100:.1f} cm")

    # 3) Obstacle avoidance
    start, goal = np.array([0, 0, 1.5]), np.array([8.0, 0, 1.5])
    obstacles = [SphereObstacle(center=[4.0, 0.15, 1.5], radius=0.75)]
    ctrl = CascadedPIDController(
        avoidance=PotentialFieldAvoidance(obstacles=obstacles)
    )
    log = run_closed_loop(
        ctrl, LineReference(start, goal, cruise_speed=2.0),
        MODEL.hover_state(start), 10.0, obstacles=obstacles,
    )
    th = np.linspace(0, 2 * np.pi, 100)
    ob = obstacles[0]
    axes[2].fill(
        ob.center[0] + ob.radius * np.cos(th),
        ob.center[1] + ob.radius * np.sin(th),
        color="tab:red", alpha=0.4, label="obstacle",
    )
    axes[2].plot(log.x[:, 0], log.x[:, 1], lw=2, label="flown path")
    axes[2].plot([start[0], goal[0]], [start[1], goal[1]], "k--", lw=1,
                 label="nominal line")
    axes[2].set(title="Potential-field avoidance (top view)",
                xlabel="x [m]", ylabel="y [m]")
    axes[2].axis("equal")
    axes[2].legend()
    print(f"[avoid] min clearance = {log.min_clearance:.2f} m, "
          f"final goal error = {np.linalg.norm(log.x[-1, :3] - goal):.3f} m, "
          f"collided = {log.collided}")

    fig.suptitle("Phase 2 — cascaded PID baseline validation")
    fig.tight_layout()
    out = OUT / "phase2_pid_validation.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
