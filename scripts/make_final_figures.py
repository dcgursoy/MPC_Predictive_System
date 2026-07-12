"""Phase 7: focused comparison figure for the moving-obstacle course.

Run from the repo root (after scripts/run_experiments.py):
  python scripts/make_final_figures.py
"""

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamics import QuadrotorParams  # noqa: E402
from sim.courses import ALL_COURSES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
FIGS = ROOT / "results" / "figures"


def load(course, kind):
    with open(RAW / f"phase5_{course}_{kind}.pkl", "rb") as f:
        return pickle.load(f)["log"]


def main():
    course = ALL_COURSES["crossing"]()
    log_pid, log_mpc = load("crossing", "pid"), load("crossing", "mpc")
    params = QuadrotorParams()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) top view with mover stroboscope
    ax = axes[0]
    th = np.linspace(0, 2 * np.pi, 80)
    for ob in course.obstacles:
        static = np.all(np.abs(ob.velocity(0.0)) < 1e-9) and "osc" not in ob.name
        if static:
            p = ob.position(0.0)
            ax.fill(p[0] + ob.radius * np.cos(th), p[1] + ob.radius * np.sin(th),
                    color="tab:red", alpha=0.4)
        else:
            for tt in np.arange(0.0, 7.0, 1.0):
                p = ob.position(tt)
                a = 0.15 + 0.5 * tt / 7.0
                ax.plot(p[0] + ob.radius * np.cos(th),
                        p[1] + ob.radius * np.sin(th),
                        color="tab:orange", lw=0.9, alpha=a)
                if ob.name == "moving_sphere" and tt in (0.0, 2.0, 4.0, 6.0):
                    ax.annotate(f"t={tt:.0f}", (p[0] + 0.65, p[1]), fontsize=7,
                                color="tab:orange")
    ax.plot(log_pid.x[:, 0], log_pid.x[:, 1], "C1", lw=2,
            label="PID + potential field")
    ax.plot(log_mpc.x[:, 0], log_mpc.x[:, 1], "C0", lw=2, label="nonlinear MPC")
    ax.plot([course.start[0], course.goal[0]], [course.start[1], course.goal[1]],
            "k--", lw=0.9, label="reference")
    ax.plot(course.goal[0], course.goal[1], "k*", ms=13)
    ax.set(title="Crossing course — top view\n(orange: moving obstacles, fading = earlier)",
           xlabel="x [m]", ylabel="y [m]")
    ax.axis("equal")
    ax.legend(fontsize=8, loc="upper left")

    # (b) clearance vs time
    ax = axes[1]
    ax.plot(log_pid.t, log_pid.clearance, "C1", lw=1.8, label="PID")
    ax.plot(log_mpc.t, log_mpc.clearance, "C0", lw=1.8, label="MPC")
    ax.axhline(params.radius, color="r", ls="--", lw=1,
               label=f"collision (< {params.radius} m)")
    ax.axhline(params.radius + 0.10, color="gray", ls=":", lw=1,
               label="MPC keep-out margin")
    ax.set(title="Distance to nearest obstacle surface", xlabel="t [s]",
           ylabel="clearance [m]", ylim=(0, 4.0), xlim=(0, 8.0))
    ax.legend(fontsize=8)

    # (c) progress + MPC solve time
    ax = axes[2]
    d_pid = np.linalg.norm(log_pid.x[:, :3] - course.goal, axis=1)
    d_mpc = np.linalg.norm(log_mpc.x[:, :3] - course.goal, axis=1)
    ax.plot(log_pid.t, d_pid, "C1", lw=1.8, label="PID dist to goal")
    ax.plot(log_mpc.t, d_mpc, "C0", lw=1.8, label="MPC dist to goal")
    ax.set(title="Progress to goal / MPC solve time", xlabel="t [s]",
           ylabel="distance to goal [m]", xlim=(0, 8.0))
    ax.legend(fontsize=8, loc="upper right")
    ax2 = ax.twinx()
    st = np.array([i["solve_time"] for i in log_mpc.info]) * 1e3
    ax2.fill_between(log_mpc.t, st, color="C2", alpha=0.25, step="mid")
    ax2.set_ylabel("MPC solve time [ms]", color="C2")
    ax2.tick_params(axis="y", labelcolor="C2")
    ax2.set_ylim(0, max(25, np.percentile(st, 99) * 1.4))

    fig.suptitle("MPC vs PID on the moving-obstacle crossing course")
    fig.tight_layout()
    out = FIGS / "phase7_crossing_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
