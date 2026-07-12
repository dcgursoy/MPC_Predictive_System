"""Phase 3 validation figure for the nonlinear MPC.

Panels:
  1. Static obstacle run (top view) with predicted-horizon snapshots —
     the receding-horizon plan visibly bends around the keep-out zone
     long before the drone arrives.
  2. Head-on moving obstacle: trajectory snapshots at three times.
  3. Solve-time distribution across all runs.
Run from the repo root:  python scripts/validate_mpc.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.mpc import NonlinearMPC  # noqa: E402
from dynamics import QuadrotorDynamics  # noqa: E402
from sim import (  # noqa: E402
    LineReference,
    MovingSphereObstacle,
    SphereObstacle,
    run_closed_loop,
)

OUT = Path(__file__).resolve().parents[1] / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = QuadrotorDynamics()


def circle(ax, c, r, **kw):
    th = np.linspace(0, 2 * np.pi, 100)
    ax.fill(c[0] + r * np.cos(th), c[1] + r * np.sin(th), **kw)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # ---- 1) static obstacle with horizon snapshots ----------------------
    start, goal = np.array([0, 0, 1.5]), np.array([8.0, 0, 1.5])
    obstacles = [SphereObstacle(center=[4.0, 0.0, 1.5], radius=0.75)]
    ref = LineReference(start, goal, cruise_speed=2.0)
    mpc = NonlinearMPC(reference=ref, obstacles=obstacles)
    log = run_closed_loop(mpc, ref, MODEL.hover_state(start), 8.0,
                          obstacles=obstacles)

    # Side view (x-z): with the obstacle dead-center on the path, the
    # optimizer's escape route is over the top, not around.
    ax = axes[0]
    ob = obstacles[0]
    circle(ax, [ob.center[0], ob.center[2]], ob.radius, color="tab:red",
           alpha=0.45, label="obstacle")
    circle(ax, [ob.center[0], ob.center[2]], ob.radius + mpc.inflation,
           color="tab:red", alpha=0.12, label="keep-out (inflated)")
    ax.plot(log.x[:, 0], log.x[:, 2], "C0", lw=2, label="flown path")
    for i, k in enumerate(range(0, len(log.info), 30)):  # every 0.6 s
        pred = log.info[k]["predicted_traj"]
        ax.plot(pred[:, 0], pred[:, 2], "C2", lw=1.2, alpha=0.7,
                label="predicted horizon" if i == 0 else None)
        ax.plot(log.x[k, 0], log.x[k, 2], "C0o", ms=4)
    ax.plot([start[0], goal[0]], [start[2], goal[2]], "k--", lw=0.8,
            label="reference line")
    ax.set(title="Static obstacle — receding-horizon plans (side view)",
           xlabel="x [m]", ylabel="z [m]")
    ax.axis("equal")
    ax.legend(loc="upper left", fontsize=8)
    st1 = np.array([i["solve_time"] for i in log.info])[2:] * 1e3
    print(f"[static] clearance {log.min_clearance:.3f} m | "
          f"solve avg {st1.mean():.1f} ms")

    # ---- 2) head-on moving obstacle -------------------------------------
    goal2 = np.array([10.0, 0, 1.5])
    obstacles2 = [MovingSphereObstacle(center=[10.0, 0.0, 1.5], radius=0.6,
                                       vel=[-1.5, 0.0, 0.0])]
    ref2 = LineReference(start, goal2, cruise_speed=2.5)
    mpc2 = NonlinearMPC(reference=ref2, obstacles=obstacles2)
    log2 = run_closed_loop(mpc2, ref2, MODEL.hover_state(start), 10.0,
                           obstacles=obstacles2)

    ax = axes[1]
    snap_times = [1.5, 2.5, 3.5]
    colors = ["#d4a0a0", "#c96666", "tab:red"]
    ob2 = obstacles2[0]
    for st, cc in zip(snap_times, colors):
        k = int(st / 0.02)
        oc = log2.obstacle_pos[k, 0]
        circle(ax, [oc[0], oc[2]], ob2.radius, color=cc, alpha=0.5)
        pred = log2.info[k]["predicted_traj"]
        ax.plot(pred[:, 0], pred[:, 2], color=cc, lw=1.4,
                label=f"plan @ t={st:.1f} s")
        ax.plot(log2.x[k, 0], log2.x[k, 2], "o", color=cc, ms=5)
    ax.plot(log2.x[:, 0], log2.x[:, 2], "C0", lw=2, label="flown path")
    ax.set(title="Head-on moving obstacle (1.5 m/s) — side view",
           xlabel="x [m]", ylabel="z [m]")
    ax.axis("equal")
    ax.legend(loc="upper left", fontsize=8)
    st2 = np.array([i["solve_time"] for i in log2.info])[2:] * 1e3
    print(f"[moving] clearance {log2.min_clearance:.3f} m | "
          f"solve avg {st2.mean():.1f} ms | collided {log2.collided}")

    # ---- 3) solve-time distribution --------------------------------------
    ax = axes[2]
    all_st = np.concatenate([st1, st2])
    ax.hist(all_st, bins=40, color="C0", alpha=0.8)
    ax.axvline(all_st.mean(), color="k", ls="--",
               label=f"mean {all_st.mean():.1f} ms")
    ax.axvline(np.percentile(all_st, 95), color="tab:orange", ls="--",
               label=f"p95 {np.percentile(all_st, 95):.1f} ms")
    ax.axvline(20.0, color="tab:red", ls=":", label="20 ms (50 Hz budget)")
    ax.set(title="IPOPT solve time (both runs)", xlabel="solve time [ms]",
           ylabel="count")
    ax.legend(fontsize=8)

    fig.suptitle("Phase 3 — nonlinear MPC validation (CasADi + IPOPT, "
                 "N=20 × 75 ms horizon)")
    fig.tight_layout()
    out = OUT / "phase3_mpc_validation.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
