"""Phase 4: fly PID and MPC through every course, save logs + figure.

Run from the repo root:  python scripts/run_courses.py
Writes per-run logs to results/raw/ (regenerable, gitignored) and the
comparison figure to results/figures/phase4_courses.png.
"""

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.courses import ALL_COURSES  # noqa: E402
from sim.experiments import fly_course  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
FIGS = ROOT / "results" / "figures"
RAW.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


def main():
    results = {}
    for cname, factory in ALL_COURSES.items():
        for kind in ("pid", "mpc"):
            course = factory()
            log, metrics = fly_course(kind, course)
            goal_err = metrics["final_goal_err"]
            success = metrics["success"]
            results[(cname, kind)] = (course, log, goal_err, success)
            solve = (np.array([i["solve_time"] for i in log.info])[2:] * 1e3
                     if kind == "mpc" else None)
            msg = (f"{cname:9s} {kind.upper():3s}: "
                   f"success={success!s:5s} goal_err={goal_err:5.2f} m "
                   f"min_clr={log.min_clearance:+5.2f} m "
                   f"collided={log.collided!s:5s}")
            if solve is not None:
                msg += f" solve avg {solve.mean():5.1f} ms max {solve.max():6.1f} ms"
            print(msg)
            with open(RAW / f"phase4_{cname}_{kind}.pkl", "wb") as f:
                pickle.dump({"course": cname, "kind": kind, "log": log,
                             "goal_err": goal_err, "success": success}, f)

    # ---- figure: one row per course, top view + side view ---------------
    fig, axes = plt.subplots(len(ALL_COURSES), 2,
                             figsize=(15, 4.2 * len(ALL_COURSES)))
    for r, cname in enumerate(ALL_COURSES):
        course, log_pid, _, _ = results[(cname, "pid")]
        _, log_mpc, _, _ = results[(cname, "mpc")]
        for c, (idx, ylab) in enumerate([(1, "y [m]"), (2, "z [m]")]):
            ax = axes[r, c]
            th = np.linspace(0, 2 * np.pi, 60)
            for ob in course.obstacles:
                moving = np.any(np.abs(ob.velocity(0)) > 0) or "oscillating" in ob.name
                if moving:
                    # stroboscope: outline every 1.5 s
                    for tt in np.arange(0, course.t_final, 1.5):
                        p = ob.position(tt)
                        ax.plot(p[0] + ob.radius * np.cos(th),
                                p[idx] + ob.radius * np.sin(th),
                                color="tab:orange", lw=0.7, alpha=0.5)
                else:
                    p = ob.position(0)
                    ax.fill(p[0] + ob.radius * np.cos(th),
                            p[idx] + ob.radius * np.sin(th),
                            color="tab:red", alpha=0.40)
            ax.plot(log_pid.x[:, 0], log_pid.x[:, idx], "C1", lw=1.8,
                    label="PID + potential field")
            ax.plot(log_mpc.x[:, 0], log_mpc.x[:, idx], "C0", lw=1.8,
                    label="MPC")
            ax.plot([course.start[0], course.goal[0]],
                    [course.start[idx], course.goal[idx]],
                    "k--", lw=0.8, label="reference")
            ax.plot(*[[course.goal[0]], [course.goal[idx]]], "k*", ms=12)
            if log_pid.collided:
                k = np.argmin(log_pid.clearance)
                ax.plot(log_pid.x[k, 0], log_pid.x[k, idx], "rx", ms=11,
                        mew=2.5, label="PID collision")
            ax.set(xlabel="x [m]", ylabel=ylab)
            view = "top view" if idx == 1 else "side view"
            ax.set_title(f"{cname} — {view}")
            ax.axis("equal")
            if c == 0:
                ax.legend(fontsize=8, loc="best")

    fig.suptitle("Phase 4 — PID vs MPC through all courses "
                 "(orange outlines: moving obstacle every 1.5 s)")
    fig.tight_layout()
    out = FIGS / "phase4_courses.png"
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
