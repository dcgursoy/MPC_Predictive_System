"""Phase 5: full instrumented comparison — every controller on every course.

Run from the repo root:  python scripts/run_experiments.py
Writes:
  results/metrics_summary.md / .csv   comparison table
  results/raw/phase5_<course>_<ctrl>.pkl   full logs (regenerable)
  results/figures/phase5_solvetimes.png    MPC solve-time distribution
"""

import csv
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
from sim.metrics import summary_table  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
FIGS = ROOT / "results" / "figures"
RAW.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


def main():
    rows = []
    solve_by_course = {}
    for cname, factory in ALL_COURSES.items():
        for kind in ("pid", "mpc"):
            course = factory()
            log, metrics = fly_course(kind, course)
            rows.append({"course": cname, "controller": kind, **metrics})
            with open(RAW / f"phase5_{cname}_{kind}.pkl", "wb") as f:
                pickle.dump({"course": cname, "kind": kind, "log": log,
                             "metrics": metrics}, f)
            if kind == "mpc":
                solve_by_course[cname] = (
                    np.array([i["solve_time"] for i in log.info])[2:] * 1e3
                )
            print(f"done: {cname} / {kind}")

    # ---- table -----------------------------------------------------------
    table = summary_table(rows)
    print("\n" + table + "\n")

    # Aggregates for the README resume bullet
    for kind in ("pid", "mpc"):
        rs = [r for r in rows if r["controller"] == kind]
        n_ok = sum(r["success"] for r in rs)
        n_col = sum(r["collided"] for r in rs)
        line = (f"{kind.upper()}: {n_ok}/{len(rs)} courses succeeded, "
                f"{n_col} collisions")
        if kind == "mpc":
            allst = np.concatenate(list(solve_by_course.values()))
            line += (f", solve mean {allst.mean():.1f} ms / "
                     f"p95 {np.percentile(allst, 95):.1f} ms / "
                     f"max {allst.max():.1f} ms over {allst.size} solves")
        print(line)

    (ROOT / "results" / "metrics_summary.md").write_text(
        "# PID vs MPC — course metrics\n\n" + table + "\n", encoding="utf-8"
    )
    with open(ROOT / "results" / "metrics_summary.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)

    # ---- solve-time figure -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    allst = np.concatenate(list(solve_by_course.values()))
    axes[0].hist(allst, bins=50, color="C0", alpha=0.85)
    axes[0].axvline(allst.mean(), color="k", ls="--",
                    label=f"mean {allst.mean():.1f} ms")
    axes[0].axvline(np.percentile(allst, 95), color="tab:orange", ls="--",
                    label=f"p95 {np.percentile(allst, 95):.1f} ms")
    axes[0].axvline(20, color="tab:red", ls=":", label="20 ms (50 Hz)")
    axes[0].set(title=f"MPC solve time, all courses ({allst.size} solves)",
                xlabel="solve time [ms]", ylabel="count")
    axes[0].legend(fontsize=8)

    data = [solve_by_course[c] for c in ALL_COURSES]
    axes[1].boxplot(data, tick_labels=list(ALL_COURSES), showfliers=True,
                    whis=(5, 95))
    axes[1].axhline(20, color="tab:red", ls=":", lw=1)
    axes[1].set(title="Solve time per course (whiskers p5-p95)",
                ylabel="solve time [ms]")

    fig.suptitle("Phase 5 — MPC solver real-time performance")
    fig.tight_layout()
    out = FIGS / "phase5_solvetimes.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
