"""Phase 6: animate a course run in 3D.

Examples (from the repo root):
  python scripts/animate_course.py --course gauntlet --mode both --save results/figures/gauntlet_side_by_side.gif
  python scripts/animate_course.py --course slalom --mode mpc          # interactive window
  python scripts/animate_course.py --course crossing --mode both --save out.mp4 --fps 25

Uses cached Phase 5 logs from results/raw when available (and not
--fresh); otherwise re-flies the course.
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.courses import ALL_COURSES  # noqa: E402
from sim.experiments import fly_course  # noqa: E402
from viz import CourseAnimator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"

LABELS = {"pid": "PID + potential field", "mpc": "Nonlinear MPC"}


def get_log(course_name: str, kind: str, fresh: bool):
    cache = RAW / f"phase5_{course_name}_{kind}.pkl"
    if not fresh and cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)["log"]
    course = ALL_COURSES[course_name]()
    log, _ = fly_course(kind, course)
    return log


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--course", choices=list(ALL_COURSES), default="gauntlet")
    ap.add_argument("--mode", choices=["mpc", "pid", "both"], default="both")
    ap.add_argument("--save", default=None,
                    help=".gif or .mp4 path; omit for interactive window")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--stride", type=int, default=2,
                    help="animate every k-th control step (default 2 = 25 fps)")
    ap.add_argument("--fresh", action="store_true",
                    help="re-fly the course instead of using cached logs")
    args = ap.parse_args()

    course = ALL_COURSES[args.course]()
    kinds = ["pid", "mpc"] if args.mode == "both" else [args.mode]
    logs = {LABELS[k]: get_log(args.course, k, args.fresh) for k in kinds}

    animator = CourseAnimator(course, logs, frame_stride=args.stride)
    if args.save:
        animator.save(args.save, fps=args.fps)
        print(f"saved {args.save}")
    else:
        animator.show()


if __name__ == "__main__":
    main()
