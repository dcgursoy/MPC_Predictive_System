"""Metrics extracted from a closed-loop run (SimLog) on a course."""

from __future__ import annotations

import numpy as np

from .simulator import SimLog


def compute_metrics(course, log: SimLog) -> dict:
    """Success, safety, efficiency, and (for MPC) solver statistics."""
    goal = course.goal
    dist_to_goal = np.linalg.norm(log.x[:, :3] - goal, axis=1)

    # Time to goal: first time inside the tolerance ball
    inside = np.flatnonzero(dist_to_goal < course.goal_tolerance)
    time_to_goal = float(log.t[inside[0]]) if inside.size else np.nan
    reached = inside.size > 0

    # Path length up to arrival (or the whole run if it never arrived)
    end = inside[0] + 1 if inside.size else len(log.t)
    seg = np.diff(log.x[:end, :3], axis=0)
    path_length = float(np.sum(np.linalg.norm(seg, axis=1)))
    # The flight ends on the goal-tolerance ball, so the ideal path is the
    # straight distance minus the tolerance (keeps efficiency <= ~1).
    straight = float(np.linalg.norm(goal - course.start)) - course.goal_tolerance
    path_efficiency = straight / path_length if path_length > 0 else np.nan

    m = {
        "success": bool(reached and not log.collided),
        "collided": bool(log.collided),
        "t_collision": log.t_collision,
        "min_clearance": float(log.min_clearance),
        "time_to_goal": time_to_goal,
        "path_length": path_length,
        "path_efficiency": path_efficiency,
        "mean_tracking_err": float(np.mean(log.pos_err)),
        "max_tracking_err": float(np.max(log.pos_err)),
        "final_goal_err": float(dist_to_goal[-1]),
    }

    solve = np.array([i["solve_time"] for i in log.info
                      if "solve_time" in i]) * 1e3
    if solve.size:
        solve = solve[2:] if solve.size > 2 else solve  # drop cold start
        m.update({
            "solve_ms_mean": float(solve.mean()),
            "solve_ms_p50": float(np.percentile(solve, 50)),
            "solve_ms_p95": float(np.percentile(solve, 95)),
            "solve_ms_max": float(solve.max()),
            "max_slack": float(max(i.get("max_slack", 0.0) for i in log.info)),
        })
    return m


def summary_table(rows: list[dict]) -> str:
    """Markdown table from a list of {course, controller, **metrics} rows."""
    def fmt(v, spec=".2f"):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return format(v, spec)

    header = ("| Course | Controller | Success | Collided | Min clear. [m] | "
              "Time to goal [s] | Path eff. | Mean track err [m] | "
              "Solve mean [ms] | Solve p95 [ms] |")
    sep = "|" + "---|" * 10
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['course']} | {r['controller'].upper()} "
            f"| {'YES' if r['success'] else 'no'} "
            f"| {'YES' if r['collided'] else 'no'} "
            f"| {fmt(r['min_clearance'])} "
            f"| {fmt(r['time_to_goal'])} "
            f"| {fmt(r['path_efficiency'])} "
            f"| {fmt(r['mean_tracking_err'])} "
            f"| {fmt(r.get('solve_ms_mean'), '.1f')} "
            f"| {fmt(r.get('solve_ms_p95'), '.1f')} |"
        )
    return "\n".join(lines)
