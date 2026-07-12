"""Unit tests for the metrics computations on synthetic logs."""

import numpy as np

from sim.courses import Course
from sim.metrics import compute_metrics, summary_table
from sim.simulator import SimLog


def _straight_line_log(start, goal, speed=2.0, dt=0.02):
    """Synthetic perfect straight-line flight at constant speed."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    d = np.linalg.norm(goal - start)
    n = int(d / (speed * dt)) + 1
    t = np.arange(n) * dt
    frac = np.minimum(t * speed / d, 1.0)
    pos = start + frac[:, None] * (goal - start)
    x = np.zeros((n, 13))
    x[:, :3] = pos
    return SimLog(
        t=t, x=x, u=np.zeros((n, 4)), ref_pos=pos.copy(),
        pos_err=np.zeros(n), clearance=np.full(n, np.inf),
        ctrl_time=np.zeros(n), obstacle_pos=np.zeros((n, 0, 3)),
        obstacle_radius=np.zeros(0), info=[{} for _ in range(n)],
    )


def test_straight_line_metrics():
    course = Course(name="unit", start=[0, 0, 1], goal=[10, 0, 1],
                    t_final=10.0)
    log = _straight_line_log(course.start, course.goal)
    m = compute_metrics(course, log)
    assert m["success"]
    assert not m["collided"]
    # 10 m at 2 m/s, tolerance ball of 0.3 m -> arrive at ~4.85 s
    assert abs(m["time_to_goal"] - (10 - 0.3) / 2.0) < 0.1
    assert m["path_efficiency"] > 0.99
    assert m["final_goal_err"] < 1e-9


def test_never_reaching_goal():
    course = Course(name="unit", start=[0, 0, 1], goal=[10, 0, 1],
                    t_final=10.0)
    log = _straight_line_log([0, 0, 1], [4.0, 0, 1])  # stops short
    m = compute_metrics(course, log)
    assert not m["success"]
    assert np.isnan(m["time_to_goal"])


def test_collision_fails_run():
    course = Course(name="unit", start=[0, 0, 1], goal=[10, 0, 1])
    log = _straight_line_log(course.start, course.goal)
    log.collided = True
    log.t_collision = 1.0
    m = compute_metrics(course, log)
    assert m["collided"] and not m["success"]


def test_summary_table_renders():
    course = Course(name="unit", start=[0, 0, 1], goal=[10, 0, 1])
    log = _straight_line_log(course.start, course.goal)
    m = compute_metrics(course, log)
    table = summary_table([{"course": "unit", "controller": "pid", **m}])
    assert "| unit | PID |" in table
    assert "—" in table  # no solve-time columns for PID
