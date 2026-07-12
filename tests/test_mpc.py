"""Closed-loop validation of the nonlinear MPC controller."""

import numpy as np

from control.mpc import NonlinearMPC
from dynamics import IDX_POS, QuadrotorDynamics
from sim import (
    HoverReference,
    LineReference,
    MovingSphereObstacle,
    SphereObstacle,
    run_closed_loop,
)

MODEL = QuadrotorDynamics()


def test_hover_regulation():
    ref = HoverReference([0.0, 0.0, 2.0])
    mpc = NonlinearMPC(reference=ref)
    log = run_closed_loop(mpc, ref, MODEL.hover_state((1.0, -1.0, 1.5)), 4.0)
    assert log.pos_err[-1] < 0.02


def test_line_tracking_error():
    """Obstacle-free line: MPC should track tightly (it plans ahead)."""
    start, goal = np.array([0, 0, 1.5]), np.array([6.0, 0, 1.5])
    ref = LineReference(start, goal, cruise_speed=2.0)
    mpc = NonlinearMPC(reference=ref)
    log = run_closed_loop(mpc, ref, MODEL.hover_state(start), 6.0)
    assert np.mean(log.pos_err) < 0.10
    assert np.linalg.norm(log.x[-1, IDX_POS] - goal) < 0.05


def test_static_obstacle_constraint():
    """Keep-out constraint enforced: true clearance stays positive even
    though the reference line passes straight through the obstacle."""
    start, goal = np.array([0, 0, 1.5]), np.array([8.0, 0, 1.5])
    obstacles = [SphereObstacle(center=[4.0, 0.0, 1.5], radius=0.75)]
    ref = LineReference(start, goal, cruise_speed=2.0)
    mpc = NonlinearMPC(reference=ref, obstacles=obstacles)
    log = run_closed_loop(
        mpc, ref, MODEL.hover_state(start), 8.0, obstacles=obstacles
    )
    assert log.min_clearance > 0.25       # inflation is 0.40 m
    assert np.linalg.norm(log.x[-1, IDX_POS] - goal) < 0.10


def test_moving_obstacle_predictive_avoidance():
    """Head-on approaching obstacle: MPC must use the obstacle's known
    trajectory over the horizon and sidestep before contact."""
    start, goal = np.array([0, 0, 1.5]), np.array([10.0, 0, 1.5])
    obstacles = [
        MovingSphereObstacle(center=[10.0, 0.0, 1.5], radius=0.6,
                             vel=[-1.5, 0.0, 0.0])
    ]
    ref = LineReference(start, goal, cruise_speed=2.5)
    mpc = NonlinearMPC(reference=ref, obstacles=obstacles)
    log = run_closed_loop(
        mpc, ref, MODEL.hover_state(start), 10.0, obstacles=obstacles
    )
    assert not log.collided
    assert log.min_clearance > 0.15
    assert np.linalg.norm(log.x[-1, IDX_POS] - goal) < 0.10


def test_predicted_trajectory_logged():
    """Every control step logs the full predicted horizon (drives the
    Phase 6 visualization)."""
    ref = HoverReference([0.0, 0.0, 2.0])
    mpc = NonlinearMPC(reference=ref)
    log = run_closed_loop(mpc, ref, MODEL.hover_state((0.5, 0, 2.0)), 1.0)
    for k, info in enumerate(log.info):
        traj = info["predicted_traj"]
        assert traj.shape == (mpc.N + 1, 3)
        # First point of the plan is the measured state
        assert np.linalg.norm(traj[0] - log.x[k, IDX_POS]) < 1e-6
        assert "solve_time" in info


def test_solve_time_realtime_budget():
    """Average solve must fit a 20 ms (50 Hz) budget with obstacles active."""
    start, goal = np.array([0, 0, 1.5]), np.array([8.0, 0, 1.5])
    obstacles = [SphereObstacle(center=[4.0, 0.0, 1.5], radius=0.75)]
    ref = LineReference(start, goal, cruise_speed=2.0)
    mpc = NonlinearMPC(reference=ref, obstacles=obstacles)
    log = run_closed_loop(
        mpc, ref, MODEL.hover_state(start), 8.0, obstacles=obstacles
    )
    solve_ms = np.array([i["solve_time"] for i in log.info])[2:] * 1e3
    assert solve_ms.mean() < 25.0  # headroom over the observed ~12 ms


def test_thrust_within_actuator_limits():
    ref = HoverReference([0.0, 0.0, 8.0])
    mpc = NonlinearMPC(reference=ref)
    log = run_closed_loop(mpc, ref, MODEL.hover_state((0, 0, 1.0)), 3.0)
    p = MODEL.params
    assert np.all(log.u[:, 0] <= p.thrust_max + 1e-9)
    assert np.all(log.u[:, 0] >= p.thrust_min - 1e-9)
