"""Closed-loop validation of the cascaded PID baseline."""

import numpy as np

from control.pid import CascadedPIDController, PotentialFieldAvoidance
from dynamics import IDX_POS, IDX_VEL, QuadrotorDynamics
from sim import (
    CircleReference,
    HoverReference,
    LineReference,
    SphereObstacle,
    StepReference,
    run_closed_loop,
)

MODEL = QuadrotorDynamics()


def test_hover_recovery_from_offset():
    """Released 1.5 m from the setpoint, converge and hold."""
    ref = HoverReference([0.0, 0.0, 2.0])
    x0 = MODEL.hover_state(position=(1.0, -1.0, 1.5))
    log = run_closed_loop(CascadedPIDController(), ref, x0, t_final=6.0)
    assert log.pos_err[-1] < 0.02
    assert np.linalg.norm(log.x[-1, IDX_VEL]) < 0.02


def test_position_step_response():
    """2 m lateral step: settles fast without excessive overshoot."""
    ref = StepReference([0, 0, 2.0], [2.0, 0, 2.0], t_step=1.0)
    x0 = MODEL.hover_state(position=(0, 0, 2.0))
    log = run_closed_loop(CascadedPIDController(), ref, x0, t_final=8.0)
    xpos = log.x[:, 0]
    overshoot = (np.max(xpos) - 2.0) / 2.0
    assert overshoot < 0.25
    settled = log.t[np.abs(xpos - 2.0) > 0.05]
    settle_time = (settled[-1] + 0.02 - 1.0) if settled.size else 0.0
    assert settle_time < 3.0
    assert log.pos_err[-1] < 0.03


def test_altitude_step_response():
    ref = StepReference([0, 0, 1.0], [0, 0, 3.0], t_step=1.0)
    x0 = MODEL.hover_state(position=(0, 0, 1.0))
    log = run_closed_loop(CascadedPIDController(), ref, x0, t_final=8.0)
    z = log.x[:, 2]
    assert (np.max(z) - 3.0) / 2.0 < 0.25
    assert log.pos_err[-1] < 0.03


def test_circle_tracking():
    """2 m/s circle: bounded steady-state tracking error."""
    ref = CircleReference(center=[0, 0, 2.0], radius=2.0, omega=1.0)
    x0 = MODEL.hover_state(position=(2.0, 0, 2.0))
    log = run_closed_loop(CascadedPIDController(), ref, x0, t_final=15.0)
    steady = log.pos_err[log.t > 5.0]
    assert np.mean(steady) < 0.15
    assert np.max(steady) < 0.30


def test_static_obstacle_avoidance():
    """Fly a line whose path clips a sphere; potential field must deflect
    around it with zero collisions and still reach the goal."""
    start, goal = np.array([0, 0, 1.5]), np.array([8.0, 0, 1.5])
    # Slightly offset from the path axis: a perfectly head-on obstacle is a
    # symmetric local minimum for any potential-field method.
    obstacles = [SphereObstacle(center=[4.0, 0.15, 1.5], radius=0.75)]
    ctrl = CascadedPIDController(
        avoidance=PotentialFieldAvoidance(obstacles=obstacles)
    )
    ref = LineReference(start, goal, cruise_speed=2.0)
    log = run_closed_loop(
        ctrl, ref, MODEL.hover_state(position=start), t_final=10.0,
        obstacles=obstacles,
    )
    # Body-safe: center stays at least a drone radius off the surface
    assert log.min_clearance > MODEL.params.radius, "PID clipped the obstacle"
    assert np.linalg.norm(log.x[-1, IDX_POS] - goal) < 0.15


def test_controller_respects_actuator_limits():
    """Huge initial error: commands must stay inside motor limits."""
    ref = HoverReference([0.0, 0.0, 10.0])
    x0 = MODEL.hover_state(position=(20.0, 0, 1.0))
    ctrl = CascadedPIDController()
    log = run_closed_loop(ctrl, ref, x0, t_final=2.0)
    p = MODEL.params
    assert np.all(log.u[:, 0] <= p.thrust_max + 1e-9)
    assert np.all(log.u[:, 0] >= p.thrust_min - 1e-9)
