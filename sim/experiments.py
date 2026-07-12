"""Shared experiment plumbing: build controllers, fly courses, collect metrics."""

from __future__ import annotations

import numpy as np

from control.mpc import NonlinearMPC
from control.pid import CascadedPIDController, PotentialFieldAvoidance
from dynamics import QuadrotorDynamics

from .metrics import compute_metrics
from .simulator import run_closed_loop

MODEL = QuadrotorDynamics()


def make_controller(kind: str, course):
    ref = course.make_reference()
    if kind == "pid":
        ctrl = CascadedPIDController(
            avoidance=PotentialFieldAvoidance(obstacles=course.obstacles)
        )
    elif kind == "mpc":
        ctrl = NonlinearMPC(reference=ref, obstacles=course.obstacles)
    else:
        raise ValueError(f"unknown controller kind {kind!r}")
    return ctrl, ref


def fly_course(kind: str, course):
    """Run one controller through one course; return (log, metrics)."""
    ctrl, ref = make_controller(kind, course)
    log = run_closed_loop(
        ctrl, ref, MODEL.hover_state(course.start), course.t_final,
        obstacles=course.obstacles,
        collision_margin=MODEL.params.radius,   # body collision
    )
    metrics = compute_metrics(course, log)
    return log, metrics
