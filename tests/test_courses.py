"""Course sanity checks + MPC completes every course collision-free."""

import numpy as np
import pytest

from control.mpc import NonlinearMPC
from dynamics import QuadrotorDynamics
from sim import run_closed_loop
from sim.courses import ALL_COURSES

MODEL = QuadrotorDynamics()


@pytest.mark.parametrize("name", list(ALL_COURSES))
def test_course_geometry_sane(name):
    course = ALL_COURSES[name]()
    r_body = MODEL.params.radius
    # Start must be clear while the vehicle is still near it, and the
    # goal must be clear by the time the vehicle can arrive (a mover may
    # legitimately sweep the goal region early in the run).
    for ob in course.obstacles:
        for t in np.linspace(0.0, 0.2 * course.t_final, 20):
            assert ob.distance(course.start, t) > r_body
        for t in np.linspace(0.6 * course.t_final, course.t_final, 20):
            assert ob.distance(course.goal, t) > r_body
    # At least one obstacle actually threatens the straight-line path
    ref = course.make_reference()
    threatened = False
    for t in np.linspace(0.0, course.t_final, 200):
        p = ref(t).pos
        for ob in course.obstacles:
            if ob.distance(p, t) < r_body:
                threatened = True
    assert threatened, f"course {name} never blocks the reference line"


@pytest.mark.parametrize("name", list(ALL_COURSES))
def test_mpc_completes_course(name):
    course = ALL_COURSES[name]()
    ref = course.make_reference()
    mpc = NonlinearMPC(reference=ref, obstacles=course.obstacles)
    log = run_closed_loop(
        mpc, ref, MODEL.hover_state(course.start), course.t_final,
        obstacles=course.obstacles, collision_margin=MODEL.params.radius,
    )
    assert not log.collided, f"MPC body-collided on {name}"
    assert np.linalg.norm(log.x[-1, :3] - course.goal) < course.goal_tolerance
