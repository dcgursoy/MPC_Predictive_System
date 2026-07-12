"""Test courses: start, goal, obstacles, and a reference factory.

Every course's reference is the straight start->goal line at cruise
speed — deliberately blind to obstacles — so obstacle handling is
entirely the controller's job. Moving obstacles are timed to be ON the
flight path when the vehicle arrives (a controller that only reacts to
current obstacle position must react late; one that predicts can
replan early).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .obstacles import (
    MovingSphereObstacle,
    OscillatingSphereObstacle,
    SphereObstacle,
)
from .trajectory import LineReference


@dataclass
class Course:
    name: str
    start: np.ndarray
    goal: np.ndarray
    obstacles: list = field(default_factory=list)
    cruise_speed: float = 2.5
    t_final: float = 12.0
    goal_tolerance: float = 0.30

    def __post_init__(self):
        self.start = np.asarray(self.start, dtype=float)
        self.goal = np.asarray(self.goal, dtype=float)

    def make_reference(self) -> LineReference:
        return LineReference(self.start, self.goal,
                             cruise_speed=self.cruise_speed)


def course_slalom() -> Course:
    """Static field: four spheres alternating off-axis force an S-path."""
    return Course(
        name="slalom",
        start=[0.0, 0.0, 1.5],
        goal=[12.0, 0.0, 1.5],
        obstacles=[
            SphereObstacle(center=[3.0, 0.45, 1.5], radius=0.80),
            SphereObstacle(center=[6.0, -0.65, 1.6], radius=0.90),
            SphereObstacle(center=[9.0, 0.55, 1.4], radius=0.70),
            SphereObstacle(center=[10.8, -0.90, 1.5], radius=0.55),
        ],
        cruise_speed=2.5,
        t_final=12.0,
    )


def course_crossing() -> Course:
    """A sphere crosses the path perpendicular, timed to be exactly in
    the way when the vehicle arrives; an oscillating sphere guards the
    final approach."""
    # Vehicle reaches x = 6 m at t ~ 2.9 s (trapezoidal profile,
    # 2.5 m/s cruise, 2 m/s^2 accel). Crosser arrives at y = 0 then too.
    return Course(
        name="crossing",
        start=[0.0, 0.0, 1.5],
        goal=[12.0, 0.0, 1.5],
        obstacles=[
            SphereObstacle(center=[3.0, -0.40, 1.5], radius=0.60),
            MovingSphereObstacle(center=[6.0, 3.77, 1.5], radius=0.60,
                                 vel=[0.0, -1.3, 0.0]),
            OscillatingSphereObstacle(center=[9.5, 0.0, 1.5], radius=0.55,
                                      axis=[0.0, 1.0, 0.0], amplitude=1.2,
                                      omega=0.9),
        ],
        cruise_speed=2.5,
        t_final=12.0,
    )


def course_gauntlet() -> Course:
    """Narrow gate, a head-on mover closing at 1.5 m/s, and a static
    guard hovering just before the goal."""
    return Course(
        name="gauntlet",
        start=[0.0, 0.0, 1.5],
        goal=[14.0, 0.0, 1.5],
        obstacles=[
            # Gate at x = 4: 1.5 m gap between inflated surfaces
            SphereObstacle(center=[4.0, 1.55, 1.5], radius=0.80),
            SphereObstacle(center=[4.0, -1.55, 1.5], radius=0.80),
            # Head-on: starts past the goal, drives down the path axis
            MovingSphereObstacle(center=[16.0, 0.15, 1.5], radius=0.60,
                                 vel=[-1.5, 0.0, 0.0]),
            SphereObstacle(center=[11.5, 0.35, 1.7], radius=0.65),
        ],
        cruise_speed=2.5,
        t_final=14.0,
    )


ALL_COURSES = {
    "slalom": course_slalom,
    "crossing": course_crossing,
    "gauntlet": course_gauntlet,
}
