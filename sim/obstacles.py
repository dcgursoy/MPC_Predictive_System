"""Obstacle primitives for the course environments.

All obstacles are spheres (or sphere-swept shapes) because a signed
distance to a sphere is smooth and cheap — exactly what both the
potential-field avoidance (Phase 2) and the MPC keep-out constraints
(Phase 3) need. Every obstacle exposes position(t) so moving obstacles
with known trajectories are first-class citizens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SphereObstacle:
    """Static sphere: center c, radius r."""

    center: np.ndarray
    radius: float
    name: str = "sphere"

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)

    def position(self, t: float) -> np.ndarray:
        return self.center

    def velocity(self, t: float) -> np.ndarray:
        return np.zeros(3)

    def distance(self, p: np.ndarray, t: float) -> float:
        """Signed distance from point p to the obstacle surface (>0 outside)."""
        return float(np.linalg.norm(p - self.position(t)) - self.radius)


@dataclass
class MovingSphereObstacle(SphereObstacle):
    """Sphere translating at constant velocity: c(t) = c0 + v t."""

    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    name: str = "moving_sphere"

    def __post_init__(self):
        super().__post_init__()
        self.vel = np.asarray(self.vel, dtype=float)

    def position(self, t: float) -> np.ndarray:
        return self.center + self.vel * t

    def velocity(self, t: float) -> np.ndarray:
        return self.vel


@dataclass
class OscillatingSphereObstacle(SphereObstacle):
    """Sphere oscillating sinusoidally: c(t) = c0 + A sin(w t + phase) axis."""

    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    amplitude: float = 1.0
    omega: float = 1.0
    phase: float = 0.0
    name: str = "oscillating_sphere"

    def __post_init__(self):
        super().__post_init__()
        self.axis = np.asarray(self.axis, dtype=float)
        self.axis /= np.linalg.norm(self.axis)

    def position(self, t: float) -> np.ndarray:
        return self.center + self.amplitude * np.sin(self.omega * t + self.phase) * self.axis

    def velocity(self, t: float) -> np.ndarray:
        return self.amplitude * self.omega * np.cos(self.omega * t + self.phase) * self.axis


def min_clearance(p: np.ndarray, obstacles, t: float) -> float:
    """Smallest signed distance from p to any obstacle surface."""
    if not obstacles:
        return np.inf
    return min(ob.distance(p, t) for ob in obstacles)
