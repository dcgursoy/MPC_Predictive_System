"""Reference trajectories.

A reference is a callable ref(t) -> Reference with position, velocity,
acceleration feedforward, and yaw. Controllers consume this interface;
courses (Phase 4) build goal-directed references from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Reference:
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0


class HoverReference:
    def __init__(self, position):
        self.p = np.asarray(position, dtype=float)

    def __call__(self, t: float) -> Reference:
        return Reference(pos=self.p.copy())


class StepReference:
    """Hold p0, then step to p1 at t_step (classic step-response test)."""

    def __init__(self, p0, p1, t_step=1.0):
        self.p0 = np.asarray(p0, dtype=float)
        self.p1 = np.asarray(p1, dtype=float)
        self.t_step = t_step

    def __call__(self, t: float) -> Reference:
        return Reference(pos=(self.p0 if t < self.t_step else self.p1).copy())


class CircleReference:
    """Constant-speed circle at fixed altitude, with exact vel/acc feedforward."""

    def __init__(self, center, radius, omega, yaw_follows_path=True):
        self.c = np.asarray(center, dtype=float)
        self.r = radius
        self.w = omega
        self.yaw_follows_path = yaw_follows_path

    def __call__(self, t: float) -> Reference:
        wt = self.w * t
        pos = self.c + self.r * np.array([np.cos(wt), np.sin(wt), 0.0])
        vel = self.r * self.w * np.array([-np.sin(wt), np.cos(wt), 0.0])
        acc = -self.r * self.w**2 * np.array([np.cos(wt), np.sin(wt), 0.0])
        yaw = np.arctan2(vel[1], vel[0]) if self.yaw_follows_path else 0.0
        return Reference(pos=pos, vel=vel, acc=acc, yaw=yaw)


class LineReference:
    """Straight line from start to goal with a trapezoidal speed profile."""

    def __init__(self, start, goal, cruise_speed=2.0, accel=2.0, t_start=0.0):
        self.p0 = np.asarray(start, dtype=float)
        self.p1 = np.asarray(goal, dtype=float)
        d = self.p1 - self.p0
        self.length = float(np.linalg.norm(d))
        self.dir = d / self.length if self.length > 0 else np.zeros(3)
        self.v = cruise_speed
        self.a = accel
        self.t_start = t_start

        # Trapezoid (or triangle, if too short to reach cruise speed)
        if self.length < self.v**2 / self.a:
            self.v = np.sqrt(self.length * self.a)
        self.t_acc = self.v / self.a
        self.d_acc = 0.5 * self.a * self.t_acc**2
        self.t_cruise = (self.length - 2 * self.d_acc) / self.v
        self.t_total = 2 * self.t_acc + self.t_cruise

    def __call__(self, t: float) -> Reference:
        s = t - self.t_start
        if s <= 0:
            d, v, a = 0.0, 0.0, 0.0
        elif s < self.t_acc:
            d, v, a = 0.5 * self.a * s**2, self.a * s, self.a
        elif s < self.t_acc + self.t_cruise:
            d = self.d_acc + self.v * (s - self.t_acc)
            v, a = self.v, 0.0
        elif s < self.t_total:
            s2 = self.t_total - s
            d = self.length - 0.5 * self.a * s2**2
            v, a = self.a * s2, -self.a
        else:
            d, v, a = self.length, 0.0, 0.0
        return Reference(
            pos=self.p0 + d * self.dir,
            vel=v * self.dir,
            acc=a * self.dir,
            yaw=np.arctan2(self.dir[1], self.dir[0]) if self.length > 0 else 0.0,
        )
