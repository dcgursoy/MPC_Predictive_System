"""Reactive potential-field obstacle avoidance for the PID baseline.

Classic Khatib repulsive potential plus a velocity-damping term that
only fires when closing on an obstacle (using the obstacle's velocity,
so a moving obstacle repels harder when it approaches the vehicle).
This is a competent reactive scheme — its known failure modes (local
minima, late reaction to fast obstacles) are exactly what MPC's
predictive replanning fixes, which is the point of the comparison.
"""

from __future__ import annotations

import numpy as np


class PotentialFieldAvoidance:
    def __init__(
        self,
        obstacles=(),
        influence_dist: float = 2.0,   # m beyond obstacle surface
        k_rep: float = 6.0,            # radial repulsion gain
        k_tan: float = 6.0,            # tangential (vortex) gain
        k_damp: float = 6.0,           # approach-damping gain
        max_accel: float = 12.0,       # m/s^2 cap on total avoidance accel
        eps: float = 0.05,             # m, keeps 1/d finite at contact
    ):
        self.obstacles = list(obstacles)
        self.d0 = influence_dist
        self.k_rep = k_rep
        self.k_tan = k_tan
        self.k_damp = k_damp
        self.max_accel = max_accel
        self.eps = eps

    def accel(self, t: float, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Avoidance acceleration command at position p, velocity v."""
        a = np.zeros(3)
        for ob in self.obstacles:
            c = ob.position(t)
            delta = p - c
            dist_c = np.linalg.norm(delta)
            if dist_c < 1e-9:
                continue
            d = dist_c - ob.radius  # signed distance to surface
            if d >= self.d0:
                continue
            n = delta / dist_c  # unit vector away from obstacle
            d_eff = max(d, self.eps)
            v_rel = v - ob.velocity(t)
            # Khatib repulsion: grows as 1/d^2 near the surface
            shape = (1.0 / d_eff - 1.0 / self.d0) / d_eff**2
            a += self.k_rep * shape * n
            # Vortex term: push sideways (horizontal tangent, in the
            # direction of current motion) so a near-head-on approach
            # slides around the obstacle instead of stalling against it.
            tan = np.cross([0.0, 0.0, 1.0], n)
            tn = np.linalg.norm(tan)
            if tn > 1e-6:
                tan /= tn
                if tan @ v_rel < 0.0:
                    tan = -tan
                a += self.k_tan * shape * tan
            # Damp only the closing component of relative velocity
            closing = v_rel @ n
            if closing < 0.0:
                a += -self.k_damp * closing * n * (1.0 - d_eff / self.d0)
        norm = np.linalg.norm(a)
        if norm > self.max_accel:
            a *= self.max_accel / norm
        return a
