"""Physical parameters for the simulated quadrotor.

Values are representative of a ~1 kg X-configuration research quadrotor
(similar scale to an AscTec Hummingbird / mid-size custom build).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class QuadrotorParams:
    mass: float = 1.0                    # kg
    gravity: float = 9.81                # m/s^2
    # Principal-axis inertia, X-config symmetric about body z (kg m^2)
    inertia: np.ndarray = field(
        default_factory=lambda: np.diag([8.2e-3, 8.2e-3, 1.64e-2])
    )
    arm_length: float = 0.17             # m, rotor hub to center
    # Rotor drag torque per unit thrust (yaw moment coefficient), m
    torque_coeff: float = 0.016
    # Per-motor thrust limits (N). T/W = 4*6/(1*9.81) ~ 2.4
    motor_thrust_min: float = 0.0
    motor_thrust_max: float = 6.0
    # Linear aerodynamic drag on the body, world-frame v (N per m/s).
    # Small but nonzero so hover isn't a knife-edge equilibrium.
    drag_coeff: float = 0.05

    @property
    def inertia_inv(self) -> np.ndarray:
        return np.linalg.inv(self.inertia)

    @property
    def hover_thrust(self) -> float:
        """Collective thrust that exactly balances weight."""
        return self.mass * self.gravity

    @property
    def thrust_min(self) -> float:
        return 4.0 * self.motor_thrust_min

    @property
    def thrust_max(self) -> float:
        return 4.0 * self.motor_thrust_max

    @property
    def mixer(self) -> np.ndarray:
        """Map per-motor thrusts f = [f1..f4] to u = [T, tau_x, tau_y, tau_z].

        X configuration, diagonal pairs spin together:
            1: front-right (CW),  2: back-left (CW),
            3: front-left (CCW),  4: back-right (CCW)
        Body frame: x forward, y left, z up. A CW rotor (seen from above)
        exerts a CCW (+z) reaction torque on the body.
        """
        l = self.arm_length / np.sqrt(2.0)  # moment arm about x/y in X-config
        c = self.torque_coeff
        return np.array([
            [1.0,  1.0,  1.0,  1.0],
            [-l,    l,    l,   -l],
            [-l,    l,   -l,    l],
            [c,    c,   -c,   -c],
        ])

    @property
    def mixer_inv(self) -> np.ndarray:
        return np.linalg.inv(self.mixer)

    def motor_thrusts(self, u: np.ndarray) -> np.ndarray:
        """Per-motor thrusts realizing wrench u = [T, tau_x, tau_y, tau_z]."""
        return self.mixer_inv @ u
