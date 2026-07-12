"""Nonlinear rigid-body quadrotor dynamics.

State (13,):  x = [p(3), v(3), q(4), omega(3)]
    p     : position, world frame (ENU-style: z up), m
    v     : velocity, world frame, m/s
    q     : unit quaternion [w, x, y, z], rotates body frame -> world frame
    omega : angular rate, body frame, rad/s

Input (4,):   u = [T, tau_x, tau_y, tau_z]
    T   : collective thrust along body +z, N
    tau : body torques, N m

Equations of motion:
    p_dot     = v
    v_dot     = -g e3 + (T/m) R(q) e3 - (c_d/m) v
    q_dot     = 0.5 * q  (x)  [0, omega]
    omega_dot = J^-1 (tau - omega x J omega)
"""

import numpy as np

from .params import QuadrotorParams

# State vector layout
IDX_POS = slice(0, 3)
IDX_VEL = slice(3, 6)
IDX_QUAT = slice(6, 10)
IDX_OMEGA = slice(10, 13)
STATE_DIM = 13
INPUT_DIM = 4


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 (x) q2, [w, x, y, z] convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix R such that v_world = R @ v_body."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX Euler angles (rad) to quaternion [w, x, y, z]."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ])


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Quaternion [w, x, y, z] to ZYX Euler angles [roll, pitch, yaw] (rad)."""
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


class QuadrotorDynamics:
    """Continuous-time quadrotor model with an RK4 discrete stepper."""

    def __init__(self, params: QuadrotorParams | None = None):
        self.params = params or QuadrotorParams()

    def hover_state(self, position=(0.0, 0.0, 1.0)) -> np.ndarray:
        x = np.zeros(STATE_DIM)
        x[IDX_POS] = position
        x[IDX_QUAT] = [1.0, 0.0, 0.0, 0.0]
        return x

    def hover_input(self) -> np.ndarray:
        return np.array([self.params.hover_thrust, 0.0, 0.0, 0.0])

    def continuous_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """State derivative x_dot = f(x, u)."""
        p = self.params
        v = x[IDX_VEL]
        q = x[IDX_QUAT]
        omega = x[IDX_OMEGA]
        thrust, tau = u[0], u[1:4]

        R = quat_to_rotmat(q)
        v_dot = (
            np.array([0.0, 0.0, -p.gravity])
            + (thrust / p.mass) * R[:, 2]
            - (p.drag_coeff / p.mass) * v
        )
        q_dot = 0.5 * quat_multiply(q, np.concatenate(([0.0], omega)))
        omega_dot = p.inertia_inv @ (tau - np.cross(omega, p.inertia @ omega))

        x_dot = np.empty(STATE_DIM)
        x_dot[IDX_POS] = v
        x_dot[IDX_VEL] = v_dot
        x_dot[IDX_QUAT] = q_dot
        x_dot[IDX_OMEGA] = omega_dot
        return x_dot

    def clip_input(self, u: np.ndarray) -> np.ndarray:
        """Saturate the wrench so every implied motor thrust is feasible."""
        p = self.params
        f = np.clip(p.motor_thrusts(u), p.motor_thrust_min, p.motor_thrust_max)
        return p.mixer @ f

    def step(self, x: np.ndarray, u: np.ndarray, dt: float,
             enforce_limits: bool = True) -> np.ndarray:
        """One RK4 step of the true dynamics; renormalizes the quaternion."""
        if enforce_limits:
            u = self.clip_input(u)
        f = self.continuous_dynamics
        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x_next[IDX_QUAT] /= np.linalg.norm(x_next[IDX_QUAT])
        return x_next
