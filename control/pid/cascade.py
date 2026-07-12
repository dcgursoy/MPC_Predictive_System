"""Cascaded PID flight controller (PX4-style architecture).

    position PID  ->  acceleration command
                  ->  (collective thrust, attitude setpoint)
    attitude P    ->  body-rate setpoint
    rate P + gyroscopic feedforward -> body torques

Loop bandwidths are separated ~3x per stage (position ~2.5 rad/s,
attitude ~8, rate ~20) — the standard timescale-separation argument
that makes the cascade behave like nested SISO loops.
"""

from __future__ import annotations

import numpy as np

from dynamics import (
    IDX_OMEGA,
    IDX_POS,
    IDX_QUAT,
    IDX_VEL,
    QuadrotorParams,
    quat_to_rotmat,
)
from sim.trajectory import Reference


class CascadedPIDController:
    def __init__(
        self,
        params: QuadrotorParams | None = None,
        avoidance=None,
        dt: float = 0.02,
        # Position loop (world frame, per-axis)
        kp_pos=(6.0, 6.0, 8.0),
        kd_pos=(4.0, 4.0, 5.0),
        ki_pos=(0.3, 0.3, 0.5),
        int_limit: float = 0.5,       # m s, anti-windup clamp
        int_radius: float = 0.5,      # m; integrate only near the setpoint
        # Attitude loop (P on rotation error, body axes)
        kp_att=(8.0, 8.0, 4.0),
        # Rate loop (P, body axes) — torque = J * kp_rate * rate error
        kp_rate=(20.0, 20.0, 10.0),
        max_tilt_deg: float = 35.0,
    ):
        self.p = params or QuadrotorParams()
        self.avoidance = avoidance
        self.dt = dt
        self.kp_pos = np.asarray(kp_pos)
        self.kd_pos = np.asarray(kd_pos)
        self.ki_pos = np.asarray(ki_pos)
        self.int_limit = int_limit
        self.int_radius = int_radius
        self.kp_att = np.asarray(kp_att)
        self.kp_rate = np.asarray(kp_rate)
        self.max_tilt = np.deg2rad(max_tilt_deg)
        self.reset()

    def reset(self):
        self._int_err = np.zeros(3)

    def compute(self, t: float, x: np.ndarray, ref: Reference):
        p = self.p
        pos, vel = x[IDX_POS], x[IDX_VEL]
        R = quat_to_rotmat(x[IDX_QUAT])
        omega = x[IDX_OMEGA]

        # --- Position PID -> world acceleration command -------------------
        e_p = ref.pos - pos
        e_v = ref.vel - vel
        # Conditional integration: the integral's job is trim-bias removal
        # near the setpoint. Integrating during large transients only winds
        # up and bleeds off slowly (classic windup overshoot).
        if np.linalg.norm(e_p) < self.int_radius:
            self._int_err = np.clip(
                self._int_err + e_p * self.dt, -self.int_limit, self.int_limit
            )
        else:
            self._int_err *= 1.0 - 2.0 * self.dt  # bleed while far away
        a_cmd = (
            self.kp_pos * e_p
            + self.kd_pos * e_v
            + self.ki_pos * self._int_err
            + ref.acc
        )
        a_avoid = np.zeros(3)
        if self.avoidance is not None:
            a_avoid = self.avoidance.accel(t, pos, vel)
            a_cmd = a_cmd + a_avoid

        # Total specific force the rotors must produce (adds gravity comp)
        f_des = a_cmd + np.array([0.0, 0.0, p.gravity])
        # Never command a downward thrust vector
        f_des[2] = max(f_des[2], 0.2 * p.gravity)
        # Tilt limit: cap horizontal component relative to vertical
        h = np.linalg.norm(f_des[:2])
        h_max = f_des[2] * np.tan(self.max_tilt)
        if h > h_max:
            f_des[:2] *= h_max / h

        # --- Thrust magnitude + attitude setpoint -------------------------
        # Thrust is the projection of desired force onto the CURRENT body z:
        # attitude hasn't converged yet, so this is what the rotors can give.
        thrust = p.mass * float(f_des @ R[:, 2])
        thrust = np.clip(thrust, p.thrust_min, p.thrust_max)

        z_des = f_des / np.linalg.norm(f_des)
        x_c = np.array([np.cos(ref.yaw), np.sin(ref.yaw), 0.0])
        y_des = np.cross(z_des, x_c)
        n = np.linalg.norm(y_des)
        if n < 1e-6:  # thrust vector parallel to yaw heading (near-inverted)
            y_des = np.array([0.0, 1.0, 0.0])
        else:
            y_des /= n
        x_des = np.cross(y_des, z_des)
        R_des = np.column_stack([x_des, y_des, z_des])

        # --- Attitude P -> body-rate setpoint ------------------------------
        # Rotation error as axis-angle of R^T R_des (small-angle: vee of skew)
        R_err = R.T @ R_des
        e_att = 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ])
        omega_sp = self.kp_att * e_att

        # --- Rate P + gyroscopic feedforward -> torques ---------------------
        tau = self.p.inertia @ (self.kp_rate * (omega_sp - omega)) + np.cross(
            omega, self.p.inertia @ omega
        )

        u = np.concatenate(([thrust], tau))
        info = {
            "a_cmd": a_cmd,
            "a_avoid": a_avoid,
            "omega_sp": omega_sp,
            "controller": "pid",
        }
        return u, info
