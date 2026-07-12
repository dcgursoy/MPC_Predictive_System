"""Physics validation for the quadrotor dynamics model.

Every test checks the simulation against a closed-form result, so a sign
error or frame mix-up in the dynamics fails loudly here before any
controller work builds on top of it.
"""

import numpy as np
import pytest

from dynamics import (
    IDX_OMEGA,
    IDX_POS,
    IDX_QUAT,
    IDX_VEL,
    QuadrotorDynamics,
    QuadrotorParams,
    quat_from_euler,
    quat_to_rotmat,
)

DT = 0.001


def _dragless() -> QuadrotorDynamics:
    return QuadrotorDynamics(QuadrotorParams(drag_coeff=0.0))


def _simulate(model, x0, u, t_final, dt=DT, enforce_limits=False):
    x = x0.copy()
    n = int(round(t_final / dt))
    for _ in range(n):
        x = model.step(x, u, dt, enforce_limits=enforce_limits)
    return x


def test_free_fall_matches_analytic():
    model = _dragless()
    g = model.params.gravity
    t = 2.0
    x0 = model.hover_state(position=(0.0, 0.0, 50.0))
    x = _simulate(model, x0, np.zeros(4), t)
    assert x[2] == pytest.approx(50.0 - 0.5 * g * t**2, abs=1e-8)
    assert x[5] == pytest.approx(-g * t, abs=1e-9)
    # No lateral motion, no rotation
    np.testing.assert_allclose(x[IDX_POS][:2], 0.0, atol=1e-12)
    np.testing.assert_allclose(x[IDX_OMEGA], 0.0, atol=1e-12)


def test_hover_is_equilibrium():
    model = QuadrotorDynamics()  # default params, drag on (irrelevant at v=0)
    x0 = model.hover_state(position=(1.0, -2.0, 3.0))
    x = _simulate(model, x0, model.hover_input(), t_final=5.0)
    np.testing.assert_allclose(x[IDX_POS], [1.0, -2.0, 3.0], atol=1e-9)
    np.testing.assert_allclose(x[IDX_VEL], 0.0, atol=1e-9)
    np.testing.assert_allclose(x[IDX_QUAT], [1, 0, 0, 0], atol=1e-12)


def test_constant_roll_torque_gives_linear_rate_ramp():
    model = _dragless()
    tau_x, t = 0.02, 1.0
    u = np.array([0.0, tau_x, 0.0, 0.0])
    x = _simulate(model, model.hover_state(), u, t)
    # Rotation about a principal axis: gyroscopic term vanishes exactly
    expected = tau_x / model.params.inertia[0, 0] * t
    assert x[IDX_OMEGA][0] == pytest.approx(expected, rel=1e-9)


def test_constant_yaw_torque_gives_linear_rate_ramp():
    model = _dragless()
    tau_z, t = 0.01, 1.0
    u = np.array([0.0, 0.0, 0.0, tau_z])
    x = _simulate(model, model.hover_state(), u, t)
    expected = tau_z / model.params.inertia[2, 2] * t
    assert x[IDX_OMEGA][2] == pytest.approx(expected, rel=1e-9)


def test_tilted_thrust_accelerates_laterally():
    """Roll phi with T = mg/cos(phi): altitude holds, a_y = -g tan(phi)."""
    model = _dragless()
    p = model.params
    phi = np.deg2rad(10.0)
    t = 1.0
    x0 = model.hover_state(position=(0.0, 0.0, 5.0))
    x0[IDX_QUAT] = quat_from_euler(phi, 0.0, 0.0)
    u = np.array([p.mass * p.gravity / np.cos(phi), 0.0, 0.0, 0.0])
    x = _simulate(model, x0, u, t)
    a_y = -p.gravity * np.tan(phi)
    assert x[2] == pytest.approx(5.0, abs=1e-8)               # altitude held
    assert x[1] == pytest.approx(0.5 * a_y * t**2, rel=1e-8)  # lateral drift
    assert x[4] == pytest.approx(a_y * t, rel=1e-8)


def test_torque_free_tumble_conserves_momentum_and_energy():
    """Zero external torque: world-frame angular momentum and rotational
    kinetic energy are invariants of the Euler equations."""
    model = _dragless()
    J = model.params.inertia
    x = model.hover_state()
    x[IDX_OMEGA] = [2.0, -1.5, 3.0]  # aggressive tumble, all axes

    def invariants(state):
        R = quat_to_rotmat(state[IDX_QUAT])
        w = state[IDX_OMEGA]
        return R @ (J @ w), 0.5 * w @ J @ w

    L0, E0 = invariants(x)
    u = np.zeros(4)
    for _ in range(int(5.0 / DT)):
        x = model.step(x, u, DT, enforce_limits=False)
    L, E = invariants(x)
    np.testing.assert_allclose(L, L0, rtol=1e-6)
    assert E == pytest.approx(E0, rel=1e-8)


def test_quaternion_stays_normalized():
    model = _dragless()
    x = model.hover_state()
    x[IDX_OMEGA] = [1.0, 2.0, -0.5]
    for _ in range(int(10.0 / 0.01)):
        x = model.step(x, np.zeros(4), 0.01, enforce_limits=False)
    assert np.linalg.norm(x[IDX_QUAT]) == pytest.approx(1.0, abs=1e-12)


def test_rk4_is_fourth_order():
    """Halving dt should shrink the one-step-composed error ~16x."""
    model = _dragless()
    x0 = model.hover_state()
    # Aggressive tumble so truncation error dominates float roundoff
    x0[IDX_OMEGA] = [6.0, -5.0, 4.0]
    u = np.array([15.0, 0.05, -0.04, 0.02])
    t = 1.0

    def endpoint(dt):
        return _simulate(model, x0, u, t, dt=dt)

    ref = endpoint(1e-4)
    err_coarse = np.linalg.norm(endpoint(2e-2) - ref)
    err_fine = np.linalg.norm(endpoint(1e-2) - ref)
    assert err_coarse / err_fine > 8.0  # ideal 16, allow slack


def test_mixer_hover_split_and_roundtrip():
    p = QuadrotorParams()
    f = p.motor_thrusts(np.array([p.hover_thrust, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(f, p.hover_thrust / 4.0, rtol=1e-12)
    np.testing.assert_allclose(p.mixer @ p.mixer_inv, np.eye(4), atol=1e-12)


def test_input_clipping_respects_motor_limits():
    model = QuadrotorDynamics()
    p = model.params
    u = model.clip_input(np.array([1e3, 5.0, -5.0, 2.0]))
    f = p.motor_thrusts(u)
    assert np.all(f >= p.motor_thrust_min - 1e-12)
    assert np.all(f <= p.motor_thrust_max + 1e-12)
    # A feasible wrench passes through untouched
    u_ok = np.array([p.hover_thrust, 0.05, -0.05, 0.01])
    np.testing.assert_allclose(model.clip_input(u_ok), u_ok, atol=1e-12)
