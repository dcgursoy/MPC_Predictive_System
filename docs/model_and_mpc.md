# Dynamics Model and MPC Formulation

## 1. Quadrotor rigid-body model (simulation truth)

State (13): position `p ∈ R³` (world, z-up), velocity `v ∈ R³` (world),
unit quaternion `q = [w x y z]` (body→world), body rates `ω ∈ R³`.
Inputs (4): collective thrust `T` along body +z and body torques `τ ∈ R³`.

```
ṗ = v
v̇ = -g e₃ + (T/m) R(q) e₃ - (c_d/m) v
q̇ = ½ q ⊗ [0, ω]
ω̇ = J⁻¹ (τ - ω × J ω)
```

Parameters model a ~1 kg X-config quadrotor: `m = 1.0 kg`,
`J = diag(8.2, 8.2, 16.4)·10⁻³ kg m²`, arm 0.17 m, per-motor thrust
0–6 N (thrust-to-weight ≈ 2.4), linear drag `c_d = 0.05 N·s/m`, body
collision radius 0.30 m.

**Why thrust + torques instead of per-motor thrusts?** The wrench form
keeps the input space decoupled (1 force + 3 moments) which conditions
the optimization better. Physical realism is preserved by the X-config
mixer: every commanded wrench is mapped to the four motor thrusts,
clipped to [0, 6] N per motor, and mapped back — so saturation behaves
exactly like a real vehicle's, including the roll/yaw coupling that box
bounds on the wrench would miss.

**Why quaternions instead of Euler angles?** No gimbal lock during
aggressive maneuvers, a 4-parameter chart with a single algebraic
constraint (unit norm) instead of trigonometric nonlinearities, and
cheap renormalization after each RK4 step.

Integration is classic RK4 at 500 Hz. The model is validated against
closed-form physics (tests/test_dynamics.py): free-fall parabola to
4e-13 m, hover trim as an exact equilibrium, principal-axis torque
ramps, tilted-thrust kinematics, energy + world-frame angular-momentum
conservation in torque-free tumbling (~1e-13 relative drift), a 4th-
order convergence check, and mixer round-trip/saturation tests.

## 2. Baseline: cascaded PID + potential field

PX4-style cascade, each stage ~3× slower than the one below:

```
position PID (≈2.5 rad/s) → acceleration command
  → (collective thrust, attitude setpoint)      [tilt-limited, 35°]
attitude P (≈8 rad/s)     → body-rate setpoint
rate P + ω×Jω feedforward (≈20 rad/s) → torques
```

Anti-windup by conditional integration (integrate only within 0.5 m of
the setpoint, bleed elsewhere). Reactive avoidance is a Khatib
potential field on body-surface distance with two additions that make
it a fair baseline: a tangential *vortex* term so head-on encounters
slide around obstacles rather than stalling, and closing-rate damping
computed against the obstacle's velocity so moving obstacles repel
harder on approach.

## 3. Nonlinear MPC

**Internal model (10 states).** Position, velocity, quaternion; inputs
`u = [T, ω_cmd]`. The MPC commands body *rates* and the inner rate P
loop (bandwidth 20 rad/s, ~6× the attitude timescales in the horizon)
tracks them — the standard rate-input formulation for quadrotor MPC:
it removes the stiffest dynamics from the NLP while staying genuinely
nonlinear through the attitude kinematics.

**Horizon.** N = 20 stages × 75 ms = 1.5 s lookahead (≈ 3.75 m at
cruise speed), re-solved every 20 ms control cycle from the measured
state (receding horizon). RK4 shooting inside each stage.

**NLP (direct multiple shooting):**

```
min   Σₖ ‖pₖ-p̄ₖ‖²_Qp + ‖vₖ-v̄ₖ‖²_Qv + ‖uₖ-u_hover‖²_R + ‖Δuₖ‖²_Rd + w_s sₖ
      + ‖p_N-p̄_N‖²_QpT + ‖v_N-v̄_N‖²_QvT
s.t.  xₖ₊₁ = f_RK4(xₖ, uₖ)                    (10·(N+1) equalities)
      0 ≤ T ≤ 24 N,  |ω_cmd| ≤ 3 rad/s
      ‖pₖ - cⱼ(tₖ)‖² ≥ Rⱼ² - sₖ,  sₖ ≥ 0      (keep-outs, every 2nd stage)
```

with `Rⱼ = rⱼ + 0.30 (body) + 0.10 (margin)`. Keep-outs are hard
constraints softened by one L1 slack per stage priced high enough
(w_s = 400) that nonzero slack means "physically cornered", never
"cheaper to clip the obstacle". Moving obstacles enter through
`cⱼ(tₖ)`: their known trajectory is sampled across the horizon, which
is what produces the early, predictive dodges.

## 4. Real-time solver engineering (72 ms → 5 ms mean)

The naive implementation (casadi.Opti, MX graphs, default IPOPT)
averaged 46–72 ms per solve with >700 ms spikes. Changes, in order of
impact:

1. **Build once, solve many.** The NLP is constructed once as an
   SX-expanded `casadi.nlpsol` function; per-cycle work is a single
   call with new parameter values (initial state, reference samples,
   obstacle samples). ~5× on average solve time.
2. **Warm-start primal and dual.** Previous solution shifted one
   stage + previous multipliers, with IPOPT's warm-start bound-push
   options. Most cycles converge in 2–4 iterations.
3. **Anytime settings.** `max_iter = 15`, `tol = 1e-3`,
   `acceptable_tol = 5e-2` after 1 iteration: a receding-horizon
   controller re-solves in 20 ms, so per-cycle optimality can be
   modest; capping iterations bounds tail latency.
4. **Constraint thinning.** Keep-outs every 2nd stage (the 0.10 m
   margin covers between-stage corner cutting) and parametric culling
   of obstacles the horizon cannot reach (radius set to 0 keeps the
   NLP structure fixed).
5. **Rejected alternatives.** `sqpmethod`+qrqp (RTI-flavor SQP) was
   slower *and* got stuck on the obstacle course; acados would need a
   C toolchain not present on the dev machine and wasn't needed once
   IPOPT hit 4.8 ms mean / 11.1 ms p95 across 1,894 course solves.

## 5. Honest limitations

- Obstacle trajectories are assumed known over the horizon (no
  estimation); the stated scope is control, not perception.
- The keep-out constraint set is nonconvex; IPOPT finds local optima.
  With an obstacle dead-center on the path the solver picks a side (in
  the symmetric case, the vertical one) — global route choice belongs
  to a planner layer this project deliberately omits.
- Solve-time statistics are wall-clock on Windows/Python; occasional
  30–60 ms outliers correlate with OS scheduling, not solver
  difficulty.
- No sensor noise / state estimation; the controller sees true state.
