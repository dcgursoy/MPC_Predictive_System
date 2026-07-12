"""Closed-loop simulation harness shared by all controllers.

The true dynamics run at a fast fixed step (dt_sim); the controller runs
at its own rate (dt_ctrl) with zero-order hold in between — mirroring a
real flight stack where the plant is continuous and the controller is a
discrete loop. Everything needed for Phase 5 metrics and the Phase 6
visualization is logged every control step, including whatever extra
info the controller reports (e.g. the MPC's full predicted horizon and
solve time).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from dynamics import IDX_POS, QuadrotorDynamics
from sim.obstacles import min_clearance


@dataclass
class SimLog:
    t: np.ndarray = None                 # (N,) control-step times
    x: np.ndarray = None                 # (N, 13) states at control steps
    u: np.ndarray = None                 # (N, 4) applied wrench commands
    ref_pos: np.ndarray = None           # (N, 3) reference positions
    pos_err: np.ndarray = None           # (N,) tracking error norm
    clearance: np.ndarray = None         # (N,) min signed distance to obstacles
    ctrl_time: np.ndarray = None         # (N,) wall-clock controller time [s]
    obstacle_pos: np.ndarray = None      # (N, n_obs, 3)
    obstacle_radius: np.ndarray = None   # (n_obs,)
    info: list = field(default_factory=list)   # controller info dict per step
    collided: bool = False
    t_collision: float | None = None

    @property
    def min_clearance(self) -> float:
        return float(np.min(self.clearance)) if self.clearance.size else np.inf


def run_closed_loop(
    controller,
    reference,
    x0: np.ndarray,
    t_final: float,
    obstacles=(),
    dynamics: QuadrotorDynamics | None = None,
    dt_sim: float = 0.002,
    dt_ctrl: float = 0.02,
    collision_margin: float = 0.0,
    stop_on_collision: bool = False,
) -> SimLog:
    """Simulate controller closing the loop around the true dynamics.

    controller: object with .compute(t, x, ref) -> (u, info_dict) and
                optionally .reset().
    reference:  callable t -> Reference.
    """
    model = dynamics or QuadrotorDynamics()
    if hasattr(controller, "reset"):
        controller.reset()

    n_ctrl = int(round(t_final / dt_ctrl))
    sub_steps = max(1, int(round(dt_ctrl / dt_sim)))
    dt_inner = dt_ctrl / sub_steps

    N = n_ctrl
    log = SimLog(
        t=np.empty(N), x=np.empty((N, 13)), u=np.empty((N, 4)),
        ref_pos=np.empty((N, 3)), pos_err=np.empty(N),
        clearance=np.empty(N), ctrl_time=np.empty(N),
        obstacle_pos=np.empty((N, len(obstacles), 3)),
        obstacle_radius=np.array([ob.radius for ob in obstacles]),
    )

    x = x0.copy()
    for k in range(n_ctrl):
        t = k * dt_ctrl
        ref = reference(t)

        tic = time.perf_counter()
        u, info = controller.compute(t, x, ref)
        toc = time.perf_counter() - tic

        log.t[k] = t
        log.x[k] = x
        log.u[k] = u
        log.ref_pos[k] = ref.pos
        log.pos_err[k] = np.linalg.norm(x[IDX_POS] - ref.pos)
        log.clearance[k] = min_clearance(x[IDX_POS], obstacles, t)
        log.ctrl_time[k] = toc
        for j, ob in enumerate(obstacles):
            log.obstacle_pos[k, j] = ob.position(t)
        log.info.append(info)

        if log.clearance[k] < collision_margin and not log.collided:
            log.collided = True
            log.t_collision = t
            if stop_on_collision:
                _truncate(log, k + 1)
                return log

        for i in range(sub_steps):
            x = model.step(x, u, dt_inner)

    return log


def _truncate(log: SimLog, n: int):
    for name in ("t", "x", "u", "ref_pos", "pos_err", "clearance",
                 "ctrl_time", "obstacle_pos"):
        setattr(log, name, getattr(log, name)[:n])
    log.info = log.info[:n]
