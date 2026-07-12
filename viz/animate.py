"""3D animated visualization of course runs.

Renders one or two (side-by-side) closed-loop runs: obstacles (static
translucent, moving re-meshed per frame), the drone as an oriented
X-frame glyph, its flown trail, the reference line, and — for MPC —
the predicted-horizon ribbon replanned every control cycle, plus a
live readout (time, solve time, tracking error, constraint proximity).

Usage: see scripts/animate_course.py.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from dynamics import IDX_POS, IDX_QUAT, QuadrotorParams, quat_to_rotmat

_SPH_U, _SPH_V = np.meshgrid(np.linspace(0, 2 * np.pi, 16),
                             np.linspace(0, np.pi, 12))
_SPH = np.stack([np.cos(_SPH_U) * np.sin(_SPH_V),
                 np.sin(_SPH_U) * np.sin(_SPH_V),
                 np.cos(_SPH_V)])


def _sphere_xyz(center, radius):
    return (center[0] + radius * _SPH[0],
            center[1] + radius * _SPH[1],
            center[2] + radius * _SPH[2])


class _Panel:
    """One 3D axes animating one run."""

    def __init__(self, ax, course, log, label, params: QuadrotorParams):
        self.ax = ax
        self.course = course
        self.log = log
        self.label = label
        self.arm = 2.0 * params.arm_length
        self.is_mpc = bool(log.info and log.info[0].get("controller") == "mpc")
        self.inflation = params.radius + 0.10

        # --- static scene -------------------------------------------------
        for ob in course.obstacles:
            if np.any(np.abs(ob.velocity(0.0)) > 1e-9) or "oscillating" in ob.name:
                continue
            ax.plot_surface(*_sphere_xyz(ob.position(0), ob.radius),
                            color="tab:red", alpha=0.35, linewidth=0)
        ax.plot([course.start[0], course.goal[0]],
                [course.start[1], course.goal[1]],
                [course.start[2], course.goal[2]],
                "k--", lw=0.8, alpha=0.6)
        ax.scatter(*course.goal, marker="*", s=140, color="k")

        # --- dynamic artists -----------------------------------------------
        self.moving = [ob for ob in course.obstacles
                       if np.any(np.abs(ob.velocity(0.0)) > 1e-9)
                       or "oscillating" in ob.name]
        self._mov_surf = [None] * len(self.moving)

        (self.trail,) = ax.plot([], [], [], "C0", lw=1.6, alpha=0.9)
        (self.arm1,) = ax.plot([], [], [], "k", lw=2.5)
        (self.arm2,) = ax.plot([], [], [], "k", lw=2.5)
        (self.body,) = ax.plot([], [], [], "o", color="C0", ms=5)
        (self.ribbon,) = ax.plot([], [], [], color="#00c46a", lw=2.4,
                                 alpha=0.85)
        self.ribbon_pts = ax.scatter([], [], [], color="#00c46a", s=6,
                                     alpha=0.6, depthshade=False)
        (self.crash,) = ax.plot([], [], [], "rx", ms=14, mew=3)

        self.readout = ax.text2D(
            0.02, 0.96, "", transform=ax.transAxes, fontsize=8.5,
            family="monospace", va="top",
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85),
        )

        # --- axes cosmetics --------------------------------------------------
        # Frame the flown volume + static obstacles; movers pass through the
        # view (framing on their whole sweep would shrink the scene).
        pts = [log.x[:, :3], course.start[None], course.goal[None]]
        for ob in course.obstacles:
            if ob not in self.moving:
                c, r = ob.position(0.0), ob.radius
                pts.append(np.array([c - r, c + r]))
        pts = np.vstack(pts)
        lo, hi = pts.min(0) - 0.8, pts.max(0) + 0.8
        lo[2] = min(lo[2], 0.0)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo, zoom=1.4)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title(label)
        ax.view_init(elev=22, azim=-60)

    def draw(self, k: int):
        log, ax = self.log, self.ax
        k = min(k, len(log.t) - 1)
        t = log.t[k]
        p = log.x[k, IDX_POS]
        R = quat_to_rotmat(log.x[k, IDX_QUAT])

        # moving obstacles: re-mesh
        for j, ob in enumerate(self.moving):
            if self._mov_surf[j] is not None:
                self._mov_surf[j].remove()
            self._mov_surf[j] = ax.plot_surface(
                *_sphere_xyz(ob.position(t), ob.radius),
                color="tab:orange", alpha=0.45, linewidth=0)

        # drone glyph: two arms along body x/y
        a = self.arm / 2.0
        for artist, axis in ((self.arm1, R[:, 0]), (self.arm2, R[:, 1])):
            artist.set_data_3d([p[0] - a * axis[0], p[0] + a * axis[0]],
                               [p[1] - a * axis[1], p[1] + a * axis[1]],
                               [p[2] - a * axis[2], p[2] + a * axis[2]])
        self.body.set_data_3d([p[0]], [p[1]], [p[2]])
        self.trail.set_data_3d(log.x[:k + 1, 0], log.x[:k + 1, 1],
                               log.x[:k + 1, 2])

        # predicted horizon ribbon (MPC only)
        info = log.info[k]
        if self.is_mpc and "predicted_traj" in info:
            pred = info["predicted_traj"]
            self.ribbon.set_data_3d(pred[:, 0], pred[:, 1], pred[:, 2])
            self.ribbon_pts._offsets3d = (pred[:, 0], pred[:, 1], pred[:, 2])

        # collision marker
        if log.collided and log.t_collision is not None and t >= log.t_collision:
            kc = int(round(log.t_collision / (log.t[1] - log.t[0])))
            pc = log.x[min(kc, len(log.t) - 1), IDX_POS]
            self.crash.set_data_3d([pc[0]], [pc[1]], [pc[2]])

        # readout
        lines = [f"t={t:5.2f} s"]
        if self.is_mpc:
            lines.append(f"solve  {info['solve_time']*1e3:6.1f} ms")
        lines.append(f"trk err {log.pos_err[k]:5.2f} m")
        clr = log.clearance[k]
        if np.isfinite(clr):
            near = clr < self.inflation + 0.05
            flag = " <ACTIVE>" if near else ""
            lines.append(f"clear  {clr:5.2f} m{flag}")
        if log.collided and t >= (log.t_collision or np.inf):
            lines.append("** COLLISION **")
        self.readout.set_text("\n".join(lines))
        self.readout.set_color(
            "red" if log.collided and t >= (log.t_collision or np.inf)
            else "black")


class CourseAnimator:
    """Animate one run, or two runs side by side on the same course."""

    def __init__(self, course, logs: dict, params: QuadrotorParams | None = None,
                 frame_stride: int = 2, figsize_per_panel=(7.2, 6.0)):
        params = params or QuadrotorParams()
        n = len(logs)
        self.fig = plt.figure(figsize=(figsize_per_panel[0] * n,
                                       figsize_per_panel[1]))
        self.panels = []
        for i, (label, log) in enumerate(logs.items()):
            ax = self.fig.add_subplot(1, n, i + 1, projection="3d")
            self.panels.append(_Panel(ax, course, log, label, params))
        self.fig.suptitle(f"course: {course.name}", fontsize=13)
        self.fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.92,
                                 wspace=0.02)
        self.stride = frame_stride
        dt_ctrl = self.panels[0].log.t[1] - self.panels[0].log.t[0]
        # Stop shortly after the last drone stops moving — no point
        # animating everyone hovering at the goal.
        last_active = 0
        for p in self.panels:
            drift = np.linalg.norm(p.log.x[:, :3] - p.log.x[-1, :3], axis=1)
            moving = np.flatnonzero(drift > 0.15)
            last_active = max(last_active,
                              moving[-1] if moving.size else len(p.log.t) - 1)
        n_steps = min(max(len(p.log.t) for p in self.panels),
                      last_active + int(1.5 / dt_ctrl))
        self.n_frames = int(n_steps) // frame_stride
        self.dt_frame = dt_ctrl * frame_stride

    def _draw_frame(self, f: int):
        for p in self.panels:
            p.draw(f * self.stride)

    def animate(self, fps: float | None = None):
        interval = 1000.0 * self.dt_frame if fps is None else 1000.0 / fps
        self.anim = animation.FuncAnimation(
            self.fig, self._draw_frame, frames=self.n_frames,
            interval=interval, blit=False, repeat=True)
        return self.anim

    def save(self, path, fps: float | None = None, dpi: int = 90):
        fps = fps or 1.0 / self.dt_frame
        self.animate(fps=fps)
        path = str(path)
        if path.endswith(".gif"):
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
        self.anim.save(path, writer=writer, dpi=dpi)

    def show(self):
        self.animate()
        plt.show()
