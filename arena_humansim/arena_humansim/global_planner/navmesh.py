from __future__ import annotations

import time

import numpy as np

from arena_humansim.utils.mesh import TOL, Mesh
from arena_humansim.utils.router import Router
from arena_humansim.utils.types import Pose2D, Segment, Segments

from . import GlobalPlanner, PlanRequest

SNAP_MARGIN = 0.01  # a snapped point ends up this far outside the inflation band [m]
SNAP_TRIES = 8
SNAP_CACHE = 4096
FRAME_INSET = 0.01


class NavMeshPlanner(GlobalPlanner):
    def __init__(
        self,
        replan_distance: float = 1.0,
        inflation_radius: float = 0.38,
        comfort_radius: float = 0.6,
    ) -> None:
        super().__init__(replan_distance)
        self._inflation_radius = inflation_radius
        self._comfort_radius = comfort_radius

        self._wall_segments: list[Segment] = []
        self._mesh: Mesh | None = None
        self._router: Router | None = None
        self._snapped: dict[tuple[float, float], tuple[float, float] | None] = {}

    def configure(self, *, inflation_radius: float, resolution: float, comfort_radius: float) -> None:
        if (inflation_radius, comfort_radius) == (self._inflation_radius, self._comfort_radius):
            return
        self._inflation_radius = inflation_radius
        self._comfort_radius = comfort_radius
        self.set_walls(self._wall_segments)

    def set_walls(self, segments: Segments) -> None:
        self._forget_paths()
        self._snapped.clear()
        self._wall_segments = list(segments)
        self._mesh = None
        self._router = None
        segs = np.array(segments, dtype=np.float64).reshape(-1, 4)
        segs = segs[np.hypot(*(segs[:, 2:] - segs[:, :2]).T) > TOL]
        if not len(segs):
            return

        began = time.perf_counter()
        reach = 2 * max(self._comfort_radius, self._inflation_radius * 1.05)
        self._mesh = Mesh.build(segs, reach)
        self._router = Router(self._mesh, self._inflation_radius, self._comfort_radius)
        build_ms = (time.perf_counter() - began) * 1e3
        self._logger.info(f"Walls meshed: {len(self._mesh.V)} triangles, {len(self._mesh.P)} vertices, {len(segments)} segment(s), inflation={self._inflation_radius}m, comfort={self._comfort_radius}m, {build_ms:.0f} ms")

    def _has_map(self) -> bool:
        return self._router is not None

    def _snap(self, p: np.ndarray) -> np.ndarray | None:
        """p itself when it keeps the inflation radius, else p pushed off the nearest walls, None when that fails."""
        assert self._mesh is not None
        r = self._inflation_radius
        for flip in (1.0, -1.0):
            q = p
            for _ in range(SNAP_TRIES):
                hit = self._mesh.index.nearest(q, r)
                if hit is None or hit[0] >= r:
                    return q
                d, wall, along = hit
                away = (q - wall) / d if d > 1e-9 else flip * np.array([-along[1], along[0]]) / np.hypot(*along)
                q = wall + away * (r + SNAP_MARGIN)
        return None

    def snap_terminal(self, pose: Pose2D) -> Pose2D:
        if self._mesh is None:
            return pose
        key = (pose.x, pose.y)
        if key not in self._snapped:
            if len(self._snapped) >= SNAP_CACHE:
                self._snapped.clear()
            p = np.array(key, dtype=np.float64)
            q = self._snap(p)
            self._snapped[key] = None if q is None or q is p else (float(q[0]), float(q[1]))
        hit = self._snapped[key]
        return pose if hit is None else Pose2D(x=hit[0], y=hit[1], theta=pose.theta)

    def _inside(self, p: np.ndarray) -> np.ndarray:
        """p moved onto the meshed frame when it lies beyond it."""
        assert self._mesh is not None
        frame = self._mesh.P[-self._mesh.n_frame :]
        return np.clip(p, frame.min(0) + FRAME_INSET, frame.max(0) - FRAME_INSET)

    def _rejoin(self, start: np.ndarray, goal: np.ndarray, polyline: np.ndarray) -> np.ndarray:
        """Route between the framed points extended to the real end points."""
        assert self._router is not None
        if (polyline[0] != start).any():
            seen = self._router.line_of_sight(start[None], polyline[1][None])[0]
            polyline = np.vstack([start, polyline[1 if seen else 0 :]])
        if (polyline[-1] != goal).any():
            seen = self._router.line_of_sight(polyline[-2][None], goal[None])[0]
            polyline = np.vstack([polyline[: -1 if seen else len(polyline)], goal])
        return polyline

    def _plan(self, requests: list[PlanRequest]) -> dict[int, list[Pose2D] | None]:
        assert self._router is not None
        plans: dict[int, list[Pose2D] | None] = {agent_id: None for agent_id, _, _ in requests}
        by_goal: dict[tuple[float, float], list[tuple[int, Pose2D, Pose2D, np.ndarray]]] = {}
        for agent_id, agent_pos, target in requests:
            start = self._snap(np.array([agent_pos.x, agent_pos.y], dtype=np.float64))
            if start is not None:
                by_goal.setdefault((target.x, target.y), []).append((agent_id, agent_pos, target, start))

        for group in by_goal.values():
            snapped = self.snap_terminal(group[0][2])
            goal = np.array([snapped.x, snapped.y])
            starts = np.array([start for _, _, _, start in group])
            framed = self._inside(starts)
            beyond = (framed != starts).any(axis=1) | (self._inside(goal) != goal).any()
            direct = beyond & self._router.line_of_sight(starts, np.repeat(goal[None], len(starts), 0))
            routes = self._router.plan_many(framed[~direct], self._inside(goal))
            polylines: list[np.ndarray | None] = [None] * len(group)
            for i, route in zip(np.nonzero(~direct)[0], routes, strict=True):
                if route is not None:
                    polylines[i] = self._rejoin(starts[i], goal, route[0]) if beyond[i] else route[0]
            for i in np.nonzero(direct)[0]:
                polylines[i] = np.array([starts[i], goal])

            for (agent_id, agent_pos, target, _), polyline in zip(group, polylines, strict=True):
                if polyline is None:
                    continue
                waypoints = [Pose2D(x=float(x), y=float(y)) for x, y in polyline]
                if polyline[0, 0] == agent_pos.x and polyline[0, 1] == agent_pos.y:
                    waypoints.pop(0)
                waypoints.insert(0, Pose2D(x=agent_pos.x, y=agent_pos.y))
                waypoints[-1] = Pose2D(x=snapped.x, y=snapped.y, theta=target.theta)
                plans[agent_id] = waypoints
        return plans
