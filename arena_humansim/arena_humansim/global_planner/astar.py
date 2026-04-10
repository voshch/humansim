from __future__ import annotations

import math
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import numpy as np
import pyastar2d
from scipy.ndimage import binary_dilation

from arena_humansim.agents import BaseAgent
from arena_humansim.utils.types import HighLevelCommand, Pose2D

from . import GlobalPlanner, simplify_path

_SQRT2 = math.sqrt(2)


def _nearest_free_cell(
    grid: np.ndarray,
    row: int,
    col: int,
    max_radius: int = 200,
) -> Optional[tuple[int, int]]:
    rows, cols = grid.shape
    if 0 <= row < rows and 0 <= col < cols and grid[row, col] == 0:
        return (row, col)
    for r in range(1, max_radius + 1):
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if abs(dr) != r and abs(dc) != r:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                    return (nr, nc)
    return None


def _astar_path(
    weights: np.ndarray,
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> Optional[list[tuple[int, int]]]:
    actual_start = _nearest_free_cell(grid, start[0], start[1])
    actual_goal = _nearest_free_cell(grid, goal[0], goal[1])
    if actual_start is None or actual_goal is None:
        return None

    prepend_start = actual_start != start
    append_goal = actual_goal != goal

    if actual_start == actual_goal:
        path = [actual_start]
        if prepend_start:
            path.insert(0, start)
        if append_goal:
            path.append(goal)
        return path

    result = pyastar2d.astar_path(weights, actual_start, actual_goal, allow_diagonal=True)
    if result is None:
        return None

    path = [(int(r), int(c)) for r, c in result]
    if prepend_start:
        path.insert(0, start)
    if append_goal:
        path.append(goal)
    return path


class AStarPlanner(GlobalPlanner):
    def __init__(
        self,
        replan_distance: float = 1.0,
        inflation_radius: float = 0.3,
    ):
        self._replan_distance = replan_distance
        self._inflation_radius = inflation_radius

        self._occupancy_grid: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None
        self._resolution: float = 0.2
        self._origin: Pose2D = Pose2D()
        self._wall_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

        self._path_cache: dict[int, tuple[tuple[float, float], list[Pose2D], int]] = {}
        self._cached_results: dict[int, Any] = {}
        self._pool = ThreadPoolExecutor(max_workers=max((os.cpu_count() or 2) - 1, 1))

    def set_walls(self, segments: list) -> None:
        self._path_cache.clear()
        self._weights = None
        self._wall_segments = list(segments)
        if not segments:
            self._occupancy_grid = None
            return

        arr = np.array(segments, dtype=np.float64).reshape(-1, 2, 2)
        all_points = arr.reshape(-1, 2)
        margin = self._inflation_radius + self._resolution * 2
        x_min = float(all_points[:, 0].min()) - margin
        y_min = float(all_points[:, 1].min()) - margin
        x_max = float(all_points[:, 0].max()) + margin
        y_max = float(all_points[:, 1].max()) + margin

        self._origin = Pose2D(x=x_min, y=y_min)
        res = self._resolution
        cols = int(math.ceil((x_max - x_min) / res)) + 1
        rows = int(math.ceil((y_max - y_min) / res)) + 1
        grid = np.zeros((rows, cols), dtype=np.uint8)

        for (x1, y1), (x2, y2) in segments:
            c1, r1 = (x1 - x_min) / res, (y1 - y_min) / res
            c2, r2 = (x2 - x_min) / res, (y2 - y_min) / res
            n = int(max(abs(c2 - c1), abs(r2 - r1))) + 1
            for t in np.linspace(0.0, 1.0, n):
                c = int(round(c1 + t * (c2 - c1)))
                r = int(round(r1 + t * (r2 - r1)))
                if 0 <= r < rows and 0 <= c < cols:
                    grid[r, c] = 1

        radius_cells = int(math.ceil(self._inflation_radius / res))
        if radius_cells > 0:
            y, x = np.ogrid[-radius_cells : radius_cells + 1, -radius_cells : radius_cells + 1]
            kernel = (x * x + y * y) <= radius_cells * radius_cells
            grid = binary_dilation(grid, structure=kernel).astype(np.uint8)

        self._occupancy_grid = grid
        self._weights = np.where(grid == 0, 1.0, np.inf).astype(np.float32)
        self._logger.info(
            f"Walls rasterized: {cols}x{rows} ({cols * rows} cells), res={res}m, "
            f"{len(segments)} segment(s), inflation={self._inflation_radius}m ({radius_cells} cells)"
        )

    def get_cached_goals(self) -> dict[int, Any]:
        return dict(self._cached_results)

    def get_cached_paths(self) -> dict[int, list[Pose2D]]:
        return {aid: wps for aid, (_, wps, _) in self._path_cache.items()}

    def _world_to_grid(self, wx: float, wy: float) -> tuple[int, int]:
        col = int(round((wx - self._origin.x) / self._resolution))
        row = int(round((wy - self._origin.y) / self._resolution))
        return row, col

    def _grid_to_world(self, row: int, col: int) -> Pose2D:
        wx = col * self._resolution + self._origin.x
        wy = row * self._resolution + self._origin.y
        return Pose2D(x=wx, y=wy)

    def _needs_replan(self, agent_id: int, goal: Pose2D, agent_pos: Pose2D) -> bool:
        if agent_id not in self._path_cache:
            return True
        cached_goal, waypoints, _ = self._path_cache[agent_id]
        goal_key = (round(goal.x, 3), round(goal.y, 3))
        if cached_goal != goal_key:
            return True
        if not waypoints:
            return True
        min_dist = self._min_distance_to_path(agent_pos, waypoints)
        if min_dist > self._replan_distance:
            return True
        return False

    @staticmethod
    def _min_distance_to_path(pos: Pose2D, waypoints: Iterable[Pose2D]) -> float:
        best = math.inf
        for wp in waypoints:
            d = math.hypot(pos.x - wp.x, pos.y - wp.y)
            if d < best:
                best = d
        return best

    def _line_of_sight(self, p1: Pose2D, p2: Pose2D) -> bool:
        grid = self._occupancy_grid
        if grid is None:
            return True
        r1, c1 = self._world_to_grid(p1.x, p1.y)
        r2, c2 = self._world_to_grid(p2.x, p2.y)
        rows, cols = grid.shape
        dr, dc = abs(r2 - r1), abs(c2 - c1)
        steps = max(dr, dc)
        if steps == 0:
            return True
        for i in range(steps + 1):
            t = i / steps
            r = int(round(r1 + t * (r2 - r1)))
            c = int(round(c1 + t * (c2 - c1)))
            if not (0 <= r < rows and 0 <= c < cols) or grid[r, c] != 0:
                return False
        return True

    def _simplify_with_los(self, waypoints: list[Pose2D]) -> list[Pose2D]:
        if len(waypoints) <= 2:
            return waypoints
        result = [waypoints[0]]
        i = 0
        while i < len(waypoints) - 1:
            farthest = i + 1
            for j in range(len(waypoints) - 1, i + 1, -1):
                if self._line_of_sight(waypoints[i], waypoints[j]):
                    farthest = j
                    break
            result.append(waypoints[farthest])
            i = farthest
        return result

    def _push_from_walls(self, waypoints: list[Pose2D]) -> list[Pose2D]:
        if len(waypoints) <= 2:
            return waypoints
        margin = self._inflation_radius
        margin_sq = margin * margin
        segs = self._wall_segments
        result = [waypoints[0]]
        for wp in waypoints[1:-1]:
            px, py = wp.x, wp.y
            push_x, push_y = 0.0, 0.0
            for (x1, y1), (x2, y2) in segs:
                sx, sy = x2 - x1, y2 - y1
                seg_len_sq = sx * sx + sy * sy
                if seg_len_sq < 1e-12:
                    cx, cy = x1, y1
                else:
                    t = max(0.0, min(1.0, ((px - x1) * sx + (py - y1) * sy) / seg_len_sq))
                    cx, cy = x1 + t * sx, y1 + t * sy
                dx, dy = px - cx, py - cy
                dist_sq = dx * dx + dy * dy
                if dist_sq < margin_sq and dist_sq > 1e-12:
                    dist = math.sqrt(dist_sq)
                    nx, ny = dx / dist, dy / dist
                    push_x += nx * (margin - dist)
                    push_y += ny * (margin - dist)
            result.append(Pose2D(x=px + push_x, y=py + push_y))
        result.append(waypoints[-1])
        return result

    def _next_waypoint(self, waypoints: Sequence[Pose2D], idx: int) -> Pose2D:
        target = idx + 1
        if target < len(waypoints):
            return waypoints[target]
        return waypoints[-1]

    def compute(
        self,
        agents: Iterable[BaseAgent],
        high_level_commands: dict[int, Any],
    ) -> dict[int, Any]:
        agent_positions: dict[int, Pose2D] = {agent.state.agent_id: agent.state.pose for agent in agents}
        goals: dict[int, Pose2D] = {}
        has_grid = self._occupancy_grid is not None and self._weights is not None

        replan_requests: list[tuple[int, Pose2D, Pose2D, tuple[int, int], tuple[int, int]]] = []

        for agent_id, cmd in high_level_commands.items():
            if not isinstance(cmd, HighLevelCommand):
                continue

            target = cmd.target_pose
            agent_pos = agent_positions.get(agent_id)

            if agent_pos is None or not has_grid:
                goals[agent_id] = target
                continue

            if not self._needs_replan(agent_id, target, agent_pos):
                cached_goal, waypoints, idx = self._path_cache[agent_id]
                idx = self.advance_along_path(agent_pos, waypoints, idx)
                self._path_cache[agent_id] = (cached_goal, waypoints, idx)
                goals[agent_id] = self._next_waypoint(waypoints, idx)
                continue

            start_rc = self._world_to_grid(agent_pos.x, agent_pos.y)
            goal_rc = self._world_to_grid(target.x, target.y)
            replan_requests.append((agent_id, agent_pos, target, start_rc, goal_rc))

        if replan_requests:
            weights = self._weights
            grid = self._occupancy_grid
            futures = {
                agent_id: self._pool.submit(_astar_path, weights, grid, start_rc, goal_rc)
                for agent_id, _, _, start_rc, goal_rc in replan_requests
            }

            for agent_id, agent_pos, target, start_rc, goal_rc in replan_requests:
                raw_path = futures[agent_id].result()

                if raw_path is None:
                    self._logger.debug(f"No path for agent {agent_id} ({start_rc} -> {goal_rc}), using direct goal")
                    goals[agent_id] = target
                    self._path_cache.pop(agent_id, None)
                    continue

                waypoints = [self._grid_to_world(r, c) for r, c in raw_path]
                waypoints[0] = Pose2D(x=agent_pos.x, y=agent_pos.y)
                waypoints[-1] = Pose2D(x=target.x, y=target.y)
                waypoints = self._simplify_with_los(waypoints)
                if self._wall_segments:
                    waypoints = self._push_from_walls(waypoints)

                goal_key = (round(target.x, 3), round(target.y, 3))
                idx = self.advance_along_path(agent_pos, waypoints, 0)
                self._path_cache[agent_id] = (goal_key, waypoints, idx)
                goals[agent_id] = self._next_waypoint(waypoints, idx)

        self._cached_results = goals
        return goals
