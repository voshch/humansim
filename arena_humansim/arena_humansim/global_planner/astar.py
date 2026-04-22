from __future__ import annotations

import math
import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyastar2d
from scipy.ndimage import binary_dilation

from arena_humansim.core.agents import BaseAgent
from arena_humansim.utils.types import CommandType, HighLevelCommand, Pose2D, Segment, Segments

from . import GlobalPlanner
from ._grid import (
    grid_to_world,
    needs_replan,
    next_waypoint,
    push_from_walls,
    simplify_with_los,
    world_to_grid,
)

_SQRT2 = math.sqrt(2)


def _nearest_free_cell(
    grid: np.ndarray,
    row: int,
    col: int,
    max_radius: int = 200,
) -> tuple[int, int] | None:
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
) -> list[tuple[int, int]] | None:
    actual_start = _nearest_free_cell(grid, start[0], start[1])
    actual_goal = _nearest_free_cell(grid, goal[0], goal[1])
    if actual_start is None or actual_goal is None:
        return None

    prepend_start = actual_start != start

    if actual_start == actual_goal:
        path = [actual_start]
        if prepend_start:
            path.insert(0, start)
        return path

    result = pyastar2d.astar_path(weights, actual_start, actual_goal, allow_diagonal=True)
    if result is None:
        return None

    path = [(int(r), int(c)) for r, c in result]
    if prepend_start:
        path.insert(0, start)
    return path


class AStarPlanner(GlobalPlanner):
    def __init__(
        self,
        replan_distance: float = 1.0,
        inflation_radius: float = 0.38,
    ):
        self._replan_distance = replan_distance
        self._inflation_radius = inflation_radius

        self._occupancy_grid: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._resolution: float = 0.2
        self._origin: Pose2D = Pose2D()
        self._wall_segments: list[Segment] = []

        self._path_cache: dict[int, tuple[tuple[float, float], list[Pose2D], int]] = {}
        self._cached_results: dict[int, Pose2D] = {}
        self._pool = ThreadPoolExecutor(max_workers=max((os.cpu_count() or 2) - 1, 1))

    def set_walls(self, segments: Segments) -> None:
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
        self._logger.info(f"Walls rasterized: {cols}x{rows} ({cols * rows} cells), res={res}m, {len(segments)} segment(s), inflation={self._inflation_radius}m ({radius_cells} cells)")

    def get_cached_goals(self) -> dict[int, Pose2D]:
        return dict(self._cached_results)

    def get_cached_paths(self) -> dict[int, list[Pose2D]]:
        return {aid: wps for aid, (_, wps, _) in self._path_cache.items()}

    def invalidate_paths(self, agent_ids: Iterable[int]) -> None:
        for aid in agent_ids:
            self._path_cache.pop(aid, None)

    def snap_terminal(self, pose: Pose2D) -> Pose2D:
        if self._occupancy_grid is None:
            return pose
        rows, cols = self._occupancy_grid.shape
        rc = world_to_grid(self._origin, self._resolution, pose.x, pose.y)
        if not (0 <= rc[0] < rows and 0 <= rc[1] < cols):
            return pose
        snapped = _nearest_free_cell(self._occupancy_grid, rc[0], rc[1])
        if snapped is None or snapped == rc:
            return pose
        cell = grid_to_world(self._origin, self._resolution, snapped[0], snapped[1])
        return Pose2D(x=cell.x, y=cell.y, theta=pose.theta)

    def compute(
        self,
        agents: Iterable[BaseAgent],
        high_level_commands: dict[int, HighLevelCommand],
    ) -> dict[int, Pose2D]:
        agent_positions: dict[int, Pose2D] = {agent.state.agent_id: agent.state.pose for agent in agents}
        goals: dict[int, Pose2D] = {}
        has_grid = self._occupancy_grid is not None and self._weights is not None

        replan_requests: list[tuple[int, Pose2D, Pose2D, tuple[int, int], tuple[int, int]]] = []

        for agent_id, cmd in high_level_commands.items():
            if not isinstance(cmd, HighLevelCommand):
                continue
            if cmd.type != CommandType.NAVIGATE:
                continue

            target = cmd.target_pose
            agent_pos = agent_positions.get(agent_id)

            if agent_pos is None or not has_grid:
                goals[agent_id] = target
                continue

            if not needs_replan(self._path_cache, agent_id, target, agent_pos, self._replan_distance):
                cached_goal, waypoints, idx = self._path_cache[agent_id]
                idx = self.advance_along_path(agent_pos, waypoints, idx)
                self._path_cache[agent_id] = (cached_goal, waypoints, idx)
                goals[agent_id] = next_waypoint(waypoints, idx)
                continue

            start_rc = world_to_grid(self._origin, self._resolution, agent_pos.x, agent_pos.y)
            goal_rc = world_to_grid(self._origin, self._resolution, target.x, target.y)
            replan_requests.append((agent_id, agent_pos, target, start_rc, goal_rc))

        if replan_requests:
            weights = self._weights
            grid = self._occupancy_grid
            futures = {agent_id: self._pool.submit(_astar_path, weights, grid, start_rc, goal_rc) for agent_id, _, _, start_rc, goal_rc in replan_requests}

            for agent_id, agent_pos, target, start_rc, goal_rc in replan_requests:
                raw_path = futures[agent_id].result()

                if raw_path is None:
                    self._logger.debug(f"No path for agent {agent_id} ({start_rc} -> {goal_rc}), using direct goal")
                    goals[agent_id] = self.snap_terminal(target)
                    self._path_cache.pop(agent_id, None)
                    continue

                waypoints = [grid_to_world(self._origin, self._resolution, r, c) for r, c in raw_path]
                waypoints[0] = Pose2D(x=agent_pos.x, y=agent_pos.y)
                waypoints[-1] = self.snap_terminal(target)
                assert self._occupancy_grid is not None
                waypoints = simplify_with_los(self._occupancy_grid, self._origin, self._resolution, waypoints)
                if self._wall_segments:
                    waypoints = push_from_walls(self._wall_segments, self._inflation_radius, waypoints)

                goal_key = (round(target.x, 3), round(target.y, 3))
                idx = self.advance_along_path(agent_pos, waypoints, 0)
                self._path_cache[agent_id] = (goal_key, waypoints, idx)
                goals[agent_id] = next_waypoint(waypoints, idx)

        self._cached_results = goals
        return goals
