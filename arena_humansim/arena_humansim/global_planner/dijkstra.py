from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Optional

import attrs
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

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


def _build_grid_graph(grid: np.ndarray) -> csr_matrix:
    rows, cols = grid.shape
    free = grid == 0
    free_flat = free.ravel()
    n_cells = rows * cols

    src = []
    dst = []
    weights = []

    for dr, dc, cost in (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2),
    ):
        # build index arrays for the valid region of the shift
        r_src = slice(max(-dr, 0), rows + min(-dr, 0))
        c_src = slice(max(-dc, 0), cols + min(-dc, 0))
        r_dst = slice(max(dr, 0), rows + min(dr, 0))
        c_dst = slice(max(dc, 0), cols + min(dc, 0))

        src_free = free[r_src, c_src]
        dst_free = free[r_dst, c_dst]
        both_free = src_free & dst_free

        r_idx, c_idx = np.where(both_free)

        src_r = r_idx + max(-dr, 0)
        src_c = c_idx + max(-dc, 0)
        dst_r = r_idx + max(dr, 0)
        dst_c = c_idx + max(dc, 0)

        src_flat = src_r * cols + src_c
        dst_flat = dst_r * cols + dst_c

        src.append(src_flat)
        dst.append(dst_flat)
        weights.append(np.full(src_flat.shape[0], cost, dtype=np.float32))

    src = np.concatenate(src)
    dst = np.concatenate(dst)
    weights = np.concatenate(weights)

    return csr_matrix((weights, (src, dst)), shape=(n_cells, n_cells))


def _dijkstra_path(
    graph: csr_matrix,
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> Optional[list[tuple[int, int]]]:
    rows, cols = grid.shape

    actual_start = _nearest_free_cell(grid, start[0], start[1])
    actual_goal = _nearest_free_cell(grid, goal[0], goal[1])
    if actual_start is None or actual_goal is None:
        return None

    prepend_start = actual_start != start
    append_goal = actual_goal != goal
    sr, sc = actual_start
    gr, gc = actual_goal

    if (sr, sc) == (gr, gc):
        path = [(sr, sc)]
        if prepend_start:
            path.insert(0, start)
        if append_goal:
            path.append(goal)
        return path

    goal_flat = gr * cols + gc
    start_flat = sr * cols + sc

    # straight-line distance as lower bound; allow 4x detour before giving up
    straight = math.sqrt((sr - gr) ** 2 + (sc - gc) ** 2)
    limit = max(straight * 4.0, 200.0)

    dist, predecessors = dijkstra(
        graph, directed=False, indices=goal_flat,
        return_predecessors=True, limit=limit,
    )

    if np.isinf(dist[start_flat]):
        # retry without limit in case path requires long detour
        dist, predecessors = dijkstra(
            graph, directed=False, indices=goal_flat,
            return_predecessors=True,
        )

    if np.isinf(dist[start_flat]):
        return None

    # trace back from start to goal
    path_flat = []
    node = start_flat
    while node != goal_flat:
        path_flat.append(node)
        node = predecessors[node]
        if node < 0:
            return None
    path_flat.append(goal_flat)

    path = [(idx // cols, idx % cols) for idx in path_flat]

    if prepend_start:
        path.insert(0, start)
    if append_goal:
        path.append(goal)
    return path


class DijkstraPlanner(GlobalPlanner):
    def __init__(
        self,
        replan_distance: float = 1.0,
        inflation_radius: float = 0.5,
    ):
        self._replan_distance = replan_distance
        self._inflation_radius = inflation_radius

        self._occupancy_grid: Optional[np.ndarray] = None
        self._grid_graph: Optional[csr_matrix] = None
        self._resolution: float = 0.2
        self._origin: Pose2D = Pose2D()
        self._wall_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

        self._path_cache: dict[int, tuple[tuple[float, float], list[Pose2D], int]] = {}

        self._cached_results: dict[int, Pose2D] = {}

    def set_walls(self, segments: list) -> None:
        self._path_cache.clear()
        self._grid_graph = None
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
        self._grid_graph = _build_grid_graph(grid)
        self._logger.info(
            f"Walls rasterized: {cols}x{rows} ({cols * rows} cells), res={res}m, "
            f"{len(segments)} segment(s), inflation={self._inflation_radius}m ({radius_cells} cells), "
            f"graph edges={self._grid_graph.nnz}"
        )

    def get_cached_goals(self) -> dict[int, Pose2D]:
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

    def _needs_replan(
        self,
        agent_id: int,
        goal: Pose2D,
        agent_pos: Pose2D,
    ) -> bool:
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
        # step count must cover both axes to avoid skipping thin walls diagonally
        steps = abs(r2 - r1) + abs(c2 - c1)
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

    def _next_waypoint(
        self,
        waypoints: Sequence[Pose2D],
        idx: int,
    ) -> Pose2D:
        target = idx + 1
        if target < len(waypoints):
            return waypoints[target]
        return waypoints[-1]

    def compute(
        self,
        agents: Iterable[BaseAgent],
        high_level_commands: dict[int, HighLevelCommand],
    ) -> dict[int, Pose2D]:
        agent_positions: dict[int, Pose2D] = {agent.state.agent_id: agent.state.pose for agent in agents}

        goals: dict[int, Pose2D] = {}

        for agent_id, cmd in high_level_commands.items():
            if not isinstance(cmd, HighLevelCommand):
                continue

            target = cmd.target_pose
            agent_pos = agent_positions.get(agent_id)

            if agent_pos is None:
                goals[agent_id] = target
                continue

            if self._occupancy_grid is None or self._grid_graph is None:
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

            raw_path = _dijkstra_path(self._grid_graph, self._occupancy_grid, start_rc, goal_rc)

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
