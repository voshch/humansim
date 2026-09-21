from __future__ import annotations

import heapq
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.pool import PoolAware
from arena_humansim.utils import ModuleRegistry
from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.types import CommandType, HighLevelCommand, Pose2D, WallAware

from ._grid import needs_replan, next_waypoint

if TYPE_CHECKING:
    from arena_humansim.core.viz import MarkerPublisher

_registry: ModuleRegistry[GlobalPlanner] = ModuleRegistry()

PlanRequest = tuple[int, Pose2D, Pose2D]


def simplify_path(
    waypoints: Sequence[Pose2D],
    min_area: float = 0.01,
) -> list[Pose2D]:
    n = len(waypoints)
    if n <= 2:
        return list(waypoints)

    xs = [w.x for w in waypoints]
    ys = [w.y for w in waypoints]
    prev_idx = list(range(-1, n - 1))
    next_idx = list(range(1, n + 1))

    _inf = math.inf
    _heappush = heapq.heappush
    _heappop = heapq.heappop

    def _area(i: int) -> float:
        p, nx = prev_idx[i], next_idx[i]
        return 0.5 * abs((xs[i] - xs[p]) * (ys[nx] - ys[p]) - (xs[nx] - xs[p]) * (ys[i] - ys[p]))

    areas = [_inf] + [_area(i) for i in range(1, n - 1)] + [_inf]
    heap = [(areas[i], i) for i in range(1, n - 1)]
    heapq.heapify(heap)
    removed = bytearray(n)

    while heap:
        area, i = _heappop(heap)
        if removed[i] or areas[i] != area:
            continue
        if area >= min_area:
            break

        removed[i] = 1
        p, nx = prev_idx[i], next_idx[i]
        next_idx[p] = nx
        prev_idx[nx] = p

        if p > 0:
            a = max(_area(p), area)
            areas[p] = a
            _heappush(heap, (a, p))
        if nx < n - 1:
            a = max(_area(nx), area)
            areas[nx] = a
            _heappush(heap, (a, nx))

    return [waypoints[i] for i in range(n) if not removed[i]]


class GlobalPlanner(PoolAware, WallAware, Loggable, ABC):
    def __init__(self, replan_distance: float = 1.0) -> None:
        self._replan_distance = replan_distance
        self._path_cache: dict[int, tuple[tuple[float, float], list[Pose2D], int]] = {}
        self._cached_results: dict[int, Pose2D] = {}

    @abstractmethod
    def _has_map(self) -> bool: ...

    @abstractmethod
    def _plan(self, requests: list[PlanRequest]) -> dict[int, list[Pose2D] | None]: ...

    def compute(
        self,
        agents: Iterable[BaseAgent],
        high_level_commands: dict[int, HighLevelCommand],
    ) -> dict[int, Pose2D]:
        agent_positions: dict[int, Pose2D] = {agent.state.agent_id: agent.state.pose for agent in agents}
        goals: dict[int, Pose2D] = {}
        requests: dict[int, tuple[Pose2D, Pose2D]] = {}

        for agent_id, cmd in high_level_commands.items():
            if not isinstance(cmd, HighLevelCommand):
                continue
            if cmd.type != CommandType.NAVIGATE:
                continue

            target = cmd.target_pose
            agent_pos = agent_positions.get(agent_id)

            if agent_pos is None or not self._has_map():
                goals[agent_id] = target
                continue

            if not needs_replan(self._path_cache, agent_id, target, agent_pos, self._replan_distance):
                cached_goal, waypoints, idx = self._path_cache[agent_id]
                idx = self.advance_along_path(agent_pos, waypoints, idx)
                self._path_cache[agent_id] = (cached_goal, waypoints, idx)
                goals[agent_id] = next_waypoint(waypoints, idx)
                continue

            requests[agent_id] = (agent_pos, target)

        if requests:
            plans = self._plan([(agent_id, agent_pos, target) for agent_id, (agent_pos, target) in requests.items()])

            for agent_id, (agent_pos, target) in requests.items():
                waypoints = plans[agent_id]

                if waypoints is None:
                    self._logger.debug(f"No path for agent {agent_id} ({agent_pos} -> {target}), using direct goal")
                    goals[agent_id] = self.snap_terminal(target)
                    self._path_cache.pop(agent_id, None)
                    continue

                goal_key = (round(target.x, 3), round(target.y, 3))
                idx = self.advance_along_path(agent_pos, waypoints, 0)
                self._path_cache[agent_id] = (goal_key, waypoints, idx)
                goals[agent_id] = next_waypoint(waypoints, idx)

        self._cached_results = goals
        return goals

    def get_cached_goals(self) -> dict[int, Pose2D]:
        return dict(self._cached_results)

    def get_cached_paths(self) -> dict[int, list[Pose2D]]:
        return {aid: wps for aid, (_, wps, _) in self._path_cache.items()}

    def invalidate_paths(self, agent_ids: Iterable[int]) -> None:
        for aid in agent_ids:
            self._path_cache.pop(aid, None)

    def _forget_paths(self) -> None:
        self._path_cache.clear()

    def configure(self, *, inflation_radius: float, resolution: float, comfort_radius: float) -> None:
        pass

    def snap_terminal(self, pose: Pose2D) -> Pose2D:
        return pose

    def publish_markers(self, pub: MarkerPublisher) -> None:
        pass

    @staticmethod
    def advance_along_path(
        agent_pos: Pose2D,
        waypoints: Sequence[Pose2D],
        current_idx: int,
    ) -> int:
        idx = current_idx
        while idx < len(waypoints) - 1:
            wp = waypoints[idx]
            nxt = waypoints[idx + 1]
            dx, dy = nxt.x - wp.x, nxt.y - wp.y
            tx, ty = agent_pos.x - wp.x, agent_pos.y - wp.y
            if dx * tx + dy * ty >= dx * dx + dy * dy:
                idx += 1
            else:
                break
        return idx

    @classmethod
    def register(cls, name: str) -> Callable[[Callable[[], type[GlobalPlanner]]], Callable[[], type[GlobalPlanner]]]:
        return _registry.register(name)

    @classmethod
    def create(cls, name: str, *args: Any, **kwargs: Any) -> GlobalPlanner:
        return _registry.get(name)(*args, **kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        return _registry.list_available()


def _load_dijkstra() -> type[GlobalPlanner]:
    from .dijkstra import DijkstraPlanner

    return DijkstraPlanner


def _load_astar() -> type[GlobalPlanner]:
    from .astar import AStarPlanner

    return AStarPlanner


def _load_navmesh() -> type[GlobalPlanner]:
    from .navmesh import NavMeshPlanner

    return NavMeshPlanner


_registry.register("dijkstra")(_load_dijkstra)
_registry.register("astar")(_load_astar)
_registry.register("navmesh")(_load_navmesh)
