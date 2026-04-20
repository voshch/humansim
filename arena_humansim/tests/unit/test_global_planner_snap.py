from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.global_planner.astar import AStarPlanner
from arena_humansim.global_planner.dijkstra import DijkstraPlanner
from arena_humansim.utils.types import Pose2D, Segments

_INFLATION = 0.38
_RESOLUTION = 0.2
_CELL_DIAG_HALF = _RESOLUTION * math.sqrt(2) / 2

_PLANNERS = [AStarPlanner, DijkstraPlanner]


def _dist_to_walls(pose: Pose2D, segments: Segments) -> float:
    best = math.inf
    for (x1, y1), (x2, y2) in segments:
        sx, sy = x2 - x1, y2 - y1
        seg_len_sq = sx * sx + sy * sy
        if seg_len_sq < 1e-12:
            cx, cy = x1, y1
        else:
            t = max(0.0, min(1.0, ((pose.x - x1) * sx + (pose.y - y1) * sy) / seg_len_sq))
            cx, cy = x1 + t * sx, y1 + t * sy
        best = min(best, math.hypot(pose.x - cx, pose.y - cy))
    return best


def _wall_on_y_axis() -> Segments:
    return [((0.0, -3.0), (0.0, 3.0))]


@pytest.fixture(params=_PLANNERS, ids=lambda c: c.__name__)
def planner_cls(request: pytest.FixtureRequest) -> type:
    return request.param


def test_snap_terminal_identity_without_walls(planner_cls: type) -> None:
    planner = planner_cls()
    target = Pose2D(x=1.0, y=2.0, theta=0.5)
    out = planner.snap_terminal(target)
    assert out.x == pytest.approx(target.x)
    assert out.y == pytest.approx(target.y)
    assert out.theta == pytest.approx(target.theta)


def test_snap_terminal_identity_when_target_already_free(planner_cls: type) -> None:
    planner = planner_cls(inflation_radius=_INFLATION)
    planner.set_walls(_wall_on_y_axis())
    target = Pose2D(x=2.0, y=0.0, theta=0.3)
    out = planner.snap_terminal(target)
    assert out.x == pytest.approx(target.x)
    assert out.y == pytest.approx(target.y)
    assert out.theta == pytest.approx(target.theta)


def test_snap_terminal_moves_out_of_wall(planner_cls: type) -> None:
    planner = planner_cls(inflation_radius=_INFLATION)
    walls = _wall_on_y_axis()
    planner.set_walls(walls)
    target = Pose2D(x=0.0, y=0.0, theta=1.0)
    out = planner.snap_terminal(target)
    assert _dist_to_walls(out, walls) >= _INFLATION - _CELL_DIAG_HALF - 1e-6
    assert out.theta == pytest.approx(target.theta)


def test_snap_terminal_moves_out_of_inflation_band(planner_cls: type) -> None:
    planner = planner_cls(inflation_radius=_INFLATION)
    walls = _wall_on_y_axis()
    planner.set_walls(walls)
    target = Pose2D(x=0.1, y=0.0, theta=0.0)
    out = planner.snap_terminal(target)
    assert _dist_to_walls(out, walls) >= _INFLATION - _CELL_DIAG_HALF - 1e-6


def test_subgoal_in_free_space_when_target_inside_wall(
    planner_cls: type,
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> None:
    planner = planner_cls(inflation_radius=_INFLATION)
    walls = _wall_on_y_axis()
    planner.set_walls(walls)
    agent = agent_factory(agent_id=1, x=-2.0, y=0.0)
    cmds = commands_factory(agent_ids=[1], target=(0.0, 0.0))
    goals = planner.compute([agent], cmds)
    assert 1 in goals
    assert _dist_to_walls(goals[1], walls) >= _INFLATION - _CELL_DIAG_HALF - 1e-6


def test_subgoal_in_free_space_when_target_in_inflation_band(
    planner_cls: type,
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> None:
    planner = planner_cls(inflation_radius=_INFLATION)
    walls = _wall_on_y_axis()
    planner.set_walls(walls)
    agent = agent_factory(agent_id=1, x=-2.0, y=0.0)
    cmds = commands_factory(agent_ids=[1], target=(0.1, 0.0))
    goals = planner.compute([agent], cmds)
    assert _dist_to_walls(goals[1], walls) >= _INFLATION - _CELL_DIAG_HALF - 1e-6


def test_cached_path_last_waypoint_not_in_wall(
    planner_cls: type,
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> None:
    planner = planner_cls(inflation_radius=_INFLATION)
    walls = _wall_on_y_axis()
    planner.set_walls(walls)
    agent = agent_factory(agent_id=1, x=-2.0, y=0.0)
    cmds = commands_factory(agent_ids=[1], target=(0.0, 0.0))
    planner.compute([agent], cmds)
    paths = planner.get_cached_paths()
    assert 1 in paths and len(paths[1]) >= 1
    last = paths[1][-1]
    assert _dist_to_walls(last, walls) >= _INFLATION - _CELL_DIAG_HALF - 1e-6
