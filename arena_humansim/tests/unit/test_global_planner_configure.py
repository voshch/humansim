from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.global_planner.astar import AStarPlanner
from arena_humansim.global_planner.dijkstra import DijkstraPlanner
from arena_humansim.utils.types import Segments

_PLANNERS = [AStarPlanner, DijkstraPlanner]


def _two_rooms_with_metre_door() -> Segments:
    return [
        ((-3.0, -3.0), (3.0, -3.0)),
        ((3.0, -3.0), (3.0, 3.0)),
        ((3.0, 3.0), (-3.0, 3.0)),
        ((-3.0, 3.0), (-3.0, -3.0)),
        ((0.0, -3.0), (0.0, -0.5)),
        ((0.0, 0.5), (0.0, 3.0)),
    ]


@pytest.fixture(params=_PLANNERS, ids=lambda c: c.__name__)
def planner_cls(request: pytest.FixtureRequest) -> type:
    return request.param


def _crosses_door(
    planner: Any,
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> bool:
    agent = agent_factory(agent_id=1, x=-2.0, y=2.0)
    cmds = commands_factory(agent_ids=[1], target=(2.0, 2.0))
    planner.compute([agent], cmds)
    return 1 in planner.get_cached_paths()


def test_metre_door_closed_on_coarse_grid(
    planner_cls: type,
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> None:
    planner = planner_cls(inflation_radius=0.38, resolution=0.2)
    planner.set_walls(_two_rooms_with_metre_door())
    assert not _crosses_door(planner, agent_factory, commands_factory)


def test_metre_door_open_on_fine_grid(
    planner_cls: type,
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> None:
    planner = planner_cls(inflation_radius=0.38, resolution=0.1)
    planner.set_walls(_two_rooms_with_metre_door())
    assert _crosses_door(planner, agent_factory, commands_factory)


def test_configure_rebuilds_grid_from_current_walls(
    planner_cls: type,
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> None:
    planner = planner_cls(inflation_radius=0.38, resolution=0.2)
    planner.set_walls(_two_rooms_with_metre_door())
    planner.configure(inflation_radius=0.38, resolution=0.1, comfort_radius=0.6)
    assert _crosses_door(planner, agent_factory, commands_factory)
    planner.configure(inflation_radius=0.38, resolution=0.2, comfort_radius=0.6)
    assert not _crosses_door(planner, agent_factory, commands_factory)
