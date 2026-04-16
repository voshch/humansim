from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest

pytest.importorskip("arena_humansim_msgs.msg")

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.global_planner import GlobalPlanner, _registry
from arena_humansim.utils.types import Pose2D, Segments

from ._util import registry_ids

_PLANNER_KWARGS: dict[str, dict[str, Any]] = {}

_IMPL_IDS = registry_ids(_registry)


@pytest.fixture(params=_IMPL_IDS, ids=lambda k: f"impl={k}")
def planner_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def planner(planner_name: str, walls_empty: Segments) -> GlobalPlanner:
    kwargs = _PLANNER_KWARGS.get(planner_name, {})
    p = GlobalPlanner.create(planner_name, **kwargs)
    p.set_walls(walls_empty)
    return p


@pytest.fixture
def agents(agent_factory: Callable[..., BaseAgent]) -> list[BaseAgent]:
    return [agent_factory(agent_id=i + 1, x=float(i), y=0.0) for i in range(3)]


@pytest.fixture
def commands(commands_factory: Callable[..., dict[int, Any]], agents: list[BaseAgent]) -> Any:
    return commands_factory(agent_ids=[a.state.agent_id for a in agents], target=(5.0, 0.0))


def test_compute_keys_subset_of_agents(planner: GlobalPlanner, agents: list[BaseAgent], commands: Any) -> None:
    goals = planner.compute(agents, commands)
    agent_ids = {a.state.agent_id for a in agents}
    assert set(goals.keys()).issubset(agent_ids)


def test_outputs_are_finite(planner: GlobalPlanner, agents: list[BaseAgent], commands: Any) -> None:
    goals = planner.compute(agents, commands)
    for pose in goals.values():
        assert isinstance(pose, Pose2D)
        assert math.isfinite(pose.x)
        assert math.isfinite(pose.y)
        assert math.isfinite(pose.theta)


def test_empty_agents_returns_empty(planner: GlobalPlanner) -> None:
    assert planner.compute([], {}) == {}


def test_set_walls_idempotent(planner: GlobalPlanner, walls_empty: Segments, agents: list[BaseAgent], commands: Any) -> None:
    planner.set_walls(walls_empty)
    first = planner.compute(agents, commands)
    planner.set_walls(walls_empty)
    second = planner.compute(agents, commands)
    assert set(first.keys()) == set(second.keys())
    for aid in first:
        assert first[aid].x == pytest.approx(second[aid].x)
        assert first[aid].y == pytest.approx(second[aid].y)
        assert first[aid].theta == pytest.approx(second[aid].theta)


def test_cached_goals_match_compute(planner: GlobalPlanner, agents: list[BaseAgent], commands: Any) -> None:
    goals = planner.compute(agents, commands)
    cached = planner.get_cached_goals()
    assert set(cached.keys()) == set(goals.keys())
    for aid in goals:
        assert cached[aid].x == pytest.approx(goals[aid].x)
        assert cached[aid].y == pytest.approx(goals[aid].y)


def test_determinism(planner_name: str, walls_empty: Segments, agents: list[BaseAgent], commands: Any) -> None:
    kwargs = _PLANNER_KWARGS.get(planner_name, {})
    p1 = GlobalPlanner.create(planner_name, **kwargs)
    p2 = GlobalPlanner.create(planner_name, **kwargs)
    p1.set_walls(walls_empty)
    p2.set_walls(walls_empty)
    g1 = p1.compute(agents, commands)
    g2 = p2.compute(agents, commands)
    assert set(g1.keys()) == set(g2.keys())
    for aid in g1:
        assert g1[aid].x == pytest.approx(g2[aid].x)
        assert g1[aid].y == pytest.approx(g2[aid].y)
        assert g1[aid].theta == pytest.approx(g2[aid].theta)
