from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.local_planner import LocalPlanner, _registry
from arena_humansim.utils.types import Pose2D, Segments

from ._util import registry_ids

_IMPL_IDS = registry_ids(_registry)


@pytest.fixture(params=_IMPL_IDS, ids=lambda k: f"impl={k}")
def planner_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def planner(planner_name: str, walls_empty: Segments) -> LocalPlanner:
    p = LocalPlanner.create(planner_name)
    p.set_walls(walls_empty)
    return p


@pytest.fixture
def agents(agent_factory: Callable[..., BaseAgent]) -> list[BaseAgent]:
    return [agent_factory(agent_id=i + 1, x=float(i), y=0.0) for i in range(3)]


@pytest.fixture
def global_goals(agents: list[BaseAgent]) -> dict[int, Pose2D]:
    return {a.state.agent_id: Pose2D(x=5.0, y=0.0, theta=0.0) for a in agents}


def test_velocity_keys_subset_of_agents(planner: LocalPlanner, agents: list[BaseAgent], global_goals: dict[int, Pose2D]) -> None:
    velocities = planner.compute(agents, global_goals, dt=0.05)
    agent_ids = {a.state.agent_id for a in agents}
    assert set(velocities.keys()).issubset(agent_ids)


def test_magnitudes_bounded_by_max_velocity(planner: LocalPlanner, agents: list[BaseAgent], global_goals: dict[int, Pose2D]) -> None:
    velocities = planner.compute(agents, global_goals, dt=0.05)
    max_vels = {a.state.agent_id: a.params.max_velocity for a in agents}
    tol = 1e-6
    for aid, (vx, vy) in velocities.items():
        mag = math.hypot(vx, vy)
        assert mag <= max_vels[aid] + tol, f"impl returned vel magnitude {mag} > max_velocity {max_vels[aid]} for agent {aid}"


def test_components_finite(planner: LocalPlanner, agents: list[BaseAgent], global_goals: dict[int, Pose2D]) -> None:
    velocities = planner.compute(agents, global_goals, dt=0.05)
    for vx, vy in velocities.values():
        assert math.isfinite(vx)
        assert math.isfinite(vy)


def test_determinism(planner_name: str, walls_empty: Segments, agents: list[BaseAgent], global_goals: dict[int, Pose2D]) -> None:
    p1 = LocalPlanner.create(planner_name)
    p2 = LocalPlanner.create(planner_name)
    p1.set_walls(walls_empty)
    p2.set_walls(walls_empty)
    v1 = p1.compute(agents, global_goals, dt=0.05)
    v2 = p2.compute(agents, global_goals, dt=0.05)
    assert set(v1.keys()) == set(v2.keys())
    for aid in v1:
        assert v1[aid][0] == pytest.approx(v2[aid][0])
        assert v1[aid][1] == pytest.approx(v2[aid][1])
