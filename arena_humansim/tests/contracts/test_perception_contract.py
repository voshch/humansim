from __future__ import annotations

import copy
from collections.abc import Callable

import pytest

from arena_humansim.agents.base import BaseAgent
from arena_humansim.perception import Perception, _registry
from arena_humansim.utils.types import AgentState, BeliefState, Pose2D, WorldAgentState, WorldState

from ._util import registry_ids

_IMPL_IDS = registry_ids(_registry)


@pytest.fixture(params=_IMPL_IDS, ids=lambda k: f"impl={k}")
def perception_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def perception(perception_name: str) -> Perception:
    cls = Perception.get(perception_name)
    return cls()


@pytest.fixture
def agents(agent_factory: Callable[..., BaseAgent]) -> list[BaseAgent]:
    return [agent_factory(agent_id=i + 1, x=float(i), y=0.0) for i in range(3)]


@pytest.fixture
def all_agents(agents: list[BaseAgent]) -> dict[int, AgentState]:
    return {a.state.agent_id: a.state for a in agents}


@pytest.fixture
def world_state() -> WorldState:
    return {}


def _prepare(perception: Perception, all_agents: dict[int, AgentState]) -> None:
    prepare = getattr(perception, "prepare_tick", None)
    if callable(prepare):
        prepare(all_agents)


def test_returns_belief_state(perception: Perception, agents: list[BaseAgent], all_agents: dict[int, AgentState], world_state: WorldState) -> None:
    _prepare(perception, all_agents)
    agent = agents[0]
    belief = BeliefState(agent_id=agent.state.agent_id)
    result = perception.compute(agent, all_agents, world_state, belief)
    assert isinstance(result, BeliefState)


def test_does_not_mutate_all_agents(perception: Perception, agents: list[BaseAgent], all_agents: dict[int, AgentState], world_state: WorldState) -> None:
    _prepare(perception, all_agents)
    snapshot = copy.deepcopy(all_agents)
    for agent in agents:
        belief = BeliefState(agent_id=agent.state.agent_id)
        perception.compute(agent, all_agents, world_state, belief)
    assert set(all_agents.keys()) == set(snapshot.keys())
    for aid, state in snapshot.items():
        cur = all_agents[aid]
        assert cur.pose.x == state.pose.x
        assert cur.pose.y == state.pose.y
        assert cur.pose.theta == state.pose.theta
        assert cur.velocity == state.velocity


def test_determinism(perception_name: str, agents: list[BaseAgent], all_agents: dict[int, AgentState], world_state: WorldState) -> None:
    cls = Perception.get(perception_name)
    p1 = cls()
    p2 = cls()
    _prepare(p1, all_agents)
    _prepare(p2, all_agents)
    agent = agents[0]
    b1 = p1.compute(agent, all_agents, world_state, BeliefState(agent_id=agent.state.agent_id))
    b2 = p2.compute(agent, all_agents, world_state, BeliefState(agent_id=agent.state.agent_id))
    ids1 = sorted(s.agent_id for s in b1.observed_agents)
    ids2 = sorted(s.agent_id for s in b2.observed_agents)
    assert ids1 == ids2


def _silence_unused() -> None:
    _ = Pose2D, WorldAgentState
