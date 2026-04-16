from __future__ import annotations

from typing import Any, cast

import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.agents.types import SampledLocalPlanner, SampledParams, SampledPerception
from arena_humansim.perception import Perception, _registry
from arena_humansim.utils.types import AgentState, BeliefState, Pose2D, WorldState

from tests.contracts._util import registry_ids

_IMPL_IDS = registry_ids(_registry)


def _params_with_vision(vision_range: float, vision_fov: float) -> SampledParams:
    return SampledParams(
        name="adult",
        desired_velocity=1.1,
        agent_radius=0.25,
        max_velocity=1.5,
        max_acceleration=1.5,
        max_deceleration=2.5,
        min_turning_radius=0.3,
        pivot_angular_velocity=2.0,
        perception=SampledPerception(vision_range=vision_range, vision_fov=vision_fov),
        local_planner_params=SampledLocalPlanner(
            relaxation_time=0.5,
            repulsion_strength=2.1,
            repulsion_range=0.3,
            anisotropy=0.5,
        ),
    )


def _make_agent(agent_id: int, x: float, y: float, theta: float, vision_range: float, vision_fov: float) -> BaseAgent:
    state = AgentState(agent_id=agent_id, pose=Pose2D(x=x, y=y, theta=theta), velocity=(0.0, 0.0), desired_velocity=1.3)
    return BaseAgent(
        state=state,
        params=_params_with_vision(vision_range=vision_range, vision_fov=vision_fov),
        global_planner=cast(Any, None),
        local_planner=cast(Any, None),
        animation=cast(Any, None),
    )


@pytest.fixture(params=_IMPL_IDS, ids=lambda k: f"impl={k}")
def perception_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def perception(perception_name: str) -> Perception:
    return Perception.get(perception_name)()


def _prepare(perception: Perception, all_agents: dict[int, AgentState]) -> None:
    prep = getattr(perception, "prepare_tick", None)
    if callable(prep):
        prep(all_agents)


def test_sees_nearby_agent_in_fov(perception: Perception) -> None:
    observer = _make_agent(1, 0.0, 0.0, theta=0.0, vision_range=5.0, vision_fov=180.0)
    other = _make_agent(2, 1.0, 0.0, theta=0.0, vision_range=5.0, vision_fov=180.0)
    all_agents: dict[int, AgentState] = {1: observer.state, 2: other.state}
    world_state: WorldState = {}
    _prepare(perception, all_agents)
    belief = perception.compute(observer, all_agents, world_state, BeliefState(agent_id=1))
    observed_ids = [s.agent_id for s in belief.observed_agents]
    assert 2 in observed_ids, f"nearby agent missing from belief: {observed_ids}"
    assert len(belief.observed_agents) == 1, f"expected exactly one observed agent, got {observed_ids}"
    obs = belief.observed_agents[0]
    assert obs.pose.x != 0.0 or obs.pose.y != 0.0, f"observed agent has zero pose: {obs.pose}"


def test_misses_out_of_range_agent(perception: Perception) -> None:
    observer = _make_agent(1, 0.0, 0.0, theta=0.0, vision_range=5.0, vision_fov=360.0)
    other = _make_agent(2, 100.0, 0.0, theta=0.0, vision_range=5.0, vision_fov=360.0)
    all_agents: dict[int, AgentState] = {1: observer.state, 2: other.state}
    world_state: WorldState = {}
    _prepare(perception, all_agents)
    belief = perception.compute(observer, all_agents, world_state, BeliefState(agent_id=1))
    observed_ids = [s.agent_id for s in belief.observed_agents]
    assert 2 not in observed_ids, f"out-of-range agent in belief: {observed_ids}"


def test_misses_behind_observer(perception: Perception) -> None:
    fov = 180.0
    if fov >= 360.0:
        pytest.skip("observer is omnidirectional")
    observer = _make_agent(1, 0.0, 0.0, theta=0.0, vision_range=5.0, vision_fov=fov)
    other = _make_agent(2, -1.0, 0.0, theta=0.0, vision_range=5.0, vision_fov=fov)
    all_agents: dict[int, AgentState] = {1: observer.state, 2: other.state}
    world_state: WorldState = {}
    _prepare(perception, all_agents)
    belief = perception.compute(observer, all_agents, world_state, BeliefState(agent_id=1))
    observed_ids = [s.agent_id for s in belief.observed_agents]
    assert 2 not in observed_ids, f"behind-observer agent in belief: {observed_ids}"


def test_idempotent_on_stable_world(perception_name: str) -> None:
    observer = _make_agent(1, 0.0, 0.0, theta=0.0, vision_range=5.0, vision_fov=180.0)
    other = _make_agent(2, 1.0, 0.0, theta=0.0, vision_range=5.0, vision_fov=180.0)
    all_agents: dict[int, AgentState] = {1: observer.state, 2: other.state}
    world_state: WorldState = {}

    p1 = Perception.get(perception_name)()
    p2 = Perception.get(perception_name)()
    _prepare(p1, all_agents)
    _prepare(p2, all_agents)

    b1 = p1.compute(observer, all_agents, world_state, BeliefState(agent_id=1))
    b2 = p2.compute(observer, all_agents, world_state, BeliefState(agent_id=1))

    ids1 = sorted(s.agent_id for s in b1.observed_agents)
    ids2 = sorted(s.agent_id for s in b2.observed_agents)
    assert ids1 == ids2
