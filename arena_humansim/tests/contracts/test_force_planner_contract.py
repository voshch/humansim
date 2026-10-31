from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.local_planner import LocalPlanner, _info, _registry
from arena_humansim.local_planner.force import ForcePlanner
from arena_humansim.utils.types import BeliefState, Pose2D

from ._util import registry_ids

_DT = 0.05
_IMPL_IDS = [name for name in registry_ids(_registry) if _info[name].family == "force" and issubclass(_registry.get(name), ForcePlanner)]


@pytest.fixture(params=_IMPL_IDS, ids=lambda k: f"impl={k}")
def planner(request: pytest.FixtureRequest) -> LocalPlanner:
    return LocalPlanner.create(request.param)


def test_helbing_is_in_the_contract() -> None:
    assert "helbing" in _IMPL_IDS


def test_pooled_and_per_agent_paths_agree(planner: LocalPlanner, pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner.set_walls([((-5.0, -0.9), (5.0, -0.9))])
    agents = [agent_factory(agent_id=1, x=0.0, y=0.0), agent_factory(agent_id=2, x=2.5, y=0.3), agent_factory(agent_id=3, x=0.6, y=0.5)]
    agents[0].state.velocity = (0.6, 0.0)
    agents[1].state.velocity = (-0.9, 0.1)
    agents[1].state.pose.theta = 2.5
    agents[2].state.velocity = (0.3, -0.2)
    for agent in agents:
        agent.belief = BeliefState(observed_agents=[o.state for o in agents if o is not agent])
    goals = {1: Pose2D(x=10.0, y=0.0), 2: Pose2D(x=-10.0, y=0.0), 3: Pose2D(x=0.6, y=8.0)}

    pool = pool_empty(capacity=4)
    planner.attach(pool)
    for agent in agents:
        pool.add_agent(agent)
    pool.set_goals(goals)
    pool.set_neighbor_csr(np.array([0, 2, 4, 6], dtype=np.int32), np.array([1, 2, 0, 2, 0, 1], dtype=np.int32))
    planner.compute_pool(pool, dt=_DT)

    per_agent = planner.compute(agents, goals, dt=_DT)
    for i, agent in enumerate(agents):
        assert np.allclose(per_agent[agent.state.agent_id], pool.vel[i])
    assert not np.allclose(pool.vel[0], pool.vel[1])


def test_coincident_agents_stay_finite(planner: LocalPlanner, pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_empty(capacity=4)
    planner.attach(pool)
    pool.add_agent(agent_factory(agent_id=1))
    pool.add_agent(agent_factory(agent_id=2))
    pool.vel[1] = (0.5, 0.0)
    pool.store_prev_vel()
    pool.set_goals({1: Pose2D(x=10.0, y=0.0), 2: Pose2D(x=-10.0, y=0.0)})
    pool.set_neighbor_csr(np.array([0, 1, 2], dtype=np.int32), np.array([1, 0], dtype=np.int32))
    planner.compute_pool(pool, store_forces=True, dt=_DT)
    assert isinstance(planner, ForcePlanner)
    assert planner._last_force_arrays is not None
    assert np.all(np.isfinite(planner._last_force_arrays[2]))
    assert np.all(np.isfinite(pool.vel[:2]))
