from __future__ import annotations

import math
from collections.abc import Callable

import pytest
from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.local_planner import LocalPlanner
from arena_humansim.local_planner.hsfm import HSFMPlanner
from arena_humansim.utils.types import Pose2D, Segments


@pytest.fixture
def hsfm(walls_empty: Segments) -> HSFMPlanner:
    p = LocalPlanner.create("hsfm")
    assert isinstance(p, HSFMPlanner)
    p.set_walls(walls_empty)
    return p


def test_provides_heading_flag(hsfm: HSFMPlanner) -> None:
    assert hsfm.provides_heading is True
    assert hsfm.supports_pool is True


def _heading_err(theta: float, target: float) -> float:
    return abs(math.atan2(math.sin(target - theta), math.cos(target - theta)))


def test_heading_converges_to_goal_direction(hsfm: HSFMPlanner, agent_factory: Callable[..., BaseAgent]) -> None:
    agent = agent_factory(agent_id=1, x=0.0, y=0.0)
    agent.state.pose.theta = math.pi / 2  # facing +y, goal is +x
    goal = Pose2D(x=100.0, y=0.0)  # far enough that the agent can't overshoot in the sim window
    aid = agent.state.agent_id

    initial_target = math.atan2(goal.y - agent.state.pose.y, goal.x - agent.state.pose.x)
    initial_err = _heading_err(agent.state.pose.theta, initial_target)

    dt = 0.05
    for _ in range(120):  # 6 simulated seconds
        velocities = hsfm.compute([agent], {aid: goal}, dt=dt)
        vx, vy = velocities[aid]
        agent.state.pose.x += vx * dt
        agent.state.pose.y += vy * dt
        agent.state.velocity = (vx, vy)

    final_target = math.atan2(goal.y - agent.state.pose.y, goal.x - agent.state.pose.x)
    final_err = _heading_err(agent.state.pose.theta, final_target)
    assert final_err < initial_err * 0.5, f"heading did not converge: initial err {initial_err:.3f}, final {final_err:.3f}"


def test_lateral_attenuation_vs_sfm(agent_factory: Callable[..., BaseAgent]) -> None:
    walls: Segments = [((0.0, 0.0), (5.0, 0.0))]

    sfm = LocalPlanner.create("sfm")
    sfm.set_walls(walls)
    hsfm = LocalPlanner.create("hsfm")
    hsfm.set_walls(walls)

    a_sfm = agent_factory(agent_id=1, x=0.0, y=0.5)
    a_hsfm = agent_factory(agent_id=1, x=0.0, y=0.5)
    goal = Pose2D(x=5.0, y=0.5)

    v_sfm = sfm.compute([a_sfm], {a_sfm.state.agent_id: goal}, dt=0.05)[a_sfm.state.agent_id]
    v_hsfm = hsfm.compute([a_hsfm], {a_hsfm.state.agent_id: goal}, dt=0.05)[a_hsfm.state.agent_id]

    assert v_sfm[1] > 1e-4, f"sanity: SFM should have positive y-velocity from wall, got {v_sfm[1]}"
    assert 0.0 < v_hsfm[1] < v_sfm[1], f"HSFM lateral velocity {v_hsfm[1]} should be less than SFM {v_sfm[1]}"
