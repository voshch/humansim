from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.local_planner import LocalPlanner
from arena_humansim.local_planner.straight import StraightToGoalPlanner
from arena_humansim.utils.types import Pose2D


def test_needs_global_subgoal_is_false() -> None:
    assert StraightToGoalPlanner.needs_global_subgoal is False


def test_single_agent_points_toward_goal(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = LocalPlanner.create("straight")
    a = agent_factory(agent_id=1, x=0.0, y=0.0)
    goals = {1: Pose2D(x=3.0, y=4.0)}
    vels = planner.compute([a], goals, dt=0.1)
    vx, vy = vels[1]
    desired = min(a.params.desired_velocity, a.params.max_velocity)
    mag = math.hypot(vx, vy)
    assert mag == pytest.approx(desired, rel=1e-6)
    assert vx == pytest.approx(desired * 3.0 / 5.0, rel=1e-6)
    assert vy == pytest.approx(desired * 4.0 / 5.0, rel=1e-6)


def test_at_goal_returns_zero(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = LocalPlanner.create("straight")
    a = agent_factory(agent_id=1, x=1.0, y=1.0)
    goals = {1: Pose2D(x=1.0, y=1.0)}
    vels = planner.compute([a], goals, dt=0.1)
    assert vels[1] == (0.0, 0.0)


def test_agents_independent(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = LocalPlanner.create("straight")
    a = agent_factory(agent_id=1, x=0.0, y=0.0)
    b = agent_factory(agent_id=2, x=10.0, y=10.0)
    goals_solo = {1: Pose2D(x=5.0, y=0.0)}
    v_solo = planner.compute([a], goals_solo, dt=0.1)[1]

    goals_pair = {1: Pose2D(x=5.0, y=0.0), 2: Pose2D(x=20.0, y=20.0)}
    v_pair = planner.compute([a, b], goals_pair, dt=0.1)[1]

    assert v_solo == v_pair


def test_no_goal_returns_zero(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = LocalPlanner.create("straight")
    a = agent_factory(agent_id=1, x=0.0, y=0.0)
    vels = planner.compute([a], {}, dt=0.1)
    assert vels[1] == (0.0, 0.0)
