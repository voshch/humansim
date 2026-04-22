from __future__ import annotations

import math
from typing import cast, Any

import pytest

from arena_humansim.animation.kinematic import KinematicAnimation
from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.agents.types import (
    SampledLocalPlanner,
    SampledParams,
    SampledPerception,
)
from arena_humansim.utils.types import AgentState, BehaviorTreeMovement, Pose2D


def _make_agent(agent_id: int = 1, vx: float = 0.0, vy: float = 0.0, theta: float = 0.0, idle_gaze_rate_hz: float = 0.0) -> BaseAgent:
    state = AgentState(
        agent_id=agent_id,
        pose=Pose2D(x=0.0, y=0.0, theta=theta),
        velocity=(vx, vy),
        desired_velocity=1.1,
    )
    params = SampledParams(
        name="adult",
        desired_velocity=1.1,
        agent_radius=0.25,
        max_velocity=1.5,
        max_acceleration=1.5,
        max_deceleration=2.5,
        min_turning_radius=0.3,
        pivot_angular_velocity=2.0,
        reaction_time=0.4,
        personal_space_min=0.6,
        perception=SampledPerception(vision_range=5.0, vision_fov=180.0),
        local_planner_params=SampledLocalPlanner(
            relaxation_time=0.5,
            repulsion_strength=2.1,
            repulsion_range=0.3,
            anisotropy=0.5,
        ),
        perception_stack=("default",),
        idle_gaze_rate_hz=idle_gaze_rate_hz,
    )
    return BaseAgent(
        state=state,
        params=params,
        global_planner=cast(Any, None),
        local_planner=cast(Any, None),
        animation=cast(Any, None),
        movement=BehaviorTreeMovement(),
    )


def test_idle_gaze_rotates_theta_when_stationary_and_rate_positive() -> None:
    anim = KinematicAnimation()
    agent = _make_agent(agent_id=1, vx=0.0, vy=0.0, theta=0.0, idle_gaze_rate_hz=0.5)
    aid = agent.state.agent_id
    dt = 0.1
    peak = 0.0
    for _ in range(20):
        anim.compute_batch([agent], {aid: (0.0, 0.0)}, {}, dt=dt)
        peak = max(peak, abs(agent.state.pose.theta))
    assert peak > 0.3


def test_idle_gaze_disabled_when_rate_zero() -> None:
    anim = KinematicAnimation()
    agent = _make_agent(agent_id=2, vx=0.0, vy=0.0, theta=0.0, idle_gaze_rate_hz=0.0)
    aid = agent.state.agent_id
    for _ in range(10):
        anim.compute_batch([agent], {aid: (0.0, 0.0)}, {}, dt=0.1)
    assert agent.state.pose.theta == pytest.approx(0.0)


def test_idle_gaze_inactive_when_moving() -> None:
    anim = KinematicAnimation()
    agent = _make_agent(agent_id=3, vx=1.0, vy=0.0, theta=0.0, idle_gaze_rate_hz=0.5)
    aid = agent.state.agent_id
    for _ in range(10):
        anim.compute_batch([agent], {aid: (1.0, 0.0)}, {}, dt=0.1)
    assert agent.state.pose.theta == pytest.approx(0.0)
