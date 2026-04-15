from __future__ import annotations

from collections.abc import Callable

import pytest

from arena_humansim.agents.base import BaseAgent
from arena_humansim.animation import MotionAnimation, _registry

from tests.contracts._util import registry_ids

_IMPL_IDS = registry_ids(_registry)


@pytest.fixture(params=_IMPL_IDS, ids=lambda k: f"impl={k}")
def animation_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def animation(animation_name: str) -> MotionAnimation:
    return MotionAnimation.create(animation_name)


def test_zero_velocity_preserves_pose(animation_name: str, animation: MotionAnimation, agent_factory: Callable[..., BaseAgent]) -> None:
    agent = agent_factory(agent_id=1, x=2.0, y=3.0)
    aid = agent.state.agent_id
    motions = animation.compute_batch([agent], {aid: (0.0, 0.0)}, {}, dt=0.1)
    if animation_name == "noop":
        assert motions == {} or aid not in motions
        return
    pose = motions[aid]
    assert abs(pose.x) < 1e-9
    assert abs(pose.y) < 1e-9
    assert abs(pose.theta) < 1e-9


def test_forward_velocity_advances_position(animation_name: str, animation: MotionAnimation, agent_factory: Callable[..., BaseAgent]) -> None:
    if animation_name == "noop":
        pytest.skip("noop animation doesn't integrate")
    agent = agent_factory(agent_id=1, x=0.0, y=0.0)
    aid = agent.state.agent_id
    motions = animation.compute_batch([agent], {aid: (1.0, 0.0)}, {}, dt=0.1)
    pose = motions[aid]
    assert pose.x > 0.0, f"forward velocity did not advance x: {pose.x}"


def test_integration_magnitude(animation_name: str, animation: MotionAnimation, agent_factory: Callable[..., BaseAgent]) -> None:
    if animation_name == "noop":
        pytest.skip("noop animation doesn't integrate")
    agent = agent_factory(agent_id=1, x=0.0, y=0.0)
    aid = agent.state.agent_id
    motions = animation.compute_batch([agent], {aid: (1.0, 0.0)}, {}, dt=0.1)
    dx = motions[aid].x
    assert 0.095 <= dx <= 0.105, f"dx={dx} outside integration bounds [0.095, 0.105]"
