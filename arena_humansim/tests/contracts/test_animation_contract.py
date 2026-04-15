from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from arena_humansim.agents.base import BaseAgent
from arena_humansim.animation import MotionAnimation, _registry
from arena_humansim.pool import AgentPool
from arena_humansim.utils.types import Pose2D

from ._util import registry_ids

_IMPL_IDS = registry_ids(_registry)


@pytest.fixture(params=_IMPL_IDS, ids=lambda k: f"impl={k}")
def animation_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def animation(animation_name: str) -> MotionAnimation:
    return MotionAnimation.create(animation_name)


@pytest.fixture
def agents(agent_factory: Callable[..., BaseAgent]) -> list[BaseAgent]:
    return [agent_factory(agent_id=i + 1, x=float(i), y=0.0) for i in range(3)]


@pytest.fixture
def velocities(agents: list[BaseAgent]) -> dict[int, tuple[float, float]]:
    return {a.state.agent_id: (0.5, 0.0) for a in agents}


def test_compute_batch_keys_subset_of_velocities(animation: MotionAnimation, agents: list[BaseAgent], velocities: dict[int, tuple[float, float]]) -> None:
    motions = animation.compute_batch(agents, velocities, {}, dt=0.05)
    assert set(motions.keys()).issubset(set(velocities.keys()))


def test_poses_finite(animation: MotionAnimation, agents: list[BaseAgent], velocities: dict[int, tuple[float, float]]) -> None:
    motions = animation.compute_batch(agents, velocities, {}, dt=0.05)
    for pose in motions.values():
        assert isinstance(pose, Pose2D)
        assert math.isfinite(pose.x)
        assert math.isfinite(pose.y)
        assert math.isfinite(pose.theta)


def test_compute_batch_pool_preserves_shape(animation: MotionAnimation, pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=4)
    capacity = pool.capacity
    n_before = pool.n
    animation.compute_batch_pool(pool, {}, dt=0.05)
    assert pool.n == n_before
    assert pool.pos.shape == (capacity, 2)
    assert pool.vel.shape == (capacity, 2)
