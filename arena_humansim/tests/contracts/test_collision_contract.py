from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from arena_humansim.collision import CollisionResolver, _registry
from arena_humansim.pool import AgentPool
from arena_humansim.utils.types import Segments, WallAware

from ._util import registry_ids

_IMPL_IDS = registry_ids(_registry)


@pytest.fixture(params=_IMPL_IDS, ids=lambda k: f"impl={k}")
def resolver_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def resolver(resolver_name: str) -> CollisionResolver:
    return CollisionResolver.create(resolver_name)


def _is_wall_aware(obj: object) -> bool:
    return isinstance(obj, WallAware) and type(obj).set_walls is not WallAware.set_walls


def test_preserves_pool_size(resolver: CollisionResolver, pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=4)
    n_before = pool.n
    resolver.resolve(pool)
    assert pool.n == n_before


def test_preserves_array_shapes(resolver: CollisionResolver, pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=4)
    capacity = pool.capacity
    resolver.resolve(pool)
    assert pool.pos.shape == (capacity, 2)
    assert pool.vel.shape == (capacity, 2)


def test_produces_finite_state(resolver: CollisionResolver, pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=4)
    resolver.resolve(pool)
    assert np.all(np.isfinite(pool.pos))
    assert np.all(np.isfinite(pool.vel))


def test_idempotent_when_no_motion(resolver: CollisionResolver, pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=4)
    pool.vel[:] = 0.0
    resolver.resolve(pool)
    pos_after_first = pool.pos.copy()
    vel_after_first = pool.vel.copy()
    resolver.resolve(pool)
    assert np.allclose(pool.pos, pos_after_first, atol=1e-9)
    assert np.allclose(pool.vel, vel_after_first, atol=1e-9)


def test_accepts_empty_and_simple_walls(resolver: CollisionResolver, pool_with_agents: Callable[..., AgentPool], walls_empty: Segments, walls_simple: Segments) -> None:
    pool = pool_with_agents(n=4)
    if _is_wall_aware(resolver):
        resolver.set_walls(walls_empty)
    resolver.resolve(pool)
    if _is_wall_aware(resolver):
        resolver.set_walls(walls_simple)
    resolver.resolve(pool)
    assert np.all(np.isfinite(pool.pos))
    assert np.all(np.isfinite(pool.vel))
