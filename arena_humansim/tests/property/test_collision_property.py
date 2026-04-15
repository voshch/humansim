from __future__ import annotations

"""Collision resolver invariants."""

from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from arena_humansim.collision import CollisionResolver, _registry
from arena_humansim.pool import AgentPool

_IMPL_IDS = sorted(_registry._registry.keys())

_coord = st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False)
_vel = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
_state_list = st.lists(st.tuples(_coord, _coord, _vel, _vel), min_size=0, max_size=8)


@pytest.mark.parametrize("impl", _IMPL_IDS)
@given(states=_state_list)
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_resolve_produces_finite_state(
    impl: str,
    states: list[tuple[float, float, float, float]],
    pool_with_agents: Callable[..., AgentPool],
) -> None:
    """Resolver preserves finiteness and pool shape for arbitrary state."""
    if not states:
        return
    pool = pool_with_agents(n=len(states))
    for i, (x, y, vx, vy) in enumerate(states):
        pool.pos[i, 0] = x
        pool.pos[i, 1] = y
        pool.vel[i, 0] = vx
        pool.vel[i, 1] = vy

    resolver = CollisionResolver.create(impl)
    resolver.resolve(pool)

    assert np.all(np.isfinite(pool.pos))
    assert np.all(np.isfinite(pool.vel))
    assert pool.pos.shape == (pool.capacity, 2)
    assert pool.vel.shape == (pool.capacity, 2)
