from __future__ import annotations

"""AgentPool structural invariants."""

from collections.abc import Callable

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool

_op_strategy = st.lists(
    st.tuples(st.sampled_from(["add", "remove"]), st.integers(min_value=1, max_value=50)),
    min_size=0,
    max_size=80,
)


@given(ops=_op_strategy)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_random_ops_preserve_invariants(
    ops: list[tuple[str, int]],
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
) -> None:
    """Add/remove ops preserve pool.n == len(id_to_idx) and index validity."""
    pool = pool_empty(capacity=8)
    for op, aid in ops:
        if op == "add":
            if aid in pool._id_to_idx:
                continue
            pool.add_agent(agent_factory(agent_id=aid))
        else:
            if aid not in pool._id_to_idx:
                continue
            pool.swap_remove(aid)

        assert pool.n == len(pool._id_to_idx)
        assert pool.pos.shape == (pool.capacity, 2)
        assert pool.vel.shape == (pool.capacity, 2)
        assert pool.prev_vel.shape == (pool.capacity, 2)
        for a, idx in pool._id_to_idx.items():
            assert 0 <= idx < pool.n
            assert int(pool.agent_ids[idx]) == a


@given(n=st.integers(min_value=0, max_value=30))
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_sequential_add_then_remove_empties(
    n: int,
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
) -> None:
    """N adds followed by N removes empties the pool."""
    pool = pool_empty(capacity=8)
    for i in range(n):
        pool.add_agent(agent_factory(agent_id=i + 1))
    for i in range(n):
        pool.swap_remove(i + 1)
    assert pool.n == 0
    assert pool._id_to_idx == {}
