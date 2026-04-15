from __future__ import annotations

from collections.abc import Callable

import pytest

from arena_humansim.agents.base import BaseAgent
from arena_humansim.pool import AgentPool


def test_add_agent_grows_n(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_empty(capacity=8)
    assert pool.n == 0
    for i in range(5):
        idx = pool.add_agent(agent_factory(agent_id=i + 1, x=float(i)))
        assert idx == i
        assert pool.n == i + 1


def test_multiple_adds_preserve_monotonic_n(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=6)
    assert pool.n == 6
    assert set(pool._id_to_idx.keys()) == {1, 2, 3, 4, 5, 6}


def test_swap_remove_reduces_n(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=4)
    pool.swap_remove(2)
    assert pool.n == 3
    assert 2 not in pool._id_to_idx


def test_swap_remove_last_gets_removed_index(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=4)
    removed_idx = pool._id_to_idx[2]
    swapped = pool.swap_remove(2)
    assert swapped == 4
    assert pool._id_to_idx[4] == removed_idx


def test_swap_remove_last_element_returns_none(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=3)
    last_id = int(pool.agent_ids[pool.n - 1])
    swapped = pool.swap_remove(last_id)
    assert swapped is None
    assert pool.n == 2


def test_id_to_idx_consistent_after_arbitrary_removes(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=6)
    for aid in (3, 1, 5):
        pool.swap_remove(aid)
    assert pool.n == 3
    for aid, idx in pool._id_to_idx.items():
        assert int(pool.agent_ids[idx]) == aid
    assert set(pool._id_to_idx.keys()) == {2, 4, 6}


def test_pos_vel_shape_constant(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_empty(capacity=8)
    assert pool.pos.shape == (8, 2)
    assert pool.vel.shape == (8, 2)
    for i in range(5):
        pool.add_agent(agent_factory(agent_id=i + 1))
    assert pool.pos.shape == (8, 2)
    assert pool.vel.shape == (8, 2)
    pool.swap_remove(2)
    assert pool.pos.shape == (8, 2)
    assert pool.vel.shape == (8, 2)


def test_reset_clears_n_and_id_map(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=4)
    pool.reset()
    assert pool.n == 0
    assert pool._id_to_idx == {}


def test_add_remove_add_cycle_keeps_bijection(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_empty(capacity=8)
    for i in range(5):
        pool.add_agent(agent_factory(agent_id=i + 1))
    pool.swap_remove(2)
    pool.swap_remove(4)
    pool.add_agent(agent_factory(agent_id=100))
    pool.add_agent(agent_factory(agent_id=101))
    assert pool.n == 5
    for aid, idx in pool._id_to_idx.items():
        assert int(pool.agent_ids[idx]) == aid
    seen_idx = sorted(pool._id_to_idx.values())
    assert seen_idx == list(range(pool.n))


def test_idx_returns_stored_index(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=3)
    assert pool.idx(1) == pool._id_to_idx[1]


def test_swap_remove_missing_raises(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=3)
    with pytest.raises(KeyError):
        pool.swap_remove(999)


def test_grow_beyond_capacity(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_empty(capacity=2)
    for i in range(5):
        pool.add_agent(agent_factory(agent_id=i + 1))
    assert pool.n == 5
    assert pool.capacity >= 5
    assert pool.pos.shape == (pool.capacity, 2)
    assert pool.vel.shape == (pool.capacity, 2)
