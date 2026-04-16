from __future__ import annotations

from typing import cast

import pytest

pytest.importorskip("arena_humansim_msgs")
pytest.importorskip("rclpy")

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.agent_manager import _MSG_BLOCK, _AgentStateMsgPool, _group_by


class _Tagged:
    def __init__(self, tag: str, seq: int) -> None:
        self.tag = tag
        self.seq = seq


def _agents(*items: tuple[str, int]) -> list[BaseAgent]:
    return cast(list[BaseAgent], [_Tagged(t, s) for t, s in items])


def test_group_by_groups_by_key() -> None:
    items = _agents(("a", 1), ("b", 2), ("a", 3), ("c", 4), ("b", 5))
    groups = dict(_group_by(items, lambda x: cast(_Tagged, x).tag))
    seqs = {k: [cast(_Tagged, a).seq for a in v] for k, v in groups.items()}
    assert seqs == {"a": [1, 3], "b": [2, 5], "c": [4]}


def test_group_by_preserves_insertion_order_within_groups() -> None:
    items = _agents(("a", 1), ("a", 2), ("a", 3), ("a", 4))
    groups = dict(_group_by(items, lambda x: cast(_Tagged, x).tag))
    assert [cast(_Tagged, a).seq for a in groups["a"]] == [1, 2, 3, 4]


def test_group_by_empty_returns_empty() -> None:
    assert dict(_group_by(cast(list[BaseAgent], []), lambda _: 0)) == {}


def test_group_by_single_item_per_key() -> None:
    items = _agents(("a", 1), ("b", 2), ("c", 3))
    groups = dict(_group_by(items, lambda x: cast(_Tagged, x).tag))
    assert {k: len(v) for k, v in groups.items()} == {"a": 1, "b": 1, "c": 1}


def test_msg_pool_initial_capacity_matches_block() -> None:
    pool = _AgentStateMsgPool()
    msg = pool.get(_MSG_BLOCK)
    assert len(msg.agents) == _MSG_BLOCK


def test_msg_pool_grows_on_demand() -> None:
    pool = _AgentStateMsgPool()
    msg = pool.get(_MSG_BLOCK * 3 + 1)
    assert len(msg.agents) == _MSG_BLOCK * 3 + 1


def test_msg_pool_reuses_underlying_messages_under_watermark() -> None:
    pool = _AgentStateMsgPool()
    pool.get(_MSG_BLOCK * 4)
    pool.get(_MSG_BLOCK * 4)
    first_snapshot = [id(m) for m in pool._pools[0]]
    second_snapshot = [id(m) for m in pool._pools[1]]
    pool.get(_MSG_BLOCK)
    pool.get(_MSG_BLOCK)
    assert [id(m) for m in pool._pools[0]] == first_snapshot
    assert [id(m) for m in pool._pools[1]] == second_snapshot


def test_msg_pool_double_buffers_returned_message() -> None:
    pool = _AgentStateMsgPool()
    a = pool.get(4)
    b = pool.get(4)
    c = pool.get(4)
    assert a is not b
    assert a is c


def test_msg_pool_returns_exactly_n_agents() -> None:
    pool = _AgentStateMsgPool()
    for n in (0, 1, 7, _MSG_BLOCK, _MSG_BLOCK + 1, _MSG_BLOCK * 5):
        msg = pool.get(n)
        assert len(msg.agents) == n


def test_msg_pool_frame_id_is_map() -> None:
    pool = _AgentStateMsgPool()
    assert pool.get(1).header.frame_id == "map"
    assert pool.get(1).header.frame_id == "map"
