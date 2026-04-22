"""SeekNode wait_for_outcome=True behavior."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

pytest.importorskip("rclpy")
py_trees = pytest.importorskip("py_trees")

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.behavior.nodes import SeekNode
from arena_humansim.core.behavior.step_context import StepContext
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.utils.types import BehaviorTreeMovement, InteractionOutcome, SeekSpec


@pytest.fixture(autouse=True)
def _clear_blackboard() -> None:
    py_trees.blackboard.Blackboard.clear()


@pytest.fixture
def rng_np() -> np.random.Generator:
    return np.random.default_rng(42)


def _mv(agent: BaseAgent) -> BehaviorTreeMovement:
    return cast(BehaviorTreeMovement, agent.movement)


def _with_bt(agent: BaseAgent) -> BaseAgent:
    agent.movement = BehaviorTreeMovement()
    return agent


def _mk_seek(
    agent: BaseAgent,
    *,
    bound: bool,
    wait_for_outcome: bool,
    rng: np.random.Generator,
) -> SeekNode:
    spec = SeekSpec(interaction_type=InteractionType.SERVICE, target="escort_ride")
    ctx = StepContext(is_bound_lookup=lambda _aid: bound)
    return SeekNode("seek", agent, spec=spec, ctx=ctx, duration_source=None, rng=rng, wait_for_outcome=wait_for_outcome)


def test_seek_returns_running_while_bound_when_wait_for_outcome_true(
    agent_factory: Callable[..., BaseAgent],
    rng_np: np.random.Generator,
) -> None:
    agent = _with_bt(agent_factory(agent_id=1))
    node = _mk_seek(agent, bound=True, wait_for_outcome=True, rng=rng_np)
    node.tick_once()
    assert node.status == py_trees.common.Status.RUNNING


def test_seek_returns_success_on_bind_when_wait_for_outcome_false(
    agent_factory: Callable[..., BaseAgent],
    rng_np: np.random.Generator,
) -> None:
    agent = _with_bt(agent_factory(agent_id=2))
    node = _mk_seek(agent, bound=True, wait_for_outcome=False, rng=rng_np)
    node.tick_once()
    assert node.status == py_trees.common.Status.SUCCESS


def test_seek_returns_success_on_interaction_end_when_wait_for_outcome_true(
    agent_factory: Callable[..., BaseAgent],
    rng_np: np.random.Generator,
) -> None:
    agent = _with_bt(agent_factory(agent_id=3))
    node = _mk_seek(agent, bound=True, wait_for_outcome=True, rng=rng_np)
    node.tick_once()
    assert node.status == py_trees.common.Status.RUNNING
    _mv(agent).last_outcome = InteractionOutcome.COMPLETED
    node.tick_once()
    assert node.status == py_trees.common.Status.SUCCESS
    assert _mv(agent).last_outcome is None


def test_seek_returns_failure_on_interaction_interrupt_when_wait_for_outcome_true(
    agent_factory: Callable[..., BaseAgent],
    rng_np: np.random.Generator,
) -> None:
    agent = _with_bt(agent_factory(agent_id=4))
    node = _mk_seek(agent, bound=True, wait_for_outcome=True, rng=rng_np)
    node.tick_once()
    assert node.status == py_trees.common.Status.RUNNING
    _mv(agent).last_outcome = InteractionOutcome.INTERRUPTED
    node.tick_once()
    assert node.status == py_trees.common.Status.FAILURE
    assert _mv(agent).last_outcome is None
