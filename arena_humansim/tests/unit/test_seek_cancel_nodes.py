"""SeekNode and CancelNode matrix coverage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("rclpy")
py_trees = pytest.importorskip("py_trees")

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.agents.types import ParamDist
from arena_humansim.core.behavior.nodes import CancelNode, SeekNode
from arena_humansim.core.behavior.step_context import StepContext
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.core.interaction_manager import InteractionManager
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.types import (
    AgentKind,
    AgentState,
    BehaviorTreeMovement,
    CommandType,
    HighLevelCommand,
    InteractionOutcome,
    Pose2D,
    SeekSpec,
)


@dataclass
class _FakeParams:
    reaction_time: float = 0.4
    personal_space_min: float = 0.6


class _FakeAgent:
    def __init__(self, agent_id: int, x: float = 0.0, y: float = 0.0) -> None:
        self.state = AgentState(agent_id=agent_id, pose=Pose2D(x=x, y=y), kind=int(AgentKind.HUMAN))
        self.params = _FakeParams()
        self.movement = BehaviorTreeMovement()
        self.needs = None


def _mk_mgr(agents: dict[int, Any]) -> InteractionManager:
    mgr = InteractionManager(RNG(0))
    mgr.set_context(
        agent_lookup=lambda aid: agents.get(aid),  # type: ignore[arg-type]
        visibility_lookup=lambda aid: set(agents) - {aid},
    )
    return mgr


def _seed_active(mgr: InteractionManager, participants: list[int], itype: InteractionType) -> int:
    spec = SeekSpec(interaction_type=itype)
    interaction = mgr._create_interaction(creator_id=participants[0], spec=spec)
    for pid in participants[1:]:
        mgr.accept(pid, interaction.id)
    interaction.outcome = InteractionOutcome.ACTIVE
    interaction.contract.formation = None
    return interaction.id


@pytest.fixture(autouse=True)
def _clear_blackboard() -> None:
    py_trees.blackboard.Blackboard.clear()


@pytest.fixture
def rng_np() -> np.random.Generator:
    return np.random.default_rng(123)


def _mv(agent: BaseAgent) -> BehaviorTreeMovement:
    return cast(BehaviorTreeMovement, agent.movement)


def _with_bt(agent: BaseAgent) -> BaseAgent:
    agent.movement = BehaviorTreeMovement()
    return agent


def _mk_seek(agent: BaseAgent, *, interaction: InteractionType, target: str | int | None = None, offer: bool = False, duration: ParamDist | None = None, rng: np.random.Generator, bound: bool = False, ctx: StepContext | None = None) -> SeekNode:
    spec = SeekSpec(interaction_type=interaction, target=target, offer=offer)
    if ctx is None:
        ctx = StepContext(is_bound_lookup=lambda _aid: bound)
    return SeekNode("seek", agent, spec=spec, ctx=ctx, duration_source=duration, rng=rng)


@pytest.mark.parametrize(
    "interaction,target,offer,creates_interaction",
    [
        (InteractionType.TALK_TO, None, False, True),
        (InteractionType.GROUP_CONVERSATION, None, False, True),
        (InteractionType.SERVICE, "water", True, True),
        # SERVICE seeker: no offer, no existing provider — IM returns None, no interaction created.
        (InteractionType.SERVICE, "water", False, False),
        (InteractionType.BLOCK, 99, False, True),
    ],
)
def test_seek_calls_im_seek_with_threaded_spec(
    rng_np: np.random.Generator,
    interaction: InteractionType,
    target: str | int | None,
    offer: bool,
    creates_interaction: bool,
) -> None:
    agent = _FakeAgent(agent_id=1)
    agents: dict[int, Any] = {1: agent}
    mgr = _mk_mgr(agents)
    spec = SeekSpec(interaction_type=interaction, target=target, offer=offer)
    ctx = StepContext(im=mgr, is_bound_lookup=mgr.is_bound)
    node = SeekNode("seek", agent, spec=spec, ctx=ctx, duration_source=None, rng=rng_np)  # type: ignore[arg-type]
    node.tick_once()
    assert node.status == py_trees.common.Status.RUNNING
    if creates_interaction:
        assert len(mgr.interactions) == 1
        istate = next(iter(mgr.interactions.values()))
        assert istate.type == int(interaction)
        if isinstance(target, str) and interaction == InteractionType.SERVICE:
            assert istate.service_tag == target
        if isinstance(target, int):
            assert istate.target_agent == target
    else:
        assert len(mgr.interactions) == 0


def test_seek_bound_short_circuits_to_success(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    agent = _with_bt(agent_factory(agent_id=1))
    node = _mk_seek(agent, interaction=InteractionType.TALK_TO, rng=rng_np, bound=True)
    node.tick_once()
    assert node.status == py_trees.common.Status.SUCCESS


def test_seek_completed_outcome_returns_success(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    agent = _with_bt(agent_factory(agent_id=3))
    node = _mk_seek(agent, interaction=InteractionType.TALK_TO, rng=rng_np)
    node.tick_once()
    _mv(agent).last_outcome = InteractionOutcome.COMPLETED
    node.tick_once()
    assert node.status == py_trees.common.Status.SUCCESS
    assert _mv(agent).last_outcome is None


def test_seek_canceled_outcome_returns_success(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    agent = _with_bt(agent_factory(agent_id=3))
    node = _mk_seek(agent, interaction=InteractionType.TALK_TO, rng=rng_np)
    node.tick_once()
    _mv(agent).last_outcome = InteractionOutcome.CANCELED
    node.tick_once()
    assert node.status == py_trees.common.Status.SUCCESS


def test_seek_interrupted_outcome_returns_failure(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    agent = _with_bt(agent_factory(agent_id=4))
    node = _mk_seek(agent, interaction=InteractionType.TALK_TO, rng=rng_np)
    node.tick_once()
    _mv(agent).last_outcome = InteractionOutcome.INTERRUPTED
    node.tick_once()
    assert node.status == py_trees.common.Status.FAILURE


def test_seek_duration_is_sampled_on_initialise(rng_np: np.random.Generator) -> None:
    agent = _FakeAgent(agent_id=5)
    agents: dict[int, Any] = {5: agent}
    mgr = _mk_mgr(agents)
    spec = SeekSpec(interaction_type=InteractionType.TALK_TO)
    ctx = StepContext(im=mgr, is_bound_lookup=mgr.is_bound)
    node = SeekNode("seek", agent, spec=spec, ctx=ctx, duration_source=ParamDist(3.5), rng=rng_np)  # type: ignore[arg-type]
    node.tick_once()
    assert len(mgr.interactions) == 1
    istate = next(iter(mgr.interactions.values()))
    assert istate.member_durations.get(5) == pytest.approx(3.5)


def test_seek_returns_success_once_bound_matching(rng_np: np.random.Generator) -> None:
    agent_id = 6
    other_id = 7
    agents: dict[int, Any] = {
        agent_id: _FakeAgent(agent_id),
        other_id: _FakeAgent(other_id, x=0.2),
    }
    mgr = _mk_mgr(agents)
    spec = SeekSpec(interaction_type=InteractionType.TALK_TO)
    ctx = StepContext(im=mgr, is_bound_lookup=mgr.is_bound)
    node = SeekNode("seek", agents[agent_id], spec=spec, ctx=ctx, duration_source=None, rng=rng_np)  # type: ignore[arg-type]
    # First tick: not yet bound, seeks and creates interaction (FORMING, only one participant).
    node.tick_once()
    assert node.status == py_trees.common.Status.RUNNING
    # Seed other_id into the same interaction so agent_id is now ACTIVE (bound matching).
    iid = next(iter(mgr.interactions))
    mgr.accept(other_id, iid)
    assert mgr.is_bound(agent_id)
    # Second tick: is_bound_matching → SUCCESS without re-seeking.
    node.tick_once()
    assert node.status == py_trees.common.Status.SUCCESS
    assert len(mgr.interactions) == 1


def test_cancel_stops_bound_interaction_via_im() -> None:
    agent_id = 7
    other_id = 8
    agents: dict[int, Any] = {
        agent_id: _FakeAgent(agent_id),
        other_id: _FakeAgent(other_id, x=0.2),
    }
    mgr = _mk_mgr(agents)
    iid = _seed_active(mgr, [agent_id, other_id], InteractionType.GROUP_CONVERSATION)
    assert mgr.is_bound(agent_id)
    agent: _FakeAgent = agents[agent_id]
    agent.movement.interaction_id = iid
    node = CancelNode("cancel", agent, im=mgr)  # type: ignore[arg-type]
    node.tick_once()
    assert node.status == py_trees.common.Status.SUCCESS
    assert not mgr.is_bound(agent_id)
    assert agent.movement.last_outcome == InteractionOutcome.CANCELED


def test_cancel_without_bound_interaction_force_stops_via_im() -> None:
    agent_id = 8
    agents: dict[int, Any] = {agent_id: _FakeAgent(agent_id)}
    mgr = _mk_mgr(agents)
    agent: _FakeAgent = agents[agent_id]
    agent.movement.interaction_id = None
    node = CancelNode("cancel", agent, im=mgr)  # type: ignore[arg-type]
    node.tick_once()
    assert node.status == py_trees.common.Status.SUCCESS
    assert not mgr.is_bound(agent_id)


def test_seek_when_bound_preserves_formation_navigate(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    # The IM writes NAVIGATE to participants' movement.command after each tick (formation).
    # On the next BT tick the SeekNode must NOT overwrite that NAVIGATE with a new SEEK
    # (or with None), otherwise bound followers stop tracking the leader.
    agent = _with_bt(agent_factory(agent_id=9))
    ctx = StepContext(is_bound_lookup=lambda _aid: True)
    spec = SeekSpec(interaction_type=InteractionType.SERVICE, target="escort")
    node = SeekNode("seek", agent, spec=spec, ctx=ctx, duration_source=None, rng=rng_np)
    formation_nav = HighLevelCommand(agent_id=9, type=CommandType.NAVIGATE, target_pose=Pose2D(x=4.2, y=0.1))
    _mv(agent).command = formation_nav
    node.tick_once()
    assert node.status == py_trees.common.Status.SUCCESS
    assert _mv(agent).command is formation_nav


def test_seek_reinitialised_after_loop_still_sees_bound(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    # After a hail → then: default loop, the SeekNode is initialise()'d again. It must
    # re-check is_bound and short-circuit to SUCCESS without emitting a fresh SEEK that
    # would clobber the formation NAVIGATE from the previous tick.
    agent = _with_bt(agent_factory(agent_id=11))
    ctx = StepContext(is_bound_lookup=lambda _aid: True)
    spec = SeekSpec(interaction_type=InteractionType.SERVICE, target="escort")
    node = SeekNode("seek", agent, spec=spec, ctx=ctx, duration_source=None, rng=rng_np)
    formation_nav = HighLevelCommand(agent_id=11, type=CommandType.NAVIGATE, target_pose=Pose2D(x=1.0))
    _mv(agent).command = formation_nav
    node.tick_once()
    # Simulate sequence loop: re-initialise mimics a fresh `hail` after SUCCESS.
    node.initialise()
    node.tick_once()
    assert node.status == py_trees.common.Status.SUCCESS
    assert _mv(agent).command is formation_nav
