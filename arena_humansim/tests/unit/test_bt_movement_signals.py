"""BehaviorTreeMovement.interaction_id write/clear invariants around join/leave/teardown."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.core.interaction_manager import InteractionManager
from arena_humansim.core.world_knowledge import WorldKnowledge, WorldObject
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.types import (
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
    def __init__(self, agent_id: int, x: float = 0.0) -> None:
        self.state = AgentState(agent_id=agent_id, pose=Pose2D(x=x))
        self.params = _FakeParams()
        self.movement = BehaviorTreeMovement()


def _mk_mgr(agents: dict[int, Any], *, world: WorldKnowledge | None = None) -> InteractionManager:
    mgr = InteractionManager(RNG(0))
    mgr.set_context(
        world_knowledge=world or WorldKnowledge(),
        agent_lookup=lambda aid: agents.get(aid),
        visibility_lookup=lambda aid: set(agents) - {aid},
    )
    return mgr


def _seek(agent_id: int, itype: InteractionType, *, target: str | int | None = None, offer: bool = False) -> HighLevelCommand:
    return HighLevelCommand(
        agent_id=agent_id,
        type=CommandType.SEEK,
        spec=SeekSpec(interaction_type=itype, target=target, offer=offer),
    )


def test_interaction_id_set_on_create() -> None:
    agents = {1: _FakeAgent(1)}
    mgr = _mk_mgr(agents)
    mgr.update({1: _seek(1, InteractionType.TALK_TO)})
    iid = next(iter(mgr.interactions))
    assert agents[1].movement.interaction_id == iid


def test_interaction_id_set_on_accept() -> None:
    agents = {1: _FakeAgent(1, x=0.0), 2: _FakeAgent(2, x=1.0)}
    mgr = _mk_mgr(agents)
    mgr.update({1: _seek(1, InteractionType.GROUP_CONVERSATION), 2: _seek(2, InteractionType.GROUP_CONVERSATION)})
    iid = next(iter(mgr.interactions))
    assert agents[1].movement.interaction_id == iid
    assert agents[2].movement.interaction_id == iid


def test_interaction_id_cleared_on_stop() -> None:
    agents = {1: _FakeAgent(1), 2: _FakeAgent(2)}
    mgr = _mk_mgr(agents)
    mgr.update({1: _seek(1, InteractionType.GROUP_CONVERSATION), 2: _seek(2, InteractionType.GROUP_CONVERSATION)})
    iid = next(iter(mgr.interactions))
    assert agents[1].movement.interaction_id == iid

    mgr.stop(1, iid)
    assert agents[1].movement.interaction_id is None


def test_interaction_id_cleared_on_teardown_for_all_members() -> None:
    agents = {1: _FakeAgent(1), 2: _FakeAgent(2)}
    mgr = _mk_mgr(agents)
    mgr.update({1: _seek(1, InteractionType.TALK_TO), 2: _seek(2, InteractionType.TALK_TO)})
    iid = next(iter(mgr.interactions))
    assert agents[1].movement.interaction_id == iid
    assert agents[2].movement.interaction_id == iid

    mgr._teardown(iid, InteractionOutcome.COMPLETED)
    assert agents[1].movement.interaction_id is None
    assert agents[2].movement.interaction_id is None


def test_interaction_id_cleared_on_force_stop() -> None:
    agents = {1: _FakeAgent(1)}
    mgr = _mk_mgr(agents)
    mgr.update({1: _seek(1, InteractionType.TALK_TO)})
    assert agents[1].movement.interaction_id is not None
    mgr.force_stop(1)
    assert agents[1].movement.interaction_id is None


def test_last_outcome_emitted_on_teardown() -> None:
    agents = {1: _FakeAgent(1), 2: _FakeAgent(2)}
    mgr = _mk_mgr(agents)
    mgr.update({1: _seek(1, InteractionType.TALK_TO), 2: _seek(2, InteractionType.TALK_TO)})
    iid = next(iter(mgr.interactions))
    mgr._teardown(iid, InteractionOutcome.COMPLETED)
    assert agents[1].movement.last_outcome == InteractionOutcome.COMPLETED
    assert agents[2].movement.last_outcome == InteractionOutcome.COMPLETED


def test_last_outcome_emitted_on_stop_when_interaction_torn_down() -> None:
    agents = {1: _FakeAgent(1), 2: _FakeAgent(2)}
    mgr = _mk_mgr(agents)
    mgr.update({1: _seek(1, InteractionType.TALK_TO), 2: _seek(2, InteractionType.TALK_TO)})
    iid = next(iter(mgr.interactions))
    mgr.stop(1, iid, reason=InteractionOutcome.CANCELED)
    # Both participants dropped below min=2 -> interaction torn down, both see CANCELED reason.
    assert agents[1].movement.last_outcome == InteractionOutcome.CANCELED
    assert agents[2].movement.last_outcome == InteractionOutcome.CANCELED


def test_queued_agent_gets_interaction_id_too() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D()))
    agents = {1: _FakeAgent(1), 2: _FakeAgent(2)}
    mgr = _mk_mgr(agents, world=wk)
    mgr.update({1: _seek(1, InteractionType.QUEUE_USE, target="atm")})
    iid = next(iter(mgr.interactions))
    mgr.update({2: _seek(2, InteractionType.QUEUE_USE, target="atm")})
    assert 2 in mgr.interactions[iid].contract.queue
    assert agents[2].movement.interaction_id == iid


def test_command_cleared_on_teardown() -> None:
    agents = {1: _FakeAgent(1), 2: _FakeAgent(2)}
    mgr = _mk_mgr(agents)
    mgr.update({1: _seek(1, InteractionType.TALK_TO), 2: _seek(2, InteractionType.TALK_TO)})
    iid = next(iter(mgr.interactions))
    agents[1].movement.command = HighLevelCommand(agent_id=1, type=CommandType.NAVIGATE)
    mgr._teardown(iid, InteractionOutcome.INTERRUPTED)
    assert agents[1].movement.command is None
