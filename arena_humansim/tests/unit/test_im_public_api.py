from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.interaction_kinds import InteractionType, MembershipRole
from arena_humansim.core.interaction_manager import InteractionManager
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.types import (
    AgentKind,
    AgentState,
    BehaviorTreeMovement,
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


def _seed_forming(mgr: InteractionManager, participant: int, itype: InteractionType) -> int:
    spec = SeekSpec(interaction_type=itype)
    interaction = mgr._create_interaction(creator_id=participant, spec=spec)
    assert interaction.outcome == InteractionOutcome.FORMING
    interaction.contract.formation = None
    return interaction.id


def test_seek_creates_matching_interaction_when_none_exists() -> None:
    agents: dict[int, Any] = {1: _FakeAgent(1, x=0.0)}
    mgr = _mk_mgr(agents)

    spec = SeekSpec(interaction_type=InteractionType.GROUP_CONVERSATION)
    iid = mgr.seek(1, spec)

    assert iid is not None
    assert iid in mgr.interactions
    assert 1 in mgr.interactions[iid].participants
    roles = mgr._agent_membership.get(1, {})
    assert roles.get(iid) == MembershipRole.PARTICIPANT


def test_seek_joins_existing_matching_interaction() -> None:
    agents: dict[int, Any] = {1: _FakeAgent(1, x=0.0), 2: _FakeAgent(2, x=0.5)}
    mgr = _mk_mgr(agents)

    # Agent 1 creates a FORMING interaction (min_participants=2 so stays FORMING alone).
    iid = _seed_forming(mgr, 1, InteractionType.GROUP_CONVERSATION)

    # Agent 2 seeks — _scan_symmetric finds agent 1's interaction because 2 sees 1.
    spec = SeekSpec(interaction_type=InteractionType.GROUP_CONVERSATION)
    returned_iid = mgr.seek(2, spec)

    assert returned_iid == iid
    assert 2 in mgr.interactions[iid].participants
    # min_participants=2 now met; _maybe_activate promotes to ACTIVE.
    assert mgr.interactions[iid].outcome == InteractionOutcome.ACTIVE


def test_seek_returns_none_when_no_match_and_cannot_create() -> None:
    # SERVICE requires offer=True to create; no existing provider interaction.
    agents: dict[int, Any] = {1: _FakeAgent(1, x=0.0)}
    mgr = _mk_mgr(agents)

    spec = SeekSpec(interaction_type=InteractionType.SERVICE, offer=False)
    result = mgr.seek(1, spec)

    assert result is None
    assert mgr._agent_membership.get(1) is None


def test_seek_enforces_at_most_one_participant_membership() -> None:
    agents: dict[int, Any] = {1: _FakeAgent(1, x=0.0), 2: _FakeAgent(2, x=0.2)}
    mgr = _mk_mgr(agents)

    group_iid = _seed_active(mgr, [1, 2], InteractionType.GROUP_CONVERSATION)
    assert 1 in mgr.interactions[group_iid].participants

    # Seek a different interaction type — SIT_ON with no world_knowledge means can_create
    # returns False, so seek returns None; but the GROUP_CONVERSATION membership is stopped first.
    sit_spec = SeekSpec(interaction_type=InteractionType.SIT_ON, target="chair")
    result = mgr.seek(1, sit_spec)

    assert result is None
    # Agent 1 must no longer be a participant of the GROUP_CONVERSATION interaction.
    roles = mgr._agent_membership.get(1, {})
    assert group_iid not in roles
    # last_outcome must reflect INTERRUPTED.
    mv = agents[1].movement
    assert mv.last_outcome == InteractionOutcome.INTERRUPTED


def test_seek_preserves_matching_membership() -> None:
    agents: dict[int, Any] = {1: _FakeAgent(1, x=0.0), 2: _FakeAgent(2, x=0.2)}
    mgr = _mk_mgr(agents)

    group_iid = _seed_active(mgr, [1, 2], InteractionType.GROUP_CONVERSATION)

    group_spec = SeekSpec(interaction_type=InteractionType.GROUP_CONVERSATION)
    returned_iid = mgr.seek(1, group_spec)

    # Matching membership must be preserved, not torn down.
    assert returned_iid == group_iid
    assert 1 in mgr.interactions[group_iid].participants
    assert mgr.interactions[group_iid].outcome == InteractionOutcome.ACTIVE
    assert agents[1].movement.last_outcome is None


def test_is_bound_matching_returns_true_for_active_matching() -> None:
    agents: dict[int, Any] = {1: _FakeAgent(1, x=0.0), 2: _FakeAgent(2, x=0.2)}
    mgr = _mk_mgr(agents)
    _seed_active(mgr, [1, 2], InteractionType.GROUP_CONVERSATION)

    spec = SeekSpec(interaction_type=InteractionType.GROUP_CONVERSATION)
    assert mgr.is_bound_matching(1, spec) is True


def test_is_bound_matching_returns_false_for_forming() -> None:
    agents: dict[int, Any] = {1: _FakeAgent(1, x=0.0)}
    mgr = _mk_mgr(agents)
    iid = _seed_forming(mgr, 1, InteractionType.GROUP_CONVERSATION)
    # Confirm it's still FORMING (min_participants=2, only one agent).
    assert mgr.interactions[iid].outcome == InteractionOutcome.FORMING

    spec = SeekSpec(interaction_type=InteractionType.GROUP_CONVERSATION)
    # is_bound_matching requires ACTIVE; FORMING must return False.
    assert mgr.is_bound_matching(1, spec) is False


def test_is_bound_matching_returns_false_for_different_type() -> None:
    agents: dict[int, Any] = {1: _FakeAgent(1, x=0.0), 2: _FakeAgent(2, x=0.2)}
    mgr = _mk_mgr(agents)
    _seed_active(mgr, [1, 2], InteractionType.GROUP_CONVERSATION)

    # Querying with a different interaction type must not match.
    sit_spec = SeekSpec(interaction_type=InteractionType.SIT_ON, target="chair")
    assert mgr.is_bound_matching(1, sit_spec) is False
