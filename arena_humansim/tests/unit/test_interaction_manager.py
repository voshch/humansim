from __future__ import annotations

import pytest

pytest.importorskip("rclpy")

from typing import Any

from dataclasses import dataclass

from arena_humansim.core.interaction_manager import CommandType, InteractionManager
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.types import (
    AgentKind,
    AgentState,
    BehaviorTreeMovement,
    HighLevelCommand,
    InteractionContract,
    InteractionOutcome,
    InteractionState,
    InteractionType,
    Pose2D,
)


@dataclass
class _FakeParams:
    reaction_time: float = 0.4
    personal_space_min: float = 0.6


class _FakeBTAgent:
    def __init__(self, agent_id: int = 0, x: float = 0.0, y: float = 0.0) -> None:
        self.state = AgentState(agent_id=agent_id, desired_velocity=1.0, pose=Pose2D(x=x, y=y))
        self.movement = BehaviorTreeMovement()
        self.params = _FakeParams()


def _fake_bt_agent(agent_id: int = 0, x: float = 0.0, y: float = 0.0) -> Any:
    return _FakeBTAgent(agent_id=agent_id, x=x, y=y)


def _cmd(
    agent_id: int,
    ctype: int,
    *,
    interaction_type: int = 0,
    interaction_target: int = -1,
    target_agent: int = -1,
    duration: float = -1.0,
) -> HighLevelCommand:
    return HighLevelCommand(
        agent_id=agent_id,
        type=ctype,
        target_pose=Pose2D(),
        desired_velocity=1.0,
        interaction_target=interaction_target,
        interaction_type=interaction_type,
        target_agent=target_agent,
        interaction_duration=duration,
    )


def _advertise(
    mgr: InteractionManager,
    agent_id: int,
    itype: int,
    *,
    target_agent: int = -1,
    duration: float = -1.0,
    object_id: str | None = None,
) -> int:
    """Create an interaction with `agent_id` as creator, bypassing the matcher.

    Most tests here exercise post-creation mechanics (accept, stop, queue, duration)
    and don't care about how the interaction came to exist. Using `_create_interaction`
    directly keeps those tests focused.
    """
    dur = duration if duration > 0 else None
    interaction = mgr._create_interaction(
        int(itype),
        agent_id,
        object_id=object_id,
        target_agent=target_agent,
        duration=dur,
    )
    if dur is not None:
        interaction.member_durations[agent_id] = dur
    return interaction.id


def test_accept_adds_participant_and_readvertises() -> None:
    mgr = InteractionManager(RNG(0))
    iid = _advertise(mgr, 1, InteractionType.GROUP_CONVERSATION)
    assert mgr.accept(2, iid) is True
    interaction = mgr.interactions[iid]
    assert 2 in interaction.participants
    assert 2 in interaction.contract.current_participants
    ads_by_type = mgr._ads_by_type.get(InteractionType.GROUP_CONVERSATION, [])
    participant_ads = [a for a in ads_by_type if a.agent_id == 2 and a.interaction_id == iid]
    assert len(participant_ads) == 1


def test_accept_when_full_queues_if_queueable() -> None:
    mgr = InteractionManager(RNG(0))
    iid = _advertise(mgr, 1, InteractionType.QUEUE_USE)
    assert mgr.interactions[iid].contract.is_full
    assert mgr.accept(2, iid) is True
    assert 2 in mgr.interactions[iid].contract.queue
    assert 2 not in mgr.interactions[iid].participants
    assert mgr.is_in_queue(2)


def test_accept_when_full_rejects_if_not_queueable() -> None:
    mgr = InteractionManager(RNG(0))
    iid = _advertise(mgr, 1, InteractionType.TALK_TO)
    assert mgr.accept(2, iid) is True
    assert mgr.interactions[iid].contract.is_full
    assert mgr.accept(3, iid) is False
    assert 3 not in mgr.interactions[iid].participants
    assert 3 not in mgr.interactions[iid].contract.queue


def test_target_agent_scoping_blocks_unrelated_accepter() -> None:
    mgr = InteractionManager(RNG(0))
    iid = _advertise(mgr, 1, InteractionType.TALK_TO, target_agent=5)
    assert mgr.accept(3, iid) is False
    assert 3 not in mgr.interactions[iid].participants
    assert mgr.accept(5, iid) is True
    assert 5 in mgr.interactions[iid].participants


def test_stop_below_min_tears_down_with_interrupted() -> None:
    mgr = InteractionManager(RNG(0))
    iid = _advertise(mgr, 1, InteractionType.TALK_TO)
    assert mgr.accept(2, iid) is True
    assert mgr.stop(1, iid) is None
    interaction = mgr.interactions[iid]
    assert interaction.outcome == InteractionOutcome.INTERRUPTED


def test_stop_removes_from_queue_cleanly() -> None:
    mgr = InteractionManager(RNG(0))
    iid = _advertise(mgr, 1, InteractionType.QUEUE_USE)
    assert mgr.accept(2, iid) is True
    assert 2 in mgr.interactions[iid].contract.queue
    mgr.stop(2, iid)
    assert 2 not in mgr.interactions[iid].contract.queue
    assert not mgr.is_in_queue(2)
    assert mgr.interactions[iid].outcome == InteractionOutcome.ACTIVE


def test_tick_promotes_next_from_queue_when_slot_opens() -> None:
    mgr = InteractionManager(RNG(0))
    contract = InteractionContract(
        type=int(InteractionType.QUEUE_USE),
        min_participants=1,
        max_participants=2,
        queueable=True,
        current_participants=[1],
        queue=[2],
    )
    interaction = InteractionState(
        id=0,
        type=int(InteractionType.QUEUE_USE),
        contract=contract,
        participants=[1],
        outcome=InteractionOutcome.ACTIVE,
    )
    mgr.interactions[0] = interaction
    mgr.next_interaction_id = 1
    mgr._agent_to_interactions.setdefault(1, set()).add(0)
    mgr._agent_to_queues.setdefault(2, set()).add(0)

    mgr.update({})

    assert 2 in interaction.participants
    assert 2 in interaction.contract.current_participants
    assert 2 not in interaction.contract.queue
    assert not mgr.is_in_queue(2)
    assert mgr.is_in_interaction(2)


def test_tick_durations_time_out_to_completed() -> None:
    mgr = InteractionManager(RNG(0))
    iid = _advertise(mgr, 1, InteractionType.TALK_TO, duration=0.5)
    assert mgr.accept(2, iid) is True
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE

    mgr.update({}, dt=0.3)
    assert iid in mgr.interactions
    assert mgr.interactions[iid].outcome == InteractionOutcome.ACTIVE

    mgr.update({}, dt=0.3)
    assert iid not in mgr.interactions


def test_force_stop_clears_interactions_and_queues() -> None:
    mgr = InteractionManager(RNG(0))
    iid_a = _advertise(mgr, 1, InteractionType.GROUP_CONVERSATION)
    mgr.interactions[iid_a].outcome = InteractionOutcome.ACTIVE
    assert mgr.accept(7, iid_a) is True
    iid_q = _advertise(mgr, 2, InteractionType.QUEUE_USE)
    mgr.interactions[iid_q].outcome = InteractionOutcome.ACTIVE
    assert mgr.accept(7, iid_q) is True
    assert mgr.is_in_interaction(7)
    assert mgr.is_in_queue(7)

    mgr.force_stop(7)

    assert not mgr.is_in_interaction(7)
    assert not mgr.is_in_queue(7)
    assert 7 not in mgr.interactions[iid_a].participants
    assert 7 not in mgr.interactions[iid_q].contract.queue


def test_queue_length_for_object_sums_across_interactions() -> None:
    mgr = InteractionManager(RNG(0))
    c1 = InteractionContract(type=int(InteractionType.QUEUE_USE), min_participants=1, max_participants=1, queueable=True, current_participants=[1], queue=[10, 11])
    i1 = InteractionState(id=0, type=int(InteractionType.QUEUE_USE), contract=c1, participants=[1], object_id="bench", outcome=InteractionOutcome.ACTIVE)
    c2 = InteractionContract(type=int(InteractionType.QUEUE_USE), min_participants=1, max_participants=1, queueable=True, current_participants=[2], queue=[12])
    i2 = InteractionState(id=1, type=int(InteractionType.QUEUE_USE), contract=c2, participants=[2], object_id="bench", outcome=InteractionOutcome.ACTIVE)
    c3 = InteractionContract(type=int(InteractionType.QUEUE_USE), min_participants=1, max_participants=1, queueable=True, current_participants=[3], queue=[99])
    i3 = InteractionState(id=2, type=int(InteractionType.QUEUE_USE), contract=c3, participants=[3], object_id="other", outcome=InteractionOutcome.ACTIVE)
    mgr.interactions[0] = i1
    mgr.interactions[1] = i2
    mgr.interactions[2] = i3

    assert mgr.queue_length_for_object("bench") == 3
    assert mgr.queue_length_for_object("other") == 1
    assert mgr.queue_length_for_object("missing") == 0


def test_update_with_seeded_rng_is_deterministic() -> None:
    def run(seed: int) -> list[int]:
        mgr = InteractionManager(RNG(seed))
        iid1 = _advertise(mgr, 1, InteractionType.GROUP_CONVERSATION)
        mgr.interactions[iid1].outcome = InteractionOutcome.ACTIVE
        iid2 = _advertise(mgr, 2, InteractionType.GROUP_CONVERSATION)
        mgr.interactions[iid2].outcome = InteractionOutcome.ACTIVE
        cmds = {
            3: _cmd(3, CommandType.ADVERTISE, interaction_type=InteractionType.GROUP_CONVERSATION, interaction_target=iid1),
            4: _cmd(4, CommandType.ADVERTISE, interaction_type=InteractionType.GROUP_CONVERSATION, interaction_target=iid1),
            5: _cmd(5, CommandType.ADVERTISE, interaction_type=InteractionType.GROUP_CONVERSATION, interaction_target=iid2),
        }
        mgr.update(cmds)
        out: list[int] = []
        for iid in sorted(mgr.interactions.keys()):
            out.extend(mgr.interactions[iid].participants)
        return out

    assert run(123) == run(123)


def test_follow_records_leader_role() -> None:
    mgr = InteractionManager(RNG(0))
    iid = _advertise(mgr, 42, InteractionType.FOLLOW)
    interaction = mgr.interactions[iid]
    assert interaction.state.get("asymmetric_roles") is True
    assert interaction.state.get("leader") == 42


def test_talk_caps_at_two_participants() -> None:
    mgr = InteractionManager(RNG(0))
    iid = _advertise(mgr, 1, InteractionType.TALK_TO)
    assert mgr.accept(2, iid) is True
    assert mgr.interactions[iid].contract.is_full
    assert mgr.accept(3, iid) is False
    assert len(mgr.interactions[iid].participants) == 2


def test_queue_use_gets_fifo_access_and_line_formation() -> None:
    mgr = InteractionManager(RNG(0))
    iid = _advertise(mgr, 1, InteractionType.QUEUE_USE)
    contract = mgr.interactions[iid].contract
    assert contract.access is not None
    assert contract.formation is not None


def test_group_conversation_gets_f_formation() -> None:
    mgr = InteractionManager(RNG(0))
    iid = _advertise(mgr, 1, InteractionType.GROUP_CONVERSATION)
    contract = mgr.interactions[iid].contract
    assert contract.formation is not None
    assert contract.access is None  # social, not a resource


def test_talk_to_gets_dyad_and_no_access() -> None:
    mgr = InteractionManager(RNG(0))
    iid = _advertise(mgr, 1, InteractionType.TALK_TO)
    contract = mgr.interactions[iid].contract
    assert contract.formation is not None
    assert contract.access is None


def test_sit_on_is_queueable_by_default() -> None:
    mgr = InteractionManager(RNG(0))
    iid = _advertise(mgr, 1, InteractionType.SIT_ON)
    assert mgr.interactions[iid].contract.is_full
    assert mgr.accept(2, iid) is True
    assert 2 in mgr.interactions[iid].contract.queue


def test_second_advertise_with_same_object_joins_existing() -> None:
    mgr = InteractionManager(RNG(0))
    cmd_a = _cmd(1, CommandType.ADVERTISE, interaction_type=InteractionType.USE)
    cmd_a.object_id = "atm"
    cmd_b = _cmd(2, CommandType.ADVERTISE, interaction_type=InteractionType.USE)
    cmd_b.object_id = "atm"

    mgr.update({1: cmd_a})
    assert len(mgr.interactions) == 1
    iid = next(iter(mgr.interactions))
    assert mgr.interactions[iid].participants == [1]

    mgr.update({2: cmd_b})
    # Still only one interaction for object "atm"; agent 2 joined the queue
    assert len(mgr.interactions) == 1
    assert 2 in mgr.interactions[iid].contract.queue
    assert 2 not in mgr.interactions[iid].participants


def test_advertise_dedups_only_on_matching_object_id() -> None:
    mgr = InteractionManager(RNG(0))
    cmd_a = _cmd(1, CommandType.ADVERTISE, interaction_type=InteractionType.USE)
    cmd_a.object_id = "atm_1"
    cmd_b = _cmd(2, CommandType.ADVERTISE, interaction_type=InteractionType.USE)
    cmd_b.object_id = "atm_2"
    mgr.update({1: cmd_a})
    mgr.update({2: cmd_b})
    # Different object_ids => two separate interactions
    assert len(mgr.interactions) == 2


def test_advertise_dedups_only_on_matching_type() -> None:
    mgr = InteractionManager(RNG(0))
    cmd_a = _cmd(1, CommandType.ADVERTISE, interaction_type=InteractionType.USE)
    cmd_a.object_id = "kiosk"
    cmd_b = _cmd(2, CommandType.ADVERTISE, interaction_type=InteractionType.SIT_ON)
    cmd_b.object_id = "kiosk"
    mgr.update({1: cmd_a})
    mgr.update({2: cmd_b})
    # Same object, different types => two interactions
    assert len(mgr.interactions) == 2


def test_duration_expiry_promotes_queue_instead_of_teardown() -> None:
    mgr = InteractionManager(RNG(0))
    iid = _advertise(mgr, 1, InteractionType.QUEUE_USE, duration=0.5)
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE
    assert mgr.accept(2, iid) is True
    assert 2 in mgr.interactions[iid].contract.queue

    # Tick past duration; should promote agent 2, not tear down
    mgr.update({}, dt=0.6)
    assert iid in mgr.interactions
    assert mgr.interactions[iid].outcome == InteractionOutcome.ACTIVE
    assert 1 not in mgr.interactions[iid].participants
    assert 2 in mgr.interactions[iid].participants
    assert mgr.interactions[iid].contract.queue == []
    assert mgr.interactions[iid].contract.elapsed == 0.0


def test_duration_expiry_tears_down_when_queue_empty() -> None:
    mgr = InteractionManager(RNG(0))
    iid = _advertise(mgr, 1, InteractionType.QUEUE_USE, duration=0.3)
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE

    mgr.update({}, dt=0.4)
    # No queue => complete normally
    assert iid not in mgr.interactions


def test_multiple_promotions_serial_service() -> None:
    mgr = InteractionManager(RNG(0))
    iid = _advertise(mgr, 1, InteractionType.QUEUE_USE, duration=0.2)
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE
    assert mgr.accept(2, iid)
    assert mgr.accept(3, iid)

    # First promotion: 1 served, 2 takes over
    mgr.update({}, dt=0.25)
    assert 2 in mgr.interactions[iid].participants
    assert 3 in mgr.interactions[iid].contract.queue

    # Second promotion: 2 served, 3 takes over
    mgr.update({}, dt=0.25)
    assert 3 in mgr.interactions[iid].participants
    assert mgr.interactions[iid].contract.queue == []

    # Third tick: 3 served, queue empty => tear down
    mgr.update({}, dt=0.25)
    assert iid not in mgr.interactions


def test_release_participant_signals_completed_to_released_agent() -> None:
    agents: dict[int, Any] = {1: _fake_bt_agent(), 2: _fake_bt_agent()}
    mgr = InteractionManager(RNG(0), agent_lookup=lambda aid: agents.get(aid))
    iid = _advertise(mgr, 1, InteractionType.QUEUE_USE, duration=0.5)
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE
    assert mgr.accept(2, iid) is True
    assert 2 in mgr.interactions[iid].contract.queue

    mgr.update({}, dt=0.6)

    assert 1 not in mgr.interactions[iid].participants
    assert 2 in mgr.interactions[iid].participants
    assert agents[1].movement.last_outcome == InteractionOutcome.COMPLETED


def test_teardown_signals_interrupted_to_queued_agents() -> None:
    agents: dict[int, Any] = {1: _fake_bt_agent(), 2: _fake_bt_agent()}
    mgr = InteractionManager(RNG(0), agent_lookup=lambda aid: agents.get(aid))
    iid = _advertise(mgr, 1, InteractionType.QUEUE_USE)
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE
    assert mgr.accept(2, iid) is True
    assert 2 in mgr.interactions[iid].contract.queue

    mgr.stop(1, iid)

    assert mgr.interactions[iid].outcome == InteractionOutcome.INTERRUPTED
    assert agents[2].movement.last_outcome == InteractionOutcome.INTERRUPTED


def test_multiple_promotions_signal_completed_to_released() -> None:
    agents: dict[int, Any] = {1: _fake_bt_agent(), 2: _fake_bt_agent(), 3: _fake_bt_agent()}
    mgr = InteractionManager(RNG(0), agent_lookup=lambda aid: agents.get(aid))
    iid = _advertise(mgr, 1, InteractionType.QUEUE_USE, duration=0.2)
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE
    assert mgr.accept(2, iid)
    assert mgr.accept(3, iid)

    mgr.update({}, dt=0.25)
    assert 2 in mgr.interactions[iid].participants
    assert agents[1].movement.last_outcome == InteractionOutcome.COMPLETED

    mgr.update({}, dt=0.25)
    assert 3 in mgr.interactions[iid].participants
    assert agents[2].movement.last_outcome == InteractionOutcome.COMPLETED


def test_queue_promotion_uses_promoted_agents_own_duration() -> None:
    mgr = InteractionManager(RNG(0))
    cmd_a = _cmd(1, CommandType.ADVERTISE, interaction_type=InteractionType.USE, duration=0.5)
    cmd_a.object_id = "atm"
    cmd_b = _cmd(2, CommandType.ADVERTISE, interaction_type=InteractionType.USE, duration=1.0)
    cmd_b.object_id = "atm"

    mgr.update({1: cmd_a})
    iid = next(iter(mgr.interactions))
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE
    mgr.update({2: cmd_b})
    assert 2 in mgr.interactions[iid].contract.queue
    assert mgr.interactions[iid].member_durations[1] == pytest.approx(0.5)
    assert mgr.interactions[iid].member_durations[2] == pytest.approx(1.0)

    mgr.update({}, dt=0.6)
    assert 2 in mgr.interactions[iid].participants
    assert mgr.interactions[iid].contract.duration == pytest.approx(1.0)
    assert mgr.interactions[iid].contract.elapsed == pytest.approx(0.0)
    assert 1 not in mgr.interactions[iid].member_durations

    mgr.update({}, dt=0.7)
    assert iid in mgr.interactions
    assert mgr.interactions[iid].contract.elapsed == pytest.approx(0.7)

    mgr.update({}, dt=0.4)
    assert iid not in mgr.interactions


def test_find_interaction_for_object_uses_index_and_reindexes_after_teardown() -> None:
    mgr = InteractionManager(RNG(0))
    cmd_a = _cmd(1, CommandType.ADVERTISE, interaction_type=InteractionType.USE)
    cmd_a.object_id = "atm_a"
    cmd_b = _cmd(2, CommandType.ADVERTISE, interaction_type=InteractionType.USE)
    cmd_b.object_id = "atm_b"
    mgr.update({1: cmd_a})
    mgr.update({2: cmd_b})
    assert len(mgr.interactions) == 2

    found_a = mgr._find_interaction_for_object(int(InteractionType.USE), "atm_a")
    assert found_a is not None and found_a.participants == [1]
    found_b = mgr._find_interaction_for_object(int(InteractionType.USE), "atm_b")
    assert found_b is not None and found_b.participants == [2]

    # Tear down interaction A; index must forget it so a new advertise re-creates.
    mgr._teardown(found_a.id, InteractionOutcome.COMPLETED)
    assert mgr._find_interaction_for_object(int(InteractionType.USE), "atm_a") is None

    cmd_a2 = _cmd(3, CommandType.ADVERTISE, interaction_type=InteractionType.USE)
    cmd_a2.object_id = "atm_a"
    mgr.update({3: cmd_a2})
    found_a2 = mgr._find_interaction_for_object(int(InteractionType.USE), "atm_a")
    assert found_a2 is not None and found_a2.participants == [3]
    assert found_a2.id != found_a.id


class _FakeFormation:
    def __init__(self) -> None:
        self._arrived: dict[int, bool] = {}

    def set_arrived(self, agent_id: int, value: bool) -> None:
        self._arrived[agent_id] = value

    def arrived(self, agent_id: int) -> bool:
        return self._arrived.get(agent_id, False)

    def on_join(self, agent_id: int) -> None:
        self._arrived.setdefault(agent_id, False)

    def on_leave(self, agent_id: int) -> None:
        self._arrived.pop(agent_id, None)

    def tick(self, _dt: float) -> dict[int, Pose2D]:
        return {}


def test_duration_does_not_advance_until_arrived() -> None:
    mgr = InteractionManager(RNG(0))
    cmd = _cmd(1, CommandType.ADVERTISE, interaction_type=InteractionType.USE, duration=0.5)
    cmd.object_id = "atm"
    mgr.update({1: cmd})
    interaction = next(iter(mgr.interactions.values()))
    interaction.contract.duration = 0.5
    fake = _FakeFormation()
    fake.on_join(1)
    interaction.contract.formation = fake  # type: ignore[assignment]
    interaction.outcome = InteractionOutcome.ACTIVE

    mgr.update({}, dt=0.2)
    mgr.update({}, dt=0.2)
    mgr.update({}, dt=0.2)
    assert interaction.contract.elapsed == 0.0

    fake.set_arrived(1, True)
    mgr.update({}, dt=0.2)
    assert interaction.contract.elapsed == pytest.approx(0.2)


def test_duration_expiry_rotates_fifo_queue() -> None:
    from arena_humansim.core.access.fifo_queue import FIFOQueue

    mgr = InteractionManager(RNG(0))
    contract = InteractionContract(
        type=int(InteractionType.USE),
        min_participants=1,
        max_participants=1,
        queueable=True,
        current_participants=[1],
        queue=[2, 3],
        duration=0.5,
    )
    contract.access = FIFOQueue()
    interaction = InteractionState(
        id=0,
        type=int(InteractionType.USE),
        contract=contract,
        participants=[1],
        outcome=InteractionOutcome.ACTIVE,
        object_id="atm",
    )
    mgr.interactions[0] = interaction
    mgr.next_interaction_id = 1
    mgr._agent_to_interactions.setdefault(1, set()).add(0)
    mgr._agent_to_queues.setdefault(2, set()).add(0)
    mgr._agent_to_queues.setdefault(3, set()).add(0)
    mgr._interaction_by_object_type[("atm", int(InteractionType.USE))] = 0

    mgr.update({}, dt=0.6)

    assert interaction.participants == [2], "agent 2 should be promoted from queue"
    assert interaction.contract.queue == [3], "agent 3 should still be queued"
    assert mgr.is_in_interaction(2)
    assert not mgr.is_in_interaction(1), "agent 1 should have been released"

    mgr.update({}, dt=0.6)

    assert interaction.participants == [3], "agent 3 should be promoted on next expiry"
    assert interaction.contract.queue == []
    assert not mgr.is_in_interaction(2)


# ---------------------------------------------------------------------------
# Matcher tests (rules 1-4 of _try_bind).
# ---------------------------------------------------------------------------


def _mk_matcher_mgr(
    agents: dict[int, Any],
    visibility: dict[int, set[int]] | None = None,
) -> InteractionManager:
    mgr = InteractionManager(RNG(0))
    vis = visibility if visibility is not None else {aid: set(agents) - {aid} for aid in agents}
    mgr.set_context(
        agent_lookup=lambda aid: agents.get(aid),
        visibility_lookup=lambda aid: vis.get(aid, set()),
    )
    return mgr


def test_matcher_targeted_interaction_joins_specific() -> None:
    agents = {1: _fake_bt_agent(1), 2: _fake_bt_agent(2, x=100.0)}  # far away
    mgr = _mk_matcher_mgr(agents)
    iid = _advertise(mgr, 1, InteractionType.GROUP_CONVERSATION)
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE

    # Agent 2 is far from the interaction, but targets it explicitly -> joins.
    cmd = _cmd(2, CommandType.ADVERTISE, interaction_type=InteractionType.GROUP_CONVERSATION, interaction_target=iid)
    mgr.update({2: cmd})

    assert 2 in mgr.interactions[iid].participants


def test_matcher_object_anchored_creates_and_joins() -> None:
    mgr = _mk_matcher_mgr({1: _fake_bt_agent(1), 2: _fake_bt_agent(2)})
    cmd_a = _cmd(1, CommandType.ADVERTISE, interaction_type=InteractionType.USE)
    cmd_a.object_id = "atm"
    mgr.update({1: cmd_a})
    assert len(mgr.interactions) == 1
    iid = next(iter(mgr.interactions))
    assert mgr.interactions[iid].object_id == "atm"
    assert mgr.interactions[iid].participants == [1]

    cmd_b = _cmd(2, CommandType.ADVERTISE, interaction_type=InteractionType.USE)
    cmd_b.object_id = "atm"
    mgr.update({2: cmd_b})
    # Same object -> same interaction; agent 2 queues (USE caps at 1).
    assert len(mgr.interactions) == 1
    assert 2 in mgr.interactions[iid].contract.queue


def test_matcher_targeted_agent_pairs_two_unbound() -> None:
    agents = {1: _fake_bt_agent(1), 2: _fake_bt_agent(2)}
    mgr = _mk_matcher_mgr(agents)
    # Both advertise TALK_TO targeting each other.
    cmd_a = _cmd(1, CommandType.ADVERTISE, interaction_type=InteractionType.TALK_TO, target_agent=2)
    cmd_b = _cmd(2, CommandType.ADVERTISE, interaction_type=InteractionType.TALK_TO, target_agent=1)
    mgr.update({1: cmd_a, 2: cmd_b})

    assert len(mgr.interactions) == 1
    iid = next(iter(mgr.interactions))
    participants = set(mgr.interactions[iid].participants)
    assert participants == {1, 2}


def test_matcher_targeted_agent_joins_bound() -> None:
    # Agent 1 already in an interaction; agent 3 ads targeting 1 -> joins it.
    agents = {1: _fake_bt_agent(1), 2: _fake_bt_agent(2), 3: _fake_bt_agent(3, x=50.0)}
    mgr = _mk_matcher_mgr(agents)
    iid = _advertise(mgr, 1, InteractionType.GROUP_CONVERSATION)
    mgr.accept(2, iid)
    mgr.interactions[iid].outcome = InteractionOutcome.ACTIVE

    cmd = _cmd(3, CommandType.ADVERTISE, interaction_type=InteractionType.GROUP_CONVERSATION, target_agent=1)
    mgr.update({3: cmd})

    assert 3 in mgr.interactions[iid].participants


def test_matcher_open_pairs_visible_unbound() -> None:
    agents = {1: _fake_bt_agent(1, x=0.0), 2: _fake_bt_agent(2, x=1.0)}
    mgr = _mk_matcher_mgr(agents)  # default visibility: both see each other
    cmd_a = _cmd(1, CommandType.ADVERTISE, interaction_type=InteractionType.TALK_TO)
    cmd_b = _cmd(2, CommandType.ADVERTISE, interaction_type=InteractionType.TALK_TO)

    # First tick: agent 1 alone, stays unbound (no visible open ad).
    mgr.update({1: cmd_a})
    assert len(mgr.interactions) == 0
    assert mgr._advertisements.get(1, [])[0].interaction_id is None

    # Second tick: agent 2 advertises, they see each other -> pair.
    mgr.update({2: cmd_b})
    assert len(mgr.interactions) == 1
    iid = next(iter(mgr.interactions))
    assert set(mgr.interactions[iid].participants) == {1, 2}


def test_matcher_open_skips_invisible() -> None:
    agents = {1: _fake_bt_agent(1, x=0.0), 2: _fake_bt_agent(2, x=100.0)}
    # Empty visibility: neither agent perceives the other.
    mgr = _mk_matcher_mgr(agents, visibility={1: set(), 2: set()})
    cmd_a = _cmd(1, CommandType.ADVERTISE, interaction_type=InteractionType.TALK_TO)
    cmd_b = _cmd(2, CommandType.ADVERTISE, interaction_type=InteractionType.TALK_TO)

    mgr.update({1: cmd_a, 2: cmd_b})
    assert len(mgr.interactions) == 0
    assert mgr._advertisements.get(1, [])[0].interaction_id is None
    assert mgr._advertisements.get(2, [])[0].interaction_id is None


def test_matcher_open_joins_visible_bound() -> None:
    agents = {1: _fake_bt_agent(1, x=0.0), 2: _fake_bt_agent(2, x=0.5), 3: _fake_bt_agent(3, x=1.0)}
    mgr = _mk_matcher_mgr(agents)  # default visibility: all see each other
    # Two agents form a GROUP_CONVERSATION via pairing.
    cmd_a = _cmd(1, CommandType.ADVERTISE, interaction_type=InteractionType.GROUP_CONVERSATION)
    cmd_b = _cmd(2, CommandType.ADVERTISE, interaction_type=InteractionType.GROUP_CONVERSATION)
    mgr.update({1: cmd_a, 2: cmd_b})
    assert len(mgr.interactions) == 1
    iid = next(iter(mgr.interactions))

    # Third agent can see a bound participant -> joins existing.
    cmd_c = _cmd(3, CommandType.ADVERTISE, interaction_type=InteractionType.GROUP_CONVERSATION)
    mgr.update({3: cmd_c})
    assert len(mgr.interactions) == 1
    assert set(mgr.interactions[iid].participants) == {1, 2, 3}


def test_matcher_open_skips_full_interactions() -> None:
    # TALK_TO caps at 2. Two agents pair, then a third visible peer must stay unbound.
    agents = {1: _fake_bt_agent(1, x=0.0), 2: _fake_bt_agent(2, x=0.5), 3: _fake_bt_agent(3, x=1.0)}
    mgr = _mk_matcher_mgr(agents)  # default visibility: all see each other
    cmd_a = _cmd(1, CommandType.ADVERTISE, interaction_type=InteractionType.TALK_TO)
    cmd_b = _cmd(2, CommandType.ADVERTISE, interaction_type=InteractionType.TALK_TO)
    mgr.update({1: cmd_a, 2: cmd_b})
    assert len(mgr.interactions) == 1

    cmd_c = _cmd(3, CommandType.ADVERTISE, interaction_type=InteractionType.TALK_TO)
    mgr.update({3: cmd_c})
    # Bound TALK_TO is full -> agent 3 can't join it, and nothing else to pair with.
    assert len(mgr.interactions) == 1
    assert 3 not in next(iter(mgr.interactions.values())).participants


def test_matcher_post_ad_idempotent_re_advertise() -> None:
    agents = {1: _fake_bt_agent(1)}
    mgr = _mk_matcher_mgr(agents)
    cmd = _cmd(1, CommandType.ADVERTISE, interaction_type=InteractionType.TALK_TO)

    mgr.update({1: cmd})
    mgr.update({1: cmd})
    mgr.update({1: cmd})
    # Should still have exactly one unbound ad.
    assert len(mgr._advertisements.get(1, [])) == 1


def test_anchorless_sit_on_anchor_falls_back_to_creator_pose() -> None:
    agents = {1: _fake_bt_agent(1, x=7.0, y=3.0)}
    mgr = _mk_matcher_mgr(agents)
    # SIT_ON has a cluster formation by default, so this exercises _build_anchor_for
    # with no object_id -> creator-pose fallback (previously snapped to origin).
    interaction = mgr._create_interaction(int(InteractionType.SIT_ON), 1, object_id=None)
    formation = interaction.contract.formation
    assert formation is not None
    anchor = formation.anchor
    pose = anchor.pose()
    assert pose.x == pytest.approx(7.0)
    assert pose.y == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# SERVICE matcher tests.
# ---------------------------------------------------------------------------


class _FakeRobotAgent:
    def __init__(self, agent_id: int, x: float = 0.0, y: float = 0.0) -> None:
        self.state = AgentState(
            agent_id=agent_id,
            desired_velocity=0.0,
            pose=Pose2D(x=x, y=y),
            kind=int(AgentKind.ROBOT),
        )
        self.movement = BehaviorTreeMovement()
        self.params = _FakeParams()


def _fake_robot_agent(agent_id: int = 0, x: float = 0.0, y: float = 0.0) -> Any:
    return _FakeRobotAgent(agent_id=agent_id, x=x, y=y)


def _service_ad(
    agent_id: int,
    *,
    service_tag: str,
    max_participants: int | None = None,
) -> HighLevelCommand:
    cmd = _cmd(agent_id, CommandType.ADVERTISE, interaction_type=int(InteractionType.SERVICE))
    cmd.service_tag = service_tag
    cmd.max_participants = max_participants
    return cmd


def test_service_matcher_binds_visible_robot_and_human() -> None:
    agents = {1: _fake_robot_agent(1, x=0.0, y=0.0), 2: _fake_bt_agent(2, x=1.0, y=0.0)}
    mgr = _mk_matcher_mgr(agents)
    cmd_robot = _service_ad(1, service_tag="water", max_participants=2)
    cmd_human = _service_ad(2, service_tag="water")

    mgr.update({1: cmd_robot, 2: cmd_human})

    assert len(mgr.interactions) == 1
    iid = next(iter(mgr.interactions))
    interaction = mgr.interactions[iid]
    assert interaction.type == int(InteractionType.SERVICE)
    assert set(interaction.participants) == {1, 2}
    assert interaction.participants[0] == 1  # robot is anchor/initiator
    assert interaction.contract.max_participants == 2


def test_service_fifo_queues_overflow() -> None:
    agents = {
        1: _fake_robot_agent(1, x=0.0, y=0.0),
        2: _fake_bt_agent(2, x=1.0, y=0.0),
        3: _fake_bt_agent(3, x=1.5, y=0.0),
    }
    mgr = _mk_matcher_mgr(agents)
    cmd_robot = _service_ad(1, service_tag="water", max_participants=1)
    cmd_a = _service_ad(2, service_tag="water")
    cmd_b = _service_ad(3, service_tag="water")

    mgr.update({1: cmd_robot, 2: cmd_a, 3: cmd_b})

    assert len(mgr.interactions) == 1
    iid = next(iter(mgr.interactions))
    interaction = mgr.interactions[iid]
    assert interaction.contract.max_participants == 1
    # Robot occupies the only slot; one human is participant-free, one queued.
    participants = set(interaction.participants)
    queue = set(interaction.contract.queue)
    assert 1 in participants
    humans_in = participants - {1}
    humans_queued = queue
    assert len(humans_in) + len(humans_queued) == 2
    assert len(humans_queued) >= 1
    # Queueing routed through the contract's FIFOQueue.
    from arena_humansim.core.access.fifo_queue import FIFOQueue
    assert isinstance(interaction.contract.access, FIFOQueue)


def test_service_two_tags_two_interactions() -> None:
    agents = {
        1: _fake_robot_agent(1, x=0.0, y=0.0),
        2: _fake_bt_agent(2, x=1.0, y=0.0),
        3: _fake_bt_agent(3, x=-1.0, y=0.0),
    }
    mgr = _mk_matcher_mgr(agents)
    robot_water = _service_ad(1, service_tag="water", max_participants=2)
    robot_trash = _service_ad(1, service_tag="trash", max_participants=2)
    cmd_human_water = _service_ad(2, service_tag="water")
    cmd_human_trash = _service_ad(3, service_tag="trash")

    mgr.update(
        {2: cmd_human_water, 3: cmd_human_trash},
        extra_commands=[robot_water, robot_trash],
    )

    assert len(mgr.interactions) == 2
    by_tag: dict[int, list[int]] = {}
    for interaction in mgr.interactions.values():
        by_tag[interaction.id] = list(interaction.participants)
    # Collect tag via the advertisements attached to the interaction.
    tag_to_participants: dict[str, set[int]] = {}
    for ads in mgr._advertisements.values():
        for ad in ads:
            if ad.interaction_id is None:
                continue
            tag_to_participants.setdefault(ad.service_tag or "", set()).update(
                mgr.interactions[ad.interaction_id].participants,
            )
    assert tag_to_participants["water"] == {1, 2}
    assert tag_to_participants["trash"] == {1, 3}
    # Participants are disjoint except for the shared robot.
    water_humans = tag_to_participants["water"] - {1}
    trash_humans = tag_to_participants["trash"] - {1}
    assert water_humans.isdisjoint(trash_humans)


def test_service_two_robots_same_tag_parallel_interactions() -> None:
    agents = {
        1: _fake_robot_agent(1, x=0.0, y=0.0),
        2: _fake_robot_agent(2, x=100.0, y=0.0),
        3: _fake_bt_agent(3, x=1.0, y=0.0),
        4: _fake_bt_agent(4, x=101.0, y=0.0),
    }
    # Each human only sees its nearby robot and vice-versa.
    visibility = {
        1: {3},
        2: {4},
        3: {1},
        4: {2},
    }
    mgr = _mk_matcher_mgr(agents, visibility=visibility)
    cmds = {
        1: _service_ad(1, service_tag="water", max_participants=2),
        2: _service_ad(2, service_tag="water", max_participants=2),
        3: _service_ad(3, service_tag="water"),
        4: _service_ad(4, service_tag="water"),
    }

    mgr.update(cmds)

    assert len(mgr.interactions) == 2
    pairs = {frozenset(i.participants) for i in mgr.interactions.values()}
    assert pairs == {frozenset({1, 3}), frozenset({2, 4})}


def test_service_max_participants_threads_from_ad() -> None:
    agents = {1: _fake_robot_agent(1, x=0.0, y=0.0), 2: _fake_bt_agent(2, x=1.0, y=0.0)}
    mgr = _mk_matcher_mgr(agents)
    mgr.update({
        1: _service_ad(1, service_tag="water", max_participants=3),
        2: _service_ad(2, service_tag="water"),
    })
    assert len(mgr.interactions) == 1
    interaction = next(iter(mgr.interactions.values()))
    assert interaction.contract.max_participants == 3

    agents2 = {10: _fake_robot_agent(10, x=0.0, y=0.0), 20: _fake_bt_agent(20, x=1.0, y=0.0)}
    mgr2 = _mk_matcher_mgr(agents2)
    mgr2.update({
        10: _service_ad(10, service_tag="water", max_participants=None),
        20: _service_ad(20, service_tag="water"),
    })
    assert len(mgr2.interactions) == 1
    interaction2 = next(iter(mgr2.interactions.values()))
    assert interaction2.contract.max_participants == -1
