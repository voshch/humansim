from __future__ import annotations

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.interaction_manager import CommandType, InteractionManager
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.types import (
    HighLevelCommand,
    InteractionContract,
    InteractionOutcome,
    InteractionState,
    InteractionType,
    Pose2D,
)


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
) -> int:
    cmd = _cmd(
        agent_id,
        CommandType.ADVERTISE,
        interaction_type=itype,
        target_agent=target_agent,
        duration=duration,
    )
    mgr.update({agent_id: cmd})
    ads = mgr._advertisements.get(agent_id, [])
    assert ads and ads[-1].interaction_id is not None
    return ads[-1].interaction_id


def test_advertise_then_search_excludes_self() -> None:
    mgr = InteractionManager(RNG(0))
    mgr.advertise(1, InteractionType.TALK_TO)
    mgr.advertise(2, InteractionType.TALK_TO)
    mgr.advertise(3, InteractionType.FOLLOW)

    results_for_1 = mgr.search(1, InteractionType.TALK_TO)
    assert [a.agent_id for a in results_for_1] == [2]

    results_for_3 = mgr.search(3, InteractionType.TALK_TO)
    assert {a.agent_id for a in results_for_3} == {1, 2}

    assert mgr.search(1, InteractionType.FOLLOW) == [ad for ad in mgr._ads_by_type[InteractionType.FOLLOW]]


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
    assert mgr.interactions[iid].outcome == InteractionOutcome.COMPLETED


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
            3: _cmd(3, CommandType.ACCEPT, interaction_type=InteractionType.GROUP_CONVERSATION),
            4: _cmd(4, CommandType.ACCEPT, interaction_type=InteractionType.GROUP_CONVERSATION),
            5: _cmd(5, CommandType.ACCEPT, interaction_type=InteractionType.GROUP_CONVERSATION),
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
