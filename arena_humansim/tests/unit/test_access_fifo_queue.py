from __future__ import annotations

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.access import AcceptResult
from arena_humansim.core.access.fifo_queue import FIFOQueue
from arena_humansim.utils.types import InteractionContract, InteractionState


def _make_interaction(max_participants: int = 1, participants: list[int] | None = None) -> InteractionState:
    contract = InteractionContract(
        max_participants=max_participants,
        current_participants=list(participants or []),
        queueable=True,
    )
    return InteractionState(contract=contract, participants=list(participants or []))


def test_on_accept_adds_participant_when_not_full() -> None:
    access = FIFOQueue()
    inter = _make_interaction(max_participants=2, participants=[1])
    assert access.on_accept(inter, 2) == AcceptResult.BECAME_PARTICIPANT
    assert 2 in inter.participants
    assert 2 in inter.contract.current_participants


def test_on_accept_queues_when_full() -> None:
    access = FIFOQueue()
    inter = _make_interaction(max_participants=1, participants=[1])
    assert access.on_accept(inter, 2) == AcceptResult.QUEUED
    assert inter.contract.queue == [2]


def test_on_accept_rejects_when_queue_cap_reached() -> None:
    access = FIFOQueue(max_queue=1)
    inter = _make_interaction(max_participants=1, participants=[1])
    inter.contract.queue.append(99)
    assert access.on_accept(inter, 2) == AcceptResult.REJECTED
    assert 2 not in inter.contract.queue


def test_tick_promotes_until_full() -> None:
    access = FIFOQueue()
    inter = _make_interaction(max_participants=2, participants=[1])
    inter.contract.queue.extend([2, 3, 4])
    promoted = access.tick(inter, dt=0.01)
    assert promoted == [2]
    assert inter.contract.queue == [3, 4]
    assert 2 in inter.participants


def test_tick_noop_when_full() -> None:
    access = FIFOQueue()
    inter = _make_interaction(max_participants=1, participants=[1])
    inter.contract.queue.append(2)
    assert access.tick(inter, dt=0.01) == []
    assert inter.contract.queue == [2]


def test_on_stop_removes_from_queue() -> None:
    access = FIFOQueue()
    inter = _make_interaction(max_participants=1, participants=[1])
    inter.contract.queue.extend([2, 3, 4])
    access.on_stop(inter, 3)
    assert inter.contract.queue == [2, 4]


def test_duplicate_accept_is_idempotent() -> None:
    access = FIFOQueue()
    inter = _make_interaction(max_participants=1, participants=[1])
    inter.contract.queue.append(2)
    assert access.on_accept(inter, 2) == AcceptResult.QUEUED
    assert inter.contract.queue == [2]
