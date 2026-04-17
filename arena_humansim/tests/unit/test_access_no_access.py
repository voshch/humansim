from __future__ import annotations

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.access import AcceptResult
from arena_humansim.core.access.no_access import NoAccess
from arena_humansim.utils.types import InteractionContract, InteractionState


def _make_interaction(max_participants: int = 1, participants: list[int] | None = None) -> InteractionState:
    contract = InteractionContract(
        max_participants=max_participants,
        current_participants=list(participants or []),
    )
    return InteractionState(contract=contract, participants=list(participants or []))


def test_on_accept_adds_when_not_full() -> None:
    access = NoAccess()
    inter = _make_interaction(max_participants=2, participants=[1])
    assert access.on_accept(inter, 2) == AcceptResult.BECAME_PARTICIPANT
    assert 2 in inter.participants
    assert 2 in inter.contract.current_participants


def test_on_accept_rejects_when_full() -> None:
    access = NoAccess()
    inter = _make_interaction(max_participants=1, participants=[1])
    assert access.on_accept(inter, 2) == AcceptResult.REJECTED
    assert 2 not in inter.participants
    assert inter.contract.queue == []


def test_tick_returns_no_promotions() -> None:
    access = NoAccess()
    inter = _make_interaction(max_participants=1, participants=[1])
    assert access.tick(inter, dt=0.1) == []


def test_on_stop_is_noop() -> None:
    access = NoAccess()
    inter = _make_interaction(max_participants=2, participants=[1, 2])
    access.on_stop(inter, 1)
    # NoAccess doesn't track queue or remove participants directly
    assert inter.participants == [1, 2]
