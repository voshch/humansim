from __future__ import annotations

from arena_humansim.utils.types import InteractionContract


def test_is_full_at_boundary() -> None:
    c = InteractionContract(max_participants=3, current_participants=[1, 2])
    assert not c.is_full
    c.current_participants.append(3)
    assert c.is_full


def test_is_full_false_for_unbounded() -> None:
    c = InteractionContract(max_participants=-1, current_participants=[1, 2, 3, 4])
    assert not c.is_full


def test_remaining_slots_nonnegative_when_bounded() -> None:
    c = InteractionContract(max_participants=4, current_participants=[1, 2])
    assert c.remaining_slots == 2
    c.current_participants.extend([3, 4])
    assert c.remaining_slots == 0


def test_remaining_slots_unbounded_returns_sentinel() -> None:
    c = InteractionContract(max_participants=-1)
    assert c.remaining_slots == -1


def test_queue_ordering_preserved() -> None:
    c = InteractionContract(queueable=True)
    for aid in (5, 3, 7, 1):
        c.queue.append(aid)
    assert c.queue == [5, 3, 7, 1]
    assert c.queue_length == 4


def test_defaults() -> None:
    c = InteractionContract()
    assert c.type == 0
    assert c.min_participants == 2
    assert c.max_participants == 2
    assert c.current_participants == []
    assert c.queue == []
    assert c.queue_length == 0
    assert not c.is_full
    assert c.remaining_slots == 2
    assert c.elapsed == 0.0
    assert c.duration is None
