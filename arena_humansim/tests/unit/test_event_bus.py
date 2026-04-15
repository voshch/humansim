from __future__ import annotations

from arena_humansim.utils.event_bus import EventBus


def test_fire_then_has() -> None:
    bus = EventBus()
    bus.fire("arrived", 42)
    assert bus.has("arrived", 42)
    assert not bus.has("arrived", 7)
    assert not bus.has("departed", 42)


def test_consume_removes_agent() -> None:
    bus = EventBus()
    bus.fire("arrived", 42)
    assert bus.consume("arrived", 42) is True
    assert not bus.has("arrived", 42)


def test_consume_missing_returns_false() -> None:
    bus = EventBus()
    assert bus.consume("arrived", 42) is False
    bus.fire("arrived", 1)
    assert bus.consume("arrived", 999) is False


def test_fire_broadcast_global_without_ids() -> None:
    bus = EventBus()
    bus.fire_broadcast("global_halt")
    assert bus.has("global_halt", 1)
    assert bus.has("global_halt", 999)
    assert bus.consume("global_halt", 1) is True
    assert bus.has("global_halt", 2)


def test_fire_broadcast_with_ids_unions() -> None:
    bus = EventBus()
    bus.fire_broadcast("wave", {1, 2})
    bus.fire_broadcast("wave", {3, 4})
    for aid in (1, 2, 3, 4):
        assert bus.has("wave", aid)
    assert not bus.has("wave", 5)


def test_broadcast_beats_targeted() -> None:
    bus = EventBus()
    bus.fire_broadcast("shutdown")
    bus.fire("shutdown", 7)
    assert bus.has("shutdown", 123)
    assert bus.consume("shutdown", 7) is True
    assert bus.has("shutdown", 123)


def test_clear_agent_auto_prunes_last_listener() -> None:
    bus = EventBus()
    bus.fire("arrived", 1)
    bus.fire("arrived", 2)
    bus.fire("bored", 1)
    bus.clear_agent(1)
    assert not bus.has("bored", 1)
    assert "bored" not in bus._events
    assert bus.has("arrived", 2)
    assert not bus.has("arrived", 1)


def test_clear_agent_skips_broadcast() -> None:
    bus = EventBus()
    bus.fire_broadcast("global")
    bus.clear_agent(5)
    assert bus.has("global", 5)


def test_clear_empties_bus() -> None:
    bus = EventBus()
    bus.fire("a", 1)
    bus.fire_broadcast("b")
    assert len(bus) == 2
    assert bool(bus)
    bus.clear()
    assert len(bus) == 0
    assert not bus


def test_len_and_bool_track_events() -> None:
    bus = EventBus()
    assert len(bus) == 0
    assert not bus
    bus.fire("x", 1)
    assert len(bus) == 1
    assert bool(bus)
    bus.consume("x", 1)
    assert len(bus) == 0
    assert not bus
