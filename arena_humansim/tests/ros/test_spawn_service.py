from __future__ import annotations

import pytest

from tests.ros._helpers import (
    RemoveAgents,
    RosTestSystem,
    SpawnAgents,
    make_remove_request,
    make_spawn_request,
)

pytestmark = pytest.mark.ros


@pytest.fixture(scope="module")
def system(ros_system: RosTestSystem) -> RosTestSystem:
    ros_system.call(RemoveAgents, "remove_agents", make_remove_request([-1]))
    return ros_system


def test_spawn_agents_round_trip(system: RosTestSystem) -> None:
    specs = [
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": 1.0},
        {"x": -2.0, "y": 3.0},
    ]
    req = make_spawn_request(specs)
    resp = system.call(SpawnAgents, "spawn_agents", req)

    assert resp is not None
    assert resp.success is True
    assert len(resp.spawned_ids) == len(specs)
    for aid in resp.spawned_ids:
        assert aid > 0


def test_spawned_ids_appear_in_agent_states_topic(system: RosTestSystem) -> None:
    system.call(RemoveAgents, "remove_agents", make_remove_request([-1]))
    specs = [{"x": float(i), "y": 0.0} for i in range(3)]
    resp = system.call(SpawnAgents, "spawn_agents", make_spawn_request(specs))
    expected = set(resp.spawned_ids)

    system.subscribe_agent_states()
    system.tick_manager(1)
    msg = system.wait_for_agent_states(timeout=5.0)

    published_ids = {a.agent_id for a in msg.agents}
    assert expected.issubset(published_ids), f"expected {expected} ⊆ {published_ids}"


def test_spawn_accepts_explicit_agent_id(system: RosTestSystem) -> None:
    system.call(RemoveAgents, "remove_agents", make_remove_request([-1]))
    specs = [{"agent_id": 42, "x": 0.0, "y": 0.0}]
    resp = system.call(SpawnAgents, "spawn_agents", make_spawn_request(specs))
    assert resp.success is True
    assert 42 in resp.spawned_ids
