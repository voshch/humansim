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


def test_remove_subset_of_agents(system: RosTestSystem) -> None:
    system.call(RemoveAgents, "remove_agents", make_remove_request([-1]))
    specs = [{"x": float(i), "y": 0.0} for i in range(4)]
    spawn_resp = system.call(SpawnAgents, "spawn_agents", make_spawn_request(specs))
    spawned = list(spawn_resp.spawned_ids)
    to_remove = spawned[:2]
    remaining = set(spawned[2:])

    remove_resp = system.call(RemoveAgents, "remove_agents", make_remove_request(to_remove))
    assert remove_resp.success is True

    system.subscribe_agent_states()
    system.tick_manager(1)
    msg = system.wait_for_agent_states(timeout=5.0)

    published_ids = {a.agent_id for a in msg.agents}
    assert remaining.issubset(published_ids)
    for rid in to_remove:
        assert rid not in published_ids


def test_remove_all_with_negative_one(system: RosTestSystem) -> None:
    system.call(RemoveAgents, "remove_agents", make_remove_request([-1]))
    specs = [{"x": float(i), "y": 0.0} for i in range(3)]
    system.call(SpawnAgents, "spawn_agents", make_spawn_request(specs))

    resp = system.call(RemoveAgents, "remove_agents", make_remove_request([-1]))
    assert resp.success is True

    system.subscribe_agent_states()
    system.tick_manager(1)
    msg = system.wait_for_agent_states(timeout=5.0)
    assert len(msg.agents) == 0
