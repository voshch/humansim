from __future__ import annotations

import pytest

from tests.ros._helpers import (
    RemoveAgents,
    ResetSimulation,
    RosTestSystem,
    SpawnAgents,
    make_remove_request,
    make_reset_request,
    make_spawn_request,
)

pytestmark = pytest.mark.ros


@pytest.fixture(scope="module")
def system(ros_system: RosTestSystem) -> RosTestSystem:
    ros_system.call(RemoveAgents, "remove_agents", make_remove_request([-1]))
    return ros_system


def test_reset_empties_pool(system: RosTestSystem) -> None:
    system.call(RemoveAgents, "remove_agents", make_remove_request([-1]))
    specs = [{"x": float(i), "y": 0.0} for i in range(3)]
    system.call(SpawnAgents, "spawn_agents", make_spawn_request(specs))

    resp = system.call(ResetSimulation, "reset", make_reset_request())
    assert resp.success is True
    assert system.manager._pool.n == 0
    assert len(system.manager._agents) == 0
    assert system.manager._tick_count == 0


def test_reset_reseeds_ids(system: RosTestSystem) -> None:
    system.call(ResetSimulation, "reset", make_reset_request())
    first = system.call(
        SpawnAgents,
        "spawn_agents",
        make_spawn_request([{"x": 0.0, "y": 0.0}]),
    )
    first_ids = list(first.spawned_ids)

    system.call(ResetSimulation, "reset", make_reset_request())
    second = system.call(
        SpawnAgents,
        "spawn_agents",
        make_spawn_request([{"x": 0.0, "y": 0.0}]),
    )
    second_ids = list(second.spawned_ids)

    assert first_ids == second_ids


def test_reset_deterministic_trajectory(system: RosTestSystem) -> None:
    def run_one_pass() -> list[tuple[float, float]]:
        system.call(ResetSimulation, "reset", make_reset_request())
        system.call(
            SpawnAgents,
            "spawn_agents",
            make_spawn_request([{"x": 0.0, "y": 0.0, "waypoints": [(5.0, 0.0)]}]),
        )
        poses: list[tuple[float, float]] = []
        for _ in range(10):
            system.tick_manager(1)
            pool = system.manager._pool
            if pool.n > 0:
                poses.append((float(pool.pos[0, 0]), float(pool.pos[0, 1])))
        return poses

    first = run_one_pass()
    second = run_one_pass()

    assert first == second, "reset+reseed must produce identical trajectory"
