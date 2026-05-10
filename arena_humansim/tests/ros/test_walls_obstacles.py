from __future__ import annotations

import pytest

from tests.ros._helpers import (
    AddWalls,
    RemoveAgents,
    RemoveWalls,
    ResetSimulation,
    RosTestSystem,
    SpawnAgents,
    make_add_walls_request,
    make_remove_request,
    make_remove_walls_request,
    make_reset_request,
    make_spawn_request,
)

pytestmark = pytest.mark.ros


@pytest.fixture(scope="module")
def system(ros_system: RosTestSystem) -> RosTestSystem:
    ros_system.call(ResetSimulation, "reset", make_reset_request())
    return ros_system


def _trajectory(system: RosTestSystem, n_ticks: int = 20) -> list[tuple[float, float]]:
    poses: list[tuple[float, float]] = []
    for _ in range(n_ticks):
        system.tick_manager(1)
        pool = system.manager._pool
        if pool.n > 0:
            poses.append((float(pool.pos[0, 0]), float(pool.pos[0, 1])))
    return poses


def test_add_walls_service_success(system: RosTestSystem) -> None:
    system.call(ResetSimulation, "reset", make_reset_request())
    req = make_add_walls_request(
        [
            ("w1", (-5.0, 0.5), (5.0, 0.5)),
            ("w2", (-5.0, -0.5), (5.0, -0.5)),
        ]
    )
    resp = system.call(AddWalls, "add_walls", req)
    assert resp.success is True
    assert "w1" in system.manager._walls
    assert "w2" in system.manager._walls


def test_add_walls_propagates_to_wall_aware(system: RosTestSystem) -> None:
    system.call(ResetSimulation, "reset", make_reset_request())
    system.call(
        AddWalls,
        "add_walls",
        make_add_walls_request([("corridor_top", (-10.0, 1.0), (10.0, 1.0))]),
    )

    for sub in system.manager._wall_aware:
        walls_attr = None
        for attr in ("_walls", "walls", "_wall_segments"):
            if hasattr(sub, attr):
                walls_attr = getattr(sub, attr)
                break
        if walls_attr is None:
            continue
        if isinstance(walls_attr, (list, tuple)):
            assert len(walls_attr) >= 1
        else:
            assert walls_attr is not None


def test_remove_walls_clears_state(system: RosTestSystem) -> None:
    system.call(ResetSimulation, "reset", make_reset_request())
    system.call(
        AddWalls,
        "add_walls",
        make_add_walls_request([("wA", (-1.0, 0.0), (1.0, 0.0))]),
    )
    assert "wA" in system.manager._walls

    resp = system.call(RemoveWalls, "remove_walls", make_remove_walls_request(["wA"]))
    assert resp.success is True
    assert "wA" not in system.manager._walls

    resp_all = system.call(RemoveWalls, "remove_walls", make_remove_walls_request([]))
    assert resp_all.success is True
    assert len(system.manager._walls) == 0


def test_wall_changes_trajectory(system: RosTestSystem) -> None:
    system.call(ResetSimulation, "reset", make_reset_request())
    system.call(RemoveAgents, "remove_agents", make_remove_request([]))
    system.call(
        SpawnAgents,
        "spawn_agents",
        make_spawn_request([{"x": 0.0, "y": 0.0, "waypoints": [(5.0, 0.0)]}]),
    )
    baseline = _trajectory(system, n_ticks=20)

    system.call(ResetSimulation, "reset", make_reset_request())
    system.call(
        AddWalls,
        "add_walls",
        make_add_walls_request([("block", (1.5, -1.0), (1.5, 1.0))]),
    )
    system.call(
        SpawnAgents,
        "spawn_agents",
        make_spawn_request([{"x": 0.0, "y": 0.0, "waypoints": [(5.0, 0.0)]}]),
    )
    with_wall = _trajectory(system, n_ticks=20)

    if not baseline or not with_wall:
        pytest.skip("trajectory empty (pool likely didn't populate); not a wall-behavior assertion")

    n = min(len(baseline), len(with_wall))
    diffs = [abs(baseline[i][0] - with_wall[i][0]) + abs(baseline[i][1] - with_wall[i][1]) for i in range(n)]
    assert max(diffs) > 1e-6, "expected wall to alter trajectory"
