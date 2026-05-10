from __future__ import annotations

import math

import pytest

from tests.ros._helpers import (
    AddWalls,
    RemoveAgents,
    ResetSimulation,
    RosTestSystem,
    SpawnAgents,
    make_add_walls_request,
    make_remove_request,
    make_reset_request,
    make_spawn_request,
)

pytestmark = pytest.mark.ros


@pytest.fixture(scope="module")
def system(ros_system: RosTestSystem) -> RosTestSystem:
    ros_system.call(ResetSimulation, "reset", make_reset_request())
    return ros_system


def _terminal_distance(pool, idx: int) -> float:
    dx = float(pool.terminal_pos[idx, 0] - pool.pos[idx, 0])
    dy = float(pool.terminal_pos[idx, 1] - pool.pos[idx, 1])
    return math.hypot(dx, dy)


def test_no_latch_while_subgoal_near_and_terminal_far(system: RosTestSystem) -> None:
    system.call(ResetSimulation, "reset", make_reset_request())
    system.call(RemoveAgents, "remove_agents", make_remove_request([]))
    system.call(
        AddWalls,
        "add_walls",
        make_add_walls_request(
            [
                ("obstacle", (1.5, -1.0), (1.5, 1.2)),
            ]
        ),
    )
    resp = system.call(
        SpawnAgents,
        "spawn_agents",
        make_spawn_request([{"x": 0.0, "y": 0.0, "waypoints": [(6.0, 0.0)]}]),
    )
    aid = resp.spawned_ids[0]
    pool = system.manager._pool
    idx = pool._id_to_idx[aid]
    r_exit = system.manager._arrival_r_exit

    latched_while_far = False
    for _ in range(60):
        system.tick_manager(1)
        if not bool(pool.has_terminal[idx]):
            continue
        d_term = _terminal_distance(pool, idx)
        if d_term > r_exit and bool(pool.latched[idx]):
            latched_while_far = True
            break
        if d_term < system.manager._arrival_r_enter:
            break

    assert latched_while_far is False


def test_latch_engages_once_terminal_reached(system: RosTestSystem) -> None:
    system.call(ResetSimulation, "reset", make_reset_request())
    system.call(RemoveAgents, "remove_agents", make_remove_request([]))
    resp = system.call(
        SpawnAgents,
        "spawn_agents",
        make_spawn_request([{"x": 0.0, "y": 0.0, "waypoints": [(2.0, 0.0)]}]),
    )
    aid = resp.spawned_ids[0]
    pool = system.manager._pool
    idx = pool._id_to_idx[aid]
    r_enter = system.manager._arrival_r_enter

    latched = False
    for _ in range(200):
        system.tick_manager(1)
        if bool(pool.latched[idx]):
            latched = True
            break

    assert latched, "agent never latched after reaching terminal"
    assert _terminal_distance(pool, idx) < r_enter
