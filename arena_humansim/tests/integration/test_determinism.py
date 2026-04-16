from __future__ import annotations

from collections.abc import Callable

import numpy as np
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import ScenarioConfig
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import Waypoint as WaypointMsg
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg
from arena_humansim_msgs.srv import SpawnAgents
from geometry_msgs.msg import Pose2D as RosPose2D
from geometry_msgs.msg import Vector3


def _scripted_spawn(mgr: AgentManager, specs: list[tuple[float, float, float, float]]) -> list[int]:
    req = SpawnAgents.Request()
    for x, y, gx, gy in specs:
        msg = AgentStateMsg()
        msg.agent_id = 0
        msg.pose = RosPose2D(x=x, y=y, theta=0.0)
        msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
        msg.desired_velocity = 1.3
        msg.radius = 0.0
        msg.agent_type = "adult"
        wp = WaypointMsg()
        wp.pose = RosPose2D(x=gx, y=gy, theta=0.0)
        msg.waypoints = WaypointsMsg(points=[wp], mode=WaypointsMsg.MODE_ONCE)
        req.agents.append(msg)
    resp = SpawnAgents.Response()
    mgr._spawn_agents_callback(req, resp)
    return list(resp.spawned_ids)


def _snapshot(mgr: AgentManager) -> np.ndarray:
    n = mgr._pool.n
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)
    out = np.empty((n, 3), dtype=np.float64)
    out[:, 0] = mgr._pool.pos[:n, 0]
    out[:, 1] = mgr._pool.pos[:n, 1]
    out[:, 2] = mgr._pool.theta[:n]
    return out


def test_two_managers_same_seed_produce_identical_trajectories(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    specs = [
        (0.0, 0.0, 5.0, 0.0),
        (0.0, 2.0, 5.0, 2.0),
        (0.0, -2.0, 5.0, -2.0),
    ]
    n_ticks = 50

    mgr_a = manager_factory(minimal_scenario, node_name="test_determinism_a")
    ids_a = _scripted_spawn(mgr_a, specs)
    traj_a: list[np.ndarray] = []
    for _ in range(n_ticks):
        mgr_a.tick()
        traj_a.append(_snapshot(mgr_a))

    mgr_b = manager_factory(minimal_scenario, node_name="test_determinism_b")
    ids_b = _scripted_spawn(mgr_b, specs)
    traj_b: list[np.ndarray] = []
    for _ in range(n_ticks):
        mgr_b.tick()
        traj_b.append(_snapshot(mgr_b))

    assert ids_a == ids_b
    assert len(traj_a) == len(traj_b) == n_ticks
    for i, (a, b) in enumerate(zip(traj_a, traj_b, strict=True)):
        assert a.shape == b.shape, f"tick {i}: shape {a.shape} vs {b.shape}"
        assert np.array_equal(a, b), f"tick {i}: trajectory diverged"
