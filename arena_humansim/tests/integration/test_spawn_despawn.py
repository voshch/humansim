from __future__ import annotations

from collections.abc import Callable

from arena_humansim.manager.agent_manager import AgentManager
from arena_humansim.utils.scenario import ScenarioConfig
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import Waypoint as WaypointMsg
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg
from arena_humansim_msgs.srv import RemoveAgents, SpawnAgents
from geometry_msgs.msg import Pose2D as RosPose2D
from geometry_msgs.msg import Vector3


def _spawn(mgr: AgentManager, n: int, base_x: float = 0.0) -> list[int]:
    req = SpawnAgents.Request()
    for i in range(n):
        msg = AgentStateMsg()
        msg.agent_id = 0
        msg.pose = RosPose2D(x=base_x + float(i), y=0.0, theta=0.0)
        msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
        msg.desired_velocity = 1.3
        msg.radius = 0.0
        msg.agent_type = "adult"
        wp = WaypointMsg()
        wp.pose = RosPose2D(x=base_x + float(i) + 10.0, y=0.0, theta=0.0)
        msg.waypoints = WaypointsMsg(points=[wp], mode=WaypointsMsg.MODE_ONCE)
        req.agents.append(msg)
    resp = SpawnAgents.Response()
    mgr._spawn_agents_callback(req, resp)
    return list(resp.spawned_ids)


def _remove(mgr: AgentManager, ids: list[int]) -> None:
    req = RemoveAgents.Request()
    req.agent_ids = list(ids)
    resp = RemoveAgents.Response()
    mgr._remove_agents_callback(req, resp)


def _assert_invariants(mgr: AgentManager) -> None:
    pool = mgr._pool
    assert pool.n == len(pool._id_to_idx)
    assert pool.n == len(mgr._agents)
    assert pool.n == len(mgr._pool_agent_ids)
    for aid, idx in pool._id_to_idx.items():
        assert 0 <= idx < pool.n
        assert int(pool.agent_ids[idx]) == aid
        assert aid in mgr._agents
    seen = sorted(pool._id_to_idx.values())
    assert seen == list(range(pool.n))


def test_spawn_despawn_cycles_preserve_invariants(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = manager_factory(minimal_scenario, node_name="test_spawn_despawn")
    _assert_invariants(mgr)

    for cycle in range(10):
        spawned = _spawn(mgr, n=10, base_x=float(cycle) * 100.0)
        assert len(spawned) == 10
        assert mgr._pool.n == 10
        _assert_invariants(mgr)

        first_half = spawned[:5]
        _remove(mgr, first_half)
        assert mgr._pool.n == 5
        _assert_invariants(mgr)

        more = _spawn(mgr, n=5, base_x=float(cycle) * 100.0 + 50.0)
        assert mgr._pool.n == 10
        assert len(more) == 5
        _assert_invariants(mgr)

        _remove(mgr, [])
        assert mgr._pool.n == 0
        _assert_invariants(mgr)
