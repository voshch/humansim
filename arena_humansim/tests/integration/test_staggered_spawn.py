from __future__ import annotations

from collections import deque
from collections.abc import Callable

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import ScenarioConfig
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg
from geometry_msgs.msg import Pose2D as RosPose2D
from geometry_msgs.msg import Vector3


def _pending_msg(agent_id: int, x: float) -> AgentStateMsg:
    msg = AgentStateMsg()
    msg.agent_id = agent_id
    msg.pose = RosPose2D(x=x, y=0.0, theta=0.0)
    msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
    msg.desired_velocity = 1.3
    msg.radius = 0.35
    msg.agent_type = "adult"
    msg.waypoints = WaypointsMsg(points=[], mode=WaypointsMsg.MODE_ONCE)
    return msg


def test_pending_scenario_spawns_fire_on_schedule(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = manager_factory(minimal_scenario, node_name="test_staggered_spawn")
    mgr._pending_scenario_spawns = deque([
        (0, _pending_msg(101, 0.0)),
        (5, _pending_msg(102, 1.0)),
        (10, _pending_msg(103, 2.0)),
    ])

    mgr.tick()
    assert 101 in mgr._agents
    assert 102 not in mgr._agents
    assert 103 not in mgr._agents

    for _ in range(5):
        mgr.tick()
    assert 102 in mgr._agents
    assert 103 not in mgr._agents

    for _ in range(5):
        mgr.tick()
    assert 103 in mgr._agents
    assert len(mgr._pending_scenario_spawns) == 0


def test_pending_scenario_spawns_empty_is_noop(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = manager_factory(minimal_scenario, node_name="test_staggered_spawn_noop")
    assert len(mgr._pending_scenario_spawns) == 0
    for _ in range(5):
        mgr.tick()
    assert mgr._agents == {}
