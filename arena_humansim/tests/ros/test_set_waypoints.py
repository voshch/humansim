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


def _make_waypoints(points: list[tuple[float, float]], mode: int = 0):
    from arena_humansim_msgs.msg import Waypoint, Waypoints
    from geometry_msgs.msg import Pose2D as Pose2DMsg

    wps = Waypoints()
    wps.mode = mode
    for x, y in points:
        wp = Waypoint()
        wp.pose = Pose2DMsg(x=float(x), y=float(y), theta=0.0)
        wp.radius = 0.0
        wps.points.append(wp)
    return wps


def test_set_waypoints_by_agent_id(system: RosTestSystem) -> None:
    from arena_humansim_msgs.srv import SetWaypoints

    system.call(RemoveAgents, "remove_agents", make_remove_request([-1]))
    specs = [{"x": 0.0, "y": 0.0, "waypoints": [(5.0, 0.0)]}]
    resp = system.call(SpawnAgents, "spawn_agents", make_spawn_request(specs))
    aid = resp.spawned_ids[0]
    idx = system.manager._pool._id_to_idx[aid]

    req = SetWaypoints.Request()
    req.agent_id = aid
    req.name = ""
    req.waypoints = _make_waypoints([(2.0, 3.0), (4.0, 1.0)])
    sw_resp = system.call(SetWaypoints, "set_waypoints", req)
    assert sw_resp.success is True

    pool = system.manager._pool
    assert bool(pool.has_goal[idx]) is True
    assert float(pool.goal_pos[idx, 0]) == pytest.approx(2.0)
    assert float(pool.goal_pos[idx, 1]) == pytest.approx(3.0)

    mv = system.manager._agents[aid].movement
    assert [(p.x, p.y) for p in mv.waypoints] == [(2.0, 3.0), (4.0, 1.0)]
    assert mv.index == 0


def test_set_waypoints_empty_clears_goals(system: RosTestSystem) -> None:
    from arena_humansim_msgs.srv import SetWaypoints

    system.call(RemoveAgents, "remove_agents", make_remove_request([-1]))
    specs = [{"x": 0.0, "y": 0.0, "waypoints": [(5.0, 0.0)]}]
    resp = system.call(SpawnAgents, "spawn_agents", make_spawn_request(specs))
    aid = resp.spawned_ids[0]
    idx = system.manager._pool._id_to_idx[aid]

    system.manager._pool.has_goal[idx] = True

    req = SetWaypoints.Request()
    req.agent_id = aid
    req.name = ""
    req.waypoints = _make_waypoints([])
    sw_resp = system.call(SetWaypoints, "set_waypoints", req)
    assert sw_resp.success is True
    assert bool(system.manager._pool.has_goal[idx]) is False
    assert aid not in system.manager._high_level_cmds


def test_set_waypoints_by_name_for_robot(system: RosTestSystem) -> None:
    from arena_humansim_msgs.msg import AgentState as AgentStateMsg
    from arena_humansim_msgs.msg import Waypoints
    from arena_humansim_msgs.srv import SetWaypoints
    from arena_humansim_msgs.srv import SpawnAgents as SpawnAgentsSrv
    from geometry_msgs.msg import Pose2D as Pose2DMsg
    from geometry_msgs.msg import Vector3

    system.call(RemoveAgents, "remove_agents", make_remove_request([-1]))

    msg = AgentStateMsg()
    msg.agent_id = 0
    msg.pose = Pose2DMsg(x=0.0, y=0.0, theta=0.0)
    msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
    msg.desired_velocity = 1.0
    msg.radius = 0.3
    msg.agent_type = "robot_named"
    msg.kind = AgentStateMsg.KIND_ROBOT
    msg.policy = ""
    msg.waypoints = Waypoints()

    spawn_req = SpawnAgentsSrv.Request()
    spawn_req.agents.append(msg)
    spawn_resp = system.call(SpawnAgentsSrv, "spawn_agents", spawn_req)
    aid = spawn_resp.spawned_ids[0]
    idx = system.manager._pool._id_to_idx[aid]

    req = SetWaypoints.Request()
    req.agent_id = 0
    req.name = "robot_named"
    req.waypoints = _make_waypoints([(7.0, -1.0)])
    sw_resp = system.call(SetWaypoints, "set_waypoints", req)
    assert sw_resp.success is True
    assert bool(system.manager._pool.has_goal[idx]) is True
    assert float(system.manager._pool.goal_pos[idx, 0]) == pytest.approx(7.0)


def test_set_waypoints_unknown_returns_false(system: RosTestSystem) -> None:
    from arena_humansim_msgs.srv import SetWaypoints

    req = SetWaypoints.Request()
    req.agent_id = 99999
    req.name = ""
    req.waypoints = _make_waypoints([(1.0, 1.0)])
    resp = system.call(SetWaypoints, "set_waypoints", req)
    assert resp.success is False
