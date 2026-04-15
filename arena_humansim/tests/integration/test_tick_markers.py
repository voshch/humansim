from __future__ import annotations

from arena_humansim.utils.scenario import ScenarioConfig
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import Waypoint as WaypointMsg
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg
from arena_humansim_msgs.srv import SpawnAgents
from geometry_msgs.msg import Pose2D as RosPose2D
from geometry_msgs.msg import Vector3
from rclpy.parameter import Parameter

from ._helpers import build_manager


def test_tick_markers_full_detail_ticks_cleanly(rclpy_context: object, minimal_scenario: ScenarioConfig) -> None:  # noqa: ARG001
    extra = [Parameter("publish_markers", Parameter.Type.INTEGER, 2)]
    mgr = build_manager(minimal_scenario, node_name="test_tick_markers_full", extra_params=extra)
    try:
        req = SpawnAgents.Request()
        for i in range(2):
            msg = AgentStateMsg()
            msg.agent_id = 0
            msg.pose = RosPose2D(x=float(i) * 2.0, y=0.0, theta=0.0)
            msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
            msg.desired_velocity = 1.3
            msg.radius = 0.0
            msg.agent_type = "adult"
            wp = WaypointMsg()
            wp.pose = RosPose2D(x=float(i) * 2.0 + 10.0, y=0.0, theta=0.0)
            msg.waypoints = WaypointsMsg(points=[wp], mode=WaypointsMsg.MODE_ONCE)
            req.agents.append(msg)
        resp = SpawnAgents.Response()
        mgr._spawn_agents_callback(req, resp)
        assert len(resp.spawned_ids) == 2

        for _ in range(20):
            mgr.tick()

        assert mgr._tick_count == 20
        assert mgr._publish_markers == 2
        assert mgr._marker_pub is not None
    finally:
        mgr.destroy_node()
