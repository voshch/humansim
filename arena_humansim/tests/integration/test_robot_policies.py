from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from arena_humansim.manager.agent_manager import AgentManager
from arena_humansim.utils.scenario import ScenarioConfig
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import Waypoint as WaypointMsg
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg
from arena_humansim_msgs.srv import Feedback, SpawnAgents
from geometry_msgs.msg import Pose2D as RosPose2D
from geometry_msgs.msg import Vector3


def _spawn(
    mgr: AgentManager,
    *,
    x: float,
    y: float,
    gx: float,
    gy: float,
    kind: int = 0,
    policy: str = "",
    radius: float = 0.3,
) -> int:
    req = SpawnAgents.Request()
    msg = AgentStateMsg()
    msg.agent_id = 0
    msg.pose = RosPose2D(x=x, y=y, theta=0.0)
    msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
    msg.desired_velocity = 1.3
    msg.radius = radius
    msg.agent_type = "adult"
    msg.kind = kind
    msg.policy = policy
    msg.policy_params = ""
    wp = WaypointMsg()
    wp.pose = RosPose2D(x=gx, y=gy, theta=0.0)
    wp.radius = radius
    msg.waypoints = WaypointsMsg(points=[wp], mode=WaypointsMsg.MODE_ONCE)
    req.agents.append(msg)
    resp = SpawnAgents.Response()
    mgr._spawn_agents_callback(req, resp)
    return list(resp.spawned_ids)[0]


def _pose_of(mgr: AgentManager, aid: int) -> tuple[float, float]:
    idx = mgr._pool.idx(aid)
    return float(mgr._pool.pos[idx, 0]), float(mgr._pool.pos[idx, 1])


def _vel_of(mgr: AgentManager, aid: int) -> tuple[float, float]:
    idx = mgr._pool.idx(aid)
    return float(mgr._pool.vel[idx, 0]), float(mgr._pool.vel[idx, 1])


def test_robot_straight_policy_reaches_goal_no_collisions(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = manager_factory(minimal_scenario, node_name="test_robot_straight")
    robot_id = _spawn(mgr, x=0.0, y=0.0, gx=10.0, gy=0.0, kind=1, policy="straight", radius=0.3)
    human_ids = []
    for y in [-0.5, 0.5]:
        hid = _spawn(mgr, x=3.0, y=y, gx=-30.0, gy=y, radius=0.3)
        human_ids.append(hid)

    initial_human_ys = {hid: _pose_of(mgr, hid)[1] for hid in human_ids}

    for _ in range(400):
        mgr.tick()
        rx, ry = _pose_of(mgr, robot_id)
        for hid in human_ids:
            hx, hy = _pose_of(mgr, hid)
            d = math.hypot(rx - hx, ry - hy)
            assert d > 0.3, f"robot collided with human {hid}: d={d}"

    rx, ry = _pose_of(mgr, robot_id)
    assert rx > 5.0, f"robot did not progress toward goal: x={rx}"

    for hid in human_ids:
        _, hy = _pose_of(mgr, hid)
        assert abs(hy - initial_human_ys[hid]) > 0.05, f"human {hid} did not detour"


def test_robot_sfm_policy_symmetric_yield(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = manager_factory(minimal_scenario, node_name="test_robot_sfm")
    robot_id = _spawn(mgr, x=0.0, y=0.0, gx=6.0, gy=0.0, kind=1, policy="sfm", radius=0.3)
    human_id = _spawn(mgr, x=3.0, y=0.6, gx=-3.0, gy=0.6, radius=0.3)

    for _ in range(200):
        mgr.tick()
        rx, ry = _pose_of(mgr, robot_id)
        hx, hy = _pose_of(mgr, human_id)
        d = math.hypot(rx - hx, ry - hy)
        assert d > 0.6, f"collision: d={d}"


def test_robot_teleport_only_policy_stays_still_until_feedback(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = manager_factory(minimal_scenario, node_name="test_robot_teleport")
    aid = _spawn(mgr, x=0.0, y=0.0, gx=5.0, gy=0.0, kind=1, policy="", radius=0.3)
    robot_name = next(name for name, rid in mgr._robot_name_to_id.items() if rid == aid)

    for _ in range(10):
        mgr.tick()
        vx, vy = _vel_of(mgr, aid)
        assert abs(vx) < 1e-9 and abs(vy) < 1e-9

    from arena_humansim_msgs.msg import RobotState

    fb_req = Feedback.Request()
    rs = RobotState()
    rs.name = robot_name
    rs.pose = RosPose2D(x=2.5, y=1.0, theta=0.0)
    rs.radius = 0.3
    fb_req.robots.append(rs)

    fb_resp = Feedback.Response()
    mgr._feedback_callback(fb_req, fb_resp)

    px, py = _pose_of(mgr, aid)
    assert px == pytest.approx(2.5)
    assert py == pytest.approx(1.0)

    mgr.tick()
    vx, vy = _vel_of(mgr, aid)
    assert abs(vx) < 1e-9 and abs(vy) < 1e-9


