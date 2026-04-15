from __future__ import annotations

import time

import pytest
from arena_humansim_msgs.msg import ObstacleConfig, WorldGeometry
from arena_humansim_msgs.srv import AddObstacles
from geometry_msgs.msg import Pose2D as Pose2DMsg
from rclpy.qos import DurabilityPolicy, QoSProfile

from tests.ros._helpers import (
    AddWalls,
    ResetSimulation,
    RosTestSystem,
    make_add_walls_request,
    make_reset_request,
)

pytestmark = pytest.mark.ros


def _latched_qos() -> QoSProfile:
    return QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def _wait_for_world_geometry(system: RosTestSystem, received: list[WorldGeometry], start_len: int, timeout: float = 3.0) -> WorldGeometry:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        system.executor.spin_once(timeout_sec=0.05)
        if len(received) > start_len:
            return received[-1]
    raise TimeoutError("no WorldGeometry message received")


@pytest.fixture(scope="module")
def system_with_sub(ros_system: RosTestSystem):
    received: list[WorldGeometry] = []
    sub = ros_system.client_node.create_subscription(
        WorldGeometry,
        "world_geometry",
        lambda m: received.append(m),
        _latched_qos(),
    )
    try:
        yield ros_system, received
    finally:
        try:
            ros_system.client_node.destroy_subscription(sub)
        except Exception:
            pass


def test_add_walls_triggers_republish(system_with_sub) -> None:
    system, received = system_with_sub
    system.call(ResetSimulation, "reset", make_reset_request())
    _wait_for_world_geometry(system, received, start_len=len(received) - 1 if received else -1)
    start = len(received)

    system.call(
        AddWalls,
        "add_walls",
        make_add_walls_request([("w_gtest", (-1.0, 0.0), (1.0, 0.0))]),
    )
    msg = _wait_for_world_geometry(system, received, start_len=start)

    assert "w_gtest" in list(msg.wall_names)


def test_add_obstacles_triggers_republish(system_with_sub) -> None:
    system, received = system_with_sub
    system.call(ResetSimulation, "reset", make_reset_request())
    start = len(received)

    obs = ObstacleConfig()
    obs.name = "o_gtest"
    obs.pose = Pose2DMsg(x=2.0, y=3.0, theta=0.0)
    obs.bb_x_min, obs.bb_x_max = -0.5, 0.5
    obs.bb_y_min, obs.bb_y_max = -0.5, 0.5
    obs.bb_z_min, obs.bb_z_max = 0.0, 1.0
    obs.obstacle_type = "test"

    req = AddObstacles.Request()
    req.obstacles.append(obs)
    system.call(AddObstacles, "add_obstacles", req)

    msg = _wait_for_world_geometry(system, received, start_len=start)
    assert any(o.name == "o_gtest" for o in msg.obstacles)
