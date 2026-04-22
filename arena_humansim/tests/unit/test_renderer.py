from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
rosbag2_py = pytest.importorskip("rosbag2_py")
pytest.importorskip("rclpy")

from arena_humansim.utils.renderer import main as render_main
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import AgentStates as AgentStatesMsg
from arena_humansim_msgs.msg import WorldGeometry as WorldGeometryMsg
from geometry_msgs.msg import Point32
from geometry_msgs.msg import Pose2D as Pose2DMsg
from geometry_msgs.msg import Vector3
from rclpy.serialization import serialize_message
from rosbag2_py import ConverterOptions, SequentialWriter, StorageOptions, TopicMetadata


def _agent(aid: int, x: float, y: float, vx: float = 0.1) -> AgentStateMsg:
    a = AgentStateMsg()
    a.agent_id = aid
    a.pose = Pose2DMsg(x=x, y=y, theta=0.0)
    a.velocity = Vector3(x=vx, y=0.0, z=0.0)
    a.radius = 0.35
    a.policy = "sfm"
    return a


def _build_bag(bag_dir: Path, with_geometry: bool = True, frames: int = 3) -> None:
    writer = SequentialWriter()
    writer.open(
        StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    writer.create_topic(TopicMetadata(id=0, name="/agent_states", type="arena_humansim_msgs/msg/AgentStates", serialization_format="cdr"))
    if with_geometry:
        writer.create_topic(TopicMetadata(id=1, name="/arena_humansim/world_geometry", type="arena_humansim_msgs/msg/WorldGeometry", serialization_format="cdr"))
        geom = WorldGeometryMsg()
        geom.wall_names = ["w1"]
        geom.wall_starts = [Point32(x=-2.0, y=0.0, z=0.0)]
        geom.wall_ends = [Point32(x=2.0, y=0.0, z=0.0)]
        writer.write("/arena_humansim/world_geometry", serialize_message(geom), 0)

    for i in range(frames):
        msg = AgentStatesMsg()
        msg.agents.append(_agent(1, x=float(i), y=1.0))
        if i >= 1:
            msg.agents.append(_agent(2, x=float(i) * -0.5, y=-1.0))
        writer.write("/agent_states", serialize_message(msg), i * 50_000_000)

    del writer


def test_renderer_produces_gif(tmp_path: Path) -> None:
    bag_dir = tmp_path / "bag"
    output = tmp_path / "scenario.gif"
    _build_bag(bag_dir)

    rc = render_main([str(bag_dir), "--output", str(output), "--format", "gif", "--fps", "10"])

    assert rc == 0
    assert output.exists()
    assert output.stat().st_size > 0


def test_renderer_handles_missing_geometry(tmp_path: Path) -> None:
    bag_dir = tmp_path / "bag"
    output = tmp_path / "scenario.gif"
    _build_bag(bag_dir, with_geometry=False)

    rc = render_main([str(bag_dir), "--output", str(output), "--format", "gif", "--fps", "10"])

    assert rc == 0
    assert output.exists()


def test_renderer_handles_spawn_despawn(tmp_path: Path) -> None:
    bag_dir = tmp_path / "bag"
    output = tmp_path / "scenario.gif"
    _build_bag(bag_dir, frames=4)

    rc = render_main([str(bag_dir), "--output", str(output), "--format", "gif", "--fps", "10"])

    assert rc == 0
