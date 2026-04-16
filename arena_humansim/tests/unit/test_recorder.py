from __future__ import annotations

from pathlib import Path

import pytest

rclpy = pytest.importorskip("rclpy")
rosbag2_py = pytest.importorskip("rosbag2_py")

from arena_humansim.core.recorder import BagRecorder, default_record_dir
from arena_humansim_msgs.msg import AgentStates as AgentStatesMsg
from arena_humansim_msgs.msg import WorldGeometry as WorldGeometryMsg
from rclpy.node import Node
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosgraph_msgs.msg import Clock


@pytest.fixture
def node(rclpy_context):
    if rclpy_context is None:
        pytest.skip("rclpy unavailable")
    n = Node("recorder_unit_test")
    try:
        yield n
    finally:
        n.destroy_node()


def test_default_record_dir_under_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = default_record_dir()
    assert d.parent == tmp_path / "recordings"


def test_recorder_writes_messages(node: Node, tmp_path: Path) -> None:
    rec = BagRecorder(node, tmp_path / "run")

    pub_agents = node.create_publisher(AgentStatesMsg, "/agent_states", 10)
    pub_geom = node.create_publisher(
        WorldGeometryMsg,
        "/world_geometry",
        rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL),
    )
    pub_clock = node.create_publisher(Clock, "/clock", 10)

    for _ in range(3):
        pub_agents.publish(AgentStatesMsg())
        pub_geom.publish(WorldGeometryMsg())
        pub_clock.publish(Clock())
        rclpy.spin_once(node, timeout_sec=0.1)

    rec.close()

    bag_dir = tmp_path / "run" / "bag"
    assert bag_dir.exists()

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topics_seen: set[str] = set()
    while reader.has_next():
        topic, _, _ = reader.read_next()
        topics_seen.add(topic)

    assert "/agent_states" in topics_seen
    assert "/world_geometry" in topics_seen
    assert "/clock" in topics_seen


def test_recorder_close_idempotent(node: Node, tmp_path: Path) -> None:
    rec = BagRecorder(node, tmp_path / "run")
    rec.close()
    rec.close()
