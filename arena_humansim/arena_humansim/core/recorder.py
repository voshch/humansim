from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from arena_humansim_msgs.msg import AgentStates as AgentStatesMsg
from arena_humansim_msgs.msg import WorldGeometry as WorldGeometryMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.serialization import serialize_message
from rosbag2_py import ConverterOptions, SequentialWriter, StorageOptions, TopicMetadata
from rosgraph_msgs.msg import Clock


def default_record_dir() -> Path:
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return Path(os.getcwd()) / "recordings" / ts


class BagRecorder:
    def __init__(self, node: Node, record_dir: Path) -> None:
        self._node = node
        self._record_dir = record_dir
        self._bag_dir = record_dir / "bag"
        self._record_dir.mkdir(parents=True, exist_ok=True)

        self._writer = SequentialWriter()
        self._writer.open(
            StorageOptions(uri=str(self._bag_dir), storage_id="mcap"),
            ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
        )

        ns = node.get_namespace().rstrip("/")
        topics = [
            ("agent_states", "arena_humansim_msgs/msg/AgentStates", AgentStatesMsg, QoSProfile(depth=10)),
            ("world_geometry", "arena_humansim_msgs/msg/WorldGeometry", WorldGeometryMsg, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)),
            ("/clock", "rosgraph_msgs/msg/Clock", Clock, QoSProfile(depth=10)),
        ]
        self._subs = []
        for i, (topic_name, type_name, msg_type, qos) in enumerate(topics):
            resolved = topic_name if topic_name.startswith("/") else f"{ns}/{topic_name}"
            sub = node.create_subscription(msg_type, topic_name, lambda msg, t=resolved: self._write(t, msg), qos)
            self._writer.create_topic(TopicMetadata(id=i, name=resolved, type=type_name, serialization_format="cdr"))
            self._subs.append(sub)

        self._closed = False
        node.get_logger().info(f"BagRecorder writing to {self._bag_dir}")

    def _write(self, topic_name: str, msg) -> None:
        if self._closed:
            return
        self._writer.write(topic_name, serialize_message(msg), self._node.get_clock().now().nanoseconds)

    @property
    def record_dir(self) -> Path:
        return self._record_dir

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for sub in self._subs:
            try:
                self._node.destroy_subscription(sub)
            except Exception:
                pass
        del self._writer
