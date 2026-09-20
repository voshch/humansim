from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("rosbag2_py")
pytest.importorskip("rclpy")

from arena_humansim.utils.evaluation import bag_cache
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import AgentStates as AgentStatesMsg
from geometry_msgs.msg import Pose2D as Pose2DMsg
from rclpy.serialization import serialize_message
from rosbag2_py import ConverterOptions, SequentialWriter, StorageOptions, TopicMetadata


def _record(trial: Path, x: float, mtime: float) -> None:
    bag = trial / "bag"
    shutil.rmtree(bag, ignore_errors=True)
    trial.mkdir(parents=True, exist_ok=True)
    writer = SequentialWriter()
    writer.open(StorageOptions(uri=str(bag), storage_id="mcap"), ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    writer.create_topic(TopicMetadata(id=0, name="/agent_states", type="arena_humansim_msgs/msg/AgentStates", serialization_format="cdr"))
    agent = AgentStateMsg()
    agent.agent_id = 1
    agent.pose = Pose2DMsg(x=x, y=0.0, theta=0.0)
    msg = AgentStatesMsg()
    msg.agents.append(agent)
    writer.write("/agent_states", serialize_message(msg), 0)
    del writer
    os.utime(bag / "metadata.yaml", (mtime, mtime))


def test_rerecorded_trial_invalidates_multi_cache(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    trial = sweep / "scen__sfm__1"
    _record(trial, x=1.0, mtime=1_000.0)

    first = bag_cache.load_multi([sweep])
    assert first["sweep"]["scen__sfm__1"]["x"].iloc[0] == 1.0

    _record(trial, x=2.0, mtime=(sweep / "_sweep_extracted.pkl").stat().st_mtime + 100.0)

    second = bag_cache.load_multi([sweep])
    assert second["sweep"]["scen__sfm__1"]["x"].iloc[0] == 2.0
