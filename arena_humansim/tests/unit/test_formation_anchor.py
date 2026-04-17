from __future__ import annotations

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.formation.anchor import (
    AgentAnchor,
    CentroidAnchor,
    ObjectAnchor,
    PoseAnchor,
)
from arena_humansim.core.world_knowledge import WorldKnowledge, WorldObject
from arena_humansim.utils.types import Pose2D


def test_pose_anchor_returns_fixed() -> None:
    p = Pose2D(x=1.0, y=2.0, theta=0.5)
    a = PoseAnchor(fixed=p)
    assert a.pose() == p


def test_object_anchor_reads_world_knowledge() -> None:
    wk = WorldKnowledge()
    wk.add_object(WorldObject(object_id="atm", type="atm", pose=Pose2D(x=5.0, y=3.0, theta=1.0)))
    a = ObjectAnchor(world_knowledge=wk, object_id="atm")
    assert a.pose() == Pose2D(x=5.0, y=3.0, theta=1.0)


def test_object_anchor_missing_object_returns_default_pose() -> None:
    wk = WorldKnowledge()
    a = ObjectAnchor(world_knowledge=wk, object_id="missing")
    assert a.pose() == Pose2D()


def test_agent_anchor_calls_lookup() -> None:
    poses = {1: Pose2D(x=7.0, y=8.0, theta=0.0)}
    a = AgentAnchor(pose_lookup=poses.get, agent_id=1)
    assert a.pose().x == 7.0
    assert a.pose().y == 8.0


def test_centroid_anchor_averages_member_poses() -> None:
    poses = {
        1: Pose2D(x=0.0, y=0.0),
        2: Pose2D(x=4.0, y=0.0),
        3: Pose2D(x=2.0, y=6.0),
    }
    a = CentroidAnchor(pose_lookup=poses.get, members_fn=lambda: [1, 2, 3])
    p = a.pose()
    assert p.x == pytest.approx(2.0)
    assert p.y == pytest.approx(2.0)


def test_centroid_anchor_updates_as_members_change() -> None:
    poses = {1: Pose2D(x=0.0, y=0.0), 2: Pose2D(x=10.0, y=0.0)}
    members: list[int] = [1]
    a = CentroidAnchor(pose_lookup=poses.get, members_fn=lambda: list(members))
    assert a.pose().x == pytest.approx(0.0)
    members.append(2)
    assert a.pose().x == pytest.approx(5.0)


def test_centroid_anchor_empty_returns_default() -> None:
    a = CentroidAnchor(pose_lookup=lambda _aid: None, members_fn=lambda: [])
    assert a.pose() == Pose2D()
