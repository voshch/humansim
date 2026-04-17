from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.formation.anchor import PoseAnchor
from arena_humansim.core.formation.cluster import ClusterFormation
from arena_humansim.utils.types import Pose2D


@dataclass
class _FakeState:
    pose: Pose2D = field(default_factory=Pose2D)


@dataclass
class _FakeAgent:
    state: _FakeState


def _lookup_for(poses: dict[int, Pose2D]) -> Callable[[int], _FakeAgent | None]:
    def inner(aid: int) -> _FakeAgent | None:
        p = poses.get(aid)
        return _FakeAgent(state=_FakeState(pose=p)) if p is not None else None
    return inner


def test_cluster_assigns_explicit_slots() -> None:
    slots = [Pose2D(x=0.0, y=0.0), Pose2D(x=1.0, y=0.0), Pose2D(x=0.0, y=1.0)]
    f = ClusterFormation(
        anchor=PoseAnchor(fixed=Pose2D()),
        agent_lookup=_lookup_for({1: Pose2D(x=0.9, y=0.1)}),
        slot_poses=slots,
    )
    f.on_join(1)
    targets = f.tick(dt=0.01)
    # Agent 1 is nearest to slot [1.0, 0.0]
    assert targets[1] == Pose2D(x=1.0, y=0.0)


def test_cluster_reassigns_freed_slot_on_leave() -> None:
    slots = [Pose2D(x=0.0, y=0.0), Pose2D(x=1.0, y=0.0)]
    poses = {1: Pose2D(x=0.0, y=0.0), 2: Pose2D(x=2.0, y=0.0)}
    f = ClusterFormation(
        anchor=PoseAnchor(fixed=Pose2D()),
        agent_lookup=_lookup_for(poses),
        slot_poses=slots,
    )
    f.on_join(1)
    f.on_join(2)
    f.on_leave(1)
    poses[3] = Pose2D(x=0.0, y=0.1)
    f.on_join(3)
    targets = f.tick(dt=0.01)
    assert targets[3] == Pose2D(x=0.0, y=0.0)
    # Original agent 2's slot unchanged
    assert targets[2] == Pose2D(x=1.0, y=0.0)


def test_cluster_generated_slots_capacity_bounded() -> None:
    f = ClusterFormation(
        anchor=PoseAnchor(fixed=Pose2D()),
        agent_lookup=_lookup_for({1: Pose2D(), 2: Pose2D(), 3: Pose2D(), 4: Pose2D()}),
        slot_poses=None,
        capacity=3,
    )
    for aid in (1, 2, 3, 4):
        f.on_join(aid)
    targets = f.tick(dt=0.01)
    # Only 3 agents placed; 4th has no free slot
    assert len(targets) == 3
