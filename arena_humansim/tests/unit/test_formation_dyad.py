from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.formation.anchor import CentroidAnchor, PoseAnchor
from arena_humansim.core.formation.dyad import DyadFormation
from arena_humansim.utils.types import Pose2D


@dataclass
class _FakeState:
    pose: Pose2D = field(default_factory=Pose2D)


@dataclass
class _FakeAgent:
    state: _FakeState


SEPARATION = 2.0


def _lookup(poses: dict[int, Pose2D]) -> Callable[[int], _FakeAgent | None]:
    def inner(aid: int) -> _FakeAgent | None:
        p = poses.get(aid)
        return _FakeAgent(state=_FakeState(pose=p)) if p is not None else None

    return inner


def test_dyad_pair_separation_matches_configured() -> None:
    poses = {1: Pose2D(x=-0.5, y=0.0), 2: Pose2D(x=0.5, y=0.0)}
    f = DyadFormation(
        anchor=CentroidAnchor(pose_lookup=lambda aid: poses.get(aid), members_fn=lambda: [1, 2]),
        agent_lookup=_lookup(poses),
        separation=SEPARATION,
    )
    f.on_join(1)
    f.on_join(2)
    targets = f.tick(dt=0.01)
    d = math.hypot(targets[1].x - targets[2].x, targets[1].y - targets[2].y)
    assert d == pytest.approx(SEPARATION)


def test_dyad_members_face_each_other() -> None:
    poses = {1: Pose2D(x=-1.0, y=0.0), 2: Pose2D(x=1.0, y=0.0)}
    f = DyadFormation(
        anchor=CentroidAnchor(pose_lookup=lambda aid: poses.get(aid), members_fn=lambda: [1, 2]),
        agent_lookup=_lookup(poses),
        separation=SEPARATION,
    )
    f.on_join(1)
    f.on_join(2)
    targets = f.tick(dt=0.01)
    # The two yaws should differ by ~pi (opposite directions along the axis)
    delta = abs((targets[1].theta - targets[2].theta + math.pi) % (2 * math.pi) - math.pi)
    assert delta == pytest.approx(math.pi, abs=1e-6) or delta == pytest.approx(0.0, abs=1e-6)


def test_dyad_caps_at_two_members() -> None:
    poses = {1: Pose2D(), 2: Pose2D(x=1.0), 3: Pose2D(x=2.0)}
    f = DyadFormation(
        anchor=PoseAnchor(fixed=Pose2D()),
        agent_lookup=_lookup(poses),
        separation=1.0,
    )
    f.on_join(1)
    f.on_join(2)
    f.on_join(3)
    targets = f.tick(dt=0.01)
    assert set(targets.keys()) == {1, 2}
