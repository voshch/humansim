from __future__ import annotations

import math

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.formation.anchor import CentroidAnchor, PoseAnchor
from arena_humansim.core.formation.f_formation import FFormation
from arena_humansim.utils.types import Pose2D


def test_f_formation_members_on_circle_of_expected_radius() -> None:
    anchor = PoseAnchor(fixed=Pose2D(x=0.0, y=0.0))
    f = FFormation(anchor=anchor, agent_lookup=lambda _aid: None, base_radius=0.8, radius_per_member=0.1)
    for aid in (1, 2, 3):
        f.on_join(aid)
    expected_radius = 0.8 + 0.1 * (3 - 1)
    targets = f.tick(dt=0.01)
    for _aid, pose in targets.items():
        r = math.hypot(pose.x, pose.y)
        assert r == pytest.approx(expected_radius)


def test_f_formation_equiangular_spacing() -> None:
    anchor = PoseAnchor(fixed=Pose2D())
    f = FFormation(anchor=anchor, agent_lookup=lambda _aid: None, base_radius=1.0, radius_per_member=0.0)
    for aid in (10, 20, 30, 40):
        f.on_join(aid)
    targets = f.tick(dt=0.01)
    angles = sorted(math.atan2(p.y, p.x) % (2 * math.pi) for p in targets.values())
    # Gaps between consecutive angles should all equal 2pi/4
    gaps = [angles[(i + 1) % 4] - angles[i] for i in range(3)]
    for g in gaps:
        assert g == pytest.approx(math.pi / 2, abs=1e-6)


def test_f_formation_members_face_inward() -> None:
    anchor = PoseAnchor(fixed=Pose2D(x=0.0, y=0.0))
    f = FFormation(anchor=anchor, agent_lookup=lambda _aid: None, base_radius=1.0, radius_per_member=0.0)
    f.on_join(1)
    f.on_join(2)
    targets = f.tick(dt=0.01)
    for _aid, pose in targets.items():
        # Theta points inward (back toward origin)
        expected = math.atan2(-pose.y, -pose.x)
        # Allow for equivalent angles modulo 2pi
        diff = (pose.theta - expected + math.pi) % (2 * math.pi) - math.pi
        assert abs(diff) < 1e-6


def test_f_formation_shrinks_on_leave() -> None:
    anchor = PoseAnchor(fixed=Pose2D())
    f = FFormation(anchor=anchor, agent_lookup=lambda _aid: None, base_radius=0.5, radius_per_member=0.3)
    for aid in (1, 2, 3):
        f.on_join(aid)
    r_before = math.hypot(f.tick(0.01)[1].x, f.tick(0.01)[1].y)
    f.on_leave(3)
    r_after_pose = next(iter(f.tick(0.01).values()))
    r_after = math.hypot(r_after_pose.x, r_after_pose.y)
    assert r_after < r_before


def test_f_formation_follows_centroid() -> None:
    poses = {1: Pose2D(x=0.0, y=0.0), 2: Pose2D(x=4.0, y=0.0)}
    anchor = CentroidAnchor(pose_lookup=poses.get, members_fn=lambda: [1, 2])
    f = FFormation(anchor=anchor, agent_lookup=lambda _aid: None, base_radius=1.0, radius_per_member=0.0)
    f.on_join(1)
    f.on_join(2)
    targets = f.tick(dt=0.01)
    # All members should be on a circle around (2.0, 0.0)
    for _aid, p in targets.items():
        d = math.hypot(p.x - 2.0, p.y)
        assert d == pytest.approx(1.0)
