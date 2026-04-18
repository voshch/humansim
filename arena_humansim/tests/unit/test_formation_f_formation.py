from __future__ import annotations

import math

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.formation.anchor import CentroidAnchor, PoseAnchor
from arena_humansim.core.formation.f_formation import FFormation
from arena_humansim.utils.types import Pose2D

BASE_RADIUS = 0.8
RADIUS_PER_MEMBER = 0.1


def test_f_formation_members_on_circle_of_expected_radius() -> None:
    num_members = 3
    anchor = PoseAnchor(fixed=Pose2D(x=0.0, y=0.0))
    f = FFormation(anchor=anchor, agent_lookup=lambda _aid: None, base_radius=BASE_RADIUS, radius_per_member=RADIUS_PER_MEMBER)
    for aid in range(1, num_members + 1):
        f.on_join(aid)
    expected_radius = BASE_RADIUS + RADIUS_PER_MEMBER * (num_members - 1)
    targets = f.tick(dt=0.01)
    for _aid, pose in targets.items():
        r = math.hypot(pose.x, pose.y)
        assert r == pytest.approx(expected_radius)


def test_f_formation_equiangular_spacing() -> None:
    num_members = 4
    anchor = PoseAnchor(fixed=Pose2D())
    f = FFormation(anchor=anchor, agent_lookup=lambda _aid: None, base_radius=1.0, radius_per_member=0.0)
    for aid in (10, 20, 30, 40):
        f.on_join(aid)
    targets = f.tick(dt=0.01)
    angles = sorted(math.atan2(p.y, p.x) % (2 * math.pi) for p in targets.values())
    gaps = [angles[(i + 1) % num_members] - angles[i] for i in range(num_members - 1)]
    for g in gaps:
        assert g == pytest.approx(2 * math.pi / num_members, abs=1e-6)


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
    base_radius = 0.5
    radius_per_member = 0.3
    anchor = PoseAnchor(fixed=Pose2D())
    f = FFormation(anchor=anchor, agent_lookup=lambda _aid: None, base_radius=base_radius, radius_per_member=radius_per_member)
    for aid in (1, 2, 3):
        f.on_join(aid)
    expected_before = base_radius + radius_per_member * (3 - 1)
    r_before = math.hypot(f.tick(0.01)[1].x, f.tick(0.01)[1].y)
    assert r_before == pytest.approx(expected_before)
    f.on_leave(3)
    r_after_pose = next(iter(f.tick(0.01).values()))
    r_after = math.hypot(r_after_pose.x, r_after_pose.y)
    expected_after = base_radius + radius_per_member * (2 - 1)
    assert r_after == pytest.approx(expected_after)
    assert r_after < r_before


def test_f_formation_follows_centroid() -> None:
    x_a, x_b = 0.0, 4.0
    centroid_x = (x_a + x_b) / 2
    base_radius = 1.0
    poses = {1: Pose2D(x=x_a, y=0.0), 2: Pose2D(x=x_b, y=0.0)}
    anchor = CentroidAnchor(pose_lookup=poses.get, members_fn=lambda: [1, 2])
    f = FFormation(anchor=anchor, agent_lookup=lambda _aid: None, base_radius=base_radius, radius_per_member=0.0)
    f.on_join(1)
    f.on_join(2)
    targets = f.tick(dt=0.01)
    for _aid, p in targets.items():
        d = math.hypot(p.x - centroid_x, p.y)
        assert d == pytest.approx(base_radius)


from dataclasses import dataclass, field


@dataclass
class _FakeState:
    pose: Pose2D = field(default_factory=Pose2D)


@dataclass
class _FakeAgent:
    state: _FakeState = field(default_factory=_FakeState)


def test_f_formation_assigns_nearest_slot_on_join() -> None:
    # Agent 1 is near the -x side, agent 2 near the +x side.
    # Naive join-order would send agent 1 to slot 0 (+x) and agent 2 to slot 1 (-x), crossing paths.
    # Hungarian assignment should send each to the nearest slot.
    agents = {
        1: _FakeAgent(state=_FakeState(pose=Pose2D(x=-5.0, y=0.0))),
        2: _FakeAgent(state=_FakeState(pose=Pose2D(x=5.0, y=0.0))),
    }
    anchor = PoseAnchor(fixed=Pose2D(x=0.0, y=0.0))
    f = FFormation(anchor=anchor, agent_lookup=lambda aid: agents.get(aid), base_radius=1.0, radius_per_member=0.0)
    f.on_join(1)
    f.on_join(2)
    targets = f.tick(dt=0.01)
    assert targets[1].x < 0.0
    assert targets[2].x > 0.0


def test_f_formation_hungarian_minimizes_total_distance() -> None:
    # Four agents at the four cardinal directions; each should land at its own side.
    poses = {
        1: Pose2D(x=3.0, y=0.0),
        2: Pose2D(x=0.0, y=3.0),
        3: Pose2D(x=-3.0, y=0.0),
        4: Pose2D(x=0.0, y=-3.0),
    }
    agents = {aid: _FakeAgent(state=_FakeState(pose=p)) for aid, p in poses.items()}
    anchor = PoseAnchor(fixed=Pose2D(x=0.0, y=0.0))
    f = FFormation(anchor=anchor, agent_lookup=lambda aid: agents.get(aid), base_radius=1.0, radius_per_member=0.0)
    # Join in an order that would produce a bad naive matching.
    for aid in (3, 1, 4, 2):
        f.on_join(aid)
    targets = f.tick(dt=0.01)
    for aid, start in poses.items():
        target = targets[aid]
        assert math.hypot(target.x - start.x, target.y - start.y) == pytest.approx(2.0, abs=1e-6)


def test_f_formation_falls_back_to_join_order_without_poses() -> None:
    anchor = PoseAnchor(fixed=Pose2D())
    f = FFormation(anchor=anchor, agent_lookup=lambda _aid: None, base_radius=1.0, radius_per_member=0.0)
    for aid in (7, 8, 9):
        f.on_join(aid)
    targets = f.tick(dt=0.01)
    for idx, aid in enumerate((7, 8, 9)):
        angle = 2.0 * math.pi * idx / 3
        assert targets[aid].x == pytest.approx(math.cos(angle), abs=1e-6)
        assert targets[aid].y == pytest.approx(math.sin(angle), abs=1e-6)


def test_f_formation_arrived_detects_slot_distance() -> None:
    agents: dict[int, _FakeAgent] = {
        1: _FakeAgent(state=_FakeState(pose=Pose2D(x=5.0, y=5.0))),
        2: _FakeAgent(state=_FakeState(pose=Pose2D(x=5.0, y=5.0))),
    }
    anchor = PoseAnchor(fixed=Pose2D(x=0.0, y=0.0))
    f = FFormation(anchor=anchor, agent_lookup=lambda aid: agents.get(aid), base_radius=BASE_RADIUS, radius_per_member=RADIUS_PER_MEMBER)
    f.on_join(1)
    f.on_join(2)
    targets = f.tick(dt=0.01)
    assert not f.arrived(1)
    slot = targets[1]
    agents[1].state.pose = Pose2D(x=slot.x, y=slot.y)
    assert f.arrived(1)
    assert not f.arrived(2)
