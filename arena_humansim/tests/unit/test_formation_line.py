from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.formation.anchor import PoseAnchor
from arena_humansim.core.formation.line import LineFormation
from arena_humansim.utils.types import Pose2D

FRONT_OFFSET = 0.8
BASE_STEP = 1.0


@dataclass
class _FakeParams:
    reaction_time: float = 0.4
    personal_space_min: float = 0.6


@dataclass
class _FakeState:
    pose: Pose2D = field(default_factory=Pose2D)


@dataclass
class _FakeAgent:
    state: _FakeState = field(default_factory=_FakeState)
    params: _FakeParams = field(default_factory=_FakeParams)


def _mk_lookup(reaction_times: dict[int, float] | None = None) -> Callable[[int], _FakeAgent | None]:
    rt = reaction_times or {}

    def lookup(aid: int) -> _FakeAgent | None:
        return _FakeAgent(params=_FakeParams(reaction_time=rt.get(aid, 0.4)))

    return lookup


def test_line_appends_with_backward_spacing() -> None:
    anchor = PoseAnchor(fixed=Pose2D(x=0.0, y=0.0, theta=0.0))
    f = LineFormation(anchor=anchor, agent_lookup=_mk_lookup(), base_step=1.0)
    f.on_join(1)
    f.on_join(2)
    targets = f.tick(dt=0.01)
    assert targets[1] == Pose2D(x=0.0, y=0.0, theta=0.0)
    # Second slot is one step backward along (theta + pi) = west for theta=0
    assert targets[2].x == pytest.approx(-1.0)
    assert targets[2].y == pytest.approx(0.0)


def test_line_slots_are_leader_defined_and_immediate() -> None:
    # Formation must be solely leader-defined: every slot's target is a direct function
    # of the current anchor pose, with no reaction-delay echo or inter-slot dependency.
    anchor = PoseAnchor(fixed=Pose2D())
    f = LineFormation(anchor=anchor, agent_lookup=_mk_lookup(), base_step=1.0)
    for aid in (1, 2, 3):
        f.on_join(aid)
    t0 = f.tick(dt=0.0)
    assert t0[1] == Pose2D(x=0.0, y=0.0, theta=0.0)
    assert t0[2].x == pytest.approx(-1.0)
    assert t0[3].x == pytest.approx(-2.0)


def test_line_slots_retarget_immediately_when_leader_moves() -> None:
    # If the anchor jumps, every slot moves by the same delta in the same tick —
    # no reaction-delay lag, no wave propagation.
    anchor = PoseAnchor(fixed=Pose2D(x=0.0, y=0.0, theta=0.0))
    f = LineFormation(anchor=anchor, agent_lookup=_mk_lookup(), base_step=1.0)
    for aid in (1, 2, 3):
        f.on_join(aid)
    f.tick(dt=0.0)
    anchor.fixed = Pose2D(x=5.0, y=0.0, theta=0.0)
    moved = f.tick(dt=0.0)
    assert moved[1].x == pytest.approx(5.0)
    assert moved[2].x == pytest.approx(4.0)
    assert moved[3].x == pytest.approx(3.0)


def test_line_rejoin_fills_vacated_slot_without_delay() -> None:
    # After a member leaves, remaining members stay assigned to their slot indices;
    # the line shortens immediately without shuffle waves.
    anchor = PoseAnchor(fixed=Pose2D())
    f = LineFormation(anchor=anchor, agent_lookup=_mk_lookup(), base_step=1.0)
    for aid in (1, 2, 3, 4):
        f.on_join(aid)
    f.on_leave(2)
    t = f.tick(dt=0.0)
    assert t[1].x == pytest.approx(0.0)
    assert t[3].x == pytest.approx(-1.0)
    assert t[4].x == pytest.approx(-2.0)


def test_line_spacing_respects_personal_space_floor() -> None:
    anchor = PoseAnchor(fixed=Pose2D())

    # Agent 2 wants 1.5m; base_step is only 0.5m; max should win
    def lookup(aid: int) -> _FakeAgent | None:
        return _FakeAgent(params=_FakeParams(personal_space_min=1.5 if aid == 2 else 0.6))

    f = LineFormation(anchor=anchor, agent_lookup=lookup, base_step=0.5)
    f.on_join(1)
    f.on_join(2)
    targets = f.tick(dt=0.01)
    assert abs(targets[2].x) == pytest.approx(1.5)


def test_line_formation_scale_multiplies_spacing() -> None:
    anchor = PoseAnchor(fixed=Pose2D())
    f = LineFormation(anchor=anchor, agent_lookup=_mk_lookup(), base_step=1.0, formation_scale=2.0)
    f.on_join(1)
    f.on_join(2)
    targets = f.tick(dt=0.01)
    assert abs(targets[2].x) == pytest.approx(2.0)


def test_line_yaw_controls_growth_direction() -> None:
    anchor = PoseAnchor(fixed=Pose2D(theta=math.pi / 2))  # facing north -> line grows south
    f = LineFormation(anchor=anchor, agent_lookup=_mk_lookup(), base_step=1.0)
    f.on_join(1)
    f.on_join(2)
    targets = f.tick(dt=0.01)
    assert targets[2].x == pytest.approx(0.0, abs=1e-6)
    assert targets[2].y == pytest.approx(-1.0)


def test_line_front_offset_places_slot0_on_queue_side_of_anchor() -> None:
    anchor_x = 4.0
    anchor = PoseAnchor(fixed=Pose2D(x=anchor_x, y=0.0, theta=math.pi))
    f = LineFormation(anchor=anchor, agent_lookup=_mk_lookup(), base_step=BASE_STEP, front_offset=FRONT_OFFSET)
    f.on_join(1)
    f.on_join(2)
    targets = f.tick(dt=0.01)
    assert targets[1].x == pytest.approx(anchor_x + FRONT_OFFSET)
    assert targets[1].y == pytest.approx(0.0)
    assert targets[2].x == pytest.approx(anchor_x + FRONT_OFFSET + BASE_STEP)
    assert targets[2].y == pytest.approx(0.0)


def test_line_front_offset_flips_with_anchor_yaw() -> None:
    anchor = PoseAnchor(fixed=Pose2D(x=0.0, y=0.0, theta=0.0))
    f = LineFormation(anchor=anchor, agent_lookup=_mk_lookup(), base_step=BASE_STEP, front_offset=FRONT_OFFSET)
    f.on_join(1)
    f.on_join(2)
    targets = f.tick(dt=0.01)
    assert targets[1].x == pytest.approx(-FRONT_OFFSET)
    assert targets[2].x == pytest.approx(-FRONT_OFFSET - BASE_STEP)


def test_line_front_offset_zero_preserves_legacy_behavior() -> None:
    anchor = PoseAnchor(fixed=Pose2D(x=4.0, y=0.0, theta=math.pi))
    f = LineFormation(anchor=anchor, agent_lookup=_mk_lookup(), base_step=1.0)
    f.on_join(1)
    targets = f.tick(dt=0.01)
    assert targets[1].x == pytest.approx(4.0)


def test_line_join_leave_churn_does_not_leak_slots() -> None:
    anchor = PoseAnchor(fixed=Pose2D())
    f = LineFormation(anchor=anchor, agent_lookup=_mk_lookup(), base_step=1.0)
    for _ in range(10):
        f.on_join(1)
        f.on_join(2)
        f.on_leave(1)
        f.on_leave(2)
    assert f._slots == []
    f.on_join(7)
    targets = f.tick(dt=0.01)
    assert set(targets.keys()) == {7}


def test_line_front_pose_caches_across_ticks() -> None:
    anchor = PoseAnchor(fixed=Pose2D(x=4.0, y=0.0, theta=math.pi))
    f = LineFormation(anchor=anchor, agent_lookup=_mk_lookup(), base_step=BASE_STEP, front_offset=FRONT_OFFSET)
    f.on_join(1)
    first = f.tick(dt=0.01)
    second = f.tick(dt=0.01)
    assert second[1] is first[1]  # identity — cache hit


def test_line_front_pose_invalidates_on_anchor_change() -> None:
    import attrs

    new_anchor_x = 10.0
    initial = Pose2D(x=4.0, y=0.0, theta=math.pi)
    anchor = PoseAnchor(fixed=initial)
    f = LineFormation(anchor=anchor, agent_lookup=_mk_lookup(), base_step=BASE_STEP, front_offset=FRONT_OFFSET)
    f.on_join(1)
    before = f.tick(dt=0.01)[1]
    anchor.fixed = attrs.evolve(initial, x=new_anchor_x)
    after = f.tick(dt=0.01)[1]
    assert after is not before
    assert after.x == pytest.approx(new_anchor_x + FRONT_OFFSET)


def test_line_arrived_flips_true_within_tolerance() -> None:
    agents: dict[int, _FakeAgent] = {1: _FakeAgent(state=_FakeState(pose=Pose2D(x=5.0, y=0.0)))}
    anchor = PoseAnchor(fixed=Pose2D(x=0.0, y=0.0, theta=0.0))
    f = LineFormation(anchor=anchor, agent_lookup=lambda aid: agents.get(aid), base_step=1.0)
    f.on_join(1)
    f.tick(dt=0.01)
    assert not f.arrived(1)
    agents[1].state.pose = Pose2D(x=0.1, y=0.0)
    assert f.arrived(1)


def test_line_arrived_true_for_unknown_agent() -> None:
    agents: dict[int, _FakeAgent] = {}
    anchor = PoseAnchor(fixed=Pose2D(x=0.0, y=0.0, theta=0.0))
    f = LineFormation(anchor=anchor, agent_lookup=lambda aid: agents.get(aid), base_step=1.0)
    assert f.arrived(999)
