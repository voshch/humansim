from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.formation.anchor import PoseAnchor
from arena_humansim.core.formation.line import SITE_COEF_SHUFFLE, LineFormation
from arena_humansim.utils.types import Pose2D


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


def test_line_front_leaves_wave_propagates_after_reaction() -> None:
    anchor = PoseAnchor(fixed=Pose2D())
    f = LineFormation(anchor=anchor, agent_lookup=_mk_lookup({2: 0.3, 3: 0.3}), base_step=1.0)
    f.on_join(1)
    f.on_join(2)
    f.on_join(3)
    f.tick(dt=0.01)  # settle initial targets
    t2_initial = f.tick(dt=0.0)[2]

    f.on_leave(1)

    # Immediately after leave, slot-2 (now index 0) gets new target at anchor
    immediately = f.tick(dt=0.0)
    assert immediately[2] == Pose2D()  # at anchor
    # Slot-3 still at its previous target (reaction hasn't fired)
    assert immediately[3].x == pytest.approx(-2.0)

    # Advance past reaction delay for slot-3
    delay = SITE_COEF_SHUFFLE * 0.3
    f.tick(dt=delay + 0.01)
    after = f.tick(dt=0.0)
    # Slot-3 should now inherit slot-2's previous target (which was -1, 0)
    assert after[3].x == pytest.approx(t2_initial.x)


def test_line_leaves_mid_causes_wave() -> None:
    anchor = PoseAnchor(fixed=Pose2D())
    f = LineFormation(anchor=anchor, agent_lookup=_mk_lookup(), base_step=1.0)
    for aid in (1, 2, 3, 4):
        f.on_join(aid)
    f.tick(dt=0.01)
    t3_before = f.tick(dt=0.0)[3]
    f.on_leave(2)
    # Slot-3 (now at index 1) sees predecessor target (anchor) differ from its own; will shuffle after reaction
    f.tick(dt=SITE_COEF_SHUFFLE * 0.4 + 0.05)
    t3_after = f.tick(dt=0.0)[3]
    # Slot-3 moved forward toward anchor
    assert t3_after.x > t3_before.x


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
