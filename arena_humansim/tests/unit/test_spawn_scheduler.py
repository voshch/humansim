from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.spawn_scheduler import SpawnScheduler
from arena_humansim.utils.types import (
    AgentTemplate,
    Pose2D,
    RateKeyframe,
    Shape,
    ShapeType,
    SinkAffinity,
    SinkConfig,
    SourceConfig,
)


def _make_source(
    name: str = "src",
    *,
    rate: float = 1.0,
    max_concurrent: int = -1,
    max_total: int = -1,
    shape: Shape | None = None,
    pose: Pose2D | None = None,
    agent: AgentTemplate | None = None,
    profile: list[RateKeyframe] | None = None,
) -> SourceConfig:
    return SourceConfig(
        name=name,
        pose=pose if pose is not None else Pose2D(),
        shape=shape if shape is not None else Shape(type=ShapeType.POLYGON, vertices=[]),
        rate_profile=profile if profile is not None else [RateKeyframe(t=0.0, rate=rate)],
        max_concurrent=max_concurrent,
        max_total=max_total,
        agent=agent if agent is not None else AgentTemplate(),
    )


def test_interpolate_rate_empty() -> None:
    assert SpawnScheduler._interpolate_rate([], 0.0) == 0.0
    assert SpawnScheduler._interpolate_rate([], 10.0) == 0.0


def test_interpolate_rate_single() -> None:
    profile = [RateKeyframe(t=2.0, rate=3.5)]
    assert SpawnScheduler._interpolate_rate(profile, 0.0) == 3.5
    assert SpawnScheduler._interpolate_rate(profile, 10.0) == 3.5


def test_interpolate_rate_endpoints() -> None:
    profile = [RateKeyframe(t=0.0, rate=1.0), RateKeyframe(t=10.0, rate=5.0)]
    assert SpawnScheduler._interpolate_rate(profile, -1.0) == 1.0
    assert SpawnScheduler._interpolate_rate(profile, 0.0) == 1.0
    assert SpawnScheduler._interpolate_rate(profile, 10.0) == 5.0
    assert SpawnScheduler._interpolate_rate(profile, 99.0) == 5.0


def test_interpolate_rate_linear_midpoint() -> None:
    profile = [RateKeyframe(t=0.0, rate=2.0), RateKeyframe(t=4.0, rate=6.0)]
    assert SpawnScheduler._interpolate_rate(profile, 2.0) == pytest.approx(4.0)
    assert SpawnScheduler._interpolate_rate(profile, 1.0) == pytest.approx(3.0)


def test_interpolate_rate_zero_dt_tie() -> None:
    profile = [RateKeyframe(t=1.0, rate=2.0), RateKeyframe(t=1.0, rate=7.0)]
    assert SpawnScheduler._interpolate_rate(profile, 1.0) == 2.0


def test_tick_zero_rate_no_requests(rng) -> None:
    sched = SpawnScheduler(rng.get_substream("sched"))
    sched.add_source(_make_source("z", rate=0.0))
    out = sched.tick(0, 0.1)
    assert out == []
    out = sched.tick(100, 0.1)
    assert out == []


def test_tick_respects_max_total(rng) -> None:
    sched = SpawnScheduler(rng.get_substream("sched"))
    sched.add_source(_make_source("s", rate=1000.0, max_total=3))
    total: list = []
    for i in range(10):
        total.extend(sched.tick(i, 1.0))
    assert len(total) == 3
    assert sched._total_count["s"] == 3


def test_tick_respects_max_concurrent(rng) -> None:
    sched = SpawnScheduler(rng.get_substream("sched"))
    sched.add_source(_make_source("s", rate=1000.0, max_concurrent=2))
    total: list = []
    for i in range(5):
        total.extend(sched.tick(i, 1.0))
    assert len(total) == 2
    assert sched._alive_count["s"] == 2


def test_notify_despawn_frees_concurrent_slot(rng) -> None:
    sched = SpawnScheduler(rng.get_substream("sched"))
    sched.add_source(_make_source("s", rate=1000.0, max_concurrent=1))
    reqs = sched.tick(0, 1.0)
    assert len(reqs) == 1
    assert sched._alive_count["s"] == 1
    sched.register_agent(42, "s")
    sched.notify_despawn(42)
    assert sched._alive_count["s"] == 0
    reqs2 = sched.tick(1, 1.0)
    assert len(reqs2) == 1


def test_sample_pose_in_circle_all_within_radius(rng) -> None:
    sched = SpawnScheduler(rng.get_substream("sched"))
    center = Pose2D(x=2.0, y=-3.0, theta=0.0)
    shape = Shape(type=ShapeType.CIRCLE, radius=1.5, vertices=[])
    for _ in range(200):
        p = sched._sample_pose_in_shape(center, shape)
        d = math.hypot(p.x - center.x, p.y - center.y)
        assert d <= shape.radius + 1e-9


def test_sample_pose_in_polygon_collinear_returns_center(rng) -> None:
    sched = SpawnScheduler(rng.get_substream("sched"))
    center = Pose2D(x=5.0, y=7.0)
    vertices = [Pose2D(x=0.0, y=0.0), Pose2D(x=1.0, y=0.0), Pose2D(x=2.0, y=0.0)]
    p = sched._sample_pose_in_polygon(vertices, center)
    assert p.x == pytest.approx(center.x + vertices[0].x)
    assert p.y == pytest.approx(center.y + vertices[0].y)


def _point_in_triangle(px: float, py: float, a: Pose2D, b: Pose2D, c: Pose2D) -> bool:
    def sign(p1x: float, p1y: float, p2x: float, p2y: float, p3x: float, p3y: float) -> float:
        return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y)

    d1 = sign(px, py, a.x, a.y, b.x, b.y)
    d2 = sign(px, py, b.x, b.y, c.x, c.y)
    d3 = sign(px, py, c.x, c.y, a.x, a.y)
    has_neg = (d1 < -1e-9) or (d2 < -1e-9) or (d3 < -1e-9)
    has_pos = (d1 > 1e-9) or (d2 > 1e-9) or (d3 > 1e-9)
    return not (has_neg and has_pos)


def test_sample_pose_in_triangle_all_inside(rng) -> None:
    sched = SpawnScheduler(rng.get_substream("sched"))
    center = Pose2D(x=0.0, y=0.0)
    a = Pose2D(x=0.0, y=0.0)
    b = Pose2D(x=4.0, y=0.0)
    c = Pose2D(x=0.0, y=3.0)
    wa = Pose2D(x=center.x + a.x, y=center.y + a.y)
    wb = Pose2D(x=center.x + b.x, y=center.y + b.y)
    wc = Pose2D(x=center.x + c.x, y=center.y + c.y)
    for _ in range(100):
        p = sched._sample_pose_in_polygon([a, b, c], center)
        assert _point_in_triangle(p.x, p.y, wa, wb, wc)


def test_sink_affinity_sets_waypoint_and_heading(rng) -> None:
    sched = SpawnScheduler(rng.get_substream("sched"))
    sink_pose = Pose2D(x=10.0, y=0.0, theta=0.0)
    sched.set_sinks({"exit": SinkConfig(name="exit", pose=sink_pose)})
    tmpl = AgentTemplate(sink_affinity=[SinkAffinity(sink_name="exit", weight=1.0)])
    src = _make_source(
        "s",
        rate=1000.0,
        max_concurrent=1,
        pose=Pose2D(x=0.0, y=0.0),
        shape=Shape(type=ShapeType.POLYGON, vertices=[]),
        agent=tmpl,
    )
    sched.add_source(src)
    reqs = sched.tick(0, 1.0)
    assert len(reqs) == 1
    req = reqs[0]
    assert req.lifetime.target_sink_name == "exit"
    assert len(req.waypoints) == 1
    assert req.waypoints[0].x == pytest.approx(sink_pose.x)
    assert req.waypoints[0].y == pytest.approx(sink_pose.y)
    expected_theta = math.atan2(sink_pose.y - req.pose.y, sink_pose.x - req.pose.x)
    assert req.pose.theta == pytest.approx(expected_theta)


def test_add_remove_clear_sources_tracks_counts(rng) -> None:
    sched = SpawnScheduler(rng.get_substream("sched"))
    sched.add_source(_make_source("a"))
    sched.add_source(_make_source("b"))
    assert set(sched._sources.keys()) == {"a", "b"}
    assert sched._alive_count == {"a": 0, "b": 0}
    assert sched._total_count == {"a": 0, "b": 0}
    sched.remove_source("a")
    assert set(sched._sources.keys()) == {"b"}
    sched.clear_sources()
    assert sched._sources == {}


def test_reset_counts_zeros_state(rng) -> None:
    sched = SpawnScheduler(rng.get_substream("sched"))
    sched.add_source(_make_source("s", rate=1000.0, max_concurrent=5))
    sched.tick(0, 1.0)
    sched.register_agent(1, "s")
    assert sched._alive_count["s"] > 0
    assert sched._total_count["s"] > 0
    assert 1 in sched._agent_source
    sched.reset_counts()
    assert sched._alive_count == {}
    assert sched._total_count == {}
    assert sched._agent_source == {}


def test_tick_uses_default_rng_seed_deterministic() -> None:
    r1 = np.random.default_rng(0)
    r2 = np.random.default_rng(0)
    s1 = SpawnScheduler(r1)
    s2 = SpawnScheduler(r2)
    s1.add_source(_make_source("s", rate=5.0, max_concurrent=10))
    s2.add_source(_make_source("s", rate=5.0, max_concurrent=10))
    a = [len(s1.tick(i, 0.5)) for i in range(5)]
    b = [len(s2.tick(i, 0.5)) for i in range(5)]
    assert a == b
