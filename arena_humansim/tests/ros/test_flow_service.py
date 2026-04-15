from __future__ import annotations

import pytest
from arena_humansim_msgs.msg import (
    AgentTemplate,
    FlowConfig,
    RateKeyframe,
    Shape,
    SinkAffinity,
    SinkConfig,
    SourceConfig,
)
from arena_humansim_msgs.srv import SetFlow
from geometry_msgs.msg import Pose2D as Pose2DMsg

from tests.ros._helpers import (
    RemoveAgents,
    ResetSimulation,
    RosTestSystem,
    make_remove_request,
    make_reset_request,
)

pytestmark = pytest.mark.ros


def _circle_shape(radius: float) -> Shape:
    s = Shape()
    s.type = Shape.CIRCLE
    s.radius = float(radius)
    return s


def _agent_template(agent_type: str = "adult", sink_affinities: list[tuple[str, float]] | None = None) -> AgentTemplate:
    tmpl = AgentTemplate()
    tmpl.desired_velocity_min = 1.2
    tmpl.desired_velocity_max = 1.4
    tmpl.agent_radius = 0.3
    tmpl.agent_type = agent_type
    tmpl.behavior_tree = ""
    for name, weight in sink_affinities or []:
        aff = SinkAffinity()
        aff.sink_name = name
        aff.weight = float(weight)
        tmpl.sink_affinity.append(aff)
    return tmpl


def _source(name: str, x: float, y: float, rate: float, shape_radius: float = 0.5, sink_affinities: list[tuple[str, float]] | None = None, max_concurrent: int = 0, max_total: int = 0) -> SourceConfig:
    src = SourceConfig()
    src.name = name
    src.pose = Pose2DMsg(x=float(x), y=float(y), theta=0.0)
    src.shape = _circle_shape(shape_radius)
    kf = RateKeyframe()
    kf.t = 0.0
    kf.rate = float(rate)
    src.rate_profile.append(kf)
    src.max_concurrent = int(max_concurrent)
    src.max_total = int(max_total)
    src.agent = _agent_template(sink_affinities=sink_affinities)
    return src


def _sink(name: str, x: float, y: float, absorption_radius: float = 1.0, capacity: int = 0) -> SinkConfig:
    sk = SinkConfig()
    sk.name = name
    sk.pose = Pose2DMsg(x=float(x), y=float(y), theta=0.0)
    sk.shape = _circle_shape(absorption_radius)
    sk.absorption_radius = float(absorption_radius)
    sk.capacity = int(capacity)
    return sk


def _set_flow_request(sources: list[SourceConfig], sinks: list[SinkConfig]) -> SetFlow.Request:
    req = SetFlow.Request()
    flow = FlowConfig()
    for s in sources:
        flow.sources.append(s)
    for s in sinks:
        flow.sinks.append(s)
    req.flow = flow
    return req


@pytest.fixture(scope="module")
def system(ros_system: RosTestSystem) -> RosTestSystem:
    ros_system.call(ResetSimulation, "reset", make_reset_request())
    return ros_system


def test_set_flow_installs_sources(system: RosTestSystem) -> None:
    system.call(ResetSimulation, "reset", make_reset_request())
    req = _set_flow_request(
        sources=[_source("src_install", 0.0, 0.0, rate=1.0)],
        sinks=[_sink("sink_install", 5.0, 0.0, absorption_radius=1.0)],
    )
    resp = system.call(SetFlow, "set_flow", req)
    assert resp.success is True

    scheduler = system.manager._spawn_scheduler
    monitor = system.manager._despawn_monitor
    assert "src_install" in scheduler._sources
    assert "sink_install" in monitor._sinks
    assert "sink_install" in scheduler._sinks


def test_source_spawns_agents_on_tick(system: RosTestSystem) -> None:
    system.call(ResetSimulation, "reset", make_reset_request())
    system.call(RemoveAgents, "remove_agents", make_remove_request([-1]))

    req = _set_flow_request(
        sources=[_source("src_spawner", 0.0, 0.0, rate=20.0, shape_radius=0.1, sink_affinities=[("sink_spawner", 1.0)])],
        sinks=[_sink("sink_spawner", 50.0, 0.0, absorption_radius=1.0)],
    )
    resp = system.call(SetFlow, "set_flow", req)
    assert resp.success is True

    before = len(system.manager._agents)
    system.tick_manager(20)
    after = len(system.manager._agents)
    assert after - before >= 1, f"expected at least one spawn, had {before} -> {after}"

    system.call(ResetSimulation, "reset", make_reset_request())


def test_sink_removes_agents_on_tick(system: RosTestSystem) -> None:
    system.call(ResetSimulation, "reset", make_reset_request())
    system.call(RemoveAgents, "remove_agents", make_remove_request([-1]))

    req = _set_flow_request(
        sources=[_source("src_sinking", 0.0, 0.0, rate=30.0, shape_radius=0.1, sink_affinities=[("sink_near", 1.0)])],
        sinks=[_sink("sink_near", 1.5, 0.0, absorption_radius=1.0)],
    )
    resp = system.call(SetFlow, "set_flow", req)
    assert resp.success is True

    peak = 0
    despawned: list[int] = []
    for _ in range(200):
        system.tick_manager(1)
        peak = max(peak, len(system.manager._agents))
        despawned.extend(getattr(system.manager, "_last_despawned_ids", []) or [])
        if despawned:
            break

    assert peak >= 1, "no agents ever spawned"
    assert len(despawned) >= 1, f"no agents despawned via sink within 200 ticks (peak={peak})"

    system.call(ResetSimulation, "reset", make_reset_request())


def test_set_flow_replaces_previous(system: RosTestSystem) -> None:
    system.call(ResetSimulation, "reset", make_reset_request())

    first = _set_flow_request(
        sources=[_source("src_old", 0.0, 0.0, rate=1.0)],
        sinks=[_sink("sink_old", 5.0, 0.0, absorption_radius=1.0)],
    )
    assert system.call(SetFlow, "set_flow", first).success is True
    assert "src_old" in system.manager._spawn_scheduler._sources
    assert "sink_old" in system.manager._despawn_monitor._sinks

    second = _set_flow_request(
        sources=[_source("src_new", 1.0, 1.0, rate=2.0)],
        sinks=[_sink("sink_new", 6.0, 0.0, absorption_radius=1.0)],
    )
    assert system.call(SetFlow, "set_flow", second).success is True

    scheduler = system.manager._spawn_scheduler
    monitor = system.manager._despawn_monitor
    assert "src_new" in scheduler._sources
    assert "src_old" not in scheduler._sources
    assert "sink_new" in monitor._sinks
    assert "sink_old" not in monitor._sinks
    assert "sink_new" in scheduler._sinks
    assert "sink_old" not in scheduler._sinks

    system.call(ResetSimulation, "reset", make_reset_request())
