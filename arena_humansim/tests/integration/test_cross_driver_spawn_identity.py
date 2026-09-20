from __future__ import annotations

from collections.abc import Callable

import attrs
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import ScenarioConfig
from arena_humansim.utils.types import AgentTemplate, Pose2D, Shape, ShapeType, SourceConfig, SourceType


def _source() -> SourceConfig:
    return SourceConfig(
        name="src",
        pose=Pose2D(x=0.0, y=0.0, theta=0.0),
        shape=Shape(type=ShapeType.CIRCLE, radius=1.5),
        type=SourceType.MAX,
        max_concurrent=4,
        max_total=-1,
        agent=AgentTemplate(),
    )


def _population(mgr: AgentManager, n_ticks: int) -> dict[int, tuple[float, float, float, float]]:
    seen: dict[int, tuple[float, float, float, float]] = {}
    for _ in range(n_ticks):
        mgr.tick()
        for aid in mgr._last_spawned_ids:
            a = mgr._agents[aid]
            seen[aid] = (a.state.pose.x, a.state.pose.y, float(a.params.desired_velocity), float(a.params.agent_radius))
    return seen


def test_two_drivers_same_seed_spawn_identical_population(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    released = {}
    for name, planner in (("a", "sfm"), ("b", "straight")):
        scenario = attrs.evolve(minimal_scenario, modules=attrs.evolve(minimal_scenario.modules, local_planner=planner))
        mgr = manager_factory(scenario, node_name=f"test_cross_driver_{name}")
        mgr._spawn_scheduler.add_source(_source())
        released[planner] = _population(mgr, 30)
    a, b = released["sfm"], released["straight"]
    assert len(a) == 4
    assert set(a) == set(b)
    for aid in a:
        assert a[aid] == b[aid], f"agent {aid}: {a[aid]} vs {b[aid]}"
