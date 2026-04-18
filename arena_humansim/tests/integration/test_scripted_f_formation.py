from __future__ import annotations

import math
from collections.abc import Callable

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import (
    AgentConfig,
    InteractionScript,
    ModuleConfig,
    Pose2DModel,
    ScenarioConfig,
    SimulationParams,
)
from arena_humansim.utils.types import InteractionOutcome


CENTROID_X = 0.0
CENTROID_Y = -8.0
RADIUS = 6.0
FIRE_TICK = 50


def _scripted_scenario() -> ScenarioConfig:
    agents = [
        AgentConfig(agent_id=10, spawn_pose=Pose2DModel(x=-10.0, y=-8.0, theta=0.0), desired_velocity=1.0),
        AgentConfig(agent_id=11, spawn_pose=Pose2DModel(x=10.0, y=-8.0, theta=math.pi), desired_velocity=1.0),
        AgentConfig(agent_id=12, spawn_pose=Pose2DModel(x=0.0, y=-18.0, theta=math.pi / 2.0), desired_velocity=1.0),
        AgentConfig(agent_id=13, spawn_pose=Pose2DModel(x=0.0, y=2.0, theta=-math.pi / 2.0), desired_velocity=1.0),
    ]
    script = InteractionScript(
        tick=FIRE_TICK,
        interaction_type="GROUP_CONVERSATION",
        participants=[10, 11, 12, 13],
        duration_ticks=2000,
    )
    return ScenarioConfig(
        name="scripted_f_formation",
        simulation=SimulationParams(seed=42, dt=0.05, bt_tick_interval=1, max_ticks=400),
        modules=ModuleConfig(),
        agents=agents,
        interaction_scripts=[script],
    )


def _centroid_distance(mgr: AgentManager, agent_id: int) -> float:
    agent = mgr._agents[agent_id]
    dx = agent.state.pose.x - CENTROID_X
    dy = agent.state.pose.y - CENTROID_Y
    return math.hypot(dx, dy)


def test_scripted_f_formation_drives_simple_agents(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scripted_scenario(), node_name="test_scripted_f_formation")

    for _ in range(FIRE_TICK - 1):
        mgr.tick()

    pre_distances = {aid: _centroid_distance(mgr, aid) for aid in (10, 11, 12, 13)}
    assert all(d > RADIUS for d in pre_distances.values()), (
        f"agents unexpectedly close to centroid before script fires: {pre_distances}"
    )

    for _ in range(200):
        mgr.tick()

    active = [i for i in mgr._interaction_manager.interactions.values() if i.outcome == InteractionOutcome.ACTIVE]
    assert any(set(i.participants) == {10, 11, 12, 13} for i in active), (
        "scripted GROUP_CONVERSATION interaction never formed with all participants active"
    )

    post_distances = {aid: _centroid_distance(mgr, aid) for aid in (10, 11, 12, 13)}
    assert all(d < RADIUS for d in post_distances.values()), (
        f"agents did not converge to F-formation after script: {post_distances}"
    )
