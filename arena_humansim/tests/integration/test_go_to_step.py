from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import py_trees
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils import DISTANCE_TOLERANCE
from arena_humansim.utils.scenario import load_scenario

_SCENARIO_PATH = Path(__file__).resolve().parents[2] / "config" / "scenarios" / "go_to_waypoint.yaml"
_TICK_LIMIT = 1000


def test_go_to_step_reaches_target(manager_factory: Callable[..., AgentManager]) -> None:
    scenario = load_scenario(str(_SCENARIO_PATH))
    mgr = manager_factory(scenario, node_name="test_go_to_step")

    target_x, target_y = 5.0, 5.0
    agent_id = 1
    bt = mgr._behavior_trees.get(agent_id)
    assert bt is not None, "agent 1 should have a compiled BT"

    for _ in range(_TICK_LIMIT):
        mgr.tick()
        if bt.root.status == py_trees.common.Status.SUCCESS:
            break

    agent = mgr._agents[agent_id]
    dx = agent.state.pose.x - target_x
    dy = agent.state.pose.y - target_y
    dist = math.hypot(dx, dy)

    assert bt.root.status == py_trees.common.Status.SUCCESS, f"BT root did not reach SUCCESS (status={bt.root.status!r}); final pose=({agent.state.pose.x:.3f}, {agent.state.pose.y:.3f}), dist={dist:.3f}"
    assert dist <= DISTANCE_TOLERANCE, f"agent final pose ({agent.state.pose.x:.3f}, {agent.state.pose.y:.3f}) is {dist:.3f}m from target ({target_x}, {target_y}); DISTANCE_TOLERANCE={DISTANCE_TOLERANCE}"
