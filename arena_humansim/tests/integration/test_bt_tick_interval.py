from __future__ import annotations

from unittest.mock import patch

import attrs
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import ScenarioConfig

from ._helpers import build_manager


def test_bt_tick_fires_every_k_ticks(rclpy_context: object, minimal_scenario: ScenarioConfig) -> None:  # noqa: ARG001
    scenario = attrs.evolve(
        minimal_scenario,
        simulation=attrs.evolve(minimal_scenario.simulation, bt_tick_interval=3),
    )

    mgr = build_manager(scenario, node_name="test_bt_tick_interval")
    try:
        assert mgr._bt_tick_interval == 3

        original = AgentManager._process_event_scripts
        counter = {"n": 0}

        def counting(self: AgentManager) -> None:
            counter["n"] += 1
            return original(self)

        with patch.object(AgentManager, "_process_event_scripts", counting):
            for _ in range(9):
                mgr.tick()

        assert counter["n"] == 3, f"expected BT tick 3 times over 9 ticks (interval=3), got {counter['n']}"
    finally:
        mgr.destroy_node()
