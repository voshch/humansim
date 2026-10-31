from __future__ import annotations

import attrs
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

        for _ in range(9):
            mgr.tick()

        assert len(mgr._phase_accum["decide"]) == 3
    finally:
        mgr.destroy_node()
