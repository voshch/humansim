from __future__ import annotations

import pytest

pytest.importorskip("rclpy")

from arena_humansim.utils.scenario import ModuleConfig, ScenarioConfig, SimulationParams
from rclpy.parameter import Parameter

from tests.integration._helpers import build_manager


def test_subsystem_stamp_is_scheduled_sim_time_not_clock_now() -> None:
    """Subsystem mode stamps AgentStates with tick_count*dt, not get_clock().now().

    Protects against the pre-fix behavior where stale queued callbacks would all
    read the current /clock, silently producing bursty state with mis-stamped headers.
    """
    dt = 0.05
    n_ticks = 10

    scenario = ScenarioConfig(
        name="subsys_stamp",
        simulation=SimulationParams(seed=1, dt=dt, max_ticks=n_ticks),
        modules=ModuleConfig(),
    )
    mgr = build_manager(
        scenario,
        node_name="test_subsys_stamp",
        extra_params=[Parameter("mode", Parameter.Type.STRING, "subsystem")],
    )

    try:
        dt_ns = int(dt * 1e9)
        for _ in range(n_ticks):
            mgr._subsystem_timer_callback()

        assert mgr._tick_count == n_ticks
        assert mgr._sim_time_ns == (n_ticks - 1) * dt_ns, (
            f"sim_time_ns after {n_ticks} ticks should be (n-1)*dt = {(n_ticks - 1) * dt_ns}, "
            f"got {mgr._sim_time_ns}"
        )
    finally:
        mgr.destroy_node()


def test_subsystem_rejects_unimplemented_policy() -> None:
    """Unknown subsystem_overrun_policy values must raise at setup, not silently no-op."""
    scenario = ScenarioConfig(
        name="subsys_bad_policy",
        simulation=SimulationParams(seed=1, dt=0.05, max_ticks=1),
        modules=ModuleConfig(),
    )
    with pytest.raises(ValueError, match="subsystem_overrun_policy"):
        build_manager(
            scenario,
            node_name="test_subsys_bad_policy",
            extra_params=[
                Parameter("mode", Parameter.Type.STRING, "subsystem"),
                Parameter("subsystem_overrun_policy", Parameter.Type.STRING, "skip"),
            ],
        )
