from __future__ import annotations

import pytest

from tests.ros._helpers import (
    GetProfile,
    RemoveAgents,
    ResetSimulation,
    RosTestSystem,
    SpawnAgents,
    make_get_profile_request,
    make_remove_request,
    make_reset_request,
    make_spawn_request,
)

pytestmark = pytest.mark.ros


@pytest.fixture(scope="module")
def system(ros_system: RosTestSystem) -> RosTestSystem:
    ros_system.call(ResetSimulation, "reset", make_reset_request())
    ros_system.manager._profile_phases = True
    return ros_system


def test_get_profile_empty_before_ticks(system: RosTestSystem) -> None:
    system.call(ResetSimulation, "reset", make_reset_request())
    system.manager._phase_accum.clear()
    resp = system.call(GetProfile, "get_profile", make_get_profile_request(reset=False))
    assert list(resp.phase_names) == []
    assert list(resp.phase_means_ms) == []
    assert list(resp.phase_p95s_ms) == []


def test_get_profile_after_ticks(system: RosTestSystem) -> None:
    system.call(ResetSimulation, "reset", make_reset_request())
    system.call(RemoveAgents, "remove_agents", make_remove_request([-1]))
    system.call(
        SpawnAgents,
        "spawn_agents",
        make_spawn_request([{"x": 0.0, "y": 0.0}]),
    )
    system.manager._profile_phases = True
    system.manager._phase_accum.clear()
    system.tick_manager(5)

    resp = system.call(GetProfile, "get_profile", make_get_profile_request(reset=False))
    assert len(resp.phase_names) > 0
    assert len(resp.phase_names) == len(resp.phase_means_ms) == len(resp.phase_p95s_ms)
    for mean in resp.phase_means_ms:
        assert mean >= 0.0
    for p95 in resp.phase_p95s_ms:
        assert p95 >= 0.0
    assert resp.n_agents == system.manager._pool.n
    assert resp.n_ticks >= 1


def test_get_profile_reset_flag_clears_accum(system: RosTestSystem) -> None:
    system.manager._profile_phases = True
    system.manager._phase_accum.clear()
    system.tick_manager(3)
    resp = system.call(GetProfile, "get_profile", make_get_profile_request(reset=True))
    assert len(resp.phase_names) > 0
    assert system.manager._phase_accum == {}

    resp_empty = system.call(GetProfile, "get_profile", make_get_profile_request(reset=False))
    assert list(resp_empty.phase_names) == []
