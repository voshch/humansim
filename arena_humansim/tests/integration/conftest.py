from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from arena_humansim.manager.agent_manager import AgentManager
from arena_humansim.utils.scenario import ScenarioConfig

from ._helpers import build_manager


@pytest.fixture
def manager_factory(rclpy_context: object) -> Iterator[Callable[..., AgentManager]]:  # noqa: ARG001
    created: list[AgentManager] = []

    def make(scenario: ScenarioConfig, node_name: str = "test_manager") -> AgentManager:
        mgr = build_manager(scenario, node_name=node_name)
        created.append(mgr)
        return mgr

    yield make

    for mgr in created:
        try:
            mgr.destroy_node()
        except Exception:
            pass
