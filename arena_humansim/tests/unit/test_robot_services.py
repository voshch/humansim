from __future__ import annotations

import pytest

pytest.importorskip("rclpy")

from typing import Any

from arena_humansim.core.interaction_manager import CommandType
from arena_humansim.core.robot_services import RobotServiceAdvertiser
from arena_humansim.utils.scenario import ServiceSpec
from arena_humansim.utils.types import AgentKind, AgentState, InteractionType, Pose2D


class _StubAgent:
    def __init__(self, agent_id: int, kind: AgentKind) -> None:
        self.state = AgentState(agent_id=agent_id, pose=Pose2D(), kind=int(kind))


def _robot_agent(agent_id: int) -> Any:
    return _StubAgent(agent_id, AgentKind.ROBOT)


def _human_agent(agent_id: int) -> Any:
    return _StubAgent(agent_id, AgentKind.HUMAN)


def test_robot_service_advertiser_emits_per_service_and_skips_non_robots() -> None:
    adv = RobotServiceAdvertiser()
    adv.register(10, [ServiceSpec(tag="water", max_participants=3), ServiceSpec(tag="trash", max_participants=-1)])
    adv.register(20, [ServiceSpec(tag="soup", max_participants=2)])

    cmds = adv.emit({10: _robot_agent(10), 20: _human_agent(20)})

    for_10 = [c for c in cmds if c.agent_id == 10]
    for_20 = [c for c in cmds if c.agent_id == 20]

    assert len(for_10) == 2
    assert for_20 == []

    by_tag = {c.service_tag: c for c in for_10}
    assert set(by_tag) == {"water", "trash"}
    for cmd in for_10:
        assert cmd.type == int(CommandType.ADVERTISE)
        assert cmd.interaction_type == int(InteractionType.SERVICE)
    assert by_tag["water"].max_participants == 3
    assert by_tag["trash"].max_participants is None
