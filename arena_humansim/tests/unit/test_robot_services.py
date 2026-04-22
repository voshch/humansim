from __future__ import annotations

import pytest

pytest.importorskip("rclpy")

from typing import Any

from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.core.robot_services import RobotServiceAdvertiser
from arena_humansim.utils.scenario import ServiceSpec
from arena_humansim.utils.types import AgentKind, AgentState, CommandType, Pose2D


class _StubAgent:
    def __init__(self, agent_id: int, kind: AgentKind) -> None:
        self.state = AgentState(agent_id=agent_id, pose=Pose2D(), kind=int(kind))


def _robot_agent(agent_id: int) -> Any:
    return _StubAgent(agent_id, AgentKind.ROBOT)


def _human_agent(agent_id: int) -> Any:
    return _StubAgent(agent_id, AgentKind.HUMAN)


def test_robot_service_advertiser_emits_per_service() -> None:
    adv = RobotServiceAdvertiser()
    adv.register(10, [ServiceSpec(tag="water", max_participants=3), ServiceSpec(tag="trash", max_participants=-1)])
    adv.register(20, [ServiceSpec(tag="soup", max_participants=2)])

    cmds = adv.emit({10: _robot_agent(10), 20: _human_agent(20)})

    for_10 = [c for c in cmds if c.agent_id == 10]
    for_20 = [c for c in cmds if c.agent_id == 20]

    assert len(for_10) == 2
    # AgentKind.ROBOT gate in IM is gone; advertiser emits for whoever registered services.
    assert len(for_20) == 1

    by_tag = {c.spec.target: c for c in for_10 if c.spec is not None}
    assert set(by_tag) == {"water", "trash"}
    for cmd in for_10:
        assert cmd.type == CommandType.SEEK
        assert cmd.spec is not None
        assert cmd.spec.interaction_type == InteractionType.SERVICE
        assert cmd.spec.offer is True
    assert by_tag["water"].spec.max_participants == 3
    # negative max_participants becomes None (unbounded) in the spec.
    assert by_tag["trash"].spec.max_participants is None


def test_unregister_drops_services() -> None:
    adv = RobotServiceAdvertiser()
    adv.register(1, [ServiceSpec(tag="water", max_participants=1)])
    adv.unregister(1)
    assert adv.emit({1: _robot_agent(1)}) == []


def test_missing_agent_skipped() -> None:
    adv = RobotServiceAdvertiser()
    adv.register(99, [ServiceSpec(tag="water", max_participants=1)])
    # agent not in the lookup dict -> nothing emitted for them.
    assert adv.emit({}) == []


def test_empty_tag_skipped() -> None:
    adv = RobotServiceAdvertiser()
    adv.register(1, [ServiceSpec(tag="", max_participants=1), ServiceSpec(tag="water", max_participants=1)])
    cmds = adv.emit({1: _robot_agent(1)})
    tags = {c.spec.target for c in cmds if c.spec is not None}
    assert tags == {"water"}
