from __future__ import annotations

from collections.abc import Callable

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import ModuleConfig, ScenarioConfig, SimulationParams
from arena_humansim.utils.types import BehaviorTreeMovement, GestureIntent
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.srv import SpawnAgents
from geometry_msgs.msg import Pose2D as RosPose2D

if "gestures" not in AgentStateMsg.get_fields_and_field_types():
    pytest.skip("arena_humansim_msgs built without gesture fields", allow_module_level=True)


def _spawn(mgr: AgentManager, name: str, kind: int, x: float, handedness: str = "") -> SpawnAgents.Response:
    req = SpawnAgents.Request()
    msg = AgentStateMsg()
    msg.name = name
    msg.kind = kind
    msg.pose = RosPose2D(x=x, y=0.0, theta=0.0)
    msg.desired_velocity = 1.3
    msg.radius = 0.3
    msg.agent_type = "adult"
    msg.handedness = handedness
    req.agents.append(msg)
    resp = SpawnAgents.Response()
    mgr._spawn_agents_callback(req, resp)
    return resp


def test_agent_states_carry_name_handedness_and_gestures(manager_factory: Callable[..., AgentManager]) -> None:
    scenario = ScenarioConfig(name="gesture_pub", simulation=SimulationParams(seed=1, dt=0.05, max_ticks=0), modules=ModuleConfig())
    mgr = manager_factory(scenario, node_name="test_gesture_pub")

    ped = _spawn(mgr, "ped_1", AgentStateMsg.KIND_HUMAN, 0.0).spawned_ids[0]
    bot = _spawn(mgr, "bot", AgentStateMsg.KIND_ROBOT, 2.0).spawned_ids[0]
    lefty = _spawn(mgr, "", AgentStateMsg.KIND_HUMAN, 4.0, handedness="l").spawned_ids[0]
    assert mgr._agent_name_to_id["ped_1"] == ped
    assert mgr._agent_name_to_id["bot"] == bot
    assert mgr._agents[ped].params.handedness in ("l", "r")
    assert mgr._agents[lefty].params.handedness == "l"

    mv = BehaviorTreeMovement(gestures=(GestureIntent("arm", 1.0, 2.0, 3.0, hand="l"), GestureIntent("head", 4.0, 5.0, 6.0)))
    mgr._agents[ped].movement = mv

    by_id = {a.agent_id: a for a in mgr._build_agent_states_msg().agents}
    assert by_id[ped].name == "ped_1"
    assert by_id[ped].handedness == mgr._agents[ped].params.handedness
    assert [g.slot for g in by_id[ped].gestures] == ["arm", "head"]
    arm, head = by_id[ped].gestures
    assert (arm.at.x, arm.at.y, arm.at.z) == (1.0, 2.0, 3.0)
    assert arm.hand == "l"
    assert (head.at.x, head.at.y, head.at.z) == (4.0, 5.0, 6.0)
    assert head.hand == "" and head.clip == ""
    assert by_id[bot].name == "bot"
    assert list(by_id[bot].gestures) == []
    assert by_id[lefty].name == ""
    assert by_id[lefty].handedness == "l"
    assert list(by_id[lefty].gestures) == []

    mv.gestures = ()
    by_id = {a.agent_id: a for a in mgr._build_agent_states_msg().agents}
    assert list(by_id[ped].gestures) == []

    assert mgr._lookup_agent_name("bot") == bot
    assert mgr._lookup_agent_name("bot", AgentStateMsg.KIND_ROBOT) == bot
    assert mgr._lookup_agent_name("ped_1", AgentStateMsg.KIND_ROBOT) is None
    assert mgr._lookup_agent_name("ped_1", AgentStateMsg.KIND_HUMAN) == ped

    mgr._remove_agent(bot)
    assert mgr._agent_name_to_id == {"ped_1": ped}


def test_spawn_rejects_reserved_names_and_bad_handedness(manager_factory: Callable[..., AgentManager]) -> None:
    scenario = ScenarioConfig(name="gesture_reserved", simulation=SimulationParams(seed=1, dt=0.05, max_ticks=0), modules=ModuleConfig())
    mgr = manager_factory(scenario, node_name="test_gesture_reserved")

    resp = _spawn(mgr, "partner", AgentStateMsg.KIND_HUMAN, 0.0)
    assert resp.success is False
    assert "reserved" in resp.message
    assert mgr._agents == {}

    resp = _spawn(mgr, "ped_1", AgentStateMsg.KIND_HUMAN, 0.0, handedness="left")
    assert resp.success is False
    assert "handedness" in resp.message
    assert mgr._agents == {}
