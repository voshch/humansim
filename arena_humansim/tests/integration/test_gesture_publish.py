from __future__ import annotations

import json
from collections.abc import Callable

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import ModuleConfig, ScenarioConfig, SimulationParams
from arena_humansim.utils.types import BehaviorTreeMovement, GestureIntent
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.srv import SpawnAgents
from geometry_msgs.msg import Pose2D as RosPose2D

if "gesture_opts" not in AgentStateMsg.get_fields_and_field_types():
    pytest.skip("arena_humansim_msgs built without gesture fields", allow_module_level=True)


def _spawn(mgr: AgentManager, name: str, kind: int, x: float) -> int:
    req = SpawnAgents.Request()
    msg = AgentStateMsg()
    msg.name = name
    msg.kind = kind
    msg.pose = RosPose2D(x=x, y=0.0, theta=0.0)
    msg.desired_velocity = 1.3
    msg.radius = 0.3
    msg.agent_type = "adult"
    req.agents.append(msg)
    resp = SpawnAgents.Response()
    mgr._spawn_agents_callback(req, resp)
    return resp.spawned_ids[0]


def test_agent_states_carry_name_and_gesture(manager_factory: Callable[..., AgentManager]) -> None:
    scenario = ScenarioConfig(name="gesture_pub", simulation=SimulationParams(seed=1, dt=0.05, max_ticks=0), modules=ModuleConfig())
    mgr = manager_factory(scenario, node_name="test_gesture_pub")

    ped = _spawn(mgr, "ped_1", AgentStateMsg.KIND_HUMAN, 0.0)
    bot = _spawn(mgr, "bot", AgentStateMsg.KIND_ROBOT, 2.0)
    anon = _spawn(mgr, "", AgentStateMsg.KIND_HUMAN, 4.0)
    assert mgr._agent_name_to_id["ped_1"] == ped
    assert mgr._agent_name_to_id["bot"] == bot

    mv = BehaviorTreeMovement(gesture=GestureIntent("point", 1.0, 2.0, 3.0, "left"))
    mgr._agents[ped].movement = mv

    by_id = {a.agent_id: a for a in mgr._build_agent_states_msg().agents}
    assert by_id[ped].name == "ped_1"
    assert by_id[ped].gesture == "point"
    assert (by_id[ped].gesture_at.x, by_id[ped].gesture_at.y, by_id[ped].gesture_at.z) == (1.0, 2.0, 3.0)
    assert json.loads(by_id[ped].gesture_opts) == {"hand": "left"}
    assert by_id[bot].name == "bot"
    assert by_id[bot].gesture == ""
    assert by_id[bot].gesture_opts == ""
    assert by_id[anon].name == ""
    assert (by_id[anon].gesture_at.x, by_id[anon].gesture_at.y, by_id[anon].gesture_at.z) == (0.0, 0.0, 0.0)

    mv.gesture = GestureIntent("point", 1.0, 2.0, 3.0)
    by_id = {a.agent_id: a for a in mgr._build_agent_states_msg().agents}
    assert json.loads(by_id[ped].gesture_opts) == {"hand": "auto"}

    mv.gesture = None
    by_id = {a.agent_id: a for a in mgr._build_agent_states_msg().agents}
    assert by_id[ped].gesture == ""
    assert by_id[ped].gesture_opts == ""
    assert (by_id[ped].gesture_at.x, by_id[ped].gesture_at.y, by_id[ped].gesture_at.z) == (0.0, 0.0, 0.0)

    assert mgr._lookup_agent_name("bot") == bot
    assert mgr._lookup_agent_name("bot", AgentStateMsg.KIND_ROBOT) == bot
    assert mgr._lookup_agent_name("ped_1", AgentStateMsg.KIND_ROBOT) is None
    assert mgr._lookup_agent_name("ped_1", AgentStateMsg.KIND_HUMAN) == ped

    mgr._remove_agent(bot)
    assert mgr._agent_name_to_id == {"ped_1": ped}
