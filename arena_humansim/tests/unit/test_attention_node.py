from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

pytest.importorskip("rclpy")
py_trees = pytest.importorskip("py_trees")

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.agents.types import AttentionDef, ParamDist, Pose3, RelativeRef, RobotRef
from arena_humansim.core.behavior.nodes import AttentionNode, ClearGestureNode
from arena_humansim.core.behavior.nodes.attention import (
    FACE_ENTER_RAD,
    FACE_KEEP_RAD,
    FACE_TIMEOUT_S,
    GESTURE_Z_AGENT,
    RESOLVE_TIMEOUT_S,
    GESTURE_Z_OBJECT,
)
from arena_humansim.core.behavior.step_context import StepContext
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.core.interaction_manager import InteractionManager
from arena_humansim.core.pool import KIND_ROBOT
from arena_humansim.core.world_knowledge import WorldKnowledge, WorldObject
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.types import BehaviorTreeMovement, CommandType, GestureIntent, HighLevelCommand, Pose2D, SeekSpec

DT = 0.5
RUNNING = py_trees.common.Status.RUNNING
SUCCESS = py_trees.common.Status.SUCCESS
FAILURE = py_trees.common.Status.FAILURE


@pytest.fixture(autouse=True)
def _clear_blackboard() -> None:
    py_trees.blackboard.Blackboard.clear()


@pytest.fixture
def rng_np() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def world() -> WorldKnowledge:
    w = WorldKnowledge()
    w.add_object(WorldObject(object_id="bench_1", type="bench", pose=Pose2D(x=5.0, y=0.0)))
    w.add_object(WorldObject(object_id="bench_2", type="bench", pose=Pose2D(x=-5.0, y=0.0)))
    return w


def _mv(agent: BaseAgent) -> BehaviorTreeMovement:
    return cast(BehaviorTreeMovement, agent.movement)


def _tick(node: py_trees.behaviour.Behaviour) -> py_trees.common.Status:
    node.tick_once()
    return node.status


def _bt_agent(agent_factory: Callable[..., BaseAgent], agent_id: int = 1, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> BaseAgent:
    agent = agent_factory(agent_id=agent_id, x=x, y=y)
    agent.state.pose.theta = theta
    agent.movement = BehaviorTreeMovement()
    return agent


def _att(at: object, **kw: object) -> AttentionDef:
    return AttentionDef(gesture="point", at=at, **kw)  # type: ignore[arg-type]


def _node(
    agent: BaseAgent,
    att: AttentionDef,
    world: WorldKnowledge,
    rng: np.random.Generator,
    agents: dict[int, BaseAgent] | None = None,
    names: dict[str, int] | None = None,
    robots: dict[str, int] | None = None,
    ctx: StepContext | None = None,
    bare: bool = True,
    idle: bool = False,
    duration: ParamDist | None = None,
) -> AttentionNode:
    agents = {} if agents is None else agents
    names = names or {}
    robots = robots or {}
    def name_lookup(name: str, kind: int | None) -> int | None:
        return robots.get(name) if kind == KIND_ROBOT else {**names, **robots}.get(name)

    node = AttentionNode(
        "attention",
        agent,
        att,
        world,
        agent_lookup=agents.get,
        name_lookup=name_lookup,
        rng=rng,
        dt=DT,
        ctx=ctx if ctx is not None else StepContext(),
        bare=bare,
        idle=idle,
        duration=duration,
    )
    node.setup()
    return node


def _bind(agents: dict[int, BaseAgent], itype: InteractionType = InteractionType.GROUP_CONVERSATION) -> InteractionManager:
    mgr = InteractionManager(RNG(0))
    mgr.set_context(world_knowledge=WorldKnowledge(), agent_lookup=lambda aid: agents.get(aid), visibility_lookup=lambda aid: set(agents) - {aid})
    cmds = {aid: HighLevelCommand(agent_id=aid, type=CommandType.SEEK, spec=SeekSpec(interaction_type=itype)) for aid in agents}
    mgr.update(cmds)
    iid = next(iter(mgr.interactions))
    assert set(mgr.interactions[iid].participants) == set(agents)
    for a in agents.values():
        assert _mv(a).interaction_id == iid
    return mgr


# resolver table


def test_resolve_object_id(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi)
    node = _node(agent, _att("bench_2", hold="keep"), world, rng_np)
    assert _tick(node) == SUCCESS
    assert _mv(agent).gesture == GestureIntent("point", -5.0, 0.0, GESTURE_Z_OBJECT)


def test_resolve_object_id_suffix(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    world.add_object(WorldObject(object_id="env_0/lamp_1", type="lamp", pose=Pose2D(x=0.0, y=-5.0)))
    agent = _bt_agent(agent_factory, theta=-math.pi / 2)
    node = _node(agent, _att("lamp_1", hold="keep"), world, rng_np)
    assert _tick(node) == SUCCESS
    assert _mv(agent).gesture == GestureIntent("point", 0.0, -5.0, GESTURE_Z_OBJECT)


def test_resolve_object_id_wins_over_agent_name(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi)
    other = _bt_agent(agent_factory, agent_id=2, x=0.0, y=5.0)
    node = _node(agent, _att("bench_2", hold="keep"), world, rng_np, agents={2: other}, names={"bench_2": 2})
    assert _tick(node) == SUCCESS
    assert _mv(agent).gesture == GestureIntent("point", -5.0, 0.0, GESTURE_Z_OBJECT)


def test_resolve_agent_name(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi / 2)
    other = _bt_agent(agent_factory, agent_id=2, x=0.0, y=5.0)
    node = _node(agent, AttentionDef(gesture="wave", at="ped_2", hold="keep"), world, rng_np, agents={2: other}, names={"ped_2": 2})
    assert _tick(node) == SUCCESS
    assert _mv(agent).gesture == GestureIntent("wave", 0.0, 5.0, GESTURE_Z_AGENT)


def test_resolve_agent_name_wins_over_object_type(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi / 2)
    other = _bt_agent(agent_factory, agent_id=2, x=0.0, y=5.0)
    node = _node(agent, _att("bench", hold="keep"), world, rng_np, agents={2: other}, names={"bench": 2})
    assert _tick(node) == SUCCESS
    assert _mv(agent).gesture == GestureIntent("point", 0.0, 5.0, GESTURE_Z_AGENT)


def test_resolve_object_type_nearest(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, x=2.0, theta=0.0)
    node = _node(agent, _att("bench", at_z=0.3, hold="keep"), world, rng_np)
    assert _tick(node) == SUCCESS
    assert _mv(agent).gesture == GestureIntent("point", 5.0, 0.0, 0.3)


def test_resolve_int_agent_id(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi / 2)
    other = _bt_agent(agent_factory, agent_id=7, x=0.0, y=3.0)
    node = _node(agent, _att(7, at_z=1.5, hold="keep"), world, rng_np, agents={7: other})
    assert _tick(node) == SUCCESS
    assert _mv(agent).gesture == GestureIntent("point", 0.0, 3.0, 1.5)


def test_resolve_pose3_literal(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(Pose3(4.0, 0.0, 2.0), hold="keep"), world, rng_np)
    assert _tick(node) == SUCCESS
    assert _mv(agent).gesture == GestureIntent("point", 4.0, 0.0, 2.0)


def test_resolve_robot_ref_only_matches_robots(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi / 2)
    human = _bt_agent(agent_factory, agent_id=2, x=0.0, y=5.0)
    bot = _bt_agent(agent_factory, agent_id=3, x=0.0, y=-5.0)
    node = _node(agent, _att(RobotRef("bot"), face=False, hold="keep"), world, rng_np, agents={2: human, 3: bot}, names={"bot": 2})
    assert _tick(node) == RUNNING
    assert _mv(agent).gesture is None
    node = _node(agent, _att(RobotRef("bot"), face=False, hold="keep"), world, rng_np, agents={2: human, 3: bot}, names={"bot": 2}, robots={"bot": 3})
    assert _tick(node) == SUCCESS
    assert _mv(agent).gesture == GestureIntent("point", 0.0, -5.0, GESTURE_Z_AGENT)


def test_resolve_target_from_ctx_pose(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att("target", hold="keep"), world, rng_np, ctx=StepContext(target_pose=Pose2D(x=3.0, y=0.0)))
    assert _tick(node) == SUCCESS
    assert _mv(agent).gesture == GestureIntent("point", 3.0, 0.0, GESTURE_Z_OBJECT)


def test_resolve_target_prefers_ctx_object(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    ctx = StepContext(target_pose=Pose2D(x=3.0, y=0.0), target_object_id="bench_1")
    node = _node(agent, _att("target", at_z=0.5, hold="keep"), world, rng_np, ctx=ctx)
    assert _tick(node) == SUCCESS
    assert _mv(agent).gesture == GestureIntent("point", 5.0, 0.0, 0.5)


def test_resolve_goal_from_navigate_command(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att("goal"), world, rng_np, bare=False)
    assert _tick(node) == RUNNING
    assert _mv(agent).gesture is None
    _mv(agent).command = HighLevelCommand(agent_id=1, type=CommandType.NAVIGATE, target_pose=Pose2D(x=2.0, y=2.0))
    assert _tick(node) == RUNNING
    assert _mv(agent).gesture == GestureIntent("point", 2.0, 2.0, GESTURE_Z_OBJECT)
    _mv(agent).command = HighLevelCommand(agent_id=1, type=CommandType.SEEK)
    _mv(agent).gesture = None
    assert _tick(node) == RUNNING
    assert _mv(agent).gesture is None


@pytest.mark.parametrize("cmd", [None, HighLevelCommand(agent_id=1, type=CommandType.STOP), HighLevelCommand(agent_id=1, type=CommandType.NAVIGATE, target_pose=Pose2D(x=1.0, y=1.0), desired_velocity=0.0), HighLevelCommand(agent_id=1, type=CommandType.NAVIGATE, target_pose=Pose2D(x=1.0, y=1.0))])
def test_resolve_goal_unresolved_while_halted(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator, cmd: HighLevelCommand | None) -> None:
    agent = _bt_agent(agent_factory, x=1.0, y=1.0)
    _mv(agent).command = cmd
    node = _node(agent, _att("goal"), world, rng_np, bare=False)
    assert _tick(node) == RUNNING
    assert _mv(agent).gesture is None
    _mv(agent).command = HighLevelCommand(agent_id=1, type=CommandType.NAVIGATE, target_pose=Pose2D(x=3.0, y=1.0))
    assert _tick(node) == RUNNING
    assert _mv(agent).gesture == GestureIntent("point", 3.0, 1.0, GESTURE_Z_OBJECT)


def test_resolve_partner_nearest_participant(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, agent_id=1, x=0.0, y=0.0)
    near = _bt_agent(agent_factory, agent_id=2, x=1.0, y=0.0)
    far = _bt_agent(agent_factory, agent_id=3, x=-2.0, y=0.0)
    agents = {1: agent, 2: near, 3: far}
    mgr = _bind(agents)
    node = _node(agent, _att("partner", face=False, hold="keep"), world, rng_np, agents=agents, ctx=StepContext(im=mgr))
    assert _tick(node) == SUCCESS
    assert _mv(agent).gesture == GestureIntent("point", 1.0, 0.0, GESTURE_Z_AGENT)
    near.state.pose.x = 3.0
    node = _node(agent, _att("partner", face=False, hold="keep"), world, rng_np, agents=agents, ctx=StepContext(im=mgr))
    assert _tick(node) == SUCCESS
    assert _mv(agent).gesture == GestureIntent("point", -2.0, 0.0, GESTURE_Z_AGENT)


def test_resolve_partner_waits_without_interaction(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    mgr = InteractionManager(RNG(0))
    node = _node(agent, _att("partner", face=False), world, rng_np, agents={1: agent}, ctx=StepContext(im=mgr))
    assert _tick(node) == RUNNING
    assert _mv(agent).gesture is None


def test_resolve_partners_cycles_all_others(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, agent_id=1)
    b = _bt_agent(agent_factory, agent_id=2, x=1.0, y=0.0)
    c = _bt_agent(agent_factory, agent_id=3, x=-2.0, y=0.0)
    agents = {1: agent, 2: b, 3: c}
    mgr = _bind(agents)
    node = _node(agent, _att("partners", face=False, dwell=0.5), world, rng_np, agents=agents, ctx=StepContext(im=mgr), bare=False)
    seen = []
    for _ in range(4):
        assert _tick(node) == RUNNING
        seen.append(_mv(agent).gesture.x)
    assert seen == [1.0, -2.0, 1.0, -2.0]


def test_resolve_relative_ref_follows_yaw(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, x=1.0, y=1.0, theta=0.0)
    node = _node(agent, _att(RelativeRef(azimuth=90.0, elevation=0.0, distance=2.0)), world, rng_np, duration=ParamDist(5.0))
    assert _tick(node) == RUNNING
    g = _mv(agent).gesture
    assert g is not None
    assert (g.x, g.y, g.z) == pytest.approx((1.0, 3.0, GESTURE_Z_AGENT))
    assert _mv(agent).heading_goal is None
    agent.state.pose.theta = math.pi / 2
    assert _tick(node) == RUNNING
    g = _mv(agent).gesture
    assert g is not None
    assert (g.x, g.y, g.z) == pytest.approx((-1.0, 1.0, GESTURE_Z_AGENT))
    assert _mv(agent).heading_goal is None


def test_resolve_relative_ref_elevation(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(RelativeRef(azimuth=0.0, elevation=30.0, distance=2.0), hold="keep"), world, rng_np)
    assert _tick(node) == SUCCESS
    g = _mv(agent).gesture
    assert g is not None
    assert (g.x, g.y, g.z) == pytest.approx((2.0 * math.cos(math.radians(30.0)), 0.0, GESTURE_Z_AGENT + 1.0))


# unresolved


@pytest.mark.parametrize("at", ["nothing_here", 99, "target", "goal"])
def test_unresolved_waits(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator, at: object) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(at), world, rng_np, bare=False)
    for _ in range(3):
        assert _tick(node) == RUNNING
        assert _mv(agent).gesture is None
        assert _mv(agent).heading_goal is None
    assert node._warned is True


def test_unresolved_then_resolved(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi / 2)
    agents: dict[int, BaseAgent] = {}
    node = _node(agent, _att(4, face=False), world, rng_np, agents=agents, duration=ParamDist(5.0))
    assert _tick(node) == RUNNING
    assert _mv(agent).gesture is None
    agents[4] = _bt_agent(agent_factory, agent_id=4, x=0.0, y=2.0)
    assert _tick(node) == RUNNING
    assert _mv(agent).gesture == GestureIntent("point", 0.0, 2.0, GESTURE_Z_AGENT)


def test_bare_unresolved_fails_after_timeout(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att("nobody"), world, rng_np)
    for _ in range(int(RESOLVE_TIMEOUT_S / DT)):
        assert _tick(node) == RUNNING
    assert _tick(node) == FAILURE
    assert _mv(agent).gesture is None


def test_rider_unresolved_waits_past_timeout(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att("nobody"), world, rng_np, bare=False)
    for _ in range(int(RESOLVE_TIMEOUT_S / DT) + 3):
        assert _tick(node) == RUNNING
    assert _mv(agent).gesture is None


def test_bare_unresolved_halts(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, agent_id=3, x=1.0, y=1.0)
    node = _node(agent, _att("nobody"), world, rng_np)
    assert _tick(node) == RUNNING
    cmd = _mv(agent).command
    assert cmd is not None
    assert cmd.type == CommandType.NAVIGATE
    assert cmd.agent_id == 3
    assert (cmd.target_pose.x, cmd.target_pose.y) == (1.0, 1.0)
    assert cmd.desired_velocity == 0.0


# face modes


def test_bare_face_auto_faces_then_raises(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, agent_id=3, x=1.0, y=1.0, theta=math.pi)
    node = _node(agent, _att("bench_1"), world, rng_np, duration=ParamDist(1.0))
    bearing = math.atan2(0.0 - 1.0, 5.0 - 1.0)
    for _ in range(2):
        _mv(agent).command = None
        assert _tick(node) == RUNNING
        assert _mv(agent).heading_goal == pytest.approx(bearing)
        assert _mv(agent).gesture is None
        cmd = _mv(agent).command
        assert cmd is not None
        assert cmd.type == CommandType.NAVIGATE
        assert cmd.desired_velocity == 0.0
    agent.state.pose.theta = bearing + 0.2
    assert _tick(node) == RUNNING
    assert _mv(agent).gesture == GestureIntent("point", 5.0, 0.0, GESTURE_Z_OBJECT)
    assert _mv(agent).heading_goal is None
    assert _tick(node) == RUNNING
    assert _tick(node) == SUCCESS
    assert _mv(agent).heading_goal is None
    assert _mv(agent).gesture is None


def test_rider_idle_face_auto_faces(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi)
    node = _node(agent, _att("bench_1"), world, rng_np, bare=False, idle=True)
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal == pytest.approx(0.0)
    assert _mv(agent).gesture is None
    assert _mv(agent).command is None


def test_rider_busy_face_auto_skips_facing(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi)
    node = _node(agent, _att("bench_1"), world, rng_np, bare=False)
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal is None
    assert _mv(agent).gesture == GestureIntent("point", 5.0, 0.0, GESTURE_Z_OBJECT)
    assert _mv(agent).command is None


def test_rider_busy_face_true_faces(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi)
    node = _node(agent, _att("bench_1", face=True), world, rng_np, bare=False)
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal == pytest.approx(0.0)
    assert _mv(agent).gesture is None
    agent.state.pose.theta = 0.1
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal is None
    assert _mv(agent).gesture == GestureIntent("point", 5.0, 0.0, GESTURE_Z_OBJECT)


def test_face_true_never_writes_heading_while_bound(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi)
    ctx = StepContext(is_bound_lookup=lambda _aid: True)
    node = _node(agent, _att("bench_1", face=True), world, rng_np, ctx=ctx, bare=False)
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal is None
    assert _mv(agent).gesture == GestureIntent("point", 5.0, 0.0, GESTURE_Z_OBJECT)


def test_bare_bound_keeps_formation_command_and_skips_facing(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi)
    sentinel = HighLevelCommand(agent_id=1, type=CommandType.NAVIGATE, target_pose=Pose2D(x=5.0, y=0.0))
    _mv(agent).command = sentinel
    ctx = StepContext(is_bound_lookup=lambda _aid: True)
    node = _node(agent, _att("bench_1"), world, rng_np, ctx=ctx, duration=ParamDist(1.0))
    assert _tick(node) == RUNNING
    assert _mv(agent).command is sentinel
    assert _mv(agent).heading_goal is None
    assert _mv(agent).gesture == GestureIntent("point", 5.0, 0.0, GESTURE_Z_OBJECT)


def test_face_false_raises_immediately(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi)
    node = _node(agent, _att("bench_1", face=False, hold="keep"), world, rng_np)
    assert _tick(node) == SUCCESS
    assert _mv(agent).heading_goal is None
    assert _mv(agent).gesture == GestureIntent("point", 5.0, 0.0, GESTURE_Z_OBJECT)


def test_face_hysteresis_enter_keep(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=FACE_ENTER_RAD + 0.05)
    other = _bt_agent(agent_factory, agent_id=2, x=5.0, y=0.0)
    node = _node(agent, _att(2), world, rng_np, agents={2: other}, duration=ParamDist(50.0))
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal == pytest.approx(0.0)
    assert _mv(agent).gesture is None
    agent.state.pose.theta = FACE_ENTER_RAD - 0.05
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal is None
    assert _mv(agent).gesture is not None
    agent.state.pose.theta = 0.0
    # target drifts inside the keep cone: no re-face
    other.state.pose.y = 5.0 * math.tan(FACE_KEEP_RAD - 0.05)
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal is None
    assert _mv(agent).gesture is not None
    # leaves the keep cone: re-face while still pointing
    other.state.pose.y = 5.0 * math.tan(FACE_KEEP_RAD + 0.1)
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal == pytest.approx(math.atan2(other.state.pose.y, 5.0))
    assert _mv(agent).gesture is not None
    # back inside keep but outside enter: still re-facing
    other.state.pose.y = 5.0 * math.tan(FACE_KEEP_RAD - 0.05)
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal == pytest.approx(math.atan2(other.state.pose.y, 5.0))
    # inside enter: done
    other.state.pose.y = 0.0
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal is None


def test_bare_face_timeout_fails(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi)
    node = _node(agent, _att("bench_1"), world, rng_np)
    for _ in range(int(FACE_TIMEOUT_S / DT)):
        assert _tick(node) == RUNNING
    assert _tick(node) == FAILURE
    assert _mv(agent).gesture is None
    assert _mv(agent).heading_goal is None


def test_rider_face_timeout_raises_anyway(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi)
    node = _node(agent, _att("bench_1", face=True), world, rng_np, bare=False)
    for _ in range(int(FACE_TIMEOUT_S / DT)):
        assert _tick(node) == RUNNING
        assert _mv(agent).gesture is None
        assert _mv(agent).heading_goal == pytest.approx(0.0)
    assert _tick(node) == RUNNING
    assert _mv(agent).gesture == GestureIntent("point", 5.0, 0.0, GESTURE_Z_OBJECT)
    assert _mv(agent).heading_goal is None
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal is None


# hold


def test_release_clears_on_terminate(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att("bench_1"), world, rng_np, duration=ParamDist(5.0))
    assert _tick(node) == RUNNING
    assert _mv(agent).gesture is not None
    node.stop(py_trees.common.Status.INVALID)
    assert _mv(agent).gesture is None
    assert _mv(agent).heading_goal is None


def test_keep_leaves_intent_clears_heading(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi)
    node = _node(agent, _att("bench_1", hold="keep"), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal is not None
    node.stop(py_trees.common.Status.INVALID)
    assert _mv(agent).heading_goal is None
    agent.state.pose.theta = 0.0
    node = _node(agent, _att("bench_1", hold="keep"), world, rng_np)
    assert _tick(node) == SUCCESS
    assert _mv(agent).gesture == GestureIntent("point", 5.0, 0.0, GESTURE_Z_OBJECT)
    assert _mv(agent).heading_goal is None


def test_face_leaves_predecessor_intent(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi)
    stale = GestureIntent("point", -5.0, 0.0, GESTURE_Z_OBJECT)
    _mv(agent).gesture = stale
    node = _node(agent, _att("bench_1", hold="keep"), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).gesture is stale
    agent.state.pose.theta = 0.0
    assert _tick(node) == SUCCESS
    assert _mv(agent).gesture == GestureIntent("point", 5.0, 0.0, GESTURE_Z_OBJECT)


def test_terminate_clears_heading_mid_face(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi)
    node = _node(agent, _att("bench_1", face=True), world, rng_np, bare=False)
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal is not None
    node.stop(py_trees.common.Status.INVALID)
    assert _mv(agent).heading_goal is None


# lists and dwell


def test_list_one_pass_sums_dwells(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(("bench_1", "bench_2"), face=False), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).gesture.x == 5.0
    assert _tick(node) == RUNNING
    assert _mv(agent).gesture.x == 5.0
    assert _tick(node) == RUNNING
    assert _mv(agent).gesture.x == -5.0
    assert _tick(node) == SUCCESS
    assert _mv(agent).gesture is None


def test_list_with_duration_cycles(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(("bench_1", "bench_2"), face=False, dwell=0.5), world, rng_np, duration=ParamDist(2.5))
    xs = []
    while _tick(node) == RUNNING:
        xs.append(_mv(agent).gesture.x)
    assert node.status == SUCCESS
    assert xs == [5.0, -5.0, 5.0, -5.0, 5.0]


def test_rider_list_cycles_forever(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att(("bench_1", "bench_2"), face=False, dwell=0.5), world, rng_np, bare=False)
    xs = []
    for _ in range(6):
        assert _tick(node) == RUNNING
        xs.append(_mv(agent).gesture.x)
    assert xs == [5.0, -5.0, 5.0, -5.0, 5.0, -5.0]


def test_list_face_only_before_first_raise(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi)
    node = _node(agent, _att(("bench_2", "bench_1"), dwell=0.5), world, rng_np, duration=ParamDist(3.0))
    assert _tick(node) == RUNNING
    assert _mv(agent).gesture.x == -5.0
    assert _mv(agent).heading_goal is None
    assert _tick(node) == RUNNING
    assert _mv(agent).gesture.x == 5.0
    assert _mv(agent).heading_goal == pytest.approx(0.0)


def test_single_target_ignores_dwell(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att("bench_1", face=False, dwell=0.5), world, rng_np, duration=ParamDist(2.0))
    for _ in range(4):
        assert _tick(node) == RUNNING
        assert _mv(agent).gesture.x == 5.0
    assert _tick(node) == SUCCESS


# intent fields


def test_hand_in_intent(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    node = _node(agent, _att("bench_1", hand="left", hold="keep"), world, rng_np)
    assert _tick(node) == SUCCESS
    assert _mv(agent).gesture == GestureIntent("point", 5.0, 0.0, GESTURE_Z_OBJECT, "left")


def test_gesture_none_clears_without_at(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    _mv(agent).gesture = GestureIntent("point", 1.0, 2.0, 3.0)
    node = _node(agent, AttentionDef(gesture="none"), world, rng_np)
    assert _tick(node) == SUCCESS
    assert _mv(agent).gesture is None


def test_gesture_none_with_at_only_faces(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory, theta=math.pi)
    _mv(agent).gesture = GestureIntent("point", 1.0, 2.0, 3.0)
    node = _node(agent, AttentionDef(gesture="none", at="bench_1", hold="keep"), world, rng_np)
    assert _tick(node) == RUNNING
    assert _mv(agent).heading_goal == pytest.approx(0.0)
    agent.state.pose.theta = 0.0
    assert _tick(node) == SUCCESS
    assert _mv(agent).gesture is None


def test_tracked_agent_target_follows(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    other = _bt_agent(agent_factory, agent_id=2, x=4.0, y=0.0)
    node = _node(agent, _att(2, face=False), world, rng_np, agents={2: other}, duration=ParamDist(5.0))
    assert _tick(node) == RUNNING
    assert _mv(agent).gesture == GestureIntent("point", 4.0, 0.0, GESTURE_Z_AGENT)
    other.state.pose.x = 4.5
    other.state.pose.y = 0.5
    assert _tick(node) == RUNNING
    assert _mv(agent).gesture == GestureIntent("point", 4.5, 0.5, GESTURE_Z_AGENT)


def test_tracked_agent_reresolves_after_respawn(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _bt_agent(agent_factory)
    agents = {2: _bt_agent(agent_factory, agent_id=2, x=4.0, y=0.0)}
    names = {"ped_2": 2}
    node = _node(agent, _att("ped_2", face=False), world, rng_np, agents=agents, names=names, duration=ParamDist(50.0))
    assert _tick(node) == RUNNING
    assert _mv(agent).gesture == GestureIntent("point", 4.0, 0.0, GESTURE_Z_AGENT)
    del agents[2]
    del names["ped_2"]
    _mv(agent).gesture = None
    assert _tick(node) == RUNNING
    assert _mv(agent).gesture is None
    agents[9] = _bt_agent(agent_factory, agent_id=9, x=0.0, y=6.0)
    names["ped_2"] = 9
    assert _tick(node) == RUNNING
    assert _mv(agent).gesture == GestureIntent("point", 0.0, 6.0, GESTURE_Z_AGENT)


def test_clear_gesture_node_clears_both(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = _bt_agent(agent_factory)
    _mv(agent).gesture = GestureIntent("point", 1.0, 2.0, 3.0)
    _mv(agent).heading_goal = 0.5
    node = ClearGestureNode("clear", agent)
    assert _tick(node) == SUCCESS
    assert _mv(agent).gesture is None
    assert _mv(agent).heading_goal is None
