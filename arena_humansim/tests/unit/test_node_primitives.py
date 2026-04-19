from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

pytest.importorskip("rclpy")
py_trees = pytest.importorskip("py_trees")

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.agents.types import ParamDist
from arena_humansim.core.behavior.nodes import (
    AcceptInteractionNode,
    AdvertiseInteractionNode,
    BlockNode,
    ClearOutcomeNode,
    GoToNode,
    HoldNode,
    PatienceWatchdogNode,
    ResolveObjectNode,
    SatisfyNode,
    _resolve_interaction_radius,
)
from arena_humansim.core.behavior.step_context import StepContext
from arena_humansim.core.interaction_manager import CommandType
from arena_humansim.core.world_knowledge import FormationSpec, WorldKnowledge, WorldObject
from arena_humansim.utils import DISTANCE_TOLERANCE
from arena_humansim.utils.types import (
    BehaviorTreeMovement,
    HighLevelCommand,
    InteractionOutcome,
    NeedsState,
    NeedState,
    Pose2D,
)


@pytest.fixture(autouse=True)
def _clear_blackboard() -> None:
    py_trees.blackboard.Blackboard.clear()


@pytest.fixture
def rng_np() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def world() -> WorldKnowledge:
    return WorldKnowledge()


def _agent_with_bt(agent_factory: Callable[..., BaseAgent], agent_id: int = 1, x: float = 0.0, y: float = 0.0) -> BaseAgent:
    agent = agent_factory(agent_id=agent_id, x=x, y=y)
    agent.movement = BehaviorTreeMovement()
    return agent


def _mv(agent: BaseAgent) -> BehaviorTreeMovement:
    return cast(BehaviorTreeMovement, agent.movement)


def _needs(**kv: float) -> NeedsState:
    return NeedsState(needs={k: NeedState(value=v, decay_rate=1.0) for k, v in kv.items()})


def test_go_to_literal_not_at_target_reemits_every_tick(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = _agent_with_bt(agent_factory, x=0.0, y=0.0)
    node = GoToNode("go", agent, target_pose=Pose2D(x=10.0, y=0.0))
    node.setup()
    for _ in range(10):
        _mv(agent).command = None
        status = _tick(node)
        assert status == py_trees.common.Status.RUNNING
        cmd = _mv(agent).command
        assert cmd is not None
        assert cmd.type == CommandType.NAVIGATE
        assert cmd.target_pose.x == pytest.approx(10.0)


def test_go_to_literal_at_target_returns_success_without_command(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = _agent_with_bt(agent_factory, x=5.0, y=5.0)
    sentinel = HighLevelCommand(agent_id=agent.state.agent_id, type=CommandType.STOP, interaction_target=-1)
    _mv(agent).command = sentinel
    node = GoToNode("go", agent, target_pose=Pose2D(x=5.0, y=5.0))
    node.setup()
    status = _tick(node)
    assert status == py_trees.common.Status.SUCCESS
    assert _mv(agent).command is sentinel


def test_go_to_ctx_mode_reads_target_from_context(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = _agent_with_bt(agent_factory, x=0.0, y=0.0)
    ctx = StepContext(target_pose=Pose2D(x=7.0, y=0.0))

    node = GoToNode("go", agent, ctx=ctx)
    status = _tick(node)
    assert status == py_trees.common.Status.RUNNING
    cmd = _mv(agent).command
    assert cmd is not None
    assert cmd.type == CommandType.NAVIGATE
    assert cmd.target_pose.x == pytest.approx(7.0)


def test_go_to_recomputes_line_slot_each_tick(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge) -> None:
    agent = _agent_with_bt(agent_factory, x=0.0, y=0.0)
    world.add_object(
        WorldObject(
            object_id="f1",
            type="fountain",
            pose=Pose2D(x=4.0, y=0.0, theta=np.pi),
            formation=FormationSpec(type="line", params={"front_offset": 0.8, "base_step": 1.0}),
        )
    )
    ctx = StepContext(target_pose=Pose2D(x=4.8, y=0.0), target_object_id="f1")
    node = GoToNode("go", agent, ctx=ctx, world=world)
    _tick(node)
    assert ctx.target_pose is not None
    assert ctx.target_pose.x == pytest.approx(4.8)

    world.set_participants_count("f1", 1)
    world.set_queue_length("f1", 2)
    _tick(node)
    assert ctx.target_pose is not None
    assert ctx.target_pose.x == pytest.approx(7.8)


def test_resolve_object_present_success_populates_context(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge) -> None:
    agent = _agent_with_bt(agent_factory, x=0.0, y=0.0)
    world.add_object(WorldObject(object_id="b1", type="bench", pose=Pose2D(x=3.0, y=4.0)))
    ctx = StepContext()
    node = ResolveObjectNode(
        "resolve",
        agent,
        world,
        target_object_type="bench",
        target_object_id=None,
        ctx=ctx,
    )
    status = _tick(node)
    assert status == py_trees.common.Status.SUCCESS
    assert ctx.target_pose is not None
    assert ctx.target_pose.x == pytest.approx(3.0)
    assert ctx.target_pose.y == pytest.approx(4.0)
    assert ctx.target_object_id == "b1"


def test_resolve_object_line_formation_empty_targets_slot_zero(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge) -> None:
    agent = _agent_with_bt(agent_factory)
    world.add_object(
        WorldObject(
            object_id="f1",
            type="fountain",
            pose=Pose2D(x=4.0, y=0.0, theta=np.pi),
            formation=FormationSpec(type="line", params={"front_offset": 0.8, "base_step": 1.0}),
        )
    )
    ctx = StepContext()
    node = ResolveObjectNode("resolve", agent, world, target_object_type="fountain", target_object_id=None, ctx=ctx)
    _tick(node)
    assert ctx.target_pose is not None
    assert ctx.target_pose.x == pytest.approx(4.8)
    assert ctx.target_pose.y == pytest.approx(0.0)


def test_resolve_object_line_formation_occupied_targets_next_slot(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge) -> None:
    agent = _agent_with_bt(agent_factory)
    world.add_object(
        WorldObject(
            object_id="f1",
            type="fountain",
            pose=Pose2D(x=4.0, y=0.0, theta=np.pi),
            formation=FormationSpec(type="line", params={"front_offset": 0.8, "base_step": 1.0}),
        )
    )
    world.set_participants_count("f1", 1)
    world.set_queue_length("f1", 2)
    ctx = StepContext()
    node = ResolveObjectNode("resolve", agent, world, target_object_type="fountain", target_object_id=None, ctx=ctx)
    _tick(node)
    assert ctx.target_pose is not None
    assert ctx.target_pose.x == pytest.approx(7.8)
    assert ctx.target_pose.y == pytest.approx(0.0)


def test_resolve_object_missing_failure(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge) -> None:
    agent = _agent_with_bt(agent_factory)
    ctx = StepContext()
    node = ResolveObjectNode(
        "resolve",
        agent,
        world,
        target_object_type="ghost",
        target_object_id=None,
        ctx=ctx,
    )
    status = _tick(node)
    assert status == py_trees.common.Status.FAILURE


def test_resolve_interaction_radius_cascade() -> None:
    # step override beats everything
    obj_with_radius = WorldObject(object_id="y", type="bench", pose=Pose2D(), interaction_radius=1.25)
    assert _resolve_interaction_radius(obj_with_radius, step_override=0.75, interaction_name="TALK_TO") == pytest.approx(0.75)
    # object radius beats type default
    assert _resolve_interaction_radius(obj_with_radius, step_override=None, interaction_name="TALK_TO") == pytest.approx(1.25)
    # type default when object has no radius
    obj_plain = WorldObject(object_id="x", type="bench", pose=Pose2D())
    assert _resolve_interaction_radius(obj_plain, step_override=None, interaction_name="TALK_TO") == pytest.approx(2.0)
    # DISTANCE_TOLERANCE when nothing else
    assert _resolve_interaction_radius(obj_plain, step_override=None, interaction_name=None) == pytest.approx(DISTANCE_TOLERANCE)


def test_resolve_object_writes_type_default_radius_to_context(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge) -> None:
    agent = _agent_with_bt(agent_factory)
    world.add_object(WorldObject(object_id="t1", type="talker", pose=Pose2D(x=1.0, y=0.0)))
    ctx = StepContext()
    node = ResolveObjectNode(
        "resolve",
        agent,
        world,
        target_object_type="talker",
        target_object_id=None,
        ctx=ctx,
        interaction_name="TALK_TO",
    )
    _tick(node)
    assert ctx.interaction_radius == pytest.approx(2.0)


def _tick(node: py_trees.behaviour.Behaviour) -> py_trees.common.Status:
    node.tick_once()
    return node.status


def test_advertise_first_update_emits_and_sets_advertised(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, agent_id=3)
    ctx = StepContext(target_pose=Pose2D(x=1.0, y=0.0), target_object_id="obj1")
    node = AdvertiseInteractionNode(
        "advertise",
        agent,
        interaction="TALK_TO",
        ctx=ctx,
        duration_source=ParamDist(2.0),
        rng=rng_np,
    )
    status = _tick(node)
    assert status == py_trees.common.Status.RUNNING
    cmd = _mv(agent).command
    assert cmd is not None
    assert cmd.type == CommandType.ADVERTISE
    assert cmd.interaction_duration == pytest.approx(2.0)
    assert cmd.object_id == "obj1"
    assert ctx.advertised is True


@pytest.mark.parametrize("final_status", [py_trees.common.Status.FAILURE, py_trees.common.Status.SUCCESS])
def test_advertise_terminate_emits_stop_to_release_participant_slot(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator, final_status: py_trees.common.Status) -> None:
    agent = _agent_with_bt(agent_factory, agent_id=5)
    ctx = StepContext(target_pose=Pose2D())
    node = AdvertiseInteractionNode(
        "advertise",
        agent,
        interaction="TALK_TO",
        ctx=ctx,
        duration_source=None,
        rng=rng_np,
    )
    node.tick_once()
    cmd_after_tick = _mv(agent).command
    assert cmd_after_tick is not None
    assert cmd_after_tick.type == CommandType.ADVERTISE

    node.terminate(final_status)
    cmd = _mv(agent).command
    assert cmd is not None
    assert cmd.type == CommandType.STOP
    assert cmd.agent_id == 5
    assert cmd.interaction_target == -1


def test_advertise_completed_outcome_returns_success(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory)
    ctx = StepContext(target_pose=Pose2D())
    node = AdvertiseInteractionNode(
        "advertise",
        agent,
        interaction="TALK_TO",
        ctx=ctx,
        duration_source=None,
        rng=rng_np,
    )
    node.tick_once()
    _mv(agent).last_outcome = InteractionOutcome.COMPLETED
    status = _tick(node)
    assert status == py_trees.common.Status.SUCCESS
    assert _mv(agent).last_outcome is None


def test_advertise_interrupted_outcome_returns_failure(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory)
    ctx = StepContext(target_pose=Pose2D())
    node = AdvertiseInteractionNode(
        "advertise",
        agent,
        interaction="TALK_TO",
        ctx=ctx,
        duration_source=None,
        rng=rng_np,
    )
    node.tick_once()
    _mv(agent).last_outcome = InteractionOutcome.INTERRUPTED
    status = _tick(node)
    assert status == py_trees.common.Status.FAILURE
    assert _mv(agent).last_outcome is None


def test_hold_no_duration_immediate_success_no_emission(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory)
    sentinel = HighLevelCommand(agent_id=agent.state.agent_id, type=CommandType.NAVIGATE, interaction_target=-1)
    _mv(agent).command = sentinel
    node = HoldNode("hold", agent, duration_source=None, rng=rng_np, dt=0.5)
    status = _tick(node)
    assert status == py_trees.common.Status.SUCCESS
    assert _mv(agent).command is sentinel


def test_hold_with_duration_emits_stop_until_elapsed(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, agent_id=9)
    node = HoldNode("hold", agent, duration_source=ParamDist(1.0), rng=rng_np, dt=0.5)
    node.setup()

    for _ in range(3):
        _mv(agent).command = None
        status = _tick(node)
        cmd = _mv(agent).command
        assert cmd is not None
        assert cmd.type == CommandType.STOP
        assert cmd.agent_id == 9
        if status == py_trees.common.Status.SUCCESS:
            break
    else:
        pytest.fail("HoldNode never returned SUCCESS within expected ticks")


def test_satisfy_applies_deltas_and_returns_success(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = _agent_with_bt(agent_factory)
    agent.needs = _needs(hunger=50.0)
    node = SatisfyNode("satisfy", agent, satisfies={"hunger": 20.0})
    status = _tick(node)
    assert status == py_trees.common.Status.SUCCESS
    assert agent.needs is not None
    assert agent.needs.needs["hunger"].value == pytest.approx(70.0)


def test_satisfy_empty_dict_still_succeeds(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = _agent_with_bt(agent_factory)
    agent.needs = _needs(rest=30.0)
    node = SatisfyNode("satisfy", agent, satisfies={})
    assert _tick(node) == py_trees.common.Status.SUCCESS
    assert agent.needs.needs["rest"].value == pytest.approx(30.0)


def test_patience_watchdog_none_runs_forever(rng_np: np.random.Generator) -> None:
    node = PatienceWatchdogNode("watchdog", patience_source=None, rng=rng_np, dt=0.5)
    node.setup()
    for _ in range(100):
        status = _tick(node)
        assert status == py_trees.common.Status.RUNNING


def test_patience_watchdog_fails_once_elapsed_crosses_patience(rng_np: np.random.Generator) -> None:
    node = PatienceWatchdogNode("watchdog", patience_source=ParamDist(1.0), rng=rng_np, dt=0.5)
    node.setup()
    statuses = [_tick(node) for _ in range(5)]
    failure_indices = [i for i, s in enumerate(statuses) if s == py_trees.common.Status.FAILURE]
    assert len(failure_indices) >= 1
    for s in statuses[: failure_indices[0]]:
        assert s == py_trees.common.Status.RUNNING
    assert py_trees.common.Status.SUCCESS not in statuses


def test_clear_outcome_resets_from_completed(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = _agent_with_bt(agent_factory)
    _mv(agent).last_outcome = InteractionOutcome.COMPLETED
    node = ClearOutcomeNode("clear", agent)
    status = _tick(node)
    assert status == py_trees.common.Status.SUCCESS
    assert _mv(agent).last_outcome is None


def test_clear_outcome_no_prior_outcome(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = _agent_with_bt(agent_factory)
    _mv(agent).last_outcome = None
    node = ClearOutcomeNode("clear", agent)
    status = _tick(node)
    assert status == py_trees.common.Status.SUCCESS
    assert _mv(agent).last_outcome is None


def test_accept_first_update_emits_bare_advertise(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, agent_id=11)
    node = AcceptInteractionNode(
        "accept",
        agent,
        interaction="SERVICE",
        duration_source=None,
        rng=rng_np,
        service_tag="water",
    )
    status = _tick(node)
    assert status == py_trees.common.Status.RUNNING
    cmd = _mv(agent).command
    assert cmd is not None
    assert cmd.type == CommandType.ADVERTISE
    assert cmd.service_tag == "water"
    assert cmd.object_id is None
    assert cmd.target_agent == -1


def test_accept_completed_outcome_returns_success(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory)
    node = AcceptInteractionNode("accept", agent, interaction="TALK_TO", duration_source=None, rng=rng_np)
    node.tick_once()
    _mv(agent).last_outcome = InteractionOutcome.COMPLETED
    assert _tick(node) == py_trees.common.Status.SUCCESS
    assert _mv(agent).last_outcome is None


def test_accept_interrupted_outcome_returns_failure(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory)
    node = AcceptInteractionNode("accept", agent, interaction="TALK_TO", duration_source=None, rng=rng_np)
    node.tick_once()
    _mv(agent).last_outcome = InteractionOutcome.INTERRUPTED
    assert _tick(node) == py_trees.common.Status.FAILURE
    assert _mv(agent).last_outcome is None


def test_accept_terminate_emits_stop(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, agent_id=13)
    node = AcceptInteractionNode("accept", agent, interaction="TALK_TO", duration_source=None, rng=rng_np)
    node.tick_once()
    node.terminate(py_trees.common.Status.SUCCESS)
    cmd = _mv(agent).command
    assert cmd is not None
    assert cmd.type == CommandType.STOP
    assert cmd.agent_id == 13
    assert cmd.interaction_target == -1


def test_accept_duration_sampled_on_initialise(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory)
    node = AcceptInteractionNode("accept", agent, interaction="TALK_TO", duration_source=ParamDist(3.5), rng=rng_np)
    _tick(node)
    cmd = _mv(agent).command
    assert cmd is not None
    assert cmd.interaction_duration == pytest.approx(3.5)


def _block_node(agent: BaseAgent, target_lookup: Callable[[int], BaseAgent | None], rng: np.random.Generator, target_id: int = 99, **kwargs: object) -> BlockNode:
    node = BlockNode(
        "block",
        agent,
        target_agent_id=target_id,
        agent_lookup=target_lookup,
        duration_source=kwargs.pop("duration_source", None),
        rng=rng,
        **kwargs,  # type: ignore[arg-type]
    )
    node.initialise()
    return node


def test_block_missing_target_returns_failure(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory)
    node = _block_node(agent, lambda _aid: None, rng_np)
    assert _tick(node) == py_trees.common.Status.FAILURE


def test_block_pursuit_emits_navigate_to_predicted_pose(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, agent_id=1, x=0.0, y=0.0)
    target = agent_factory(agent_id=99, x=10.0, y=0.0)
    target.state.velocity = (2.0, 0.5)
    target.state.desired_velocity = 1.0

    node = _block_node(agent, lambda aid: target if aid == 99 else None, rng_np, target_id=99, lookahead=1.0)
    status = _tick(node)

    assert status == py_trees.common.Status.RUNNING
    cmd = _mv(agent).command
    assert cmd is not None
    assert cmd.type == CommandType.NAVIGATE
    assert cmd.target_pose.x == pytest.approx(12.0)
    assert cmd.target_pose.y == pytest.approx(0.5)


def test_block_boosts_desired_velocity_during_pursuit(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, x=0.0, y=0.0)
    target = agent_factory(agent_id=99, x=10.0, y=0.0)
    target.state.desired_velocity = 1.0

    node = _block_node(agent, lambda aid: target if aid == 99 else None, rng_np, velocity_boost=1.5)
    _tick(node)
    assert agent.state.desired_velocity == pytest.approx(1.5)


def test_block_within_tolerance_emits_advertise_once(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, agent_id=1, x=0.0, y=0.0)
    target = agent_factory(agent_id=99, x=0.1, y=0.0)
    target.state.velocity = (0.0, 0.0)

    node = _block_node(agent, lambda aid: target if aid == 99 else None, rng_np, duration_source=ParamDist(4.0))
    status = _tick(node)

    assert status == py_trees.common.Status.RUNNING
    cmd = _mv(agent).command
    assert cmd is not None
    assert cmd.type == CommandType.ADVERTISE
    assert cmd.target_agent == 99
    assert cmd.interaction_duration == pytest.approx(4.0)


def test_block_completed_outcome_returns_success(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, x=0.0, y=0.0)
    target = agent_factory(agent_id=99, x=0.0, y=0.0)
    node = _block_node(agent, lambda aid: target if aid == 99 else None, rng_np)
    node.tick_once()
    _mv(agent).last_outcome = InteractionOutcome.COMPLETED
    assert _tick(node) == py_trees.common.Status.SUCCESS
    assert _mv(agent).last_outcome is None


def test_block_interrupted_outcome_returns_failure(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, x=0.0, y=0.0)
    target = agent_factory(agent_id=99, x=0.0, y=0.0)
    node = _block_node(agent, lambda aid: target if aid == 99 else None, rng_np)
    node.tick_once()
    _mv(agent).last_outcome = InteractionOutcome.INTERRUPTED
    assert _tick(node) == py_trees.common.Status.FAILURE


def test_block_terminate_restores_velocity_and_emits_stop(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, agent_id=7, x=0.0, y=0.0)
    agent.state.desired_velocity = 0.9
    target = agent_factory(agent_id=99, x=0.0, y=0.0)
    target.state.desired_velocity = 1.2

    node = _block_node(agent, lambda aid: target if aid == 99 else None, rng_np)
    node.tick_once()
    assert agent.state.desired_velocity != pytest.approx(0.9)

    node.terminate(py_trees.common.Status.SUCCESS)
    assert agent.state.desired_velocity == pytest.approx(0.9)
    cmd = _mv(agent).command
    assert cmd is not None
    assert cmd.type == CommandType.STOP
    assert cmd.agent_id == 7
