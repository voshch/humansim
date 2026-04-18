from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("rclpy")
py_trees = pytest.importorskip("py_trees")

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.agents.types import (
    ActionDef,
    NeedCondition,
    ParamDist,
    SequenceDef,
    StepDef,
    TransitionDef,
)
from arena_humansim.core.behavior.nodes import (
    AutonomousNode,
    ConcreteStepNode,
    NeedsDecayNode,
    SequenceStateMachine,
    _at_target,
    _interaction_command,
    _nav_command,
    _resolve_interaction_radius,
    _sample_param_dist,
    check_condition,
    preconditions_met,
    score_actions,
)
from arena_humansim.core.interaction_manager import CommandType
from arena_humansim.core.world_knowledge import WorldKnowledge, WorldObject
from arena_humansim.utils.event_bus import EventBus
from arena_humansim.utils.types import (
    BehaviorTreeMovement,
    InteractionOutcome,
    InteractionType,
    NeedsState,
    NeedState,
    Pose2D,
)


@pytest.fixture
def rng_np() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def world() -> WorldKnowledge:
    return WorldKnowledge()


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


def _agent_with_bt(agent_factory: Callable[..., BaseAgent], agent_id: int = 1, x: float = 0.0, y: float = 0.0) -> BaseAgent:
    agent = agent_factory(agent_id=agent_id, x=x, y=y)
    agent.movement = BehaviorTreeMovement()
    return agent


def _mv(agent: BaseAgent) -> BehaviorTreeMovement:
    return cast(BehaviorTreeMovement, agent.movement)


def _needs(**kv: float) -> NeedsState:
    return NeedsState(needs={k: NeedState(value=v, decay_rate=1.0) for k, v in kv.items()})


def test_check_condition_below_bound() -> None:
    assert check_condition(10.0, NeedCondition(below=20.0)) is True
    assert check_condition(20.0, NeedCondition(below=20.0)) is False
    assert check_condition(30.0, NeedCondition(below=20.0)) is False


def test_check_condition_above_bound() -> None:
    assert check_condition(30.0, NeedCondition(above=20.0)) is True
    assert check_condition(20.0, NeedCondition(above=20.0)) is False
    assert check_condition(10.0, NeedCondition(above=20.0)) is False


def test_check_condition_in_range() -> None:
    cond = NeedCondition(above=10.0, below=50.0)
    assert check_condition(30.0, cond) is True
    assert check_condition(10.0, cond) is False
    assert check_condition(50.0, cond) is False


def test_preconditions_met_missing_need() -> None:
    needs = {"energy": NeedState(value=80.0)}
    when = {"hunger": NeedCondition(below=50.0)}
    assert preconditions_met(needs, when) is False


def test_preconditions_met_all_satisfied() -> None:
    needs = {"energy": NeedState(value=80.0), "hunger": NeedState(value=10.0)}
    when = {"energy": NeedCondition(above=50.0), "hunger": NeedCondition(below=20.0)}
    assert preconditions_met(needs, when) is True


def test_preconditions_met_empty_true() -> None:
    assert preconditions_met({}, {}) is True


def test_score_actions_filters_preconditions(world: WorldKnowledge) -> None:
    needs = {"energy": NeedState(value=80.0)}
    actions = {
        "rest": ActionDef(when={"energy": NeedCondition(below=20.0)}, satisfies={"energy": 50.0}),
        "eat": ActionDef(satisfies={"energy": 10.0}),
    }
    scored = score_actions(needs, actions, {}, world)
    assert [n for n, _ in scored] == ["eat"]


def test_score_actions_queue_penalty(world: WorldKnowledge) -> None:
    world.add_object(WorldObject(object_id="c1", type="chair", pose=Pose2D()))
    world.set_queue_length("c1", 4)

    needs = {"rest": NeedState(value=0.0)}
    actions = {
        "sit": ActionDef(target_object_type="chair", satisfies={"rest": 100.0}),
        "stand": ActionDef(satisfies={"rest": 100.0}),
    }
    scored = dict(score_actions(needs, actions, {}, world))
    assert scored["sit"] == pytest.approx(1.0 * (1.0 - 0.2))
    assert scored["stand"] == pytest.approx(1.0)
    assert scored["sit"] < scored["stand"]


def test_score_actions_queue_penalty_floor(world: WorldKnowledge) -> None:
    world.add_object(WorldObject(object_id="c1", type="chair", pose=Pose2D(), capacity=100))
    world.set_queue_length("c1", 100)

    needs = {"rest": NeedState(value=0.0)}
    actions = {"sit": ActionDef(target_object_type="chair", satisfies={"rest": 100.0})}
    scored = dict(score_actions(needs, actions, {}, world))
    assert scored["sit"] == pytest.approx(0.2)


def test_score_actions_zero_utility_excluded(world: WorldKnowledge) -> None:
    needs = {"energy": NeedState(value=100.0)}
    actions = {"noop": ActionDef(satisfies={"energy": 10.0})}
    scored = score_actions(needs, actions, {}, world)
    assert scored == []


def test_score_actions_sorted_desc(world: WorldKnowledge) -> None:
    needs = {"a": NeedState(value=0.0), "b": NeedState(value=0.0)}
    actions = {
        "low": ActionDef(satisfies={"a": 10.0}),
        "high": ActionDef(satisfies={"b": 80.0}),
        "mid": ActionDef(satisfies={"a": 40.0}),
    }
    scored = score_actions(needs, actions, {"a": 1.0, "b": 1.0}, world)
    names = [n for n, _ in scored]
    assert names == ["high", "mid", "low"]


def test_score_actions_missing_need_in_satisfies(world: WorldKnowledge) -> None:
    needs = {"a": NeedState(value=0.0)}
    actions = {"x": ActionDef(satisfies={"a": 50.0, "missing": 100.0})}
    scored = dict(score_actions(needs, actions, {}, world))
    assert scored["x"] == pytest.approx(0.5)


def test_sample_param_dist_zero_std(rng_np: np.random.Generator) -> None:
    d = ParamDist(mean=5.0, std=0.0, clip_low=0.0, clip_high=100.0)
    assert _sample_param_dist(d, rng_np) == pytest.approx(5.0)


def test_sample_param_dist_clips_low(rng_np: np.random.Generator) -> None:
    d = ParamDist(mean=-10.0, std=0.0, clip_low=0.0, clip_high=100.0)
    assert _sample_param_dist(d, rng_np) == pytest.approx(0.0)


def test_sample_param_dist_clips_high(rng_np: np.random.Generator) -> None:
    d = ParamDist(mean=500.0, std=0.0, clip_low=0.0, clip_high=100.0)
    assert _sample_param_dist(d, rng_np) == pytest.approx(100.0)


def test_nav_command_fields(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = _agent_with_bt(agent_factory, agent_id=7)
    target = Pose2D(x=3.0, y=4.0, theta=0.5)
    cmd = _nav_command(agent, target)
    assert cmd.agent_id == 7
    assert cmd.type == CommandType.NAVIGATE
    assert cmd.target_pose is target
    assert cmd.desired_velocity == pytest.approx(agent.state.desired_velocity)


def test_interaction_command_fields(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = _agent_with_bt(agent_factory, agent_id=9)
    cmd = _interaction_command(agent, "TALK_TO", target_agent=3, duration=2.5)
    assert cmd.agent_id == 9
    assert cmd.type == CommandType.ADVERTISE
    assert cmd.interaction_type == InteractionType.TALK_TO.value
    assert cmd.target_agent == 3
    assert cmd.interaction_duration == pytest.approx(2.5)
    assert cmd.desired_velocity == pytest.approx(agent.state.desired_velocity)


def test_at_target_within_tolerance(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = _agent_with_bt(agent_factory, agent_id=1, x=0.0, y=0.0)
    assert _at_target(agent, Pose2D(x=0.1, y=0.1)) is True


def test_at_target_outside_tolerance(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = _agent_with_bt(agent_factory, agent_id=1, x=0.0, y=0.0)
    assert _at_target(agent, Pose2D(x=5.0, y=0.0)) is False


def _concrete_node(
    agent: BaseAgent,
    world: WorldKnowledge,
    rng: np.random.Generator,
    *,
    duration: ParamDist | None = None,
    patience: ParamDist | None = None,
    interaction: str | None = None,
    target_object_type: str | None = None,
    target_object_id: str | None = None,
    satisfies: dict[str, float] | None = None,
    interaction_radius: float | None = None,
    dt: float = 0.5,
) -> ConcreteStepNode:
    step = StepDef(
        target_object_type=target_object_type,
        target_object_id=target_object_id,
        interaction=interaction,
        duration=duration,
        patience=patience,
        satisfies=satisfies or {},
        interaction_radius=interaction_radius,
    )
    node = ConcreteStepNode("step", step, agent, world, rng, dt=dt)
    node.initialise()
    return node


def test_concrete_target_object_id_resolves_to_specific_not_nearest(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, x=0.0, y=0.0)
    # near_atm is closest by distance; far_atm is farther but named explicitly
    world.add_object(WorldObject(object_id="near_atm", type="atm", pose=Pose2D(x=1.0, y=0.0)))
    world.add_object(WorldObject(object_id="far_atm", type="atm", pose=Pose2D(x=10.0, y=0.0)))
    node = _concrete_node(
        agent,
        world,
        rng_np,
        target_object_id="far_atm",
        patience=ParamDist(30.0),
    )
    node.update()
    cmd = _mv(agent).command
    assert cmd is not None
    assert cmd.type == CommandType.NAVIGATE
    assert cmd.target_pose.x == pytest.approx(10.0)


def test_concrete_interaction_completed_success(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory)
    agent.needs = _needs(hunger=50.0)
    node = _concrete_node(agent, world, rng_np, interaction="TALK_TO", satisfies={"hunger": 20.0})
    _mv(agent).last_outcome = InteractionOutcome.COMPLETED
    status = node.update()
    assert status == py_trees.common.Status.SUCCESS
    assert agent.needs is not None
    assert agent.needs.needs["hunger"].value == pytest.approx(70.0)


def test_concrete_interaction_interrupted_failure(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory)
    node = _concrete_node(agent, world, rng_np, interaction="TALK_TO")
    _mv(agent).last_outcome = InteractionOutcome.INTERRUPTED
    assert node.update() == py_trees.common.Status.FAILURE


def test_concrete_navigates_when_not_at_target(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, x=0.0, y=0.0)
    world.add_object(WorldObject(object_id="b1", type="bench", pose=Pose2D(x=10.0, y=0.0)))
    node = _concrete_node(
        agent,
        world,
        rng_np,
        target_object_type="bench",
        duration=ParamDist(1.0),
        patience=ParamDist(10.0),
    )
    status = node.update()
    assert status == py_trees.common.Status.RUNNING
    cmd = _mv(agent).command
    assert cmd is not None
    assert cmd.type == CommandType.NAVIGATE


def test_concrete_patience_triggers_failure_during_nav(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, x=0.0, y=0.0)
    world.add_object(WorldObject(object_id="b1", type="bench", pose=Pose2D(x=10.0, y=0.0)))
    node = _concrete_node(
        agent,
        world,
        rng_np,
        target_object_type="bench",
        patience=ParamDist(0.4),
        dt=0.5,
    )
    assert node.update() == py_trees.common.Status.FAILURE


def test_concrete_interaction_advertises_then_awaits_outcome(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, x=0.0, y=0.0)
    agent.needs = _needs(energy=30.0)
    node = _concrete_node(
        agent,
        world,
        rng_np,
        interaction="TALK_TO",
        patience=ParamDist(10.0),
        satisfies={"energy": 20.0},
        dt=0.5,
    )
    s1 = node.update()
    assert s1 == py_trees.common.Status.RUNNING
    cmd = _mv(agent).command
    assert cmd is not None
    assert cmd.type == CommandType.ADVERTISE

    assert node.update() == py_trees.common.Status.RUNNING

    _mv(agent).last_outcome = InteractionOutcome.COMPLETED
    assert node.update() == py_trees.common.Status.SUCCESS
    assert agent.needs is not None
    assert agent.needs.needs["energy"].value == pytest.approx(50.0)


def test_concrete_interaction_advertise_carries_sampled_duration(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, x=0.0, y=0.0)
    node = _concrete_node(
        agent,
        world,
        rng_np,
        interaction="TALK_TO",
        duration=ParamDist(3.0),
        patience=ParamDist(10.0),
        dt=0.5,
    )
    node.update()
    cmd = _mv(agent).command
    assert cmd is not None
    assert cmd.type == CommandType.ADVERTISE
    assert cmd.interaction_duration == pytest.approx(3.0)


def test_concrete_interaction_advertises_once_then_stays_silent(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, x=0.0, y=0.0)
    node = _concrete_node(
        agent,
        world,
        rng_np,
        interaction="TALK_TO",
        patience=ParamDist(10.0),
        dt=0.5,
    )
    assert node.update() == py_trees.common.Status.RUNNING
    first_cmd = _mv(agent).command
    assert first_cmd is not None
    assert first_cmd.type == CommandType.ADVERTISE

    _mv(agent).command = None
    assert node.update() == py_trees.common.Status.RUNNING
    assert _mv(agent).command is None


def test_interaction_step_navigates_before_advertising(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, x=10.0, y=10.0)
    world.add_object(WorldObject(object_id="atm1", type="atm", pose=Pose2D(x=0.0, y=0.0)))
    node = _concrete_node(
        agent,
        world,
        rng_np,
        interaction="USE",
        target_object_type="atm",
        patience=ParamDist(120.0),
        dt=0.5,
    )
    assert node.update() == py_trees.common.Status.RUNNING
    cmd = _mv(agent).command
    assert cmd is not None
    assert cmd.type == CommandType.NAVIGATE
    assert cmd.target_pose.x == pytest.approx(0.0)
    assert cmd.target_pose.y == pytest.approx(0.0)

    agent.state.pose.x = 0.1
    agent.state.pose.y = 0.0
    assert node.update() == py_trees.common.Status.RUNNING
    cmd = _mv(agent).command
    assert cmd is not None
    assert cmd.type == CommandType.ADVERTISE


def test_interaction_radius_cascade() -> None:
    type_default = 2.0  # TALK_TO
    step_plain = StepDef(interaction="TALK_TO")
    assert _resolve_interaction_radius(step_plain, None) == pytest.approx(type_default)

    obj_with_override = WorldObject(object_id="b", type="bench", pose=Pose2D(), interaction_radius=1.25)
    assert _resolve_interaction_radius(step_plain, obj_with_override) == pytest.approx(1.25)

    step_with_override = StepDef(interaction="TALK_TO", interaction_radius=0.75)
    assert _resolve_interaction_radius(step_with_override, obj_with_override) == pytest.approx(0.75)

    step_no_interaction = StepDef()
    assert _resolve_interaction_radius(step_no_interaction, None) == pytest.approx(0.5)


def test_concrete_interaction_local_duration_does_not_force_success(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, x=0.0, y=0.0)
    agent.needs = _needs(energy=30.0)
    node = _concrete_node(
        agent,
        world,
        rng_np,
        interaction="TALK_TO",
        duration=ParamDist(1.0),
        patience=ParamDist(10.0),
        satisfies={"energy": 20.0},
        dt=0.5,
    )
    for _ in range(4):
        assert node.update() == py_trees.common.Status.RUNNING
    assert agent.needs is not None
    assert agent.needs.needs["energy"].value == pytest.approx(30.0)


def test_concrete_no_interaction_no_duration_immediate_success(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory)
    agent.needs = _needs(rest=50.0)
    node = _concrete_node(agent, world, rng_np, satisfies={"rest": 10.0})
    assert node.update() == py_trees.common.Status.SUCCESS
    assert agent.needs is not None
    assert agent.needs.needs["rest"].value == pytest.approx(60.0)


def test_concrete_patience_without_duration_triggers_failure(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory)
    node = _concrete_node(
        agent,
        world,
        rng_np,
        interaction="TALK_TO",
        patience=ParamDist(0.4),
        dt=0.5,
    )
    assert node.update() == py_trees.common.Status.FAILURE


def _auto_node(
    agent: BaseAgent,
    world: WorldKnowledge,
    event_bus: EventBus,
    rng: np.random.Generator,
    *,
    step: StepDef | None = None,
    actions: dict[str, ActionDef] | None = None,
    weights: dict[str, float] | None = None,
    dt: float = 0.5,
) -> AutonomousNode:
    s = step or StepDef(autonomous=True)
    node = AutonomousNode(
        "auto",
        s,
        agent,
        actions or {},
        weights or {},
        world,
        event_bus,
        rng,
        dt=dt,
    )
    node.initialise()
    return node


def test_autonomous_allowed_filter(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(autonomous=True, allowed_actions=("keep",))
    actions = {"keep": ActionDef(), "drop": ActionDef()}
    node = _auto_node(agent_factory(agent_id=1), world, event_bus, rng_np, step=step, actions=actions)
    assert set(node._actions.keys()) == {"keep"}


def test_autonomous_blocked_filter(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(autonomous=True, blocked_actions=("drop",))
    actions = {"keep": ActionDef(), "drop": ActionDef()}
    node = _auto_node(agent_factory(agent_id=1), world, event_bus, rng_np, step=step, actions=actions)
    assert set(node._actions.keys()) == {"keep"}


def test_autonomous_default_no_filter(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    actions = {"a": ActionDef(), "b": ActionDef()}
    node = _auto_node(agent_factory(agent_id=1), world, event_bus, rng_np, actions=actions)
    assert set(node._actions.keys()) == {"a", "b"}


def test_autonomous_until_event_consumes_and_succeeds(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, agent_id=2)
    step = StepDef(autonomous=True, until="bell")
    event_bus.fire("bell", 2)
    node = _auto_node(agent, world, event_bus, rng_np, step=step)
    assert node.update() == py_trees.common.Status.SUCCESS
    assert event_bus.has("bell", 2) is False


def test_autonomous_until_need_satisfied(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, agent_id=1)
    agent.needs = _needs(energy=80.0)
    step = StepDef(autonomous=True, until_need={"energy": NeedCondition(above=50.0)})
    node = _auto_node(agent, world, event_bus, rng_np, step=step)
    assert node.update() == py_trees.common.Status.SUCCESS


def test_autonomous_duration_expiry(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, agent_id=1)
    step = StepDef(autonomous=True, duration=ParamDist(0.4))
    node = _auto_node(agent, world, event_bus, rng_np, step=step, dt=0.5)
    assert node.update() == py_trees.common.Status.RUNNING
    assert node.update() == py_trees.common.Status.SUCCESS


def test_autonomous_picks_best_target_object(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, agent_id=1, x=0.0, y=0.0)
    agent.needs = _needs(rest=0.0)
    world.add_object(WorldObject(object_id="c1", type="chair", pose=Pose2D(x=2.0, y=0.0)))

    actions = {"sit": ActionDef(target_object_type="chair", satisfies={"rest": 100.0})}
    node = _auto_node(agent, world, event_bus, rng_np, actions=actions)
    status = node.update()
    assert status == py_trees.common.Status.RUNNING
    cmd = _mv(agent).command
    assert cmd is not None
    assert cmd.type == CommandType.NAVIGATE
    assert cmd.target_pose.x == pytest.approx(2.0)


def test_autonomous_picks_best_interaction(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, agent_id=1)
    agent.needs = _needs(social=0.0)
    actions = {"chat": ActionDef(interaction="TALK_TO", satisfies={"social": 50.0})}
    node = _auto_node(agent, world, event_bus, rng_np, actions=actions)
    assert node.update() == py_trees.common.Status.RUNNING
    cmd = _mv(agent).command
    assert cmd is not None
    assert cmd.type == CommandType.ADVERTISE
    assert cmd.interaction_type == InteractionType.TALK_TO.value


def test_autonomous_no_scored_clears_command(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, agent_id=1)
    _mv(agent).command = _nav_command(agent, Pose2D(x=1.0, y=1.0))
    node = _auto_node(agent, world, event_bus, rng_np, actions={})
    status = node.update()
    assert status == py_trees.common.Status.RUNNING
    assert _mv(agent).command is None


def test_autonomous_best_action_no_target_no_interaction_clears_command(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    agent = _agent_with_bt(agent_factory, agent_id=1)
    agent.needs = _needs(rest=0.0)
    _mv(agent).command = _nav_command(agent, Pose2D(x=1.0, y=1.0))
    actions = {"idle": ActionDef(satisfies={"rest": 50.0})}
    node = _auto_node(agent, world, event_bus, rng_np, actions=actions)
    assert node.update() == py_trees.common.Status.RUNNING
    assert _mv(agent).command is None


def test_needs_decay_calls_decay(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = agent_factory(agent_id=1)
    agent.needs = _needs(energy=50.0)
    agent.needs.needs["energy"].decay_rate = 10.0
    node = NeedsDecayNode("decay", agent, dt=1.0)
    status = node.update()
    assert status == py_trees.common.Status.RUNNING
    assert agent.needs is not None
    assert agent.needs.needs["energy"].value == pytest.approx(40.0)


def test_needs_decay_no_needs_still_running(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = agent_factory(agent_id=1)
    agent.needs = None
    node = NeedsDecayNode("decay", agent, dt=1.0)
    assert node.update() == py_trees.common.Status.RUNNING


class _StubChild(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, status: Any) -> None:
        super().__init__(name=name)
        self.status_to_return = status
        self.initialised = 0
        self.terminated_with: list[Any] = []

    def initialise(self) -> None:
        self.initialised += 1

    def update(self) -> Any:
        return self.status_to_return

    def terminate(self, new_status: Any) -> None:
        self.terminated_with.append(new_status)


def test_state_machine_transition_fires_goto(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = agent_factory(agent_id=1)
    agent.needs = _needs(energy=5.0)
    a = _StubChild("a", py_trees.common.Status.RUNNING)
    b = _StubChild("b", py_trees.common.Status.RUNNING)
    seq_defs = {
        "a": SequenceDef(
            steps={"s0": StepDef()},
            transitions=(TransitionDef(when={"energy": NeedCondition(below=10.0)}, goto="b"),),
        ),
        "b": SequenceDef(steps={"s0": StepDef()}),
    }
    sm = SequenceStateMachine("sm", {"a": a, "b": b}, seq_defs, "a", agent)
    sm.initialise()
    assert sm.update() == py_trees.common.Status.RUNNING
    assert sm._current_name == "b"
    assert b.initialised >= 1
    assert py_trees.common.Status.FAILURE in a.terminated_with


def test_state_machine_success_chain_then(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = agent_factory(agent_id=1)
    a = _StubChild("a", py_trees.common.Status.SUCCESS)
    b = _StubChild("b", py_trees.common.Status.RUNNING)
    seq_defs = {
        "a": SequenceDef(steps={"s": StepDef()}, then="b"),
        "b": SequenceDef(steps={"s": StepDef()}),
    }
    sm = SequenceStateMachine("sm", {"a": a, "b": b}, seq_defs, "a", agent)
    sm.initialise()
    assert sm.update() == py_trees.common.Status.RUNNING
    assert sm._current_name == "b"
    assert py_trees.common.Status.SUCCESS in a.terminated_with


def test_state_machine_success_terminal(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = agent_factory(agent_id=1)
    a = _StubChild("a", py_trees.common.Status.SUCCESS)
    seq_defs = {"a": SequenceDef(steps={"s": StepDef()})}
    sm = SequenceStateMachine("sm", {"a": a}, seq_defs, "a", agent)
    sm.initialise()
    assert sm.update() == py_trees.common.Status.SUCCESS
    assert py_trees.common.Status.SUCCESS in a.terminated_with


def test_state_machine_failure_chain_on_failure(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = agent_factory(agent_id=1)
    a = _StubChild("a", py_trees.common.Status.FAILURE)
    b = _StubChild("b", py_trees.common.Status.RUNNING)
    seq_defs = {
        "a": SequenceDef(steps={"s": StepDef()}, on_failure="b"),
        "b": SequenceDef(steps={"s": StepDef()}),
    }
    sm = SequenceStateMachine("sm", {"a": a, "b": b}, seq_defs, "a", agent)
    sm.initialise()
    assert sm.update() == py_trees.common.Status.RUNNING
    assert sm._current_name == "b"


def test_state_machine_failure_terminal(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = agent_factory(agent_id=1)
    a = _StubChild("a", py_trees.common.Status.FAILURE)
    seq_defs = {"a": SequenceDef(steps={"s": StepDef()})}
    sm = SequenceStateMachine("sm", {"a": a}, seq_defs, "a", agent)
    sm.initialise()
    assert sm.update() == py_trees.common.Status.FAILURE


def test_state_machine_running_passthrough(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = agent_factory(agent_id=1)
    a = _StubChild("a", py_trees.common.Status.RUNNING)
    seq_defs = {"a": SequenceDef(steps={"s": StepDef()})}
    sm = SequenceStateMachine("sm", {"a": a}, seq_defs, "a", agent)
    sm.initialise()
    assert sm.update() == py_trees.common.Status.RUNNING
    assert sm._current_name == "a"


def test_state_machine_goto_unknown_fails(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = agent_factory(agent_id=1)
    agent.needs = _needs(energy=5.0)
    a = _StubChild("a", py_trees.common.Status.RUNNING)
    seq_defs = {
        "a": SequenceDef(
            steps={"s": StepDef()},
            transitions=(TransitionDef(when={"energy": NeedCondition(below=10.0)}, goto="nowhere"),),
        ),
    }
    sm = SequenceStateMachine("sm", {"a": a}, seq_defs, "a", agent)
    sm.initialise()
    assert sm.update() == py_trees.common.Status.FAILURE


def test_state_machine_terminate_forwards_to_current(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = agent_factory(agent_id=1)
    a = _StubChild("a", py_trees.common.Status.RUNNING)
    seq_defs = {"a": SequenceDef(steps={"s": StepDef()})}
    sm = SequenceStateMachine("sm", {"a": a}, seq_defs, "a", agent)
    sm.initialise()
    sm.terminate(py_trees.common.Status.INVALID)
    assert py_trees.common.Status.INVALID in a.terminated_with


def test_pool_kind_filter_excludes_robots(agent_factory: Callable[..., BaseAgent]) -> None:
    from arena_humansim.core.pool import AgentPool, human_mask, is_human

    pool = AgentPool(capacity=4)
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.add_agent(agent_factory(agent_id=2, x=1.0, y=0.0))
    pool.kind[pool.idx(2)] = 1

    mask = human_mask(pool)
    assert mask.tolist() == [True, False]
    assert is_human(pool, pool.idx(1)) is True
    assert is_human(pool, pool.idx(2)) is False
