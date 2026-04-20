from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

py_trees = pytest.importorskip("py_trees")

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.agents.types import (
    AgentType,
    GoToStepDef,
    NeedDist,
    ParamDist,
    SequenceDef,
    StepDef,
)
from arena_humansim.core.behavior.compiler import BehaviorTreeFactory
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
    SequenceStateMachine,
)
from arena_humansim.core.world_knowledge import WorldKnowledge
from arena_humansim.utils.event_bus import EventBus
from arena_humansim.utils.types import Pose2D


@pytest.fixture
def rng_np() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def world() -> WorldKnowledge:
    return WorldKnowledge()


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


def _agent_type(sequences: dict[str, SequenceDef], *, needs: dict[str, NeedDist] | None = None, initial: str = "default") -> AgentType:
    return AgentType(
        name="test_agent",
        mode="behavior_tree",
        sequences=sequences,
        initial_sequence=initial,
        needs=needs if needs is not None else {},
    )


def _compiled_root(
    agent_type: AgentType,
    agent: BaseAgent,
    world: WorldKnowledge,
    event_bus: EventBus,
    rng: np.random.Generator,
    agent_lookup: Callable[[int], BaseAgent | None] | None = None,
) -> py_trees.behaviour.Behaviour:
    factory = BehaviorTreeFactory(agent_type)
    bt = factory.build(agent, world, event_bus, rng, 0.05, agent_lookup=agent_lookup)
    sm = bt.root
    assert isinstance(sm, SequenceStateMachine)
    return sm._sequences["default"]


def _assert_outer_shape(root: py_trees.behaviour.Behaviour, seq: str, step: str) -> tuple[PatienceWatchdogNode, py_trees.composites.Sequence]:
    assert isinstance(root, py_trees.composites.Parallel)
    assert root.name == f"{seq}/{step}"
    assert isinstance(root.policy, py_trees.common.ParallelPolicy.SuccessOnOne)
    assert len(root.children) == 2
    watchdog, inner = root.children
    assert isinstance(watchdog, PatienceWatchdogNode)
    assert watchdog.name == f"{seq}/{step}/watchdog"
    assert isinstance(inner, py_trees.composites.Sequence)
    assert inner.memory is True
    assert inner.name == f"{seq}/{step}/sequence"
    return watchdog, inner


def test_go_to_step_shape(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = GoToStepDef(
        target_pose=Pose2D(x=5.0, y=5.0),
        duration=ParamDist(2.0),
        satisfies={"energy": 10.0},
    )
    sequences = {"default": SequenceDef(steps={"walk": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=1)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np)
    _, inner = _assert_outer_shape(root, "default", "walk")

    child_types = [type(c) for c in inner.children]
    assert child_types == [ClearOutcomeNode, GoToNode, HoldNode, SatisfyNode]
    assert [c.name for c in inner.children] == [
        "default/walk/clear_outcome",
        "default/walk/go_to",
        "default/walk/hold",
        "default/walk/satisfy",
    ]


def test_go_to_step_shape_minimal(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = GoToStepDef(target_pose=Pose2D(x=1.0, y=2.0))
    sequences = {"default": SequenceDef(steps={"walk2": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=2)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np)
    _, inner = _assert_outer_shape(root, "default", "walk2")

    assert [type(c) for c in inner.children] == [ClearOutcomeNode, GoToNode]


def test_object_interact_interaction_shape(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(
        target_object_type="chair",
        interaction="sit_on",
        duration=ParamDist(3.0),
        satisfies={"rest": 5.0},
    )
    sequences = {"default": SequenceDef(steps={"sit": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=3)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np)
    _, inner = _assert_outer_shape(root, "default", "sit")

    child_types = [type(c) for c in inner.children]
    assert child_types == [ClearOutcomeNode, ResolveObjectNode, GoToNode, AdvertiseInteractionNode, SatisfyNode]
    assert [c.name for c in inner.children] == [
        "default/sit/clear_outcome",
        "default/sit/resolve_object",
        "default/sit/go_to",
        "default/sit/advertise",
        "default/sit/satisfy",
    ]


def test_object_interact_interaction_no_satisfy(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(
        target_object_type="chair",
        interaction="sit_on",
        duration=ParamDist(3.0),
    )
    sequences = {"default": SequenceDef(steps={"sit2": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=4)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np)
    _, inner = _assert_outer_shape(root, "default", "sit2")

    assert [type(c) for c in inner.children] == [ClearOutcomeNode, ResolveObjectNode, GoToNode, AdvertiseInteractionNode]


def test_floating_advertise_has_no_resolve_or_goto(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(
        interaction="FOLLOW",
        duration=ParamDist(5.0),
        satisfies={"company": 1.0},
    )
    sequences = {"default": SequenceDef(steps={"lead": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=20)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np)
    _, inner = _assert_outer_shape(root, "default", "lead")

    assert [type(c) for c in inner.children] == [ClearOutcomeNode, AdvertiseInteractionNode, SatisfyNode]
    assert [c.name for c in inner.children] == [
        "default/lead/clear_outcome",
        "default/lead/advertise",
        "default/lead/satisfy",
    ]


def test_floating_advertise_minimal(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(interaction="FOLLOW")
    sequences = {"default": SequenceDef(steps={"lead2": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=21)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np)
    _, inner = _assert_outer_shape(root, "default", "lead2")

    assert [type(c) for c in inner.children] == [ClearOutcomeNode, AdvertiseInteractionNode]


def test_object_nav_only_shape(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(
        target_object_type="fountain",
        duration=ParamDist(1.5),
        satisfies={"thirst": 2.0},
    )
    sequences = {"default": SequenceDef(steps={"visit": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=5)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np)
    _, inner = _assert_outer_shape(root, "default", "visit")

    child_types = [type(c) for c in inner.children]
    assert child_types == [ClearOutcomeNode, ResolveObjectNode, GoToNode, HoldNode, SatisfyNode]
    assert [c.name for c in inner.children] == [
        "default/visit/clear_outcome",
        "default/visit/resolve_object",
        "default/visit/go_to",
        "default/visit/hold",
        "default/visit/satisfy",
    ]


def test_object_nav_only_no_duration_no_satisfy(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(target_object_type="fountain")
    sequences = {"default": SequenceDef(steps={"visit2": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=6)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np)
    _, inner = _assert_outer_shape(root, "default", "visit2")

    assert [type(c) for c in inner.children] == [ClearOutcomeNode, ResolveObjectNode, GoToNode]


def test_pure_wait_shape(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(duration=ParamDist(2.0), satisfies={"boredom": 1.0})
    sequences = {"default": SequenceDef(steps={"wait": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=7)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np)
    _, inner = _assert_outer_shape(root, "default", "wait")

    child_types = [type(c) for c in inner.children]
    assert child_types == [ClearOutcomeNode, HoldNode, SatisfyNode]
    assert [c.name for c in inner.children] == [
        "default/wait/clear_outcome",
        "default/wait/hold",
        "default/wait/satisfy",
    ]


def test_pure_wait_no_satisfy(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(duration=ParamDist(1.0))
    sequences = {"default": SequenceDef(steps={"wait2": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=8)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np)
    _, inner = _assert_outer_shape(root, "default", "wait2")

    assert [type(c) for c in inner.children] == [ClearOutcomeNode, HoldNode]


def test_accept_step_shape(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(
        accept=True,
        interaction="SERVICE",
        service_tag="water",
        patience=ParamDist(30.0),
        satisfies={"thirst": 20.0},
    )
    sequences = {"default": SequenceDef(steps={"vend": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=9)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np)
    _, inner = _assert_outer_shape(root, "default", "vend")

    child_types = [type(c) for c in inner.children]
    assert child_types == [ClearOutcomeNode, AcceptInteractionNode, SatisfyNode]
    assert [c.name for c in inner.children] == [
        "default/vend/clear_outcome",
        "default/vend/accept",
        "default/vend/satisfy",
    ]

    accept_node = inner.children[1]
    assert isinstance(accept_node, AcceptInteractionNode)
    assert accept_node._interaction == "SERVICE"
    assert accept_node._service_tag == "water"


def test_accept_step_no_satisfy(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(accept=True, interaction="TALK_TO", patience=ParamDist(10.0))
    sequences = {"default": SequenceDef(steps={"greet": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=10)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np)
    _, inner = _assert_outer_shape(root, "default", "greet")

    assert [type(c) for c in inner.children] == [ClearOutcomeNode, AcceptInteractionNode]
    accept_node = inner.children[1]
    assert isinstance(accept_node, AcceptInteractionNode)
    assert accept_node._service_tag is None


def test_block_step_shape(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(
        target_agent=99,
        duration=ParamDist(5.0),
        patience=ParamDist(15.0),
        satisfies={"mischief": 10.0},
    )
    sequences = {"default": SequenceDef(steps={"pursue": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=11)
    target = agent_factory(agent_id=99)
    lookup = lambda aid: target if aid == 99 else None

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np, agent_lookup=lookup)
    _, inner = _assert_outer_shape(root, "default", "pursue")

    assert [type(c) for c in inner.children] == [ClearOutcomeNode, BlockNode, SatisfyNode]
    assert [c.name for c in inner.children] == [
        "default/pursue/clear_outcome",
        "default/pursue/block",
        "default/pursue/satisfy",
    ]
    block_node = inner.children[1]
    assert isinstance(block_node, BlockNode)
    assert block_node._target_agent_id == 99


def test_block_step_requires_agent_lookup(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(target_agent=99)
    sequences = {"default": SequenceDef(steps={"pursue": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=11)

    with pytest.raises(ValueError, match="agent_lookup"):
        _compiled_root(agent_type, agent, world, event_bus, rng_np, agent_lookup=None)


