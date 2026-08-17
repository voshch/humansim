from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

py_trees = pytest.importorskip("py_trees")

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.agents.types import (
    AgentType,
    AttentionDef,
    AttentionStepDef,
    GoToStepDef,
    NeedDist,
    ParamDist,
    SequenceDef,
    StepDef,
)
from arena_humansim.core.behavior.compiler import BehaviorTreeFactory
from arena_humansim.core.behavior.nodes import (
    AttentionNode,
    AutonomousNode,
    BlockNode,
    CancelNode,
    ClearGestureNode,
    ClearOutcomeNode,
    GoToNode,
    HoldNode,
    PatienceWatchdogNode,
    ResolveObjectNode,
    SatisfyNode,
    SeekNode,
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
    name_lookup: Callable[[str, int | None], int | None] | None = None,
) -> py_trees.behaviour.Behaviour:
    factory = BehaviorTreeFactory(agent_type)
    bt = factory.build(agent, world, event_bus, rng, 0.05, agent_lookup=agent_lookup, name_lookup=name_lookup)
    sm = bt.root
    assert isinstance(sm, SequenceStateMachine)
    return sm._sequences["default"]


def _assert_outer_shape(root: py_trees.behaviour.Behaviour, seq: str, step: str, rider: bool = False) -> tuple[PatienceWatchdogNode, py_trees.composites.Sequence]:
    assert isinstance(root, py_trees.composites.Parallel)
    assert root.name == f"{seq}/{step}"
    assert isinstance(root.policy, py_trees.common.ParallelPolicy.SuccessOnOne)
    if rider:
        assert len(root.children) == 3
        watchdog, inner, attention = root.children
        assert isinstance(attention, AttentionNode)
        assert attention.name == f"{seq}/{step}/attention"
    else:
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
    assert child_types == [ClearOutcomeNode, ClearGestureNode, GoToNode, HoldNode, SatisfyNode]
    assert [c.name for c in inner.children] == [
        "default/walk/clear_outcome",
        "default/walk/clear_gesture",
        "default/walk/go_to",
        "default/walk/hold",
        "default/walk/satisfy",
    ]


def test_go_to_step_with_target_pose_minimal(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = GoToStepDef(target_pose=Pose2D(x=1.0, y=2.0))
    sequences = {"default": SequenceDef(steps={"walk2": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=2)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np)
    _, inner = _assert_outer_shape(root, "default", "walk2")

    assert [type(c) for c in inner.children] == [ClearOutcomeNode, ClearGestureNode, GoToNode]


def test_go_to_step_with_target_expands_resolve_plus_goto(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = GoToStepDef(target="bench")
    sequences = {"default": SequenceDef(steps={"walkto": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=12)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np)
    _, inner = _assert_outer_shape(root, "default", "walkto")
    assert [type(c) for c in inner.children] == [ClearOutcomeNode, ClearGestureNode, ResolveObjectNode, GoToNode]


def test_object_bound_interaction_shape(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(
        interaction="SIT_ON",
        target="chair",
        duration=ParamDist(3.0),
        satisfies={"rest": 5.0},
    )
    sequences = {"default": SequenceDef(steps={"sit": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=3)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np)
    _, inner = _assert_outer_shape(root, "default", "sit")

    child_types = [type(c) for c in inner.children]
    assert child_types == [ClearOutcomeNode, ClearGestureNode, ResolveObjectNode, GoToNode, SeekNode, SatisfyNode]
    assert [c.name for c in inner.children] == [
        "default/sit/clear_outcome",
        "default/sit/clear_gesture",
        "default/sit/resolve_object",
        "default/sit/go_to",
        "default/sit/seek",
        "default/sit/satisfy",
    ]


def test_object_bound_interaction_no_satisfy(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(
        interaction="SIT_ON",
        target="chair",
        duration=ParamDist(3.0),
    )
    sequences = {"default": SequenceDef(steps={"sit2": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=4)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np)
    _, inner = _assert_outer_shape(root, "default", "sit2")

    assert [type(c) for c in inner.children] == [ClearOutcomeNode, ClearGestureNode, ResolveObjectNode, GoToNode, SeekNode]


def test_non_object_interaction_has_no_resolve_or_goto(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(
        interaction="TALK_TO",
        duration=ParamDist(5.0),
        satisfies={"company": 1.0},
    )
    sequences = {"default": SequenceDef(steps={"chat": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=20)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np)
    _, inner = _assert_outer_shape(root, "default", "chat")

    assert [type(c) for c in inner.children] == [ClearOutcomeNode, ClearGestureNode, SeekNode, SatisfyNode]
    assert [c.name for c in inner.children] == [
        "default/chat/clear_outcome",
        "default/chat/clear_gesture",
        "default/chat/seek",
        "default/chat/satisfy",
    ]


def test_non_object_interaction_minimal(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(interaction="TALK_TO")
    sequences = {"default": SequenceDef(steps={"chat2": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=21)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np)
    _, inner = _assert_outer_shape(root, "default", "chat2")

    assert [type(c) for c in inner.children] == [ClearOutcomeNode, ClearGestureNode, SeekNode]


def test_service_offer_has_no_resolve(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(
        interaction="SERVICE",
        target="water",
        offer=True,
        min_participants=1,
        max_participants=3,
        queueable=True,
    )
    sequences = {"default": SequenceDef(steps={"vend": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=25)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np)
    _, inner = _assert_outer_shape(root, "default", "vend")

    assert [type(c) for c in inner.children] == [ClearOutcomeNode, ClearGestureNode, SeekNode]


def test_cancel_step_shape(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(cancel=True, satisfies={"boredom": 1.0})
    sequences = {"default": SequenceDef(steps={"bail": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=30)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np)
    _, inner = _assert_outer_shape(root, "default", "bail")

    assert [type(c) for c in inner.children] == [ClearOutcomeNode, ClearGestureNode, CancelNode, SatisfyNode]


def test_cancel_step_minimal(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(cancel=True)
    sequences = {"default": SequenceDef(steps={"bail2": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=31)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np)
    _, inner = _assert_outer_shape(root, "default", "bail2")

    assert [type(c) for c in inner.children] == [ClearOutcomeNode, ClearGestureNode, CancelNode]


def test_pure_wait_shape(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(duration=ParamDist(2.0), satisfies={"boredom": 1.0})
    sequences = {"default": SequenceDef(steps={"wait": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=7)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np)
    _, inner = _assert_outer_shape(root, "default", "wait")

    child_types = [type(c) for c in inner.children]
    assert child_types == [ClearOutcomeNode, ClearGestureNode, HoldNode, SatisfyNode]
    assert [c.name for c in inner.children] == [
        "default/wait/clear_outcome",
        "default/wait/clear_gesture",
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

    assert [type(c) for c in inner.children] == [ClearOutcomeNode, ClearGestureNode, HoldNode]


def test_block_step_shape(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(
        interaction="BLOCK",
        target=99,
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

    assert [type(c) for c in inner.children] == [ClearOutcomeNode, ClearGestureNode, BlockNode, SatisfyNode]
    assert [c.name for c in inner.children] == [
        "default/pursue/clear_outcome",
        "default/pursue/clear_gesture",
        "default/pursue/block",
        "default/pursue/satisfy",
    ]
    block_node = inner.children[2]
    assert isinstance(block_node, BlockNode)
    assert block_node._target_agent_id == 99


def test_block_step_requires_agent_lookup(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(interaction="BLOCK", target=99)
    sequences = {"default": SequenceDef(steps={"pursue": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=11)

    with pytest.raises(ValueError, match="agent_lookup"):
        _compiled_root(agent_type, agent, world, event_bus, rng_np, agent_lookup=None)


_ATT = AttentionDef(gesture="point", at="bench")


def _lookups() -> dict:
    return {"agent_lookup": lambda aid: None, "name_lookup": lambda name, kind=None: None}


def test_attention_step_shape(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = AttentionStepDef(attention=_ATT, duration=ParamDist(1.5), satisfies={"social": 5.0})
    sequences = {"default": SequenceDef(steps={"show": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=1)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np, **_lookups())
    _, inner = _assert_outer_shape(root, "default", "show")

    assert [type(c) for c in inner.children] == [ClearOutcomeNode, AttentionNode, SatisfyNode]
    assert [c.name for c in inner.children] == [
        "default/show/clear_outcome",
        "default/show/attention",
        "default/show/satisfy",
    ]
    node = inner.children[1]
    assert isinstance(node, AttentionNode)
    assert node._bare is True
    assert node._idle is True
    assert node._duration_source == ParamDist(1.5)


def test_attention_step_requires_lookups(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = AttentionStepDef(attention=_ATT)
    sequences = {"default": SequenceDef(steps={"show": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=1)

    with pytest.raises(ValueError, match="name_lookup"):
        _compiled_root(agent_type, agent, world, event_bus, rng_np, agent_lookup=lambda aid: None, name_lookup=None)


def test_go_to_attention_rider_shape(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = GoToStepDef(target_pose=Pose2D(x=5.0, y=5.0), duration=ParamDist(2.0), attention=_ATT)
    sequences = {"default": SequenceDef(steps={"walk": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=1)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np, **_lookups())
    _, inner = _assert_outer_shape(root, "default", "walk", rider=True)
    assert [type(c) for c in inner.children] == [ClearOutcomeNode, GoToNode, HoldNode]
    rider = root.children[2]
    assert isinstance(rider, AttentionNode)
    assert rider._bare is False
    assert rider._idle is False
    assert rider._ctx.target_pose == Pose2D(x=5.0, y=5.0)


def test_go_to_attention_requires_lookups(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = GoToStepDef(target_pose=Pose2D(x=5.0, y=5.0), attention=_ATT)
    sequences = {"default": SequenceDef(steps={"walk": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=1)

    with pytest.raises(ValueError, match="agent_lookup"):
        _compiled_root(agent_type, agent, world, event_bus, rng_np)


def test_interaction_attention_rider_shape(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(interaction="TALK_TO", duration=ParamDist(4.0), attention=AttentionDef(gesture="point", at="partner"))
    sequences = {"default": SequenceDef(steps={"chat": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=1)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np, **_lookups())
    _, inner = _assert_outer_shape(root, "default", "chat", rider=True)
    assert [type(c) for c in inner.children] == [ClearOutcomeNode, SeekNode]
    rider = root.children[2]
    assert isinstance(rider, AttentionNode)
    assert rider._idle is False


def test_object_interaction_attention_rider_shape(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(interaction="SIT_ON", target="bench", attention=AttentionDef(gesture="point", at="target"))
    sequences = {"default": SequenceDef(steps={"sit": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=1)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np, **_lookups())
    _, inner = _assert_outer_shape(root, "default", "sit", rider=True)
    assert [type(c) for c in inner.children] == [ClearOutcomeNode, ResolveObjectNode, GoToNode, SeekNode]
    rider = root.children[2]
    assert isinstance(rider, AttentionNode)
    assert rider._ctx is inner.children[1]._ctx


def test_wait_attention_rider_is_idle(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(duration=ParamDist(2.0), attention=_ATT)
    sequences = {"default": SequenceDef(steps={"wait": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=1)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np, **_lookups())
    _, inner = _assert_outer_shape(root, "default", "wait", rider=True)
    assert [type(c) for c in inner.children] == [ClearOutcomeNode, HoldNode]
    rider = root.children[2]
    assert isinstance(rider, AttentionNode)
    assert rider._bare is False
    assert rider._idle is True


def test_cancel_attention_rider_shape(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(cancel=True, attention=_ATT)
    sequences = {"default": SequenceDef(steps={"drop": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=1)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np, **_lookups())
    _, inner = _assert_outer_shape(root, "default", "drop", rider=True)
    assert [type(c) for c in inner.children] == [ClearOutcomeNode, CancelNode]


def test_block_attention_rider_shape(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(interaction="BLOCK", target=99, attention=AttentionDef(gesture="point", at=99))
    sequences = {"default": SequenceDef(steps={"pursue": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=1)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np, **_lookups())
    _, inner = _assert_outer_shape(root, "default", "pursue", rider=True)
    assert [type(c) for c in inner.children] == [ClearOutcomeNode, BlockNode]


def test_autonomous_step_rejects_attention(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(autonomous=True, duration=ParamDist(1.0), attention=_ATT)
    sequences = {"default": SequenceDef(steps={"roam": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=1)

    with pytest.raises(ValueError, match="autonomous"):
        _compiled_root(agent_type, agent, world, event_bus, rng_np, **_lookups())


def test_autonomous_step_clears_gesture_first(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    step = StepDef(autonomous=True, duration=ParamDist(1.0))
    sequences = {"default": SequenceDef(steps={"roam": step})}
    agent_type = _agent_type(sequences)
    agent = agent_factory(agent_id=1)

    root = _compiled_root(agent_type, agent, world, event_bus, rng_np)
    assert isinstance(root, py_trees.composites.Sequence)
    assert root.name == "default/roam"
    assert [type(c) for c in root.children] == [ClearGestureNode, AutonomousNode]
    assert [c.name for c in root.children] == ["default/roam/clear_gesture", "default/roam/autonomous"]
