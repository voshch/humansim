from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

py_trees = pytest.importorskip("py_trees")

from arena_humansim.core.agents import BaseAgent
from py_trees.trees import BehaviourTree

from arena_humansim.core.agents.types import (
    AgentType,
    AttentionDef,
    AttentionStepDef,
    ChannelDef,
    NeedCondition,
    NeedDist,
    ParamDist,
    SequenceDef,
    StepDef,
    TransitionDef,
    sample_agent_type,
)
from arena_humansim.core.behavior.compiler import compile_agent_behavior
from arena_humansim.core.behavior.nodes import (
    NeedsDecayNode,
    SequenceStateMachine,
)
from arena_humansim.core.behavior.nodes.attention import RESOLVE_TIMEOUT_S
from arena_humansim.core.world_knowledge import WorldKnowledge, WorldObject
from arena_humansim.utils.event_bus import EventBus
from arena_humansim.utils.types import BehaviorTreeMovement, NeedsState, NeedState, Pose2D


@pytest.fixture
def rng_np() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def world() -> WorldKnowledge:
    return WorldKnowledge()


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


def _concrete_step() -> StepDef:
    return StepDef(duration=ParamDist(1.0), autonomous=False)


def _autonomous_step() -> StepDef:
    return StepDef(duration=ParamDist(1.0), autonomous=True)


def _agent_type(
    *,
    mode: str = "complex",
    sequences: dict[str, SequenceDef] | None = None,
    initial_sequence: str = "default",
    needs: dict[str, NeedDist] | None = None,
) -> AgentType:
    return AgentType(
        name="test_agent",
        mode=mode,
        sequences=sequences if sequences is not None else {},
        initial_sequence=initial_sequence,
        needs=needs if needs is not None else {},
    )


def test_simple_mode_returns_none(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    agent_type = _agent_type(mode="simple")
    agent = agent_factory(agent_id=1)
    assert compile_agent_behavior(agent_type, agent, world, event_bus, rng_np, 0.05) is None


def test_no_sequences_returns_none(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    agent_type = _agent_type(mode="complex", sequences={})
    agent = agent_factory(agent_id=1)
    assert compile_agent_behavior(agent_type, agent, world, event_bus, rng_np, 0.05) is None


def test_string_transition_raises_value_error(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    """String-valued transition.when is rejected; compiler only accepts dict[str, NeedCondition]."""
    bad_transition = TransitionDef(when="some_missing_state", goto="other")  # type: ignore[arg-type]
    sequences = {
        "default": SequenceDef(
            steps={"s0": _concrete_step()},
            transitions=(bad_transition,),
        ),
    }
    agent_type = _agent_type(sequences=sequences)
    agent = agent_factory(agent_id=1)
    with pytest.raises(ValueError, match="String-based transition"):
        compile_agent_behavior(agent_type, agent, world, event_bus, rng_np, 0.05)


def test_multi_step_wraps_in_sequence_with_memory(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    sequences = {
        "default": SequenceDef(
            steps={
                "s0": _concrete_step(),
                "s1": _concrete_step(),
            },
        ),
    }
    agent_type = _agent_type(sequences=sequences)
    agent = agent_factory(agent_id=1)

    bt = compile_agent_behavior(agent_type, agent, world, event_bus, rng_np, 0.05)
    assert bt is not None

    root = bt.root
    assert isinstance(root, SequenceStateMachine)
    compiled = root._sequences["default"]
    assert isinstance(compiled, py_trees.composites.Sequence)
    assert compiled.memory is True
    assert len(compiled.children) == 2


def test_needs_wraps_root_in_parallel_with_decay(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    sequences = {"default": SequenceDef(steps={"only": _concrete_step()})}
    needs = {"energy": NeedDist()}
    agent_type = _agent_type(sequences=sequences, needs=needs)

    agent = agent_factory(agent_id=1)
    agent.needs = NeedsState(needs={"energy": NeedState(value=100.0, decay_rate=0.5)})

    bt = compile_agent_behavior(agent_type, agent, world, event_bus, rng_np, 0.05)
    assert bt is not None

    root = bt.root
    assert isinstance(root, py_trees.composites.Parallel)
    kinds = [type(c) for c in root.children]
    assert NeedsDecayNode in kinds
    assert any(isinstance(c, SequenceStateMachine) for c in root.children)


def test_no_needs_uses_state_machine_as_root(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    sequences = {"default": SequenceDef(steps={"only": _concrete_step()})}
    agent_type = _agent_type(sequences=sequences, needs={})
    agent = agent_factory(agent_id=1)

    bt = compile_agent_behavior(agent_type, agent, world, event_bus, rng_np, 0.05)
    assert bt is not None
    assert isinstance(bt.root, SequenceStateMachine)
    assert not isinstance(bt.root, py_trees.composites.Parallel)


def test_transitions_with_need_condition_compile(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    sequences = {
        "default": SequenceDef(
            steps={"s0": _concrete_step()},
            transitions=(TransitionDef(when={"energy": NeedCondition(below=20.0)}, goto="rest"),),
        ),
        "rest": SequenceDef(steps={"r0": _concrete_step()}),
    }
    agent_type = _agent_type(sequences=sequences)
    agent = agent_factory(agent_id=1)

    bt = compile_agent_behavior(agent_type, agent, world, event_bus, rng_np, 0.05)
    assert bt is not None
    assert isinstance(bt.root, SequenceStateMachine)


# sequence rider and hold across steps (ticking the compiled tree)


def _rider_setup(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator, agent_type: AgentType) -> tuple[BaseAgent, BehaviourTree]:
    world.add_object(WorldObject(object_id="bench_1", type="bench", pose=Pose2D(x=5.0, y=0.0)))
    world.add_object(WorldObject(object_id="lamp_1", type="lamp", pose=Pose2D(x=5.0, y=2.0)))
    agent = agent_factory(agent_id=1)
    agent.movement = BehaviorTreeMovement()
    bt = compile_agent_behavior(agent_type, agent, world, event_bus, rng_np, 0.5, agent_lookup=lambda aid: None, name_lookup=lambda name, kind=None: None)
    assert bt is not None
    return agent, bt


def _slots(agent: BaseAgent) -> dict[str, tuple[float, float]]:
    mv = agent.movement
    assert isinstance(mv, BehaviorTreeMovement)
    return {g.slot: (g.x, g.y) for g in mv.gestures}


def test_sequence_rider_pauses_for_step_block_and_resumes_with_index(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    steps = {
        "wait": StepDef(duration=ParamDist(1.0)),
        "show": AttentionStepDef(attention=AttentionDef(point=ChannelDef(at="lamp_1"), face=False), duration=ParamDist(1.0)),
        "rest": StepDef(duration=ParamDist(2.0)),
    }
    rider = AttentionDef(gaze=ChannelDef(at=("bench_1", "lamp_1"), dwell=1.5), face=False)
    agent_type = _agent_type(sequences={"default": SequenceDef(steps=steps, attention=rider)})
    agent, bt = _rider_setup(agent_factory, world, event_bus, rng_np, agent_type)

    seen = []
    for _ in range(9):
        bt.tick()
        seen.append(_slots(agent))
    assert seen[0] == {"head": (5.0, 0.0)}
    assert seen[1] == {"head": (5.0, 0.0)}
    assert seen[2] == {"arm": (5.0, 2.0)}
    assert seen[3] == {"arm": (5.0, 2.0)}
    assert seen[4] == {"head": (5.0, 0.0)}
    assert seen[5] == {"head": (5.0, 2.0)}
    assert seen[6] == {"head": (5.0, 2.0)}
    assert bt.root.status == py_trees.common.Status.SUCCESS
    assert seen[8] == {}


def test_sequence_rider_suspended_during_autonomous_step(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    steps = {
        "wait": StepDef(duration=ParamDist(0.5)),
        "roam": StepDef(duration=ParamDist(1.0), autonomous=True),
        "rest": StepDef(duration=ParamDist(1.0)),
    }
    rider = AttentionDef(gaze=ChannelDef(at="bench_1"), face=False)
    agent_type = _agent_type(sequences={"default": SequenceDef(steps=steps, attention=rider)})
    agent, bt = _rider_setup(agent_factory, world, event_bus, rng_np, agent_type)

    seen = []
    for _ in range(6):
        bt.tick()
        seen.append(set(_slots(agent)))
    assert seen[0] == {"head"}
    assert seen[1] == set()
    assert seen[2] == set()
    assert "head" in seen[3] or "head" in seen[4]


def test_hold_keep_survives_into_next_step_and_sequence_change_clears(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    steps = {
        "show": AttentionStepDef(attention=AttentionDef(point=ChannelDef(at="bench_1", hold="keep"), face=False), duration=ParamDist(0.5)),
        "wait": StepDef(duration=ParamDist(1.0)),
    }
    sequences = {
        "default": SequenceDef(steps=steps, then="other"),
        "other": SequenceDef(steps={"idle": StepDef(duration=ParamDist(5.0))}),
    }
    agent_type = _agent_type(sequences=sequences)
    agent, bt = _rider_setup(agent_factory, world, event_bus, rng_np, agent_type)

    bt.tick()
    assert _slots(agent) == {"arm": (5.0, 0.0)}
    bt.tick()
    assert _slots(agent) == {"arm": (5.0, 0.0)}
    bt.tick()
    assert _slots(agent) == {"arm": (5.0, 0.0)}
    sm = bt.root
    assert isinstance(sm, SequenceStateMachine)
    while sm._current_name == "default":
        bt.tick()
    assert _slots(agent) == {}


def test_required_sequence_rider_failure_fails_sequence(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    rider = AttentionDef(gaze=ChannelDef(at="nobody"), face=False, required=True)
    sequences = {
        "default": SequenceDef(steps={"wait": StepDef(duration=ParamDist(50.0))}, attention=rider, on_failure="fallback"),
        "fallback": SequenceDef(steps={"idle": StepDef(duration=ParamDist(50.0))}),
    }
    agent_type = _agent_type(sequences=sequences)
    agent, bt = _rider_setup(agent_factory, world, event_bus, rng_np, agent_type)
    sm = bt.root
    assert isinstance(sm, SequenceStateMachine)
    for _ in range(int(RESOLVE_TIMEOUT_S / 0.5)):
        bt.tick()
    assert sm._current_name == "default"
    bt.tick()
    assert sm._current_name == "fallback"


def test_cosmetic_sequence_rider_never_fails(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    rider = AttentionDef(gaze=ChannelDef(at="nobody"), face=False)
    sequences = {
        "default": SequenceDef(steps={"wait": StepDef(duration=ParamDist(50.0))}, attention=rider, on_failure="fallback"),
        "fallback": SequenceDef(steps={"idle": StepDef(duration=ParamDist(50.0))}),
    }
    agent_type = _agent_type(sequences=sequences)
    agent, bt = _rider_setup(agent_factory, world, event_bus, rng_np, agent_type)
    sm = bt.root
    assert isinstance(sm, SequenceStateMachine)
    for _ in range(int(RESOLVE_TIMEOUT_S / 0.5) + 4):
        bt.tick()
    assert sm._current_name == "default"
    assert _slots(agent) == {}


def test_step_rider_released_when_step_ends(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    steps = {
        "wait": StepDef(duration=ParamDist(0.5), attention=AttentionDef(gaze=ChannelDef(at="bench_1"), face=False)),
        "rest": StepDef(duration=ParamDist(5.0)),
    }
    agent_type = _agent_type(sequences={"default": SequenceDef(steps=steps)})
    agent, bt = _rider_setup(agent_factory, world, event_bus, rng_np, agent_type)
    bt.tick()
    assert _slots(agent) == {"head": (5.0, 0.0)}
    bt.tick()
    bt.tick()
    assert _slots(agent) == {}


# handedness


def test_handedness_sampled_from_agent_type() -> None:
    right = AgentType(name="r", handedness={"right": 1.0})
    left = AgentType(name="l", handedness={"left": 1.0})
    assert sample_agent_type(right, np.random.default_rng(1)).handedness == "r"
    assert sample_agent_type(left, np.random.default_rng(1)).handedness == "l"
    default = AgentType(name="d")
    hands = [sample_agent_type(default, np.random.default_rng(i)).handedness for i in range(200)]
    assert set(hands) == {"l", "r"}
    assert hands.count("r") > hands.count("l") * 4
    assert sample_agent_type(default, np.random.default_rng(3)).handedness == sample_agent_type(default, np.random.default_rng(3)).handedness
