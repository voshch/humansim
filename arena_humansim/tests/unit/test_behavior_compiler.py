from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

py_trees = pytest.importorskip("py_trees")

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.agents.types import (
    AgentType,
    NeedCondition,
    NeedDist,
    ParamDist,
    SequenceDef,
    StepDef,
    TransitionDef,
)
from arena_humansim.core.behavior.compiler import compile_agent_behavior
from arena_humansim.core.behavior.nodes import (
    AutonomousNode,
    ConcreteStepNode,
    NeedsDecayNode,
    SequenceStateMachine,
)
from arena_humansim.core.world_knowledge import WorldKnowledge
from arena_humansim.utils.event_bus import EventBus
from arena_humansim.utils.types import NeedsState, NeedState


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


def test_single_step_inlined_no_sequence_wrapper(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    sequences = {"default": SequenceDef(steps={"only": _concrete_step()})}
    agent_type = _agent_type(sequences=sequences)
    agent = agent_factory(agent_id=1)

    bt = compile_agent_behavior(agent_type, agent, world, event_bus, rng_np, 0.05)
    assert bt is not None

    root = bt.root
    assert isinstance(root, SequenceStateMachine)
    compiled = root._sequences["default"]
    assert isinstance(compiled, ConcreteStepNode)
    assert not isinstance(compiled, py_trees.composites.Sequence)


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


def test_autonomous_vs_concrete_recipe_dispatch(agent_factory: Callable[..., BaseAgent], world: WorldKnowledge, event_bus: EventBus, rng_np: np.random.Generator) -> None:
    """Autonomous StepDef compiles to AutonomousNode; non-autonomous to ConcreteStepNode."""
    sequences = {
        "auto_seq": SequenceDef(steps={"s": _autonomous_step()}),
        "concrete_seq": SequenceDef(steps={"s": _concrete_step()}),
    }
    agent_type = _agent_type(sequences=sequences, initial_sequence="auto_seq")
    agent = agent_factory(agent_id=1)

    bt = compile_agent_behavior(agent_type, agent, world, event_bus, rng_np, 0.05)
    assert bt is not None
    sm = bt.root
    assert isinstance(sm, SequenceStateMachine)
    assert isinstance(sm._sequences["auto_seq"], AutonomousNode)
    assert isinstance(sm._sequences["concrete_seq"], ConcreteStepNode)


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
