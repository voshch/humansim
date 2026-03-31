import numpy as np
import py_trees
from rclpy.logging import get_logger

from arena_humansim.agents import AgentType, BaseAgent, SequenceDef
from arena_humansim.manager.world_knowledge import WorldKnowledge
from arena_humansim.utils.event_bus import EventBus

_logger = get_logger("behavior_compiler")

from .nodes import AutonomousNode, ConcreteStepNode, NeedsDecayNode, SequenceStateMachine


def _compile_sequence(
    seq_name: str,
    seq_def: SequenceDef,
    agent_type: AgentType,
    agent: BaseAgent,
    world: WorldKnowledge,
    event_bus: EventBus,
    rng: np.random.Generator,
    dt: float = 0.05,
) -> py_trees.behaviour.Behaviour:
    children: list[py_trees.behaviour.Behaviour] = []

    for step_name, step_def in seq_def.steps.items():
        node_name = f"{seq_name}/{step_name}"

        if step_def.autonomous:
            node = AutonomousNode(
                name=node_name,
                step_def=step_def,
                agent=agent,
                action_defs=agent_type.actions,
                utility_weights=agent_type.utility_weights,
                world=world,
                event_bus=event_bus,
                rng=rng,
                dt=dt,
            )
        else:
            node = ConcreteStepNode(
                name=node_name,
                step_def=step_def,
                agent=agent,
                world=world,
                rng=rng,
                dt=dt,
            )
        children.append(node)

    if len(children) == 1:
        return children[0]

    return py_trees.composites.Sequence(
        name=seq_name,
        memory=True,
        children=children,
    )


def compile_agent_behavior(
    agent_type: AgentType,
    agent: BaseAgent,
    world: WorldKnowledge,
    event_bus: EventBus,
    rng: np.random.Generator,
    dt: float,
) -> py_trees.trees.BehaviourTree | None:
    if agent_type.mode == "simple":
        _logger.debug(f"Agent {agent.state.agent_id} ({agent_type.name}): simple mode, no behavior tree")
        return None

    if not agent_type.sequences:
        return None

    compiled_sequences: dict[str, py_trees.behaviour.Behaviour] = {}
    for seq_name, seq_def in agent_type.sequences.items():
        compiled_sequences[seq_name] = _compile_sequence(
            seq_name=seq_name,
            seq_def=seq_def,
            agent_type=agent_type,
            agent=agent,
            world=world,
            event_bus=event_bus,
            rng=rng,
            dt=dt,
        )

    for seq_name, seq_def in agent_type.sequences.items():
        for transition in seq_def.transitions:
            if isinstance(transition.when, str):
                raise ValueError(f"String-based transition condition '{transition.when}' in sequence '{seq_name}' is not supported. Use dict[str, NeedCondition] instead.")

    state_machine = SequenceStateMachine(
        name=f"{agent_type.name}_behavior",
        sequences=compiled_sequences,
        sequence_defs=agent_type.sequences,
        initial=agent_type.initial_sequence,
        agent=agent,
    )

    if agent_type.needs and agent.needs is not None:
        root = py_trees.composites.Parallel(
            name=f"{agent_type.name}_root",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=[
                NeedsDecayNode(
                    name=f"{agent_type.name}_decay",
                    agent=agent,
                    dt=dt,
                ),
                state_machine,
            ],
        )
    else:
        root = state_machine

    _logger.debug(f"Agent {agent.state.agent_id} ({agent_type.name}): compiled {len(compiled_sequences)} sequence(s), initial={agent_type.initial_sequence}")
    return py_trees.trees.BehaviourTree(root=root)
