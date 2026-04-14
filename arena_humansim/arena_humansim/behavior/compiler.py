from collections.abc import Mapping

import numpy as np
import py_trees
from rclpy.logging import get_logger

from arena_humansim.agents import AgentType, BaseAgent
from arena_humansim.agents.types import ActionDef, StepDef
from arena_humansim.manager.world_knowledge import WorldKnowledge
from arena_humansim.utils.event_bus import EventBus

_logger = get_logger("behavior_compiler")

from .nodes import AutonomousNode, ConcreteStepNode, NeedsDecayNode, SequenceStateMachine


class _StepRecipe:
    __slots__ = ("autonomous", "node_name", "step_def", "action_defs", "utility_weights")

    def __init__(self, autonomous: bool, node_name: str, step_def: StepDef, action_defs: Mapping[str, ActionDef], utility_weights: Mapping[str, float]) -> None:
        self.autonomous = autonomous
        self.node_name = node_name
        self.step_def = step_def
        self.action_defs = action_defs
        self.utility_weights = utility_weights

    def build(self, agent: BaseAgent, world: WorldKnowledge, event_bus: EventBus, rng: np.random.Generator, dt: float) -> AutonomousNode | ConcreteStepNode:
        if self.autonomous:
            return AutonomousNode(
                name=self.node_name,
                step_def=self.step_def,
                agent=agent,
                action_defs=self.action_defs,
                utility_weights=self.utility_weights,
                world=world,
                event_bus=event_bus,
                rng=rng,
                dt=dt,
            )
        return ConcreteStepNode(
            name=self.node_name,
            step_def=self.step_def,
            agent=agent,
            world=world,
            rng=rng,
            dt=dt,
        )


class BehaviorTreeFactory:
    def __init__(self, agent_type: AgentType):
        self._agent_type = agent_type
        self._initial = agent_type.initial_sequence
        self._sequence_defs = agent_type.sequences
        self._has_needs = bool(agent_type.needs)
        self._root_name = f"{agent_type.name}_root"
        self._sm_name = f"{agent_type.name}_behavior"
        self._decay_name = f"{agent_type.name}_decay"

        for seq_name, seq_def in agent_type.sequences.items():
            for transition in seq_def.transitions:
                if isinstance(transition.when, str):
                    raise ValueError(f"String-based transition condition '{transition.when}' in sequence '{seq_name}' is not supported. Use dict[str, NeedCondition] instead.")

        self._seq_recipes: dict[str, list[_StepRecipe]] = {}
        for seq_name, seq_def in agent_type.sequences.items():
            recipes: list[_StepRecipe] = []
            for step_name, step_def in seq_def.steps.items():
                recipes.append(
                    _StepRecipe(
                        autonomous=step_def.autonomous,
                        node_name=f"{seq_name}/{step_name}",
                        step_def=step_def,
                        action_defs=agent_type.actions,
                        utility_weights=agent_type.utility_weights,
                    )
                )
            self._seq_recipes[seq_name] = recipes

    def build(
        self,
        agent: BaseAgent,
        world: WorldKnowledge,
        event_bus: EventBus,
        rng: np.random.Generator,
        dt: float,
    ) -> py_trees.trees.BehaviourTree:
        compiled_sequences: dict[str, py_trees.behaviour.Behaviour] = {}
        for seq_name, recipes in self._seq_recipes.items():
            children = [r.build(agent, world, event_bus, rng, dt) for r in recipes]
            if len(children) == 1:
                compiled_sequences[seq_name] = children[0]
            else:
                compiled_sequences[seq_name] = py_trees.composites.Sequence(
                    name=seq_name,
                    memory=True,
                    children=children,
                )

        state_machine = SequenceStateMachine(
            name=self._sm_name,
            sequences=compiled_sequences,
            sequence_defs=self._sequence_defs,
            initial=self._initial,
            agent=agent,
        )

        if self._has_needs and agent.needs is not None:
            root = py_trees.composites.Parallel(
                name=self._root_name,
                policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
                children=[
                    NeedsDecayNode(name=self._decay_name, agent=agent, dt=dt),
                    state_machine,
                ],
            )
        else:
            root = state_machine

        return py_trees.trees.BehaviourTree(root=root)


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

    factory = BehaviorTreeFactory(agent_type)
    bt = factory.build(agent, world, event_bus, rng, dt)
    _logger.debug(f"Agent {agent.state.agent_id} ({agent_type.name}): compiled {len(agent_type.sequences)} sequence(s), initial={agent_type.initial_sequence}")
    return bt
