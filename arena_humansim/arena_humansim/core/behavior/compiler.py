from collections.abc import Callable, Mapping

import numpy as np
import py_trees
from rclpy.logging import get_logger

from arena_humansim.core.agents import AgentType, BaseAgent
from arena_humansim.core.agents.types import ActionDef, GoToStepDef, StepDef
from arena_humansim.core.world_knowledge import WorldKnowledge
from arena_humansim.utils.event_bus import EventBus

_logger = get_logger("behavior_compiler")

from .nodes import (
    AcceptInteractionNode,
    AdvertiseInteractionNode,
    AutonomousNode,
    BlockNode,
    ClearOutcomeNode,
    GoToNode,
    HoldNode,
    NeedsDecayNode,
    PatienceWatchdogNode,
    ResolveObjectNode,
    SatisfyNode,
    SequenceStateMachine,
)
from .step_context import StepContext

AgentLookup = Callable[[int], "BaseAgent | None"]


def _watched(node_name: str, watchdog: py_trees.behaviour.Behaviour, sequence_children: list[py_trees.behaviour.Behaviour]) -> py_trees.composites.Parallel:
    # Watchdog never returns SUCCESS, so Parallel(SuccessOnOne) status tracks the sibling Sequence;
    # a watchdog FAILURE still propagates because Parallel returns FAILURE on any child FAILURE.
    inner = py_trees.composites.Sequence(
        name=f"{node_name}/sequence",
        memory=True,
        children=sequence_children,
    )
    return py_trees.composites.Parallel(
        name=node_name,
        policy=py_trees.common.ParallelPolicy.SuccessOnOne(),
        children=[watchdog, inner],
    )


def _expand_go_to_step(
    node_name: str,
    step: GoToStepDef,
    agent: BaseAgent,
    rng: np.random.Generator,
    dt: float,
) -> py_trees.composites.Parallel:
    watchdog = PatienceWatchdogNode(
        name=f"{node_name}/watchdog",
        patience_source=step.patience,
        rng=rng,
        dt=dt,
    )
    children: list[py_trees.behaviour.Behaviour] = [
        ClearOutcomeNode(name=f"{node_name}/clear_outcome", agent=agent),
        GoToNode(name=f"{node_name}/go_to", agent=agent, target_pose=step.target_pose),
    ]
    if step.duration is not None:
        children.append(HoldNode(name=f"{node_name}/hold", agent=agent, duration_source=step.duration, rng=rng, dt=dt))
    if step.satisfies:
        children.append(SatisfyNode(name=f"{node_name}/satisfy", agent=agent, satisfies=step.satisfies))
    return _watched(node_name, watchdog, children)


def _expand_object_interaction_step(
    node_name: str,
    step: StepDef,
    agent: BaseAgent,
    world: WorldKnowledge,
    rng: np.random.Generator,
    dt: float,
) -> py_trees.composites.Parallel:
    assert step.interaction is not None, "_expand_object_interaction_step requires step.interaction"
    ctx = StepContext()
    watchdog = PatienceWatchdogNode(
        name=f"{node_name}/watchdog",
        patience_source=step.patience,
        rng=rng,
        dt=dt,
    )
    children: list[py_trees.behaviour.Behaviour] = [
        ClearOutcomeNode(name=f"{node_name}/clear_outcome", agent=agent),
        ResolveObjectNode(
            name=f"{node_name}/resolve_object",
            agent=agent,
            world=world,
            target_object_type=step.target_object_type,
            target_object_id=step.target_object_id,
            ctx=ctx,
            step_interaction_radius=step.interaction_radius,
            interaction_name=step.interaction,
        ),
        GoToNode(name=f"{node_name}/go_to", agent=agent, ctx=ctx, world=world),
        AdvertiseInteractionNode(
            name=f"{node_name}/advertise",
            agent=agent,
            interaction=step.interaction,
            ctx=ctx,
            duration_source=step.duration,
            rng=rng,
        ),
    ]
    if step.satisfies:
        children.append(SatisfyNode(name=f"{node_name}/satisfy", agent=agent, satisfies=step.satisfies))
    return _watched(node_name, watchdog, children)


def _expand_object_nav_step(
    node_name: str,
    step: StepDef,
    agent: BaseAgent,
    world: WorldKnowledge,
    rng: np.random.Generator,
    dt: float,
) -> py_trees.composites.Parallel:
    ctx = StepContext()
    watchdog = PatienceWatchdogNode(
        name=f"{node_name}/watchdog",
        patience_source=step.patience,
        rng=rng,
        dt=dt,
    )
    children: list[py_trees.behaviour.Behaviour] = [
        ClearOutcomeNode(name=f"{node_name}/clear_outcome", agent=agent),
        ResolveObjectNode(
            name=f"{node_name}/resolve_object",
            agent=agent,
            world=world,
            target_object_type=step.target_object_type,
            target_object_id=step.target_object_id,
            ctx=ctx,
            step_interaction_radius=step.interaction_radius,
            interaction_name=step.interaction,
        ),
        GoToNode(name=f"{node_name}/go_to", agent=agent, ctx=ctx, world=world),
    ]
    if step.duration is not None:
        children.append(HoldNode(name=f"{node_name}/hold", agent=agent, duration_source=step.duration, rng=rng, dt=dt))
    if step.satisfies:
        children.append(SatisfyNode(name=f"{node_name}/satisfy", agent=agent, satisfies=step.satisfies))
    return _watched(node_name, watchdog, children)


def _expand_pure_wait_step(
    node_name: str,
    step: StepDef,
    agent: BaseAgent,
    rng: np.random.Generator,
    dt: float,
) -> py_trees.composites.Parallel:
    watchdog = PatienceWatchdogNode(
        name=f"{node_name}/watchdog",
        patience_source=step.patience,
        rng=rng,
        dt=dt,
    )
    children: list[py_trees.behaviour.Behaviour] = [
        ClearOutcomeNode(name=f"{node_name}/clear_outcome", agent=agent),
        HoldNode(name=f"{node_name}/hold", agent=agent, duration_source=step.duration, rng=rng, dt=dt),
    ]
    if step.satisfies:
        children.append(SatisfyNode(name=f"{node_name}/satisfy", agent=agent, satisfies=step.satisfies))
    return _watched(node_name, watchdog, children)


def _expand_accept_step(
    node_name: str,
    step: StepDef,
    agent: BaseAgent,
    rng: np.random.Generator,
    dt: float,
) -> py_trees.composites.Parallel:
    assert step.interaction is not None, "_expand_accept_step requires step.interaction"
    watchdog = PatienceWatchdogNode(
        name=f"{node_name}/watchdog",
        patience_source=step.patience,
        rng=rng,
        dt=dt,
    )
    children: list[py_trees.behaviour.Behaviour] = [
        ClearOutcomeNode(name=f"{node_name}/clear_outcome", agent=agent),
        AcceptInteractionNode(
            name=f"{node_name}/accept",
            agent=agent,
            interaction=step.interaction,
            duration_source=step.duration,
            rng=rng,
            service_tag=step.service_tag,
        ),
    ]
    if step.satisfies:
        children.append(SatisfyNode(name=f"{node_name}/satisfy", agent=agent, satisfies=step.satisfies))
    return _watched(node_name, watchdog, children)


def _expand_block_step(
    node_name: str,
    step: StepDef,
    agent: BaseAgent,
    agent_lookup: AgentLookup,
    rng: np.random.Generator,
    dt: float,
) -> py_trees.composites.Parallel:
    assert step.target_agent is not None, "_expand_block_step requires step.target_agent"
    watchdog = PatienceWatchdogNode(
        name=f"{node_name}/watchdog",
        patience_source=step.patience,
        rng=rng,
        dt=dt,
    )
    children: list[py_trees.behaviour.Behaviour] = [
        ClearOutcomeNode(name=f"{node_name}/clear_outcome", agent=agent),
        BlockNode(
            name=f"{node_name}/block",
            agent=agent,
            target_agent_id=step.target_agent,
            agent_lookup=agent_lookup,
            duration_source=step.duration,
            rng=rng,
        ),
    ]
    if step.satisfies:
        children.append(SatisfyNode(name=f"{node_name}/satisfy", agent=agent, satisfies=step.satisfies))
    return _watched(node_name, watchdog, children)


class _StepRecipe:
    __slots__ = ("autonomous", "node_name", "step_def", "action_defs", "utility_weights")

    def __init__(self, autonomous: bool, node_name: str, step_def: StepDef | GoToStepDef, action_defs: Mapping[str, ActionDef], utility_weights: Mapping[str, float]) -> None:
        self.autonomous = autonomous
        self.node_name = node_name
        self.step_def = step_def
        self.action_defs = action_defs
        self.utility_weights = utility_weights

    def build(self, agent: BaseAgent, world: WorldKnowledge, event_bus: EventBus, rng: np.random.Generator, dt: float, agent_lookup: AgentLookup | None = None) -> py_trees.behaviour.Behaviour:
        step = self.step_def
        if isinstance(step, GoToStepDef):
            return _expand_go_to_step(self.node_name, step, agent, rng, dt)
        if step.autonomous:
            return AutonomousNode(
                name=self.node_name,
                step_def=step,
                agent=agent,
                action_defs=dict(self.action_defs),
                utility_weights=dict(self.utility_weights),
                world=world,
                event_bus=event_bus,
                rng=rng,
                dt=dt,
            )
        if step.accept:
            return _expand_accept_step(self.node_name, step, agent, rng, dt)
        if step.target_agent is not None:
            if agent_lookup is None:
                raise ValueError(f"{self.node_name}: target_agent requires agent_lookup to be threaded into BehaviorTreeFactory.build")
            return _expand_block_step(self.node_name, step, agent, agent_lookup, rng, dt)
        if step.interaction is not None:
            return _expand_object_interaction_step(self.node_name, step, agent, world, rng, dt)
        if step.target_object_type or step.target_object_id:
            return _expand_object_nav_step(self.node_name, step, agent, world, rng, dt)
        return _expand_pure_wait_step(self.node_name, step, agent, rng, dt)


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
                autonomous = step_def.autonomous if isinstance(step_def, StepDef) else False
                recipes.append(
                    _StepRecipe(
                        autonomous=autonomous,
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
        agent_lookup: AgentLookup | None = None,
    ) -> py_trees.trees.BehaviourTree:
        compiled_sequences: dict[str, py_trees.behaviour.Behaviour] = {}
        for seq_name, recipes in self._seq_recipes.items():
            children = [r.build(agent, world, event_bus, rng, dt, agent_lookup) for r in recipes]
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
    agent_lookup: AgentLookup | None = None,
) -> py_trees.trees.BehaviourTree | None:
    if agent_type.mode == "simple":
        _logger.debug(f"Agent {agent.state.agent_id} ({agent_type.name}): simple mode, no behavior tree")
        return None

    if not agent_type.sequences:
        return None

    factory = BehaviorTreeFactory(agent_type)
    bt = factory.build(agent, world, event_bus, rng, dt, agent_lookup)
    _logger.debug(f"Agent {agent.state.agent_id} ({agent_type.name}): compiled {len(agent_type.sequences)} sequence(s), initial={agent_type.initial_sequence}")
    return bt
