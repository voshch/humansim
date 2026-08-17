from __future__ import annotations

from collections.abc import Callable, Mapping

import attrs
import numpy as np
import py_trees
from rclpy.logging import get_logger

from arena_humansim.core.agents import AgentType, BaseAgent, ParamDist
from arena_humansim.core.agents.types import ActionDef, AttentionDef, AttentionStepDef, GoToStepDef, StepDef
from arena_humansim.core.interaction_kinds import InteractionType, is_object_bound_name
from arena_humansim.core.interaction_manager import InteractionManager
from arena_humansim.core.pool import AgentPool
from arena_humansim.core.world_knowledge import WorldKnowledge
from arena_humansim.utils.event_bus import EventBus
from arena_humansim.utils.types import SeekSpec

_logger = get_logger("behavior_compiler")

IsBoundLookup = Callable[[int], bool]

from .nodes import (
    AttentionNode,
    AutonomousNode,
    BlockNode,
    CancelNode,
    ClearGestureNode,
    ClearOutcomeNode,
    GoToNode,
    HoldNode,
    NeedsDecayNode,
    PatienceWatchdogNode,
    ResolveObjectNode,
    SatisfyNode,
    SeekNode,
    SequenceStateMachine,
)
from .step_context import StepContext

AgentLookup = Callable[[int], BaseAgent | None]
NameLookup = Callable[[str, int | None], int | None]


@attrs.frozen
class _AttentionWiring:
    world: WorldKnowledge
    agent_lookup: AgentLookup
    name_lookup: NameLookup
    rng: np.random.Generator
    dt: float


def _wiring(node_name: str, world: WorldKnowledge, agent_lookup: AgentLookup | None, name_lookup: NameLookup | None, rng: np.random.Generator, dt: float) -> _AttentionWiring:
    if agent_lookup is None or name_lookup is None:
        raise ValueError(f"{node_name}: attention requires agent_lookup and name_lookup to be threaded into BehaviorTreeFactory.build")
    return _AttentionWiring(world=world, agent_lookup=agent_lookup, name_lookup=name_lookup, rng=rng, dt=dt)


def _attention_node(node_name: str, agent: BaseAgent, attention: AttentionDef, w: _AttentionWiring, ctx: StepContext, bare: bool = False, idle: bool = False, duration: ParamDist | None = None) -> AttentionNode:
    return AttentionNode(
        name=f"{node_name}/attention",
        agent=agent,
        attention=attention,
        world=w.world,
        agent_lookup=w.agent_lookup,
        name_lookup=w.name_lookup,
        rng=w.rng,
        dt=w.dt,
        ctx=ctx,
        bare=bare,
        idle=idle,
        duration=duration,
    )


def _head(node_name: str, agent: BaseAgent, attention: AttentionDef | None) -> list[py_trees.behaviour.Behaviour]:
    # Steps with attention retarget the previous intent instead of clearing it.
    children: list[py_trees.behaviour.Behaviour] = [ClearOutcomeNode(name=f"{node_name}/clear_outcome", agent=agent)]
    if attention is None:
        children.append(ClearGestureNode(name=f"{node_name}/clear_gesture", agent=agent))
    return children


def _watched(node_name: str, watchdog: py_trees.behaviour.Behaviour, sequence_children: list[py_trees.behaviour.Behaviour], attention: AttentionNode | None = None) -> py_trees.composites.Parallel:
    # Watchdog never returns SUCCESS, so Parallel(SuccessOnOne) status tracks the sibling Sequence;
    # a watchdog FAILURE still propagates because Parallel returns FAILURE on any child FAILURE.
    # A rider AttentionNode never returns SUCCESS or FAILURE either.
    inner = py_trees.composites.Sequence(
        name=f"{node_name}/sequence",
        memory=True,
        children=sequence_children,
    )
    children: list[py_trees.behaviour.Behaviour] = [watchdog, inner]
    if attention is not None:
        children.append(attention)
    return py_trees.composites.Parallel(
        name=node_name,
        policy=py_trees.common.ParallelPolicy.SuccessOnOne(),
        children=children,
    )


def _expand_go_to_step(
    node_name: str,
    step: GoToStepDef,
    agent: BaseAgent,
    world: WorldKnowledge,
    rng: np.random.Generator,
    dt: float,
    pool: AgentPool | None = None,
    is_bound_lookup: IsBoundLookup | None = None,
    im: InteractionManager | None = None,
    agent_lookup: AgentLookup | None = None,
    name_lookup: NameLookup | None = None,
) -> py_trees.composites.Parallel:
    ctx = StepContext(is_bound_lookup=is_bound_lookup, im=im, target_pose=step.target_pose)
    watchdog = PatienceWatchdogNode(
        name=f"{node_name}/watchdog",
        patience_source=step.patience,
        rng=rng,
        dt=dt,
    )
    children = _head(node_name, agent, step.attention)
    if step.target is not None:
        children.extend(
            [
                ResolveObjectNode(
                    name=f"{node_name}/resolve_object",
                    agent=agent,
                    world=world,
                    target=step.target,
                    ctx=ctx,
                ),
                GoToNode(name=f"{node_name}/go_to", agent=agent, ctx=ctx, world=world, pool=pool),
            ]
        )
    else:
        children.append(GoToNode(name=f"{node_name}/go_to", agent=agent, target_pose=step.target_pose, pool=pool))
    if step.duration is not None:
        children.append(HoldNode(name=f"{node_name}/hold", agent=agent, duration_source=step.duration, rng=rng, dt=dt, ctx=ctx))
    if step.satisfies:
        children.append(SatisfyNode(name=f"{node_name}/satisfy", agent=agent, satisfies=step.satisfies))
    rider = None
    if step.attention is not None:
        rider = _attention_node(node_name, agent, step.attention, _wiring(node_name, world, agent_lookup, name_lookup, rng, dt), ctx)
    return _watched(node_name, watchdog, children, rider)


def _expand_interaction_step(
    node_name: str,
    step: StepDef,
    agent: BaseAgent,
    world: WorldKnowledge,
    rng: np.random.Generator,
    dt: float,
    is_bound_lookup: IsBoundLookup | None = None,
    im: InteractionManager | None = None,
    agent_lookup: AgentLookup | None = None,
    name_lookup: NameLookup | None = None,
) -> py_trees.composites.Parallel:
    assert step.interaction is not None, "_expand_interaction_step requires step.interaction"
    ctx = StepContext(is_bound_lookup=is_bound_lookup, im=im)
    watchdog = PatienceWatchdogNode(
        name=f"{node_name}/watchdog",
        patience_source=step.patience,
        rng=rng,
        dt=dt,
    )
    children = _head(node_name, agent, step.attention)
    object_bound = is_object_bound_name(step.interaction) and step.target is not None
    if object_bound:
        children.extend(
            [
                ResolveObjectNode(
                    name=f"{node_name}/resolve_object",
                    agent=agent,
                    world=world,
                    target=step.target,
                    ctx=ctx,
                    step_interaction_radius=step.interaction_radius,
                    interaction_name=step.interaction,
                ),
                GoToNode(name=f"{node_name}/go_to", agent=agent, ctx=ctx, world=world),
            ]
        )
    spec = SeekSpec(
        interaction_type=InteractionType[step.interaction],
        target=step.target,
        offer=step.offer,
        min_participants=step.min_participants,
        max_participants=step.max_participants,
        queueable=step.queueable,
        formation_spec=step.formation_spec,
    )
    children.append(
        SeekNode(
            name=f"{node_name}/seek",
            agent=agent,
            spec=spec,
            ctx=ctx,
            duration_source=step.duration,
            rng=rng,
            wait_for_outcome=step.wait_for_outcome,
        )
    )
    if step.satisfies:
        children.append(SatisfyNode(name=f"{node_name}/satisfy", agent=agent, satisfies=step.satisfies))
    rider = None
    if step.attention is not None:
        rider = _attention_node(node_name, agent, step.attention, _wiring(node_name, world, agent_lookup, name_lookup, rng, dt), ctx)
    return _watched(node_name, watchdog, children, rider)


def _expand_pure_wait_step(
    node_name: str,
    step: StepDef,
    agent: BaseAgent,
    rng: np.random.Generator,
    dt: float,
    world: WorldKnowledge,
    is_bound_lookup: IsBoundLookup | None = None,
    im: InteractionManager | None = None,
    agent_lookup: AgentLookup | None = None,
    name_lookup: NameLookup | None = None,
) -> py_trees.composites.Parallel:
    ctx = StepContext(is_bound_lookup=is_bound_lookup, im=im)
    watchdog = PatienceWatchdogNode(
        name=f"{node_name}/watchdog",
        patience_source=step.patience,
        rng=rng,
        dt=dt,
    )
    children = _head(node_name, agent, step.attention)
    children.append(HoldNode(name=f"{node_name}/hold", agent=agent, duration_source=step.duration, rng=rng, dt=dt, ctx=ctx))
    if step.satisfies:
        children.append(SatisfyNode(name=f"{node_name}/satisfy", agent=agent, satisfies=step.satisfies))
    rider = None
    if step.attention is not None:
        rider = _attention_node(node_name, agent, step.attention, _wiring(node_name, world, agent_lookup, name_lookup, rng, dt), ctx, idle=True)
    return _watched(node_name, watchdog, children, rider)


def _expand_cancel_step(
    node_name: str,
    step: StepDef,
    agent: BaseAgent,
    rng: np.random.Generator,
    dt: float,
    world: WorldKnowledge,
    im: InteractionManager | None = None,
    agent_lookup: AgentLookup | None = None,
    name_lookup: NameLookup | None = None,
) -> py_trees.composites.Parallel:
    watchdog = PatienceWatchdogNode(
        name=f"{node_name}/watchdog",
        patience_source=step.patience,
        rng=rng,
        dt=dt,
    )
    children = _head(node_name, agent, step.attention)
    children.append(CancelNode(name=f"{node_name}/cancel", agent=agent, im=im))
    if step.satisfies:
        children.append(SatisfyNode(name=f"{node_name}/satisfy", agent=agent, satisfies=step.satisfies))
    rider = None
    if step.attention is not None:
        rider = _attention_node(node_name, agent, step.attention, _wiring(node_name, world, agent_lookup, name_lookup, rng, dt), StepContext(im=im))
    return _watched(node_name, watchdog, children, rider)


def _expand_block_step(
    node_name: str,
    step: StepDef,
    agent: BaseAgent,
    agent_lookup: AgentLookup,
    rng: np.random.Generator,
    dt: float,
    world: WorldKnowledge,
    im: InteractionManager | None = None,
    is_bound_lookup: IsBoundLookup | None = None,
    name_lookup: NameLookup | None = None,
) -> py_trees.composites.Parallel:
    assert isinstance(step.target, int), "_expand_block_step requires step.target: int"
    ctx = StepContext(im=im, is_bound_lookup=is_bound_lookup)
    watchdog = PatienceWatchdogNode(
        name=f"{node_name}/watchdog",
        patience_source=step.patience,
        rng=rng,
        dt=dt,
    )
    children = _head(node_name, agent, step.attention)
    children.append(
        BlockNode(
            name=f"{node_name}/block",
            agent=agent,
            target_agent_id=step.target,
            agent_lookup=agent_lookup,
            duration_source=step.duration,
            rng=rng,
            ctx=ctx,
        )
    )
    if step.satisfies:
        children.append(SatisfyNode(name=f"{node_name}/satisfy", agent=agent, satisfies=step.satisfies))
    rider = None
    if step.attention is not None:
        rider = _attention_node(node_name, agent, step.attention, _wiring(node_name, world, agent_lookup, name_lookup, rng, dt), ctx)
    return _watched(node_name, watchdog, children, rider)


def _expand_attention_step(
    node_name: str,
    step: AttentionStepDef,
    agent: BaseAgent,
    world: WorldKnowledge,
    rng: np.random.Generator,
    dt: float,
    is_bound_lookup: IsBoundLookup | None = None,
    im: InteractionManager | None = None,
    agent_lookup: AgentLookup | None = None,
    name_lookup: NameLookup | None = None,
) -> py_trees.composites.Parallel:
    ctx = StepContext(is_bound_lookup=is_bound_lookup, im=im)
    watchdog = PatienceWatchdogNode(
        name=f"{node_name}/watchdog",
        patience_source=step.patience,
        rng=rng,
        dt=dt,
    )
    children: list[py_trees.behaviour.Behaviour] = [
        ClearOutcomeNode(name=f"{node_name}/clear_outcome", agent=agent),
        _attention_node(node_name, agent, step.attention, _wiring(node_name, world, agent_lookup, name_lookup, rng, dt), ctx, bare=True, duration=step.duration),
    ]
    if step.satisfies:
        children.append(SatisfyNode(name=f"{node_name}/satisfy", agent=agent, satisfies=step.satisfies))
    return _watched(node_name, watchdog, children)


class _StepRecipe:
    __slots__ = ("autonomous", "node_name", "step_def", "action_defs", "utility_weights")

    def __init__(self, autonomous: bool, node_name: str, step_def: StepDef | GoToStepDef | AttentionStepDef, action_defs: Mapping[str, ActionDef], utility_weights: Mapping[str, float]) -> None:
        self.autonomous = autonomous
        self.node_name = node_name
        self.step_def = step_def
        self.action_defs = action_defs
        self.utility_weights = utility_weights

    def build(self, agent: BaseAgent, world: WorldKnowledge, event_bus: EventBus, rng: np.random.Generator, dt: float, agent_lookup: AgentLookup | None = None, pool: AgentPool | None = None, is_bound_lookup: IsBoundLookup | None = None, im: InteractionManager | None = None, name_lookup: NameLookup | None = None) -> py_trees.behaviour.Behaviour:
        step = self.step_def
        if isinstance(step, AttentionStepDef):
            return _expand_attention_step(self.node_name, step, agent, world, rng, dt, is_bound_lookup=is_bound_lookup, im=im, agent_lookup=agent_lookup, name_lookup=name_lookup)
        if isinstance(step, GoToStepDef):
            return _expand_go_to_step(self.node_name, step, agent, world, rng, dt, pool=pool, is_bound_lookup=is_bound_lookup, im=im, agent_lookup=agent_lookup, name_lookup=name_lookup)
        if step.autonomous:
            if step.attention is not None:
                raise ValueError(f"{self.node_name}: attention is not supported on autonomous steps")
            return py_trees.composites.Sequence(
                name=self.node_name,
                memory=True,
                children=[
                    ClearGestureNode(name=f"{self.node_name}/clear_gesture", agent=agent),
                    AutonomousNode(
                        name=f"{self.node_name}/autonomous",
                        step_def=step,
                        agent=agent,
                        action_defs=dict(self.action_defs),
                        utility_weights=dict(self.utility_weights),
                        world=world,
                        event_bus=event_bus,
                        rng=rng,
                        dt=dt,
                    ),
                ],
            )
        if step.cancel:
            return _expand_cancel_step(self.node_name, step, agent, rng, dt, world, im=im, agent_lookup=agent_lookup, name_lookup=name_lookup)
        if step.interaction == "BLOCK":
            if agent_lookup is None:
                raise ValueError(f"{self.node_name}: interaction BLOCK requires agent_lookup to be threaded into BehaviorTreeFactory.build")
            return _expand_block_step(self.node_name, step, agent, agent_lookup, rng, dt, world, im=im, is_bound_lookup=is_bound_lookup, name_lookup=name_lookup)
        if step.interaction is not None:
            return _expand_interaction_step(self.node_name, step, agent, world, rng, dt, is_bound_lookup=is_bound_lookup, im=im, agent_lookup=agent_lookup, name_lookup=name_lookup)
        return _expand_pure_wait_step(self.node_name, step, agent, rng, dt, world, is_bound_lookup=is_bound_lookup, im=im, agent_lookup=agent_lookup, name_lookup=name_lookup)


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
        pool: AgentPool | None = None,
        is_bound_lookup: IsBoundLookup | None = None,
        im: InteractionManager | None = None,
        name_lookup: NameLookup | None = None,
    ) -> py_trees.trees.BehaviourTree:
        compiled_sequences: dict[str, py_trees.behaviour.Behaviour] = {}
        for seq_name, recipes in self._seq_recipes.items():
            children = [r.build(agent, world, event_bus, rng, dt, agent_lookup, pool, is_bound_lookup, im, name_lookup) for r in recipes]
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
            im=im,
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
    pool: AgentPool | None = None,
    is_bound_lookup: IsBoundLookup | None = None,
    im: InteractionManager | None = None,
    name_lookup: NameLookup | None = None,
) -> py_trees.trees.BehaviourTree | None:
    if agent_type.mode == "simple":
        _logger.debug(f"Agent {agent.state.agent_id} ({agent_type.name}): simple mode, no behavior tree")
        return None

    if not agent_type.sequences:
        return None

    factory = BehaviorTreeFactory(agent_type)
    bt = factory.build(agent, world, event_bus, rng, dt, agent_lookup, pool, is_bound_lookup, im=im, name_lookup=name_lookup)
    _logger.debug(f"Agent {agent.state.agent_id} ({agent_type.name}): compiled {len(agent_type.sequences)} sequence(s), initial={agent_type.initial_sequence}")
    return bt
