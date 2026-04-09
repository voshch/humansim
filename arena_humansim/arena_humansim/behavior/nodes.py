import math

import numpy as np
import py_trees
from rclpy.logging import get_logger

from arena_humansim.agents import (
    ActionDef,
    BaseAgent,
    NeedCondition,
    ParamDist,
    SequenceDef,
    StepDef,
)
from arena_humansim.manager.interaction_manager import CommandType
from arena_humansim.manager.world_knowledge import WorldKnowledge
from arena_humansim.utils.event_bus import EventBus
from arena_humansim.utils.types import (
    BehaviorTreeMovement,
    HighLevelCommand,
    InteractionOutcome,
    InteractionType,
    Pose2D,
)

_bt_logger = get_logger("behavior_tree")


def check_condition(value: float, condition: NeedCondition) -> bool:
    if condition.below is not None and value >= condition.below:
        return False
    if condition.above is not None and value <= condition.above:
        return False
    return True


def preconditions_met(
    needs: dict,
    when: dict[str, NeedCondition],
) -> bool:
    for need_name, condition in when.items():
        need = needs.get(need_name)
        if need is None:
            return False
        if not check_condition(need.value, condition):
            return False
    return True


def score_actions(
    needs: dict,
    actions: dict[str, ActionDef],
    utility_weights: dict[str, float],
    world: WorldKnowledge,
) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []

    for name, action in actions.items():
        if not preconditions_met(needs, action.when):
            continue

        utility = 0.0
        for need_name, delta in action.satisfies.items():
            need = needs.get(need_name)
            if need is None:
                continue
            urgency = (100.0 - need.value) / 100.0
            weight = utility_weights.get(need_name, 1.0)
            utility += urgency * weight * (delta / 100.0)

        # Queue length penalty for object-targeted actions
        if action.target_object:
            q_len = world.queue_length(action.target_object)
            penalty = q_len * 0.05
            utility *= max(0.2, 1.0 - penalty)

        if utility > 0.0:
            scored.append((name, utility))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


_ARRIVAL_THRESHOLD = 0.5  # metres


def _sample_param_dist(dist: ParamDist, rng: np.random.Generator) -> float:
    value = rng.normal(dist.mean, dist.std) if dist.std > 0 else dist.mean
    return float(np.clip(value, dist.clip_low, dist.clip_high))


def _nav_command(agent: BaseAgent, target_pose: Pose2D) -> HighLevelCommand:
    return HighLevelCommand(
        agent_id=agent.state.agent_id,
        type=CommandType.NAVIGATE,
        target_pose=target_pose,
        desired_velocity=agent.state.desired_velocity,
    )


def _interaction_command(
    agent: BaseAgent,
    interaction_name: str,
    target_agent: int = -1,
    duration: float = -1.0,
) -> HighLevelCommand:
    return HighLevelCommand(
        agent_id=agent.state.agent_id,
        type=CommandType.ADVERTISE,
        desired_velocity=agent.state.desired_velocity,
        interaction_type=InteractionType[interaction_name].value,
        target_agent=target_agent,
        interaction_duration=duration,
    )


def _at_target(agent: BaseAgent, target_pose: Pose2D) -> bool:
    dx = agent.state.pose.x - target_pose.x
    dy = agent.state.pose.y - target_pose.y
    return math.hypot(dx, dy) < _ARRIVAL_THRESHOLD


class ConcreteStepNode(py_trees.behaviour.Behaviour):
    def __init__(
        self,
        name: str,
        step_def: StepDef,
        agent: BaseAgent,
        world: WorldKnowledge,
        rng: np.random.Generator,
        dt: float = 0.05,
    ) -> None:
        super().__init__(name=name)
        self._step = step_def
        self._agent = agent
        self._world = world
        self._rng = rng
        self._dt = dt

        self._duration: float | None = None
        self._patience: float | None = None
        self._elapsed: float = 0.0
        self._target_pose: Pose2D | None = None

    def initialise(self) -> None:
        self._elapsed = 0.0
        self._duration = _sample_param_dist(self._step.duration, self._rng) if self._step.duration is not None else None
        self._patience = _sample_param_dist(self._step.patience, self._rng) if self._step.patience is not None else None
        self._target_pose = None
        self._clear_outcome()

        if self._step.target_object:
            obj = self._world.nearest_object(self._step.target_object, self._agent.state.pose)
            if obj is not None:
                self._target_pose = obj.pose

    def update(self) -> py_trees.common.Status:
        outcome = self._read_outcome()
        if outcome == InteractionOutcome.COMPLETED:
            self._apply_satisfaction()
            return py_trees.common.Status.SUCCESS
        if outcome == InteractionOutcome.INTERRUPTED:
            return py_trees.common.Status.FAILURE

        if self._target_pose is not None and not _at_target(self._agent, self._target_pose):
            self._agent.movement.command = _nav_command(self._agent, self._target_pose)
            self._elapsed += self._dt
            if self._patience is not None and self._elapsed >= self._patience:
                return py_trees.common.Status.FAILURE
            return py_trees.common.Status.RUNNING

        if self._step.interaction is not None:
            self._agent.movement.command = _interaction_command(
                self._agent, self._step.interaction
            )

        if self._duration is not None:
            if self._elapsed >= self._duration:
                self._apply_satisfaction()
                return py_trees.common.Status.SUCCESS
            self._elapsed += self._dt
            if self._patience is not None and self._elapsed >= self._patience:
                return py_trees.common.Status.FAILURE
            return py_trees.common.Status.RUNNING

        if self._step.interaction is None:
            self._apply_satisfaction()
            return py_trees.common.Status.SUCCESS

        self._elapsed += self._dt
        if self._patience is not None and self._elapsed >= self._patience:
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING

    def _apply_satisfaction(self) -> None:
        if self._step.satisfies and self._agent.needs is not None:
            self._agent.needs.satisfy(self._step.satisfies)

    def _read_outcome(self) -> int | None:
        mv = self._agent.movement
        if isinstance(mv, BehaviorTreeMovement) and mv.last_outcome is not None:
            outcome = mv.last_outcome
            mv.last_outcome = None
            return outcome
        return None

    def _clear_outcome(self) -> None:
        mv = self._agent.movement
        if isinstance(mv, BehaviorTreeMovement):
            mv.last_outcome = None


class AutonomousNode(py_trees.behaviour.Behaviour):
    def __init__(
        self,
        name: str,
        step_def: StepDef,
        agent: BaseAgent,
        action_defs: dict[str, ActionDef],
        utility_weights: dict[str, float],
        world: WorldKnowledge,
        event_bus: EventBus,
        rng: np.random.Generator,
        dt: float = 0.05,
    ) -> None:
        super().__init__(name=name)
        self._step = step_def
        self._agent = agent
        self._world = world
        self._event_bus = event_bus
        self._rng = rng
        self._dt = dt
        self._utility_weights = utility_weights

        self._actions = self._filter_actions(action_defs)

        self._duration: float | None = None
        self._elapsed: float = 0.0

    def _filter_actions(
        self, action_defs: dict[str, ActionDef]
    ) -> dict[str, ActionDef]:
        if self._step.allowed_actions is not None:
            allowed = set(self._step.allowed_actions)
            return {k: v for k, v in action_defs.items() if k in allowed}
        if self._step.blocked_actions is not None:
            blocked = set(self._step.blocked_actions)
            return {k: v for k, v in action_defs.items() if k not in blocked}
        return dict(action_defs)

    def initialise(self) -> None:
        self._elapsed = 0.0
        self._duration = (
            _sample_param_dist(self._step.duration, self._rng)
            if self._step.duration is not None
            else None
        )

    def update(self) -> py_trees.common.Status:
        agent_id = self._agent.state.agent_id
        needs = self._agent.needs.needs if self._agent.needs else {}

        if self._step.until is not None:
            if self._event_bus.has(self._step.until, agent_id):
                self._event_bus.consume(self._step.until, agent_id)
                return py_trees.common.Status.SUCCESS

        if self._step.until_need is not None:
            if preconditions_met(needs, self._step.until_need):
                return py_trees.common.Status.SUCCESS

        if self._duration is not None and self._elapsed >= self._duration:
            return py_trees.common.Status.SUCCESS

        scored = score_actions(needs, self._actions, self._utility_weights, self._world)

        if scored:
            best_name, _score = scored[0]
            best_action = self._actions[best_name]

            if best_action.target_object:
                obj = self._world.nearest_object(
                    best_action.target_object, self._agent.state.pose
                )
                if obj is not None:
                    self._agent.movement.command = _nav_command(self._agent, obj.pose)
            elif best_action.interaction:
                self._agent.movement.command = _interaction_command(
                    self._agent, best_action.interaction
                )
            else:
                self._agent.movement.command = None
        else:
            self._agent.movement.command = None

        self._elapsed += self._dt
        return py_trees.common.Status.RUNNING


class NeedsDecayNode(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, agent: BaseAgent, dt: float) -> None:
        super().__init__(name=name)
        self._agent = agent
        self._dt = dt

    def update(self) -> py_trees.common.Status:
        if self._agent.needs is not None:
            self._agent.needs.decay(self._dt)
        return py_trees.common.Status.SUCCESS


class SequenceStateMachine(py_trees.behaviour.Behaviour):
    def __init__(
        self,
        name: str,
        sequences: dict[str, py_trees.behaviour.Behaviour],
        sequence_defs: dict[str, SequenceDef],
        initial: str,
        agent: BaseAgent,
    ) -> None:
        super().__init__(name=name)
        self._sequences = sequences
        self._sequence_defs = sequence_defs
        self._initial = initial
        self._agent = agent
        self._current_name: str = initial
        self._current_node: py_trees.behaviour.Behaviour = sequences[initial]

    def initialise(self) -> None:
        self._current_name = self._initial
        self._current_node = self._sequences[self._initial]
        self._current_node.initialise()

    def update(self) -> py_trees.common.Status:
        # 1. Check conditional transitions
        redirect = self._check_transitions()
        if redirect is not None:
            return self._goto(redirect)

        # 2. Tick current sequence
        status = self._current_node.update()

        if status == py_trees.common.Status.SUCCESS:
            seq_def = self._sequence_defs[self._current_name]
            self._current_node.terminate(status)
            if seq_def.then is None:
                return py_trees.common.Status.SUCCESS
            return self._goto(seq_def.then)

        if status == py_trees.common.Status.FAILURE:
            seq_def = self._sequence_defs[self._current_name]
            self._current_node.terminate(status)
            if seq_def.on_failure is None:
                return py_trees.common.Status.FAILURE
            return self._goto(seq_def.on_failure)

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        self._current_node.terminate(new_status)

    def _goto(self, target: str) -> py_trees.common.Status:
        if target not in self._sequences:
            _bt_logger.warning(
                f'Agent {self._agent.state.agent_id}: invalid transition target "{target}"'
            )
            return py_trees.common.Status.FAILURE
        _bt_logger.debug(
            f"Agent {self._agent.state.agent_id}: {self._current_name} -> {target}"
        )
        self._current_node.terminate(py_trees.common.Status.FAILURE)
        self._current_name = target
        self._current_node = self._sequences[target]
        self._current_node.initialise()
        return py_trees.common.Status.RUNNING

    def _check_transitions(self) -> str | None:
        seq_def = self._sequence_defs.get(self._current_name)
        if seq_def is None:
            return None
        needs = self._agent.needs.needs if self._agent.needs else {}
        for transition in seq_def.transitions:
            if preconditions_met(needs, transition.when):
                return transition.goto
        return None
