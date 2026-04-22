import numpy as np
import py_trees

from arena_humansim.core.agents import ActionDef, BaseAgent, StepDef
from arena_humansim.core.behavior.nodes.helpers import (
    _at_target,
    _bt_logger,
    _nav_command,
    _resolve_interaction_radius,
    _sample_param_dist,
    _seek_command,
)
from arena_humansim.core.behavior.nodes.utility import preconditions_met, score_actions
from arena_humansim.core.interaction_kinds import HandleKind, InteractionType
from arena_humansim.core.world_knowledge import WorldKnowledge
from arena_humansim.utils import DT
from arena_humansim.utils.event_bus import EventBus
from arena_humansim.utils.types import SeekSpec


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
        dt: float = DT,
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

    def _filter_actions(self, action_defs: dict[str, ActionDef]) -> dict[str, ActionDef]:
        if self._step.allowed_actions is not None:
            allowed = set(self._step.allowed_actions)
            return {k: v for k, v in action_defs.items() if k in allowed}
        if self._step.blocked_actions is not None:
            blocked = set(self._step.blocked_actions)
            return {k: v for k, v in action_defs.items() if k not in blocked}
        return dict(action_defs)

    def initialise(self) -> None:
        self._elapsed = 0.0
        self._duration = _sample_param_dist(self._step.duration, self._rng) if self._step.duration is not None else None

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

            interaction_type: InteractionType | None = None
            symmetric = False
            if best_action.interaction:
                try:
                    interaction_type = InteractionType[best_action.interaction]
                    symmetric = interaction_type.kind.handle.kind == HandleKind.NONE
                except KeyError:
                    interaction_type = None

            if best_action.target:
                obj = self._world.resolve(best_action.target, self._agent.state.pose, exclude_full=True)
                if obj is None:
                    _bt_logger.warning(f"Agent {agent_id}: step {self.name} could not resolve target={best_action.target!r}")
                    self._agent.movement.command = None
                else:
                    tolerance = _resolve_interaction_radius(obj, None, best_action.interaction)
                    if symmetric and interaction_type is not None and _at_target(self._agent, obj.pose, tolerance):
                        self._agent.movement.command = _seek_command(self._agent, SeekSpec(interaction_type=interaction_type))
                    else:
                        self._agent.movement.command = _nav_command(self._agent, obj.pose)
            elif interaction_type is not None:
                self._agent.movement.command = _seek_command(self._agent, SeekSpec(interaction_type=interaction_type))
            else:
                self._agent.movement.command = None
        else:
            self._agent.movement.command = None

        self._elapsed += self._dt
        return py_trees.common.Status.RUNNING
