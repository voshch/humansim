import numpy as np
import py_trees

from arena_humansim.core.agents import ActionDef, BaseAgent, StepDef
from arena_humansim.core.behavior.nodes.helpers import (
    _bt_logger,
    _interaction_command,
    _nav_command,
    _sample_param_dist,
)
from arena_humansim.core.behavior.nodes.utility import preconditions_met, score_actions
from arena_humansim.core.world_knowledge import WorldKnowledge
from arena_humansim.utils import DT
from arena_humansim.utils.event_bus import EventBus


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

            if best_action.target_object_id:
                obj = self._world.get(best_action.target_object_id)
                if obj is not None:
                    self._agent.movement.command = _nav_command(self._agent, obj.pose)
                else:
                    _bt_logger.warning(f"Agent {agent_id}: step {self.name} could not resolve target_object_id={best_action.target_object_id!r}")
            elif best_action.target_object_type:
                obj = self._world.nearest_object(best_action.target_object_type, self._agent.state.pose)
                if obj is not None:
                    self._agent.movement.command = _nav_command(self._agent, obj.pose)
                else:
                    _bt_logger.warning(f"Agent {agent_id}: step {self.name} could not resolve target_object_type={best_action.target_object_type!r}")
            elif best_action.interaction:
                self._agent.movement.command = _interaction_command(self._agent, best_action.interaction)
            else:
                self._agent.movement.command = None
        else:
            self._agent.movement.command = None

        self._elapsed += self._dt
        return py_trees.common.Status.RUNNING
