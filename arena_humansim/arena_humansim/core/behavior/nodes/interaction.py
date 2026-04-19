from collections.abc import Callable

import numpy as np
import py_trees

from arena_humansim.core.agents import BaseAgent, ParamDist
from arena_humansim.core.behavior.nodes.helpers import _at_target, _interaction_command, _nav_command, _sample_param_dist
from arena_humansim.core.behavior.step_context import StepContext
from arena_humansim.core.interaction_manager import CommandType
from arena_humansim.utils import DISTANCE_TOLERANCE
from arena_humansim.utils.types import BehaviorTreeMovement, HighLevelCommand, InteractionOutcome, Pose2D

AgentLookup = Callable[[int], "BaseAgent | None"]


class AcceptInteractionNode(py_trees.behaviour.Behaviour):
    """Bare ADVERTISE without object-resolution or navigation.

    Parks until the IM pairs this ad into an interaction, then watches outcome.
    Used for passive service roles (vendors, greeters) and any interaction that
    isn't anchored to a world object.
    """

    def __init__(
        self,
        name: str,
        agent: BaseAgent,
        interaction: str,
        duration_source: ParamDist | None,
        rng: np.random.Generator,
        service_tag: str | None = None,
    ) -> None:
        super().__init__(name)
        self._agent = agent
        self._interaction = interaction
        self._duration_source = duration_source
        self._rng = rng
        self._service_tag = service_tag
        self._duration: float | None = None
        self._advertised: bool = False

    def initialise(self) -> None:
        self._duration = _sample_param_dist(self._duration_source, self._rng) if self._duration_source is not None else None
        self._advertised = False

    def update(self) -> py_trees.common.Status:
        if not self._advertised:
            self._agent.movement.command = _interaction_command(
                self._agent,
                self._interaction,
                duration=self._duration,
                service_tag=self._service_tag,
            )
            self._advertised = True

        mv = self._agent.movement
        if isinstance(mv, BehaviorTreeMovement) and mv.last_outcome is not None:
            outcome = mv.last_outcome
            if outcome == InteractionOutcome.COMPLETED:
                mv.last_outcome = None
                return py_trees.common.Status.SUCCESS
            if outcome == InteractionOutcome.INTERRUPTED:
                mv.last_outcome = None
                return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        del new_status
        if self._advertised:
            self._agent.movement.command = HighLevelCommand(
                agent_id=self._agent.state.agent_id,
                type=CommandType.STOP,
                interaction_target=-1,
            )
            self._advertised = False


class AdvertiseInteractionNode(py_trees.behaviour.Behaviour):
    def __init__(
        self,
        name: str,
        agent: BaseAgent,
        interaction: str,
        ctx: StepContext,
        duration_source: ParamDist | None,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(name)
        self._agent = agent
        self._interaction = interaction
        self._ctx = ctx
        self._duration_source = duration_source
        self._rng = rng
        self._duration: float | None = None

    def initialise(self) -> None:
        self._duration = _sample_param_dist(self._duration_source, self._rng) if self._duration_source is not None else None
        self._ctx.advertised = False

    def update(self) -> py_trees.common.Status:
        if not self._ctx.advertised:
            self._agent.movement.command = _interaction_command(
                self._agent,
                self._interaction,
                duration=self._duration,
                object_id=self._ctx.target_object_id,
                target_pose=self._ctx.target_pose,
            )
            self._ctx.advertised = True

        mv = self._agent.movement
        if isinstance(mv, BehaviorTreeMovement) and mv.last_outcome is not None:
            outcome = mv.last_outcome
            if outcome == InteractionOutcome.COMPLETED:
                mv.last_outcome = None
                return py_trees.common.Status.SUCCESS
            if outcome == InteractionOutcome.INTERRUPTED:
                mv.last_outcome = None
                return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        # IM releases the participant slot only on STOP; fire on any exit, including SUCCESS.
        del new_status
        if self._ctx.advertised:
            self._agent.movement.command = HighLevelCommand(
                agent_id=self._agent.state.agent_id,
                type=CommandType.STOP,
                interaction_target=-1,
            )
            self._ctx.advertised = False


class BlockNode(py_trees.behaviour.Behaviour):
    """Pursue a target agent and advertise a BLOCK interaction on arrival.

    Each tick while pursuing: predicts the target's position `lookahead` seconds
    ahead from its current velocity, boosts own desired_velocity to
    `target.desired_velocity * velocity_boost`, and emits NAVIGATE toward the
    prediction. Once within `tolerance` of the prediction, emits an ADVERTISE
    (BLOCK, target_agent=target.id) exactly once and watches outcome.
    Restores pre-block desired_velocity and emits STOP on terminate.
    """

    def __init__(
        self,
        name: str,
        agent: BaseAgent,
        target_agent_id: int,
        agent_lookup: AgentLookup,
        duration_source: ParamDist | None,
        rng: np.random.Generator,
        tolerance: float = DISTANCE_TOLERANCE,
        lookahead: float = 1.0,
        velocity_boost: float = 1.5,
    ) -> None:
        super().__init__(name)
        self._agent = agent
        self._target_agent_id = target_agent_id
        self._agent_lookup = agent_lookup
        self._duration_source = duration_source
        self._rng = rng
        self._tolerance = tolerance
        self._lookahead = lookahead
        self._velocity_boost = velocity_boost
        self._target: BaseAgent | None = None
        self._duration: float | None = None
        self._advertised: bool = False
        self._prev_desired_vel: float = agent.state.desired_velocity

    def initialise(self) -> None:
        self._target = self._agent_lookup(self._target_agent_id)
        self._duration = _sample_param_dist(self._duration_source, self._rng) if self._duration_source is not None else None
        self._advertised = False
        self._prev_desired_vel = self._agent.state.desired_velocity

    def _future_pose(self, target: BaseAgent) -> Pose2D:
        state = target.state
        vx, vy = state.velocity[0], state.velocity[1]
        return Pose2D(
            x=state.pose.x + vx * self._lookahead,
            y=state.pose.y + vy * self._lookahead,
            theta=state.pose.theta,
        )

    def update(self) -> py_trees.common.Status:
        if self._target is None:
            return py_trees.common.Status.FAILURE

        if not self._advertised:
            self._agent.state.desired_velocity = self._target.state.desired_velocity * self._velocity_boost
            future = self._future_pose(self._target)
            if _at_target(self._agent, future, self._tolerance):
                self._agent.movement.command = _interaction_command(
                    self._agent,
                    "BLOCK",
                    target_agent=self._target.state.agent_id,
                    duration=self._duration,
                )
                self._advertised = True
                return py_trees.common.Status.RUNNING
            self._agent.movement.command = _nav_command(self._agent, future)
            return py_trees.common.Status.RUNNING

        mv = self._agent.movement
        if isinstance(mv, BehaviorTreeMovement) and mv.last_outcome is not None:
            outcome = mv.last_outcome
            if outcome == InteractionOutcome.COMPLETED:
                mv.last_outcome = None
                return py_trees.common.Status.SUCCESS
            if outcome == InteractionOutcome.INTERRUPTED:
                mv.last_outcome = None
                return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        del new_status
        self._agent.state.desired_velocity = self._prev_desired_vel
        if self._advertised:
            self._agent.movement.command = HighLevelCommand(
                agent_id=self._agent.state.agent_id,
                type=CommandType.STOP,
                interaction_target=-1,
            )
            self._advertised = False
