import math
from typing import TYPE_CHECKING

import py_trees

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.behavior.nodes.helpers import (
    _at_target,
    _bt_logger,
    _nav_command,
    _resolve_interaction_radius,
)
from arena_humansim.core.behavior.step_context import StepContext
from arena_humansim.core.world_knowledge import WorldKnowledge, WorldObject
from arena_humansim.utils import DISTANCE_TOLERANCE
from arena_humansim.utils.types import Pose2D

if TYPE_CHECKING:
    from arena_humansim.core.pool import AgentPool


def _approach_pose_for(obj: WorldObject, world: WorldKnowledge) -> Pose2D:
    spec = obj.formation
    if spec is None or spec.type != "line":
        return obj.pose
    params = spec.params or {}
    front_offset = float(params.get("front_offset", 0.0))
    base_step = float(params.get("base_step", 1.0))
    slot_index = world.participants_count_for_object(obj.object_id) + world.queue_length_for_object(obj.object_id)
    offset = front_offset + slot_index * base_step
    back = obj.pose.theta + math.pi
    return Pose2D(
        x=obj.pose.x + offset * math.cos(back),
        y=obj.pose.y + offset * math.sin(back),
        theta=obj.pose.theta,
    )


class ResolveObjectNode(py_trees.behaviour.Behaviour):
    def __init__(
        self,
        name: str,
        agent: BaseAgent,
        world: WorldKnowledge,
        target_object_type: str | None,
        target_object_id: str | None,
        ctx: StepContext,
        step_interaction_radius: float | None = None,
        interaction_name: str | None = None,
    ) -> None:
        super().__init__(name)
        self._agent = agent
        self._world = world
        self._target_object_type = target_object_type
        self._target_object_id = target_object_id
        self._ctx = ctx
        self._step_interaction_radius = step_interaction_radius
        self._interaction_name = interaction_name
        self._resolved: bool = False

    def initialise(self) -> None:
        self._resolved = False
        obj: WorldObject | None = None
        if self._target_object_id:
            obj = self._world.get(self._target_object_id)
            if obj is None:
                _bt_logger.warning(f"Agent {self._agent.state.agent_id}: step {self.name} could not resolve target_object_id={self._target_object_id!r}")
        elif self._target_object_type:
            obj = self._world.nearest_object(self._target_object_type, self._agent.state.pose, exclude_full=False)
            if obj is None:
                _bt_logger.warning(f"Agent {self._agent.state.agent_id}: step {self.name} could not resolve target_object_type={self._target_object_type!r}")

        if obj is not None:
            self._ctx.target_pose = _approach_pose_for(obj, self._world)
            self._ctx.target_object_id = obj.object_id
            self._ctx.interaction_radius = _resolve_interaction_radius(obj, self._step_interaction_radius, self._interaction_name)
            self._resolved = True

    def update(self) -> py_trees.common.Status:
        if self._resolved:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class GoToNode(py_trees.behaviour.Behaviour):
    def __init__(
        self,
        name: str,
        agent: BaseAgent,
        target_pose: Pose2D | None = None,
        ctx: StepContext | None = None,
        world: WorldKnowledge | None = None,
        pool: "AgentPool | None" = None,
    ) -> None:
        super().__init__(name)
        if target_pose is None and ctx is None:
            raise ValueError("GoToNode requires either target_pose or ctx")
        self._agent = agent
        self._literal_pose = target_pose
        self._ctx = ctx
        self._world = world
        self._pool = pool

    def _target(self) -> tuple[Pose2D | None, float]:
        if self._literal_pose is not None:
            return self._literal_pose, DISTANCE_TOLERANCE
        assert self._ctx is not None
        if self._world is not None and self._ctx.target_object_id is not None:
            obj = self._world.get(self._ctx.target_object_id)
            if obj is not None:
                self._ctx.target_pose = _approach_pose_for(obj, self._world)
        return self._ctx.target_pose, self._ctx.interaction_radius

    def _arrived(self, target: Pose2D, tolerance: float) -> bool:
        # Literal go_to: couple to the physics-level arrival latch so "done" matches the stop condition.
        if self._literal_pose is not None and self._pool is not None:
            idx = self._pool._id_to_idx.get(self._agent.state.agent_id)
            if idx is not None:
                return bool(self._pool.latched[idx])
        return _at_target(self._agent, target, tolerance)

    def update(self) -> py_trees.common.Status:
        target, tolerance = self._target()
        if target is None:
            return py_trees.common.Status.FAILURE
        if self._arrived(target, tolerance):
            return py_trees.common.Status.SUCCESS
        # Formations clobber command every tick; re-emit unconditionally.
        self._agent.movement.command = _nav_command(agent=self._agent, target_pose=target)
        return py_trees.common.Status.RUNNING
