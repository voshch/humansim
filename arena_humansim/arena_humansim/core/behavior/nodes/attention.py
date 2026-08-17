from __future__ import annotations

import math
from collections.abc import Callable

import attrs
import numpy as np
import py_trees

from arena_humansim.core.agents import BaseAgent, ParamDist
from arena_humansim.core.agents.types import AttentionDef, AttentionRef, Pose3, RelativeRef, RobotRef
from arena_humansim.core.behavior.nodes.helpers import _bt_logger, _nav_command, _sample_param_dist
from arena_humansim.core.behavior.step_context import StepContext
from arena_humansim.core.pool import KIND_ROBOT
from arena_humansim.core.world_knowledge import WorldKnowledge
from arena_humansim.utils.types import BehaviorTreeMovement, CommandType, GestureIntent, Pose2D, WaypointMovement

FACE_ENTER_RAD = 0.25
FACE_KEEP_RAD = 0.6
FACE_TIMEOUT_S = 4.0
RESOLVE_TIMEOUT_S = 4.0
HALT_EPS = 1e-6
GESTURE_Z_OBJECT = 0.8
GESTURE_Z_AGENT = 1.2
NO_GESTURE = "none"

REF_PARTNER = "partner"
REF_PARTNERS = "partners"
REF_TARGET = "target"
REF_GOAL = "goal"
_KEYWORDS = (REF_PARTNER, REF_PARTNERS, REF_TARGET, REF_GOAL)

AgentLookup = Callable[[int], BaseAgent | None]
NameLookup = Callable[[str, int | None], int | None]
XYZ = tuple[float, float, float]


@attrs.frozen
class _AgentTarget:
    agent_id: int


@attrs.frozen
class _ObjectTarget:
    object_id: str


_Handle = str | Pose3 | RelativeRef | _AgentTarget | _ObjectTarget


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class AttentionNode(py_trees.behaviour.Behaviour):
    """Resolve the attention refs each tick, face the current one when allowed, and raise the gesture intent. Bare mode halts and owns the step, rider mode never finishes."""

    def __init__(
        self,
        name: str,
        agent: BaseAgent,
        attention: AttentionDef,
        world: WorldKnowledge,
        agent_lookup: AgentLookup,
        name_lookup: NameLookup,
        rng: np.random.Generator,
        dt: float,
        ctx: StepContext,
        bare: bool = False,
        idle: bool = False,
        duration: ParamDist | None = None,
    ) -> None:
        super().__init__(name)
        self._agent = agent
        self._att = attention
        self._world = world
        self._agent_lookup = agent_lookup
        self._name_lookup = name_lookup
        self._rng = rng
        self._dt = dt
        self._ctx = ctx
        self._bare = bare
        self._idle = bare or idle
        self._duration_source = duration
        at = attention.at
        self._refs: tuple[AttentionRef, ...] = at if isinstance(at, tuple) else () if at is None else (at,)
        self._multi = isinstance(at, tuple) or at == REF_PARTNERS
        self._cycle = not bare or duration is not None
        self._handles: dict[int, _Handle] = {}
        self._duration: float | None = None
        self._elapsed = 0.0
        self._face_elapsed = 0.0
        self._resolve_elapsed = 0.0
        self._face_gave_up = False
        self._raised = False
        self._refacing = False
        self._idx = 0
        self._dwell_elapsed = 0.0
        self._warned = False

    def initialise(self) -> None:
        self._handles = {}
        self._duration = _sample_param_dist(self._duration_source, self._rng) if self._duration_source is not None else None
        self._elapsed = 0.0
        self._face_elapsed = 0.0
        self._resolve_elapsed = 0.0
        self._face_gave_up = False
        self._raised = False
        self._refacing = False
        self._idx = 0
        self._dwell_elapsed = 0.0
        self._warned = False

    def _bt_mv(self) -> BehaviorTreeMovement | None:
        mv = self._agent.movement
        return mv if isinstance(mv, BehaviorTreeMovement) else None

    def _bound(self) -> bool:
        lookup = self._ctx.is_bound_lookup
        return bool(lookup(self._agent.state.agent_id)) if lookup is not None else False

    def _z(self, default: float) -> float:
        return self._att.at_z if self._att.at_z is not None else default

    def _resolve_ref(self, ref: AttentionRef) -> _Handle | None:
        if isinstance(ref, (Pose3, RelativeRef)):
            return ref
        if isinstance(ref, RobotRef):
            aid = self._name_lookup(ref.name, KIND_ROBOT)
            return _AgentTarget(aid) if aid is not None else None
        if isinstance(ref, int):
            return _AgentTarget(ref) if self._agent_lookup(ref) is not None else None
        if ref in _KEYWORDS:
            return ref
        obj = self._world.find(ref)
        if obj is not None:
            return _ObjectTarget(obj.object_id)
        aid = self._name_lookup(ref, None)
        if aid is not None:
            return _AgentTarget(aid)
        obj = self._world.resolve(ref, self._agent.state.pose, exclude_full=False)
        return _ObjectTarget(obj.object_id) if obj is not None else None

    def _handle(self, i: int) -> _Handle | None:
        handle = self._handles.get(i)
        if isinstance(handle, _AgentTarget) and self._agent_lookup(handle.agent_id) is None:
            del self._handles[i]
            handle = None
        if handle is None:
            handle = self._resolve_ref(self._refs[i])
            if handle is not None:
                self._handles[i] = handle
        return handle

    def _other_participants(self) -> list[int]:
        mv = self._bt_mv()
        im = self._ctx.im
        if mv is None or im is None or mv.interaction_id is None:
            return []
        interaction = im.interactions.get(mv.interaction_id)
        if interaction is None:
            return []
        own = self._agent.state.agent_id
        return [aid for aid in interaction.participants if aid != own]

    def _agent_xyz(self, agent_id: int) -> XYZ | None:
        other = self._agent_lookup(agent_id)
        if other is None:
            return None
        pose = other.state.pose
        return (pose.x, pose.y, self._z(GESTURE_Z_AGENT))

    def _pose_xyz(self, pose: Pose2D | None) -> XYZ | None:
        if pose is None:
            return None
        return (pose.x, pose.y, self._z(GESTURE_Z_OBJECT))

    def _goal_pose(self) -> Pose2D | None:
        mv = self._agent.movement
        if isinstance(mv, BehaviorTreeMovement):
            cmd = mv.command
            if cmd is None or cmd.type != CommandType.NAVIGATE or cmd.desired_velocity == 0.0:
                return None
            own = self._agent.state.pose
            if math.hypot(cmd.target_pose.x - own.x, cmd.target_pose.y - own.y) <= HALT_EPS:
                return None
            return cmd.target_pose
        if isinstance(mv, WaypointMovement) and mv.waypoints:
            return mv.waypoints[mv.index % len(mv.waypoints)]
        return None

    def _target_pose(self) -> Pose2D | None:
        if self._ctx.target_object_id is not None:
            obj = self._world.get(self._ctx.target_object_id)
            if obj is not None:
                return obj.pose
        return self._ctx.target_pose

    def _relative_xyz(self, ref: RelativeRef) -> XYZ:
        own = self._agent.state.pose
        az = own.theta + math.radians(ref.azimuth)
        el = math.radians(ref.elevation)
        flat = ref.distance * math.cos(el)
        return (own.x + flat * math.cos(az), own.y + flat * math.sin(az), GESTURE_Z_AGENT + ref.distance * math.sin(el))

    def _expand(self, handle: _Handle) -> list[tuple[_Handle, XYZ]]:
        if isinstance(handle, Pose3):
            return [(handle, (handle.x, handle.y, handle.z))]
        if isinstance(handle, RelativeRef):
            return [(handle, self._relative_xyz(handle))]
        if isinstance(handle, _AgentTarget):
            xyz = self._agent_xyz(handle.agent_id)
            return [(handle, xyz)] if xyz is not None else []
        if isinstance(handle, _ObjectTarget):
            obj = self._world.get(handle.object_id)
            return [(handle, (obj.pose.x, obj.pose.y, self._z(GESTURE_Z_OBJECT)))] if obj is not None else []
        if handle == REF_TARGET:
            xyz = self._pose_xyz(self._target_pose())
            return [(handle, xyz)] if xyz is not None else []
        if handle == REF_GOAL:
            xyz = self._pose_xyz(self._goal_pose())
            return [(handle, xyz)] if xyz is not None else []
        own = self._agent.state.pose
        out: list[tuple[_Handle, XYZ]] = []
        for aid in self._other_participants():
            xyz = self._agent_xyz(aid)
            if xyz is not None:
                out.append((_AgentTarget(aid), xyz))
        if handle == REF_PARTNERS or not out:
            return out
        return [min(out, key=lambda t: math.hypot(t[1][0] - own.x, t[1][1] - own.y))]

    def _targets(self) -> list[tuple[_Handle, XYZ]]:
        out: list[tuple[_Handle, XYZ]] = []
        for i in range(len(self._refs)):
            handle = self._handle(i)
            if handle is not None:
                out.extend(self._expand(handle))
        return out

    def _face_enabled(self, handle: _Handle) -> bool:
        if isinstance(handle, RelativeRef) or self._face_gave_up or self._bound():
            return False
        face = self._att.face
        return self._idle if face is None else face

    def _halt(self) -> None:
        if self._bare and not self._bound():
            cmd = _nav_command(self._agent, self._agent.state.pose)
            cmd.desired_velocity = 0.0
            self._agent.movement.command = cmd

    def _finish_bare(self) -> py_trees.common.Status:
        if self._duration is None:
            return py_trees.common.Status.SUCCESS if not self._multi else py_trees.common.Status.RUNNING
        if self._elapsed >= self._duration:
            return py_trees.common.Status.SUCCESS
        self._elapsed += self._dt
        return py_trees.common.Status.RUNNING

    def update(self) -> py_trees.common.Status:
        self._halt()
        mv = self._bt_mv()
        if not self._refs:
            if mv is not None:
                mv.gesture = None
            return self._finish_bare() if self._bare else py_trees.common.Status.RUNNING

        targets = self._targets()
        if not targets:
            if not self._warned:
                _bt_logger.warning(f"Agent {self._agent.state.agent_id}: step {self.name} waiting, could not resolve at={self._att.at!r}")
                self._warned = True
            self._resolve_elapsed += self._dt
            if self._bare and self._resolve_elapsed > RESOLVE_TIMEOUT_S:
                return py_trees.common.Status.FAILURE
            return py_trees.common.Status.RUNNING
        self._resolve_elapsed = 0.0
        if self._idx >= len(targets):
            if not self._cycle:
                return py_trees.common.Status.SUCCESS
            self._idx = 0
        handle, xyz = targets[self._idx]

        own = self._agent.state.pose
        bearing = math.atan2(xyz[1] - own.y, xyz[0] - own.x)
        err = abs(_wrap(bearing - own.theta))
        heading: float | None = None
        if self._face_enabled(handle):
            if not self._raised:
                if err <= FACE_ENTER_RAD:
                    self._refacing = False
                else:
                    self._face_elapsed += self._dt
                    if self._face_elapsed <= FACE_TIMEOUT_S:
                        if mv is not None:
                            mv.heading_goal = bearing
                        return py_trees.common.Status.RUNNING
                    if self._bare:
                        return py_trees.common.Status.FAILURE
                    self._face_gave_up = True
            elif err > FACE_KEEP_RAD:
                self._refacing = True
            elif err <= FACE_ENTER_RAD:
                self._refacing = False
            if self._refacing:
                heading = bearing
        if mv is not None:
            mv.heading_goal = heading
            mv.gesture = None if self._att.gesture == NO_GESTURE else GestureIntent(self._att.gesture, xyz[0], xyz[1], xyz[2], self._att.hand)
        self._raised = True

        if self._multi:
            self._dwell_elapsed += self._dt
            if self._dwell_elapsed >= self._att.dwell:
                self._dwell_elapsed = 0.0
                self._idx += 1
                if not self._cycle and self._idx >= len(targets):
                    return py_trees.common.Status.SUCCESS
        return self._finish_bare() if self._bare else py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        del new_status
        mv = self._bt_mv()
        if mv is None:
            return
        mv.heading_goal = None
        if self._att.hold == "release":
            mv.gesture = None
