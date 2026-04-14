"""Core data structures for arena_humansim."""

import enum

import attrs

Segment = tuple[tuple[float, float], tuple[float, float]]
Segments = list[Segment]


class WallAware:
    def set_walls(self, segments: Segments) -> None:
        pass


class WaypointMode(enum.IntEnum):
    REPEAT = 0
    REVERSE = 1
    ONCE = 2
    RANDOM = 3


class InteractionType(enum.IntEnum):
    TALK_TO = 0
    GROUP_CONVERSATION = 1
    FOLLOW = 2
    SIT_ON = 3
    LIE_ON = 4
    USE = 5
    QUEUE_USE = 6
    WAVE_AT = 7
    BLOCK = 8


class InteractionOutcome(enum.IntEnum):
    FORMING = 0
    ACTIVE = 1
    COMPLETED = 2
    INTERRUPTED = 3


@attrs.define
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


Goal = Pose2D | tuple[float, float]


@attrs.define
class AgentState:
    agent_id: int = 0
    pose: Pose2D = attrs.Factory(Pose2D)
    velocity: tuple[float, float] = (0.0, 0.0)
    desired_velocity: float = 1.3


@attrs.define
class WorldAgentState:
    pose: Pose2D = attrs.Factory(Pose2D)
    velocity: tuple[float, float] = (0.0, 0.0)


WorldState = dict[int, WorldAgentState]


@attrs.define
class BeliefState:
    """Perceived world state for an agent (not ground truth)."""

    agent_id: int = 0
    observed_agents: list[AgentState] = attrs.Factory(list)
    extra: dict[str, object] = attrs.Factory(dict)


@attrs.define
class InteractionContract:
    type: int = 0
    min_participants: int = 2
    max_participants: int = 2  # -1 for unbounded
    current_participants: list[int] = attrs.Factory(list)
    queueable: bool = False
    max_queue: int = -1  # -1 = unbounded
    queue: list[int] = attrs.Factory(list)
    duration: float | None = None  # seconds; None = no contract-level timeout
    elapsed: float = 0.0

    @property
    def queue_length(self) -> int:
        return len(self.queue)

    @property
    def remaining_slots(self) -> int:
        if self.max_participants == -1:
            return -1
        return self.max_participants - len(self.current_participants)

    @property
    def is_full(self) -> bool:
        return self.max_participants != -1 and len(self.current_participants) >= self.max_participants


@attrs.define
class InteractionState:
    id: int = 0
    type: int = 0
    contract: InteractionContract = attrs.Factory(InteractionContract)
    participants: list[int] = attrs.Factory(list)
    state: dict[str, object] = attrs.Factory(dict)
    object_id: str | None = None
    outcome: int = 0  # InteractionOutcome.FORMING


@attrs.define
class HighLevelCommand:
    agent_id: int = 0
    type: int = 0
    target_pose: Pose2D = attrs.Factory(Pose2D)
    desired_velocity: float = 1.3
    interaction_target: int = -1
    interaction_type: int = 0
    target_agent: int = -1  # -1 = broadcast; >= 0 = only this agent can accept
    interaction_duration: float = -1.0  # seconds; -1 = no contract-level timeout


@attrs.define
class WaypointMovement:
    waypoints: list[Pose2D] = attrs.Factory(list)
    radii: list[float] = attrs.Factory(list)
    index: int = 0
    mode: WaypointMode = WaypointMode.REPEAT
    forward: bool = True


@attrs.define
class BehaviorTreeMovement:
    command: HighLevelCommand | None = None
    last_outcome: int | None = None  # InteractionOutcome from most recently ended interaction


class ShapeType(enum.Enum):
    CIRCLE = "circle"
    POLYGON = "polygon"


@attrs.define
class Shape:
    type: ShapeType = ShapeType.POLYGON
    radius: float = 0.0
    vertices: list[Pose2D] = attrs.field(factory=list)


@attrs.define
class RateKeyframe:
    t: float = 0.0  # seconds since sim start
    rate: float = 0.0  # agents/sec (lambda for Poisson)


@attrs.define
class SinkAffinity:
    sink_name: str = ""  # name of the target sink
    weight: float = 1.0  # relative weight (normalized at runtime)


@attrs.define
class AgentTemplate:
    desired_velocity_min: float = 1.0
    desired_velocity_max: float = 1.5
    agent_radius: float = 0.35
    agent_type: str = "adult"
    sink_affinity: list[SinkAffinity] = attrs.Factory(list)


@attrs.define
class SourceConfig:
    name: str = ""
    pose: Pose2D = attrs.Factory(Pose2D)
    shape: Shape = attrs.Factory(Shape)
    rate_profile: list[RateKeyframe] = attrs.Factory(list)
    max_concurrent: int = -1  # -1 = unlimited
    max_total: int = -1  # -1 = unlimited
    agent: AgentTemplate = attrs.Factory(AgentTemplate)


@attrs.define
class SinkConfig:
    name: str = ""
    pose: Pose2D = attrs.Factory(Pose2D)
    shape: Shape = attrs.Factory(Shape)
    absorption_radius: float = 0.5
    capacity: int = -1  # -1 = unlimited simultaneous


@attrs.define
class FlowConfig:
    sources: list[SourceConfig] = attrs.Factory(list)
    sinks: list[SinkConfig] = attrs.Factory(list)


@attrs.define
class AgentLifetime:
    agent_id: int = 0
    source_name: str = ""  # which source spawned this agent ("" if explicit)
    spawn_tick: int = 0  # tick when agent was created
    max_lifetime_s: float = -1  # -1 = no TTL
    target_sink_name: str = ""  # "" = no sink target
    pending_despawn: bool = False  # True if reached sink but in interaction


@attrs.define
class SpawnRequest:
    """Lightweight spawn descriptor — AgentManager materializes into BaseAgent."""

    pose: Pose2D = attrs.Factory(Pose2D)
    desired_velocity: float = 1.3
    agent_radius: float = 0.35
    agent_type: str = "adult"
    waypoints: list[Pose2D] = attrs.Factory(list)
    lifetime: AgentLifetime = attrs.Factory(AgentLifetime)


@attrs.define
class DespawnRequest:
    agent_id: int = 0
    force: bool = False  # True = TTL hard cutoff, force-STOP interactions
    reason: str = "sink"  # "sink" | "ttl" | "deferred"


@attrs.define
class NeedState:
    """Runtime state for a single need."""

    value: float = 100.0
    decay_rate: float = 0.5


@attrs.define
class NeedsState:
    """Per-agent mutable needs container."""

    needs: dict[str, NeedState] = attrs.Factory(dict)

    def decay(self, dt: float, modifiers: dict[str, float] | None = None) -> None:
        """Tick all needs downward. modifiers: need_name -> decay multiplier."""
        for name, need in self.needs.items():
            mod = modifiers.get(name, 1.0) if modifiers else 1.0
            need.value = max(0.0, need.value - need.decay_rate * dt * mod)

    def satisfy(self, deltas: dict[str, float]) -> None:
        """Apply satisfaction from a completed action."""
        for name, amount in deltas.items():
            if name in self.needs:
                self.needs[name].value = min(100.0, self.needs[name].value + amount)


# ---------------------------------------------------------------------------
# cattrs converter with hooks for agent type deserialization
# ---------------------------------------------------------------------------

import cattrs

from arena_humansim.agents.types import (
    ActionDef,
    AgentType,
    LocalPlannerDist,
    NeedCondition,
    NeedDist,
    ParamDist,
    PerceptionDist,
    SequenceDef,
    StepDef,
    TransitionDef,
    VarDef,
)

converter = cattrs.Converter()


def _structure_param_dist(val: object, _: type) -> ParamDist:
    if isinstance(val, (int, float)):
        return ParamDist(mean=float(val))
    if isinstance(val, ParamDist):
        return val
    return ParamDist(**val)


converter.register_structure_hook(ParamDist, _structure_param_dist)


def _structure_need_dist(val: object, _: type) -> NeedDist:
    if isinstance(val, (int, float)):
        return NeedDist(initial=ParamDist(mean=float(val)))
    if isinstance(val, NeedDist):
        return val
    d = dict(val)
    if "initial" in d:
        d["initial"] = converter.structure(d["initial"], ParamDist)
    if "decay_rate" in d:
        d["decay_rate"] = converter.structure(d["decay_rate"], ParamDist)
    return NeedDist(**d)


converter.register_structure_hook(NeedDist, _structure_need_dist)


def _structure_need_condition(val: object, _: type) -> NeedCondition:
    if isinstance(val, NeedCondition):
        return val
    return NeedCondition(**val)


converter.register_structure_hook(NeedCondition, _structure_need_condition)


def _structure_action_def(val: object, _: type) -> ActionDef:
    if isinstance(val, ActionDef):
        return val
    d = dict(val)
    if "when" in d and isinstance(d["when"], dict):
        d["when"] = {k: converter.structure(v, NeedCondition) for k, v in d["when"].items()}
    if "duration" in d and d["duration"] is not None:
        d["duration"] = converter.structure(d["duration"], ParamDist)
    if "patience" in d and d["patience"] is not None:
        d["patience"] = converter.structure(d["patience"], ParamDist)
    return ActionDef(**d)


converter.register_structure_hook(ActionDef, _structure_action_def)


def _structure_step_def(val: object, _: type) -> StepDef:
    if isinstance(val, StepDef):
        return val
    d = dict(val)
    if "duration" in d and d["duration"] is not None:
        d["duration"] = converter.structure(d["duration"], ParamDist)
    if "patience" in d and d["patience"] is not None:
        d["patience"] = converter.structure(d["patience"], ParamDist)
    if "until_need" in d and isinstance(d["until_need"], dict):
        d["until_need"] = {k: converter.structure(v, NeedCondition) for k, v in d["until_need"].items()}
    if "allowed_actions" in d and d["allowed_actions"] is not None:
        d["allowed_actions"] = tuple(d["allowed_actions"])
    if "blocked_actions" in d and d["blocked_actions"] is not None:
        d["blocked_actions"] = tuple(d["blocked_actions"])
    return StepDef(**d)


converter.register_structure_hook(StepDef, _structure_step_def)


def _structure_transition_def(val: object, _: type) -> TransitionDef:
    if isinstance(val, TransitionDef):
        return val
    d = dict(val)
    when = d["when"]
    if isinstance(when, str):
        raise ValueError(f"String-based transition condition '{when}' is not supported. Use dict[str, NeedCondition] instead.")
    d["when"] = {k: converter.structure(v, NeedCondition) for k, v in when.items()}
    return TransitionDef(**d)


converter.register_structure_hook(TransitionDef, _structure_transition_def)


def _structure_sequence_def(val: object, _: type) -> SequenceDef:
    if isinstance(val, SequenceDef):
        return val
    d = dict(val)
    if "steps" in d:
        d["steps"] = {k: converter.structure(v, StepDef) for k, v in d["steps"].items()}
    if "transitions" in d:
        d["transitions"] = tuple(converter.structure(t, TransitionDef) for t in d["transitions"])
    else:
        d["transitions"] = ()
    return SequenceDef(**d)


converter.register_structure_hook(SequenceDef, _structure_sequence_def)


def _structure_perception_dist(val: object, _: type) -> PerceptionDist:
    if isinstance(val, PerceptionDist):
        return val
    d = dict(val)
    for k in d:
        d[k] = converter.structure(d[k], ParamDist)
    return PerceptionDist(**d)


converter.register_structure_hook(PerceptionDist, _structure_perception_dist)


def _structure_local_planner_dist(val: object, _: type) -> LocalPlannerDist:
    if isinstance(val, LocalPlannerDist):
        return val
    d = dict(val)
    for k in d:
        d[k] = converter.structure(d[k], ParamDist)
    return LocalPlannerDist(**d)


converter.register_structure_hook(LocalPlannerDist, _structure_local_planner_dist)


_PARAM_DIST_FIELDS = frozenset(f.name for f in attrs.fields(AgentType) if f.type is ParamDist)


def _structure_agent_type(val: object, _: type) -> AgentType:
    if isinstance(val, AgentType):
        return val
    d = dict(val)
    for field_name in _PARAM_DIST_FIELDS:
        if field_name in d:
            d[field_name] = converter.structure(d[field_name], ParamDist)
    if "perception" in d and isinstance(d["perception"], dict):
        d["perception"] = converter.structure(d["perception"], PerceptionDist)
    if "local_planner_params" in d and isinstance(d["local_planner_params"], dict):
        d["local_planner_params"] = converter.structure(d["local_planner_params"], LocalPlannerDist)
    if "needs" in d and isinstance(d["needs"], dict):
        d["needs"] = {k: converter.structure(v, NeedDist) for k, v in d["needs"].items()}
    if "actions" in d and isinstance(d["actions"], dict):
        d["actions"] = {k: converter.structure(v, ActionDef) for k, v in d["actions"].items()}
    if "sequences" in d and isinstance(d["sequences"], dict):
        d["sequences"] = {k: converter.structure(v, SequenceDef) for k, v in d["sequences"].items()}
    if "vars" in d and isinstance(d["vars"], dict):
        d["vars"] = {k: VarDef(**v) if isinstance(v, dict) else v for k, v in d["vars"].items()}
    if "perception_stack" in d and isinstance(d["perception_stack"], list):
        d["perception_stack"] = tuple(d["perception_stack"])
    return AgentType(**d)


converter.register_structure_hook(AgentType, _structure_agent_type)


def _structure_var_def(val: object, _: type) -> VarDef:
    if isinstance(val, VarDef):
        return val
    return VarDef(**val)


converter.register_structure_hook(VarDef, _structure_var_def)
