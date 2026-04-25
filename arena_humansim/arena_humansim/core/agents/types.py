__all__ = [
    "ActionDef",
    "AgentType",
    "GoToStepDef",
    "NeedCondition",
    "NeedDist",
    "ParamDist",
    "PerceptionDist",
    "SampledNeed",
    "SampledParams",
    "SampledPerception",
    "SequenceDef",
    "StepDef",
    "TransitionDef",
    "VarDef",
    "sample_agent_type",
]

import math
from pathlib import Path

import attrs
import numpy as np

from arena_humansim.utils.types import FormationSpec, Pose2D


@attrs.frozen
class ParamDist:
    mean: float
    std: float = 0.0
    clip_low: float = 0.01
    clip_high: float = float("inf")


def _as_paramdist(val: object) -> "ParamDist | None":
    if val is None:
        return None
    if isinstance(val, ParamDist):
        return val
    if isinstance(val, (int, float)):
        return ParamDist(mean=float(val))
    if isinstance(val, dict):
        return ParamDist(**val)
    return val  # type: ignore[return-value]


@attrs.frozen
class NeedDist:
    initial: ParamDist = attrs.field(default=ParamDist(100.0), converter=_as_paramdist)
    decay_rate: ParamDist = attrs.field(default=ParamDist(0.5, 0.1), converter=_as_paramdist)


@attrs.frozen
class NeedCondition:
    below: float | None = None
    above: float | None = None


@attrs.frozen
class VarDef:
    type: str  # "int", "float", "bool", "str"
    default: int | float | bool | str
    min: float | None = None
    max: float | None = None
    description: str = ""


@attrs.frozen
class ActionDef:
    when: dict[str, NeedCondition] = attrs.Factory(dict)
    interaction: str | None = None
    target: str | None = None
    duration: ParamDist | None = attrs.field(default=None, converter=_as_paramdist)
    patience: ParamDist | None = attrs.field(default=None, converter=_as_paramdist)
    satisfies: dict[str, float] = attrs.Factory(dict)
    on_failure: str = "skip"


@attrs.frozen
class TransitionDef:
    when: dict[str, NeedCondition]
    goto: str


@attrs.frozen
class StepDef:
    target: str | None = None
    interaction: str | None = None
    duration: ParamDist | None = attrs.field(default=None, converter=_as_paramdist)
    patience: ParamDist | None = attrs.field(default=None, converter=_as_paramdist)
    satisfies: dict[str, float] = attrs.Factory(dict)
    on_failure: str = "abort"

    autonomous: bool = False
    until: str | None = None
    until_need: dict[str, NeedCondition] | None = None
    allowed_actions: tuple[str, ...] | None = None
    blocked_actions: tuple[str, ...] | None = None

    interruptible: bool | None = None

    interaction_radius: float | None = None

    offer: bool = False
    cancel: bool = False
    queueable: bool | None = None
    min_participants: int | None = None
    max_participants: int | None = None
    formation_spec: FormationSpec | None = None
    wait_for_outcome: bool = False


@attrs.frozen
class GoToStepDef:
    target_pose: Pose2D | None = None
    target: str | None = None
    duration: ParamDist | None = attrs.field(default=None, converter=_as_paramdist)
    patience: ParamDist | None = attrs.field(default=None, converter=_as_paramdist)
    satisfies: dict[str, float] = attrs.Factory(dict)
    on_failure: str = "abort"
    interruptible: bool | None = None


@attrs.frozen
class SequenceDef:
    steps: dict[str, StepDef | GoToStepDef]
    then: str | None = None
    on_failure: str | None = None
    interruptible: bool = True
    transitions: tuple[TransitionDef, ...] = ()


@attrs.frozen
class PerceptionDist:
    vision_range: ParamDist = attrs.field(default=ParamDist(5.0, 0.5), converter=_as_paramdist)
    vision_fov: ParamDist = attrs.field(default=ParamDist(180.0, 10.0), converter=_as_paramdist)
    proximity_sense: ParamDist = attrs.field(default=ParamDist(1.0, 0.2, clip_low=0.5, clip_high=2.0), converter=_as_paramdist)
    vision_occlusion: bool = True


@attrs.frozen
class AgentType:
    name: str
    mode: str = "simple"

    desired_velocity: ParamDist = attrs.field(default=ParamDist(1.1, 0.12), converter=_as_paramdist)
    agent_radius: ParamDist = attrs.field(default=ParamDist(0.35, 0.02), converter=_as_paramdist)
    max_velocity: ParamDist = attrs.field(default=ParamDist(1.5, 0.1, clip_low=0.5), converter=_as_paramdist)
    max_acceleration: ParamDist = attrs.field(default=ParamDist(1.5, 0.1, clip_low=0.3), converter=_as_paramdist)
    max_deceleration: ParamDist = attrs.field(default=ParamDist(2.5, 0.2, clip_low=0.5), converter=_as_paramdist)
    min_turning_radius: ParamDist = attrs.field(default=ParamDist(0.3, 0.03, clip_low=0.1), converter=_as_paramdist)
    pivot_angular_velocity: ParamDist = attrs.field(default=ParamDist(2.0, 0.2, clip_low=1.0), converter=_as_paramdist)

    # LogNormal: mean is interpreted as the desired median in seconds; std is the shape parameter.
    reaction_time: ParamDist = attrs.field(default=ParamDist(0.4, 0.3, clip_low=0.05, clip_high=1.5), converter=_as_paramdist)
    personal_space_min: ParamDist = attrs.field(default=ParamDist(0.6, 0.15, clip_low=0.2, clip_high=2.0), converter=_as_paramdist)

    idle_gaze_rate: ParamDist = attrs.field(default=ParamDist(0.0, 0.0, clip_low=0.0, clip_high=0.0), converter=_as_paramdist)

    perception: PerceptionDist = attrs.Factory(PerceptionDist)
    local_planner_params: dict[str, ParamDist] = attrs.Factory(dict)

    perception_stack: tuple[str, ...] = ("default",)
    local_planner: str | None = None
    global_planner: str | None = None
    animation: str | None = None

    needs: dict[str, NeedDist] = attrs.Factory(dict)
    utility_weights: dict[str, float] = attrs.Factory(dict)
    actions: dict[str, ActionDef] = attrs.Factory(dict)
    sequences: dict[str, SequenceDef] = attrs.Factory(dict)
    initial_sequence: str = "default"
    vars: dict[str, VarDef] = attrs.Factory(dict)
    extends: str | None = None

    source_path: Path | None = attrs.field(default=None, eq=False, hash=False, repr=False)


@attrs.frozen
class SampledNeed:
    initial: float
    decay_rate: float


@attrs.frozen
class SampledPerception:
    vision_range: float = 5.0
    vision_fov: float = 180.0
    proximity_sense: float = 1.0
    vision_occlusion: bool = True


@attrs.frozen
class SampledParams:
    name: str

    desired_velocity: float
    agent_radius: float
    max_velocity: float
    max_acceleration: float
    max_deceleration: float
    min_turning_radius: float
    pivot_angular_velocity: float

    reaction_time: float
    personal_space_min: float

    perception: SampledPerception = attrs.Factory(SampledPerception)
    local_planner_params: dict[str, float] = attrs.Factory(dict)

    perception_stack: tuple[str, ...] = ("default",)
    local_planner: str | None = None
    global_planner: str | None = None
    animation: str | None = None

    needs: dict[str, SampledNeed] = attrs.Factory(dict)
    utility_weights: dict[str, float] = attrs.Factory(dict)

    idle_gaze_rate_hz: float = 0.0


def _sample_dist(dist: ParamDist, rng: np.random.Generator) -> float:
    value = rng.normal(dist.mean, dist.std) if dist.std > 0 else dist.mean
    return float(np.clip(value, dist.clip_low, dist.clip_high))


def _sample_lognormal_dist(dist: ParamDist, rng: np.random.Generator) -> float:
    # dist.mean is interpreted as the desired median in linear space; dist.std is sigma of the underlying normal.
    if dist.std > 0:
        value = rng.lognormal(mean=math.log(max(dist.mean, 1e-9)), sigma=dist.std)
    else:
        value = dist.mean
    return float(np.clip(value, dist.clip_low, dist.clip_high))


def sample_agent_type(
    agent_type: AgentType,
    rng: np.random.Generator,
    default_local_planner: str = "sfm",
) -> SampledParams:
    sampled_needs: dict[str, SampledNeed] = {}
    for need_name, need_dist in agent_type.needs.items():
        sampled_needs[need_name] = SampledNeed(
            initial=_sample_dist(need_dist.initial, rng),
            decay_rate=_sample_dist(need_dist.decay_rate, rng),
        )

    desired_velocity = _sample_dist(agent_type.desired_velocity, rng)
    agent_radius = _sample_dist(agent_type.agent_radius, rng)
    max_velocity = _sample_dist(agent_type.max_velocity, rng)
    max_acceleration = _sample_dist(agent_type.max_acceleration, rng)
    max_deceleration = _sample_dist(agent_type.max_deceleration, rng)
    min_turning_radius = _sample_dist(agent_type.min_turning_radius, rng)
    pivot_angular_velocity = _sample_dist(agent_type.pivot_angular_velocity, rng)
    reaction_time = _sample_lognormal_dist(agent_type.reaction_time, rng)
    personal_space_min = _sample_dist(agent_type.personal_space_min, rng)
    sampled_perception = SampledPerception(
        vision_range=_sample_dist(agent_type.perception.vision_range, rng),
        vision_fov=_sample_dist(agent_type.perception.vision_fov, rng),
        proximity_sense=_sample_dist(agent_type.perception.proximity_sense, rng),
        vision_occlusion=agent_type.perception.vision_occlusion,
    )

    from arena_humansim.local_planner import LocalPlanner

    planner_name = agent_type.local_planner or default_local_planner
    lp_dists = {**LocalPlanner.get_class(planner_name).PARAM_DEFAULTS, **agent_type.local_planner_params}
    sampled_lp = {k: _sample_dist(d, rng) for k, d in lp_dists.items()}

    idle_gaze_rate_hz = _sample_dist(agent_type.idle_gaze_rate, rng)

    return SampledParams(
        name=agent_type.name,
        desired_velocity=desired_velocity,
        agent_radius=agent_radius,
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
        max_deceleration=max_deceleration,
        min_turning_radius=min_turning_radius,
        pivot_angular_velocity=pivot_angular_velocity,
        reaction_time=reaction_time,
        personal_space_min=personal_space_min,
        perception=sampled_perception,
        local_planner_params=sampled_lp,
        perception_stack=agent_type.perception_stack,
        local_planner=agent_type.local_planner,
        global_planner=agent_type.global_planner,
        animation=agent_type.animation,
        needs=sampled_needs,
        utility_weights=dict(agent_type.utility_weights),
        idle_gaze_rate_hz=idle_gaze_rate_hz,
    )
