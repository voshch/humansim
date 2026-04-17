__all__ = [
    "ActionDef",
    "AgentType",
    "LocalPlannerDist",
    "NeedCondition",
    "NeedDist",
    "ParamDist",
    "PerceptionDist",
    "SampledLocalPlanner",
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


@attrs.frozen
class ParamDist:
    mean: float
    std: float = 0.0
    clip_low: float = 0.01
    clip_high: float = float("inf")


@attrs.frozen
class NeedDist:
    initial: ParamDist = ParamDist(100.0)
    decay_rate: ParamDist = ParamDist(0.5, 0.1)


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
    target_object: str | None = None
    duration: ParamDist | None = None
    patience: ParamDist | None = None
    satisfies: dict[str, float] = attrs.Factory(dict)
    on_failure: str = "skip"


@attrs.frozen
class TransitionDef:
    when: dict[str, NeedCondition]
    goto: str


@attrs.frozen
class StepDef:
    target_object: str | None = None
    interaction: str | None = None
    duration: ParamDist | None = None
    patience: ParamDist | None = None
    satisfies: dict[str, float] = attrs.Factory(dict)
    on_failure: str = "abort"

    autonomous: bool = False
    until: str | None = None
    until_need: dict[str, NeedCondition] | None = None
    allowed_actions: tuple[str, ...] | None = None
    blocked_actions: tuple[str, ...] | None = None

    interruptible: bool | None = None


@attrs.frozen
class SequenceDef:
    steps: dict[str, StepDef]
    then: str | None = None
    on_failure: str | None = None
    interruptible: bool = True
    transitions: tuple[TransitionDef, ...] = ()


@attrs.frozen
class PerceptionDist:
    vision_range: ParamDist = ParamDist(5.0, 0.5)
    vision_fov: ParamDist = ParamDist(180.0, 10.0)


@attrs.frozen
class LocalPlannerDist:
    relaxation_time: ParamDist = ParamDist(0.5, 0.05)
    repulsion_strength: ParamDist = ParamDist(2.1, 0.2)
    repulsion_range: ParamDist = ParamDist(0.3, 0.03)
    anisotropy: ParamDist = ParamDist(0.5, 0.0)


@attrs.frozen
class AgentType:
    name: str
    mode: str = "simple"

    desired_velocity: ParamDist = ParamDist(1.1, 0.12)
    agent_radius: ParamDist = ParamDist(0.35, 0.02)
    max_velocity: ParamDist = ParamDist(1.5, 0.1, clip_low=0.5)
    max_acceleration: ParamDist = ParamDist(1.5, 0.1, clip_low=0.3)
    max_deceleration: ParamDist = ParamDist(2.5, 0.2, clip_low=0.5)
    min_turning_radius: ParamDist = ParamDist(0.3, 0.03, clip_low=0.1)
    pivot_angular_velocity: ParamDist = ParamDist(2.0, 0.2, clip_low=1.0)

    # LogNormal: mean is interpreted as the desired median in seconds; std is the shape parameter.
    reaction_time: ParamDist = ParamDist(0.4, 0.3, clip_low=0.05, clip_high=1.5)
    personal_space_min: ParamDist = ParamDist(0.6, 0.15, clip_low=0.2, clip_high=2.0)

    perception: PerceptionDist = attrs.Factory(PerceptionDist)
    local_planner_params: LocalPlannerDist = attrs.Factory(LocalPlannerDist)

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


@attrs.frozen
class SampledLocalPlanner:
    relaxation_time: float = 0.5
    repulsion_strength: float = 2.1
    repulsion_range: float = 0.3
    anisotropy: float = 0.5


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
    local_planner_params: SampledLocalPlanner = attrs.Factory(SampledLocalPlanner)

    perception_stack: tuple[str, ...] = ("default",)
    local_planner: str | None = None
    global_planner: str | None = None
    animation: str | None = None

    needs: dict[str, SampledNeed] = attrs.Factory(dict)
    utility_weights: dict[str, float] = attrs.Factory(dict)


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
) -> SampledParams:
    sampled_needs: dict[str, SampledNeed] = {}
    for need_name, need_dist in agent_type.needs.items():
        sampled_needs[need_name] = SampledNeed(
            initial=_sample_dist(need_dist.initial, rng),
            decay_rate=_sample_dist(need_dist.decay_rate, rng),
        )

    return SampledParams(
        name=agent_type.name,
        desired_velocity=_sample_dist(agent_type.desired_velocity, rng),
        agent_radius=_sample_dist(agent_type.agent_radius, rng),
        max_velocity=_sample_dist(agent_type.max_velocity, rng),
        max_acceleration=_sample_dist(agent_type.max_acceleration, rng),
        max_deceleration=_sample_dist(agent_type.max_deceleration, rng),
        min_turning_radius=_sample_dist(agent_type.min_turning_radius, rng),
        pivot_angular_velocity=_sample_dist(agent_type.pivot_angular_velocity, rng),
        reaction_time=_sample_lognormal_dist(agent_type.reaction_time, rng),
        personal_space_min=_sample_dist(agent_type.personal_space_min, rng),
        perception=SampledPerception(
            vision_range=_sample_dist(agent_type.perception.vision_range, rng),
            vision_fov=_sample_dist(agent_type.perception.vision_fov, rng),
        ),
        local_planner_params=SampledLocalPlanner(
            relaxation_time=_sample_dist(agent_type.local_planner_params.relaxation_time, rng),
            repulsion_strength=_sample_dist(agent_type.local_planner_params.repulsion_strength, rng),
            repulsion_range=_sample_dist(agent_type.local_planner_params.repulsion_range, rng),
            anisotropy=_sample_dist(agent_type.local_planner_params.anisotropy, rng),
        ),
        perception_stack=agent_type.perception_stack,
        local_planner=agent_type.local_planner,
        global_planner=agent_type.global_planner,
        animation=agent_type.animation,
        needs=sampled_needs,
        utility_weights=dict(agent_type.utility_weights),
    )
