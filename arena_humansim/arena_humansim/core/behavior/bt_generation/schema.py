import enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Pose2D(BaseModel):
    x: float = 0.0
    y: float = 0.0
    theta: float = Field(default=0.0, description="Yaw (rad)")


class ServiceSpec(BaseModel):
    tag: str = ""
    max_participants: int = -1


class AgentConfig(BaseModel):
    agent_id: int = 0
    agent_type: str = "adult"
    spawn_pose: Pose2D = Field(default_factory=Pose2D)
    desired_velocity: float = 1.3
    agent_radius: float = 0.35
    policy: Literal["sfm", "orca", "straight"] = Field(default="sfm", description="Local planner name.")
    policy_params: str = Field(default="", description="opaque JSON blob forwarded to the planner")
    services: list[ServiceSpec] = Field(default_factory=list)
    spawn_tick: int = Field(default=0, description="deferred spawn; 0 = spawn on first tick")


class ParamDist(BaseModel):
    mean: float
    std: float = 0.0
    clip_low: float = 0.01
    clip_high: float = 1e9  # Infinity won't work for google.genai structured output


# class PerceptionDist(BaseModel):
#     vision_range: ParamDist = Field(default=ParamDist(mean=5.0, std=0.5))
#     vision_fov: ParamDist = Field(default=ParamDist(mean=180.0, std=10.0))
#     proximity_sense: ParamDist = Field(default=ParamDist(mean=1.0, std=0.2, clip_low=0.5, clip_high=2.0))


# class LocalPlannerDist(BaseModel):
#     relaxation_time: ParamDist = Field(default=ParamDist(mean=0.5, std=0.05))
#     repulsion_strength: ParamDist = Field(default=ParamDist(mean=2.1, std=0.2))
#     repulsion_range: ParamDist = Field(default=ParamDist(mean=0.3, std=0.03))
#     anisotropy: ParamDist = Field(default=ParamDist(mean=0.5, std=0.0))


class NeedDist(BaseModel):
    """
    Scalar state in `[0, 100]` that decays every tick and is restored by steps via `satisfies`.
    `initial` and `decay_rate` are distribution dicts, sampled per agent.
    Decay is linear; `satisfies: {thirst: 100.0}` adds 100 on completion (clamped at 100)
    """

    initial: ParamDist = Field(default=ParamDist(mean=100.0, clip_low=0.0, clip_high=100.0))
    decay_rate: ParamDist = Field(default=ParamDist(mean=0.5, std=0.1))


class NeedCondition(BaseModel):
    below: float | None = None
    above: float | None = None


# class VarDef(BaseModel):
#     type: Literal["int", "float", "bool", "str"]
#     default: int | float | bool | str
#     min: float | None = None
#     max: float | None = None


class TransitionDef(BaseModel):
    when: dict[str, NeedCondition] = Field(description="Need-condition predicate that triggers the transition.")
    goto: str

    @model_validator(mode="before")
    @classmethod
    def reject_string_when(cls, data: object) -> object:
        # Constraint from scenario_loader._structure_transition_def:
        # String-based transition conditions are not supported.
        if isinstance(data, dict):
            when = data.get("when")
            if isinstance(when, str):
                raise ValueError(f"String-based transition condition '{when}' is not supported. Use dict[str, NeedCondition] instead.")
        return data


class AnchorKind(enum.StrEnum):
    OBJECT = "object"
    AGENT = "agent"
    PROVIDER = "provider"
    POSE = "pose"
    CENTROID = "centroid"


class FormationSpec(BaseModel):
    """Runtime representation of a scenario FormationConfig on a WorldObject."""

    type: str
    params: dict[str, float] = Field(default_factory=dict)
    anchor_kind: AnchorKind = AnchorKind.OBJECT
    anchor_ref: str | None = None
    anchor_pose: Pose2D | None = None


# Interaction types whose `offer: true` is valid (SERVICE-kind interactions).
# Mirrors scenario_loader: `kind.allows_offer` — only SERVICE allows offer.
_OFFER_ALLOWED_INTERACTIONS = {"SERVICE"}

# Provider-side fields that require `offer: true`.
_PROVIDER_SIDE_FIELDS = ("queueable", "min_participants", "max_participants", "formation_spec")


class StepDef(BaseModel):
    """
    Describes an interaction, pure-wait, or cancel behavior
    """

    interaction: Literal["TALK_TO", "GROUP_CONVERSATION", "SIT_ON", "LIE_ON", "USE", "QUEUE_USE", "WAVE_AT", "BLOCK", "SERVICE"] | None = Field(default=None, description="Interaction type, omit for a pure-wait step.")
    target: str | None = Field(
        default=None,
        description=(
            "Interpreted per the interaction's handle kind: object id/type for `OBJECT`; "
            "service tag (str) for `SERVICE`; agent id (int) for `BLOCK`; omit for symmetric types. "
            "Required when interaction is one of ['SIT_ON', 'LIE_ON', 'USE', 'QUEUE_USE']. "
            "Symmetric interactions (TALK_TO, GROUP_CONVERSATION) don't take a `target:`."
        ),
    )

    duration: ParamDist | None = Field(default=None, description="Duration of the interaction.")
    patience: ParamDist | None = Field(default=None, description="Patient of agent on this interaction, decay over time.")
    satisfies: dict[str, float] = Field(default_factory=dict, description="Which needs this step satisfies. `{need: amount}` applied on SUCCESS.")
    on_failure: Literal["abort", "skip"] = "abort"

    autonomous: bool = Field(default=False, description="`true` ⇒ `AutonomousNode` scores `actions` and runs the winner.")
    until: str | None = Field(default=None, description="Event-bus event name that exits the autonomous step on fire (e.g. `'agent_ready'`).")
    until_need: dict[str, NeedCondition] | None = Field(default=None, description="Need-condition predicate that exits the autonomous step (e.g. `{rest: {above: 80}}`).")
    allowed_actions: list[str] | None = Field(default=None, description="Filter the autonomous candidate pool.")
    blocked_actions: list[str] | None = Field(default=None, description="Filter the autonomous candidate pool.")

    interruptible: bool | None = None

    interaction_radius: float | None = None

    offer: bool = Field(default=False, description="SERVICE provider side. `true` makes this step create-and-wait rather than find-and-join. Required when a SERVICE interaction has no existing provider.")
    cancel: bool = Field(default=False, description="`true` ⇒ emit STOP with `reason=CANCELED` on the agent's current interaction. Mutually exclusive with `interaction:`.")
    queueable: bool | None = Field(default=None, description="Provider-side override (SERVICE with `offer: true`) — admit seekers into a FIFO queue when full.")
    min_participants: int | None = None
    max_participants: int | None = None
    formation_spec: FormationSpec | None = None
    wait_for_outcome: bool = False

    @model_validator(mode="after")
    def check_step_constraints(self) -> "StepDef":
        interaction = self.interaction
        offer = self.offer
        cancel = self.cancel

        # Constraint: `cancel: true` is mutually exclusive with `interaction:`.
        if cancel and interaction is not None:
            raise ValueError("'cancel: true' is mutually exclusive with 'interaction:'. A cancel step must not specify an interaction.")

        # Constraint: `offer: true` is only valid for SERVICE interactions.
        if offer and interaction not in _OFFER_ALLOWED_INTERACTIONS:
            raise ValueError(f"'offer: true' is not valid for interaction={interaction!r}. Only SERVICE interactions support 'offer: true'.")

        # Constraint: provider-side fields require `offer: true`.
        for field_name in _PROVIDER_SIDE_FIELDS:
            value = getattr(self, field_name)
            if value is not None and not offer:
                raise ValueError(f"step field {field_name!r} is provider-side configuration; requires 'offer: true'")

        if not cancel and interaction is not None:
            # --- HandleKind.NONE: interactions that take no target at all ---
            # WAVE_AT, BLOCK are handled separately below; NONE-kind interactions
            # would need no target. (Extend this set if new NONE-kind types are added.)

            # --- HandleKind.TAG: SERVICE ---
            # Provider side (offer: true): target must be a non-null string (the service tag).
            # Seeker side (offer: false): target is optional; if present it must be a string
            #   (already enforced by the `str | None` type annotation).
            if interaction == "SERVICE":
                if offer and not isinstance(self.target, str):
                    raise ValueError("interaction: SERVICE with 'offer: true' requires 'target: <tag:str>' (a non-null string naming the service tag, e.g. 'target: reception_desk')")

            # --- HandleKind.OBJECT: interactions that require an object target ---
            # SIT_ON, LIE_ON, USE, QUEUE_USE — target must be a non-null string.
            _target_required = {"SIT_ON", "LIE_ON", "USE", "QUEUE_USE"}
            if interaction in _target_required and self.target is None:
                raise ValueError(f"interaction: {interaction} requires 'target: <object_id|object_type>'")

            # --- Symmetric interactions: no target permitted ---
            # TALK_TO, GROUP_CONVERSATION initiate peer-to-peer; the runtime matches
            # participants automatically. A target here would be silently ignored and
            # mislead the LLM into thinking it controls who participates.
            _symmetric = {"TALK_TO", "GROUP_CONVERSATION"}
            if interaction in _symmetric and self.target is not None:
                raise ValueError(f"interaction: {interaction} is symmetric and takes no 'target:'. Use a go_to step first to navigate to the meeting area, then run the interaction without a target.")

        return self


class GoToStepDef(BaseModel):
    """
    Describes a navigation behavior
    Takes exactly ONE of `target_pose` or `target`
    """

    target_pose: Pose2D | None = Field(default=None, description="Scripted waypoint")
    target: str | None = Field(default=None, description="object id/type to navigate toward")
    duration: ParamDist | None = Field(default=None)
    patience: ParamDist | None = Field(default=None)
    satisfies: dict[str, float] = Field(default_factory=dict)
    on_failure: Literal["skip", "abort"] = "abort"
    interruptible: bool | None = None

    @model_validator(mode="after")
    def exactly_one_target(self) -> "GoToStepDef":
        has_pose = self.target_pose is not None
        has_target = self.target is not None

        # Constraint from scenario_loader._structure_go_to_step_def:
        # exactly one of target_pose / target must be set.
        if has_pose and has_target:
            raise ValueError("go_to step accepts either 'target_pose: {x, y}' or 'target: <id|type>', not both")
        if not has_pose and not has_target:
            raise ValueError("go_to step requires either 'target_pose: {x, y}' or 'target: <id|type>'")
        return self


class SequenceDef(BaseModel):
    """
    Named state machines. Each sequence runs its `steps` in declaration order; on completion the compiler routes via `then` (chain) or `transitions` (need-driven preemption) or `on_failure` (recovery).
    """

    steps: dict[str, StepDef | GoToStepDef] = Field(description="Each step is either a `StepDef` (interaction, pure-wait, cancel) or a `GoToStepDef` (explicit `kind: go_to`).")
    then: str | None = None
    on_failure: str | None = None
    interruptible: bool = True
    transitions: list[TransitionDef] = Field(
        default_factory=list,
        description="`transitions` evaluate every tick — they can cut a step short. `then` only fires on the last step succeeding. `on_failure: <seq_name>` routes to another sequence when the current one FAILs. `interruptible: false` disables `transitions` for the sequence.",
    )


class ActionDef(BaseModel):
    when: dict[str, NeedCondition] = Field(default_factory=dict, description=r"`{need: {below\|above: X}}` — preconditions gating the action.")
    interaction: Literal["TALK_TO", "GROUP_CONVERSATION", "SIT_ON", "LIE_ON", "USE", "QUEUE_USE", "WAVE_AT", "BLOCK", "SERVICE"]
    target: str | None = Field(
        default=None,
        description=(
            "Interpreted per the interaction's handle kind: object id/type for `OBJECT`; "
            "service tag (str) for `SERVICE`; agent id (int) for `BLOCK`; omit for symmetric types. "
            "Required when interaction is one of ['SIT_ON', 'LIE_ON', 'USE', 'QUEUE_USE']. "
            "Symmetric interactions (TALK_TO, GROUP_CONVERSATION) don't take a `target:`."
        ),
    )
    duration: ParamDist | None = Field(default=None)
    patience: ParamDist | None = Field(default=None)
    satisfies: dict[str, float] = Field(default_factory=dict, description="Which needs this action satisfies. `{need: amount}` applied on SUCCESS.")
    on_failure: Literal["skip", "abort"] = "skip"

    @model_validator(mode="after")
    def check_action_target_constraints(self) -> "ActionDef":
        interaction = self.interaction

        _target_required = {"SIT_ON", "LIE_ON", "USE", "QUEUE_USE"}
        if interaction in _target_required and self.target is None:
            raise ValueError(f"action interaction: {interaction} requires 'target: <object_id|object_type>'")

        _symmetric = {"TALK_TO", "GROUP_CONVERSATION"}
        if interaction in _symmetric and self.target is not None:
            raise ValueError(f"action interaction: {interaction} is symmetric and takes no 'target:'.")

        return self


class AgentType(BaseModel):
    name: str
    mode: Literal["simple", "behavior_tree"] = "behavior_tree"

    needs: dict[str, NeedDist] = Field(
        default_factory=dict,
        description="""Describes agent needs, wants, requirements, demands, e.t.c. E.g:
```yaml
needs:
  thirst:
    initial: {mean: 55.0, std: 15.0, clip_low: 25.0, clip_high: 90.0}
    decay_rate: {mean: 8.0, std: 2.0, clip_low: 4.0, clip_high: 14.0}  # units/sec
```
    """,
    )
    utility_weights: dict[str, float] = Field(default_factory=dict, description="Per-need weights used by `AutonomousNode` when a step has `autonomous: true`. Higher weight ⇒ need drives action selection more aggressively.")
    # actions is disabled, this should work fine with steps only
    actions: dict[str, ActionDef] = Field(default_factory=dict, description="Candidate actions the autonomous selector can pick from.")
    sequences: dict[str, SequenceDef] = Field(
        default_factory=dict,
        description="""Named state machines, describes agent behaviors and interactions. E.g:
```yaml
sequences:
    chat:
        steps:
        hold_conversation:
            interaction: GROUP_CONVERSATION
            duration: {mean: 45.0, std: 15.0}
            patience: {mean: 60.0}
        then: chat                       # loop
        transitions:
        - when: {thirst: {below: 30.0}}
            goto: drink                  # preempt into another sequence
    drink:
        steps:
        queue_and_drink:
            target: fountain             # object id (or type)
            interaction: USE
            duration: {mean: 6.0}
            satisfies: {thirst: 100.0}
        then: chat
```
    """,
    )
    initial_sequence: str = Field(default="default", description="Picks the entry sequence by name.")
    # vars: dict[str, VarDef] = Field(default_factory=dict)
    # extends: Literal["adult", "elder"] | None = Field(description="Inherits fields from another agent type, then this agent type's fields override.")
