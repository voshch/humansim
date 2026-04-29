"""
Pydantic schema for LLM-based behavior-tree generation.

Design principle: every constraint that the runtime (scenario_loader.py) enforces
must be expressed as a *structural* property of the JSON schema — i.e. via
`Literal` fields, `required` fields, or absent fields — so the LLM cannot emit
an invalid value even if it tries.

`@model_validator` is used only as a defence-in-depth layer for cross-field rules
that JSON Schema cannot express natively (e.g. "exactly one of A or B").
"""

import enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class Pose2D(BaseModel):
    x: float = 0.0
    y: float = 0.0
    theta: float = Field(default=0.0, description="Yaw (rad)")


class ParamDist(BaseModel):
    mean: float
    std: float = 0.0
    clip_low: float = 0.01
    clip_high: float = 1e9  # Infinity won't work for google.genai structured output


class NeedDist(BaseModel):
    """
    Scalar need in [0, 100] that decays every tick and is restored by steps via `satisfies`.
    Decay is linear; `satisfies: {thirst: 100.0}` adds 100 on completion (clamped to 100).
    """

    initial: ParamDist = Field(
        default=ParamDist(mean=100.0, clip_low=0.0, clip_high=100.0),
        description="Starting value distribution. clip_low/clip_high should be within [0, 100].",
    )
    decay_rate: ParamDist = Field(
        default=ParamDist(mean=0.5, std=0.1),
        description="Units per second drained from this need.",
    )


class NeedCondition(BaseModel):
    below: float | None = None
    above: float | None = None


# class VarDef(BaseModel):
#     type: Literal["int", "float", "bool", "str"]
#     default: int | float | bool | str
#     min: float | None = None
#     max: float | None = None


class TransitionDef(BaseModel):
    when: dict[str, NeedCondition] = Field(description=("Need-condition map that triggers the transition, e.g. `{thirst: {below: 30.0}}`."))
    goto: str = Field(description="Name of the sequence to jump to when `when` is satisfied.")


class AnchorKind(enum.Enum):
    OBJECT = "object"
    AGENT = "agent"
    PROVIDER = "provider"
    POSE = "pose"
    CENTROID = "centroid"


class FormationSpec(BaseModel):
    type: str
    params: dict[str, float] = Field(default_factory=dict)
    anchor_kind: str = AnchorKind.OBJECT.value
    anchor_ref: str | None = None
    anchor_pose: Pose2D | None = None


# ---------------------------------------------------------------------------
# Shared step timing fields (mixed into every step variant via inheritance)
# ---------------------------------------------------------------------------


class _StepTimingMixin(BaseModel):
    model_config = ConfigDict(extra="forbid")
    duration: ParamDist | None = Field(default=None, description="How long the step runs.")
    patience: ParamDist | None = Field(default=None, description="How long the agent will wait before giving up.")
    satisfies: dict[str, float] = Field(default_factory=dict, description="`{need: amount}` credited to the agent on SUCCESS.")
    on_failure: Literal["abort", "skip"] = "abort"
    interruptible: bool | None = None
    interaction_radius: float | None = None
    autonomous: bool = Field(default=False, description="`true` => AutonomousNode scores `actions` each tick and runs the winner.")
    until: str | None = Field(default=None, description="Event-bus event name that exits the autonomous step.")
    until_need: dict[str, NeedCondition] | None = Field(default=None, description="Need predicate that exits the autonomous step, e.g. `{rest: {above: 80}}`.")
    allowed_actions: list[str] | None = Field(default=None, description="Restrict the autonomous candidate pool to these actions.")
    blocked_actions: list[str] | None = Field(default=None, description="Exclude these actions from the autonomous candidate pool.")
    wait_for_outcome: bool = False


# ---------------------------------------------------------------------------
# StepDef variants — each variant encodes its own structural constraints.
#
# Split constraints into separate classes with Literal field types.
# Constraints become part of the JSON schema the LLM receives, not just runtime
# validators it never sees.
# ---------------------------------------------------------------------------


class ServiceProviderStep(_StepTimingMixin):
    """
    SERVICE interaction — provider side.
    Creates a service slot and waits for seekers to join.
    `target` is the service tag string (e.g. 'reception_counter_1' or 'SM_ReceptionDesk_01a') and is REQUIRED.
    Provider-side fields (queueable, min/max_participants, formation_spec) are only
    valid here, never on seeker steps.
    """

    interaction: Literal["SERVICE"]
    offer: Literal[True] = Field(
        default=True,
        description="Must be `true` for provider steps.",
    )
    target: str = Field(max_length=64, description=("Service tag that names this slot, must match a world objects `object_id` or `type`, e.g. 'reception_counter_1' or 'SM_ReceptionDesk_01a'. Seekers use this same tag to find and join the service. REQUIRED."))
    queueable: bool | None = Field(default=None, description="If true, seekers queue when the slot is full.")
    min_participants: int | None = None
    max_participants: int | None = None
    formation_spec: FormationSpec | None = None


class ServiceSeekerStep(_StepTimingMixin):
    """
    SERVICE interaction — seeker side.
    Finds an existing provider slot and joins it.
    `target` is an optional tag filter; omit to join any available service.
    Do NOT set queueable, min_participants, max_participants, or formation_spec here.
    """

    interaction: Literal["SERVICE"]
    offer: Literal[False] = Field(
        default=False,
        description="Must be `false` (or omitted) for seeker steps.",
    )
    target: str | None = Field(
        default=None,
        max_length=64,
        description="Optional service tag to filter by, must match a world objects `object_id` or `type`, e.g. 'reception_counter_1' or 'SM_ReceptionDesk_01a'. Omit to join any available service.",
    )


class ObjectInteractStep(_StepTimingMixin):
    """
    Object-handle interactions: SIT_ON, LIE_ON, USE, QUEUE_USE.
    `target` is REQUIRED — must be the `object_id` or `type` string (e.g. 'reception_counter_1' or 'SM_ReceptionDesk_01a').
    """

    interaction: Literal["SIT_ON", "LIE_ON", "USE", "QUEUE_USE"]
    target: str = Field(max_length=64, description="Object id or type to interact with, must match a world objects `object_id` or `type`, e.g. 'reception_counter_1' or 'SM_ReceptionDesk_01a'. REQUIRED.")


class SymmetricInteractStep(_StepTimingMixin):
    """
    Symmetric peer-to-peer interactions: TALK_TO, GROUP_CONVERSATION.
    Do NOT set `target` — the runtime matches participants automatically.
    To control where the meeting happens, precede with a GoToStepDef.
    """

    interaction: Literal["TALK_TO", "GROUP_CONVERSATION"]
    # `target` is intentionally absent from this class.
    # The LLM has no field to fill in, so it cannot accidentally set one.


class WaveAtStep(_StepTimingMixin):
    """
    An agent waves hand at another.
    Do NOT set `target` — the runtime matches participants automatically.
    """

    interaction: Literal["WAVE_AT"]


class BlockStep(_StepTimingMixin):
    """
    An agent blocks the navigation way of another.
    """

    interaction: Literal["BLOCK"]
    target: str | None = Field(
        default=None,
        max_length=64,
        description="Agent id to interact with.",
    )


class CancelStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    """
    Emits a STOP/CANCELED signal on the agent's currently active interaction.
    Do NOT include an `interaction:` field — cancel is mutually exclusive with starting one.
    """
    cancel: Literal[True] = True
    on_failure: Literal["abort", "skip"] = "abort"


class PureWaitStep(_StepTimingMixin):
    """
    A step with no interaction — the agent idles for `duration`.
    Use with `autonomous: true` to let the agent score its `actions` while waiting.
    """

    interaction: None = Field(
        default=None,
        description="Must be null (or omitted) for pure-wait steps.",
    )


# Union the LLM sees for every step in a sequence.
# Class names and docstrings propagate into the JSON schema as `title` and
# `description` fields, guiding LLM generation structurally.
StepDef = Annotated[
    ServiceProviderStep | ServiceSeekerStep | ObjectInteractStep | SymmetricInteractStep | WaveAtStep | BlockStep | CancelStep | PureWaitStep,
    Field(
        description=(
            "One step in a sequence. Choose the variant that matches your intent:\n"
            "- ServiceProviderStep:   interaction=SERVICE, offer=true, target=<tag> REQUIRED\n"
            "- ServiceSeekerStep:     interaction=SERVICE, offer=false, target optional\n"
            "- ObjectInteractStep:    interaction=SIT_ON/LIE_ON/USE/QUEUE_USE, target REQUIRED\n"
            "- SymmetricInteractStep: interaction=TALK_TO/GROUP_CONVERSATION, NO target field\n"
            "- WaveAtStep:            interaction=WAVE_AT\n"
            "- BlockStep:             interaction=BLOCK\n"
            "- CancelStep:            cancel=true, NO interaction field\n"
            "- PureWaitStep:          no interaction, idle wait"
        )
    ),
]


# ---------------------------------------------------------------------------
# GoToStepDef — navigation step
# ---------------------------------------------------------------------------


class GoToStepDef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    """
    Navigation step. Set EXACTLY ONE of:
    - `target_pose`: a fixed {x, y[, theta]} waypoint
    - `target`: an object id or type to navigate toward

    Setting both or neither is an error caught at validation time.
    """
    kind: Literal["go_to"] = Field(description="Must be 'go_to'.")
    target_pose: Pose2D | None = Field(
        default=None,
        description="Fixed waypoint. Set this OR `target`, never both.",
    )
    target: str | None = Field(
        default=None,
        max_length=64,
        description="Object id/type to navigate toward, must match a world objects `object_id` or `type`. Set this OR `target_pose`, never both.",
    )
    duration: ParamDist | None = None
    patience: ParamDist | None = None
    satisfies: dict[str, float] = Field(default_factory=dict)
    on_failure: Literal["skip", "abort"] = "abort"
    interruptible: bool | None = None

    @model_validator(mode="after")
    def exactly_one_target(self) -> "GoToStepDef":
        # JSON Schema cannot express XOR, so this stays as a model_validator.
        has_pose = self.target_pose is not None
        has_target = self.target is not None
        if has_pose and has_target:
            raise ValueError("go_to step: set either 'target_pose' or 'target', not both")
        if not has_pose and not has_target:
            raise ValueError("go_to step: must set either 'target_pose: {x, y}' or 'target: <id|type>'")
        return self


# ---------------------------------------------------------------------------
# ActionDef
# ---------------------------------------------------------------------------


class ActionDef(BaseModel):
    """Candidate action an AutonomousNode can score and execute."""

    when: dict[str, NeedCondition] = Field(
        default_factory=dict,
        description=r"Precondition map: `{need: {below|above: X}}`. Action is only eligible when all conditions hold.",
    )
    interaction: Literal["SIT_ON", "LIE_ON", "USE", "QUEUE_USE", "WAVE_AT", "BLOCK", "SERVICE"]
    target: str | None = Field(
        default=None,
        max_length=64,
        description=("Object id/type for SIT_ON/LIE_ON/USE/QUEUE_USE (REQUIRED for these), must match a world objects `object_id` or `type`. Service tag for SERVICE. Omit for WAVE_AT/BLOCK."),
    )
    duration: ParamDist | None = None
    patience: ParamDist | None = None
    satisfies: dict[str, float] = Field(default_factory=dict)
    on_failure: Literal["skip", "abort"] = "skip"

    @model_validator(mode="after")
    def check_target(self) -> "ActionDef":
        if self.interaction in {"SIT_ON", "LIE_ON", "USE", "QUEUE_USE"} and self.target is None:
            raise ValueError(f"ActionDef: interaction {self.interaction!r} requires 'target'")
        return self


# ---------------------------------------------------------------------------
# SequenceDef / AgentType
# ---------------------------------------------------------------------------

SequenceStep = Annotated[
    StepDef | GoToStepDef,
    Field(description="A StepDef variant or a GoToStepDef navigation step."),
]


class SequenceDef(BaseModel):
    """
    Named state machine. Steps run in declaration order.
    Use `then` to chain to another sequence on success.
    Use `transitions` for need-driven preemption at any tick.
    """

    steps: dict[str, SequenceStep] = Field(description="Ordered map of step_name => step.")
    then: str | None = Field(default=None, description="Sequence to run after the last step succeeds.")
    on_failure: str | None = Field(default=None, description="Sequence to run if any step fails.")
    interruptible: bool = Field(default=True, description="Set false to disable transitions mid-sequence.")
    transitions: list[TransitionDef] = Field(
        default_factory=list,
        description="Need-condition checks evaluated every tick. Can preempt the current step.",
    )


class AgentType(BaseModel):
    name: str
    mode: Literal["simple", "behavior_tree"] = "behavior_tree"

    needs: dict[str, NeedDist] = Field(
        default_factory=dict,
        description=("Named needs the agent tracks. E.g:\n```yaml\nneeds:\n  thirst:\n    initial: {mean: 55.0, std: 15.0, clip_low: 25.0, clip_high: 90.0}\n    decay_rate: {mean: 8.0, std: 2.0, clip_low: 4.0, clip_high: 14.0}\n```"),
    )
    utility_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Per-need weights for AutonomousNode scoring. Higher = need drives selection more aggressively.",
    )
    actions: dict[str, ActionDef] = Field(
        default_factory=dict,
        description="Candidate actions available to autonomous steps.",
    )
    sequences: dict[str, SequenceDef] = Field(
        default_factory=dict,
        description=(
            "Named state machines. E.g:\n"
            "```yaml\n"
            "sequences:\n"
            "  chat:\n"
            "    steps:\n"
            "      hold_conversation:\n"
            "        interaction: GROUP_CONVERSATION\n"
            "        duration: {mean: 45.0, std: 15.0}\n"
            "    then: chat\n"
            "    transitions:\n"
            "    - when: {thirst: {below: 30.0}}\n"
            "      goto: drink\n"
            "  drink:\n"
            "    steps:\n"
            "      queue_and_drink:\n"
            "        interaction: USE\n"
            "        target: fountain\n"
            "        duration: {mean: 6.0}\n"
            "        satisfies: {thirst: 100.0}\n"
            "    then: chat\n"
            "```"
        ),
    )
    initial_sequence: str = Field(default="default", description="Entry sequence name.")


# ---------------------------------------------------------------------------
# AgentConfig (spawn configuration)
# ---------------------------------------------------------------------------


class ServiceSpec(BaseModel):
    tag: str = ""
    max_participants: int = -1


class AgentConfig(BaseModel):
    agent_id: int = 0
    agent_type: str = "adult"
    spawn_pose: Pose2D = Field(default_factory=Pose2D)
    desired_velocity: float = 1.3
    agent_radius: float = 0.35
    policy: Literal["sfm", "orca", "straight"] = Field(default="sfm")
    policy_params: str = Field(default="")
    services: list[ServiceSpec] = Field(default_factory=list)
    spawn_tick: int = Field(default=0, description="0 = spawn on first tick; >0 = deferred spawn.")
