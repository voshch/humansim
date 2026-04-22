"""cattrs structuring hooks for scenario/agent-type YAML loading."""

import attrs
import cattrs

from arena_humansim.core.agents.types import (
    ActionDef,
    AgentType,
    GoToStepDef,
    LocalPlannerDist,
    NeedCondition,
    NeedDist,
    PerceptionDist,
    SequenceDef,
    StepDef,
    TransitionDef,
    VarDef,
)
from arena_humansim.core.interaction_kinds import HandleKind, InteractionType
from arena_humansim.utils.types import AnchorKind, FormationSpec, Pose2D

converter = cattrs.Converter()


def _structure_need_dist(val: object, _: type) -> NeedDist:
    if isinstance(val, (int, float)):
        return NeedDist(initial=float(val))
    if isinstance(val, NeedDist):
        return val
    return NeedDist(**dict(val))


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
    return ActionDef(**d)


converter.register_structure_hook(ActionDef, _structure_action_def)


_STEP_KINDS = ("object_interact", "go_to")


def _structure_go_to_step_def(val: object, _: type) -> GoToStepDef:
    if isinstance(val, GoToStepDef):
        return val
    d = dict(val)
    d.pop("kind", None)
    has_pose = "target_pose" in d and d["target_pose"] is not None
    has_target = "target" in d and d["target"] is not None
    if has_pose and has_target:
        raise ValueError("go_to step accepts either 'target_pose: {x, y}' or 'target: <id|type>', not both")
    if not has_pose and not has_target:
        raise ValueError("go_to step requires either 'target_pose: {x, y}' or 'target: <id|type>'")
    if has_pose:
        pose = d["target_pose"]
        if isinstance(pose, Pose2D):
            d["target_pose"] = pose
        else:
            pose_d = dict(pose)
            d["target_pose"] = Pose2D(x=float(pose_d["x"]), y=float(pose_d["y"]), theta=float(pose_d.get("theta", 0.0)))
    return GoToStepDef(**d)


converter.register_structure_hook(GoToStepDef, _structure_go_to_step_def)


def _structure_step_def(val: object, _: type) -> StepDef:
    if isinstance(val, StepDef):
        return val
    d = dict(val)
    d.pop("kind", None)
    if "accept" in d:
        raise ValueError("step field 'accept' removed; drop it and use 'interaction:' (absence of 'offer: true' means seeker)")
    if "interaction" in d and d["interaction"] == "FOLLOW":
        raise ValueError("interaction FOLLOW removed; use 'interaction: SERVICE, target: <tag>, offer: true, min_participants: 2, max_participants: 2, queueable: false, formation_spec: {type: line, anchor_kind: provider, params: {base_step: 0.8}}'")
    for legacy in ("target_object_id", "target_object_type", "target_agent", "service_tag"):
        if legacy in d:
            raise ValueError(f"step field {legacy!r} removed; use 'target:' instead (meaning determined by interaction type)")
    interaction = d.get("interaction")
    offer = bool(d.get("offer", False))
    target = d.get("target")
    kind = None
    if interaction is not None:
        try:
            kind = InteractionType[interaction].kind
        except KeyError:
            raise ValueError(f"unknown interaction type {interaction!r}") from None
    if offer and (kind is None or not kind.allows_offer):
        raise ValueError(f"'offer: true' is not valid for interaction={interaction!r}")
    provider_fields = ("queueable", "min_participants", "max_participants", "formation_spec")
    for pf in provider_fields:
        if pf in d and d[pf] is not None and not offer:
            raise ValueError(f"step field {pf!r} is provider-side configuration; requires 'offer: true'")
    if kind is not None and not d.get("cancel", False):
        hk = kind.handle.kind
        strategy = kind.handle.strategy
        shapes = strategy.target_shape
        if hk == HandleKind.NONE:
            if target is not None:
                raise ValueError(f"interaction: {interaction} takes no target; 'target' must not be set")
        elif hk == HandleKind.TAG:
            if offer and not isinstance(target, shapes):
                raise ValueError(f"interaction: {interaction} with 'offer: true' requires 'target: <tag:str>'")
            if not offer and target is not None and not isinstance(target, shapes):
                raise ValueError(f"interaction: {interaction} seeker 'target' must be a tag string or omitted")
        elif strategy.target_required_for_create and not isinstance(target, shapes):
            expected = " | ".join(t.__name__ for t in shapes)
            raise ValueError(f"interaction: {interaction} requires 'target: <{expected}>'")
    if "until_need" in d and isinstance(d["until_need"], dict):
        d["until_need"] = {k: converter.structure(v, NeedCondition) for k, v in d["until_need"].items()}
    if "allowed_actions" in d and d["allowed_actions"] is not None:
        d["allowed_actions"] = tuple(d["allowed_actions"])
    if "blocked_actions" in d and d["blocked_actions"] is not None:
        d["blocked_actions"] = tuple(d["blocked_actions"])
    if "formation_spec" in d and isinstance(d["formation_spec"], dict):
        fs = d["formation_spec"]
        pose_cfg = fs.get("anchor_pose")
        anchor_pose = None
        if pose_cfg is not None:
            anchor_pose = Pose2D(
                x=float(pose_cfg.get("x", 0.0)),
                y=float(pose_cfg.get("y", 0.0)),
                theta=float(pose_cfg.get("theta", 0.0)),
            )
        raw_kind = fs.get("anchor_kind", "object")
        try:
            anchor_kind_val = AnchorKind(raw_kind)
        except ValueError:
            raise ValueError(f"Unknown anchor_kind {raw_kind!r}. Valid: {[k.value for k in AnchorKind]}") from None
        d["formation_spec"] = FormationSpec(
            type=fs["type"],
            params=dict(fs.get("params", {})),
            anchor_kind=anchor_kind_val,
            anchor_ref=fs.get("anchor_ref"),
            anchor_pose=anchor_pose,
        )
    known = {f.name for f in attrs.fields(StepDef)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown step fields: {sorted(unknown)}")
    return StepDef(**d)


converter.register_structure_hook(StepDef, _structure_step_def)


def _structure_step_variant(val: object) -> StepDef | GoToStepDef:
    if isinstance(val, (StepDef, GoToStepDef)):
        return val
    kind = dict(val).get("kind", "object_interact")
    if kind == "go_to":
        return converter.structure(val, GoToStepDef)
    if kind == "object_interact":
        return converter.structure(val, StepDef)
    raise ValueError(f"unknown step kind {kind!r}; expected one of {_STEP_KINDS}")


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
        d["steps"] = {k: _structure_step_variant(v) for k, v in d["steps"].items()}
    if "transitions" in d:
        d["transitions"] = tuple(converter.structure(t, TransitionDef) for t in d["transitions"])
    else:
        d["transitions"] = ()
    return SequenceDef(**d)


converter.register_structure_hook(SequenceDef, _structure_sequence_def)


def _structure_perception_dist(val: object, _: type) -> PerceptionDist:
    if isinstance(val, PerceptionDist):
        return val
    return PerceptionDist(**dict(val))


converter.register_structure_hook(PerceptionDist, _structure_perception_dist)


def _structure_local_planner_dist(val: object, _: type) -> LocalPlannerDist:
    if isinstance(val, LocalPlannerDist):
        return val
    return LocalPlannerDist(**dict(val))


converter.register_structure_hook(LocalPlannerDist, _structure_local_planner_dist)


def _structure_agent_type(val: object, _: type) -> AgentType:
    if isinstance(val, AgentType):
        return val
    d = dict(val)
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
