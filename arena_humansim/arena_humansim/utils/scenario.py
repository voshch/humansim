import ast
import copy
import enum
import operator
import re
from pathlib import Path
from typing import Any

import attrs
import yaml

from ..agents import BUILTIN_AGENTS, AgentType, VarDef
from .types import converter


@attrs.define
class Pose2DModel:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


class ExecutionMode(str, enum.Enum):
    MASTER = "master"
    SUBSYSTEM = "subsystem"


@attrs.define
class SimulationParams:
    seed: int = 0
    dt: float = 0.05
    bt_tick_interval: int = 5
    max_ticks: int = 0  # 0 = run indefinitely
    execution_mode: str = "master"


@attrs.define
class ModuleConfig:
    perception: str = "default"
    global_planner: str = "dijkstra"
    local_planner: str = "sfm"
    animation: str = "noop"

    perception_params: dict[str, Any] = attrs.Factory(dict)
    global_planner_params: dict[str, Any] = attrs.Factory(dict)
    local_planner_params: dict[str, Any] = attrs.Factory(dict)
    animation_params: dict[str, Any] = attrs.Factory(dict)


@attrs.define
class AgentConfig:
    agent_id: int = 0
    agent_type: str = "adult"
    spawn_pose: Pose2DModel = attrs.Factory(Pose2DModel)
    goal_sequence: list[Pose2DModel] = attrs.Factory(list)
    desired_velocity: float = 1.3
    agent_radius: float = 0.35
    interaction_preferences: dict[str, Any] = attrs.Factory(dict)


@attrs.define
class InteractionScript:
    tick: int = 0
    interaction_type: str = "TALK_TO"
    participants: list[int] = attrs.Factory(list)
    duration_ticks: int = 0
    metadata: dict[str, Any] = attrs.Factory(dict)


@attrs.define
class ShapeModel:
    type: str = "polygon"
    radius: float = 0.0


@attrs.define
class RateKeyframeModel:
    t: float = 0.0
    rate: float = 0.0


@attrs.define
class SinkAffinityModel:
    sink_idx: int = 0
    weight: float = 1.0


@attrs.define
class AgentTemplateModel:
    desired_velocity_min: float = 1.0
    desired_velocity_max: float = 1.5
    agent_radius: float = 0.35
    agent_type: str = "adult"
    sink_affinity: list[SinkAffinityModel] = attrs.Factory(list)


@attrs.define
class SourceScenarioConfig:
    pose: Pose2DModel = attrs.Factory(Pose2DModel)
    shape: ShapeModel = attrs.Factory(ShapeModel)
    rate_profile: list[RateKeyframeModel] = attrs.Factory(list)
    max_concurrent: int = -1
    max_total: int = -1
    agent_template: AgentTemplateModel = attrs.Factory(AgentTemplateModel)


@attrs.define
class SinkScenarioConfig:
    pose: Pose2DModel = attrs.Factory(Pose2DModel)
    shape: ShapeModel = attrs.Factory(ShapeModel)
    absorption_radius: float = 0.5
    capacity: int = -1


@attrs.define
class FlowScenarioConfig:
    sources: list[SourceScenarioConfig] = attrs.Factory(list)
    sinks: list[SinkScenarioConfig] = attrs.Factory(list)


@attrs.define
class WorldObjectConfig:
    object_id: str = ""
    type: str = ""
    pose: Pose2DModel = attrs.Factory(Pose2DModel)
    capacity: int = 1
    satisfies: dict[str, float] = attrs.Factory(dict)


@attrs.define
class EventScript:
    tick: int = 0
    event: str = ""
    target_agent: int = -1  # -1 = broadcast


@attrs.define
class ScenarioConfig:
    name: str = "unnamed"
    description: str = ""
    simulation: SimulationParams = attrs.Factory(SimulationParams)
    modules: ModuleConfig = attrs.Factory(ModuleConfig)
    agents: list[AgentConfig] = attrs.Factory(list)
    interaction_scripts: list[InteractionScript] = attrs.Factory(list)
    flow: FlowScenarioConfig | None = None
    agent_types: dict[str, AgentType] = attrs.Factory(dict)
    world_objects: list[WorldObjectConfig] = attrs.Factory(list)
    event_scripts: list[EventScript] = attrs.Factory(list)


def _parse_agent_types(raw_section: dict[str, Any]) -> dict[str, AgentType]:
    from arena_humansim.utils.types import converter

    result: dict[str, AgentType] = {}
    for name, fields in raw_section.items():
        if fields is None:
            fields = {}
        fields["name"] = name
        result[name] = converter.structure(fields, AgentType)
    return result


_DICT_MERGE_FIELDS = {
    "needs",
    "utility_weights",
    "actions",
    "sequences",
    "vars",
    "perception",
    "local_planner_params",
}

_TUPLE_FIELDS = {"perception_stack"}


def resolve_extends(
    agent_types: dict[str, AgentType],
    builtins: dict[str, AgentType],
) -> dict[str, AgentType]:
    all_types: dict[str, dict[str, Any]] = {}
    for name, at in agent_types.items():
        all_types[name] = attrs.asdict(at)  # type: ignore[arg-type]

    resolved: dict[str, dict[str, Any]] = {}
    in_progress: set[str] = set()

    def _resolve(name: str) -> dict[str, Any]:
        if name in resolved:
            return resolved[name]
        if name not in all_types:
            raise ValueError(f"Agent type '{name}' referenced by extends but not defined")
        if name in in_progress:
            raise ValueError(f"Circular extends detected involving '{name}'")

        in_progress.add(name)
        child = all_types[name]
        extends = child.get("extends")

        if extends is None:
            resolved[name] = child
            in_progress.discard(name)
            return child

        parent = _resolve(extends)
        merged = _deep_merge(parent, child, name)
        resolved[name] = merged
        in_progress.discard(name)
        return merged

    for name in all_types:
        _resolve(name)

    from arena_humansim.utils.types import converter

    result: dict[str, AgentType] = {}
    for name, raw in resolved.items():
        raw["name"] = name
        result[name] = converter.structure(raw, AgentType)
    return result


def _deep_merge(
    parent: dict[str, Any],
    child: dict[str, Any],
    child_name: str,
) -> dict[str, Any]:
    merged = copy.deepcopy(parent)
    merged["name"] = child_name
    merged["extends"] = None

    for key, child_val in child.items():
        if key in ("name", "extends"):
            continue
        if key in _DICT_MERGE_FIELDS:
            if isinstance(child_val, dict) and child_val:
                parent_val = merged.get(key, {})
                if not isinstance(parent_val, dict):
                    parent_val = {}
                merged_dict = copy.deepcopy(parent_val)
                merged_dict.update(copy.deepcopy(child_val))
                merged[key] = merged_dict
        elif key in _TUPLE_FIELDS:
            merged[key] = copy.deepcopy(child_val)
        else:
            merged[key] = copy.deepcopy(child_val)

    return merged


_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST, variables: dict[str, int | float | bool | str]) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body, variables)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        raise ValueError(f"Unknown variable: {node.id}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left, variables)
        right = _safe_eval(node.right, variables)
        return _SAFE_OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval(node.operand, variables)
        return _SAFE_OPS[op_type](operand)
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def _resolve_var_string(
    s: str,
    variables: dict[str, int | float | bool | str],
) -> Any:
    pattern = r"\$\{([^}]+)\}"
    matches = list(re.finditer(pattern, s))
    if not matches:
        return s

    if len(matches) == 1 and matches[0].start() == 0 and matches[0].end() == len(s):
        expr_str = matches[0].group(1).strip()
        if expr_str in variables:
            return variables[expr_str]
        tree = ast.parse(expr_str, mode="eval")
        return _safe_eval(tree, variables)

    def _replace(m: re.Match) -> str:
        expr_str = m.group(1).strip()
        if expr_str in variables:
            return str(variables[expr_str])
        tree = ast.parse(expr_str, mode="eval")
        return str(_safe_eval(tree, variables))

    return re.sub(pattern, _replace, s)


def resolve_vars(
    raw_dict: dict[str, Any],
    var_defs: dict[str, VarDef],
    overrides: dict[str, int | float | bool | str] | None = None,
) -> dict[str, Any]:
    variables: dict[str, int | float | bool | str] = {}
    for vname, vdef in var_defs.items():
        variables[vname] = vdef.default

    if overrides:
        for vname, val in overrides.items():
            if vname not in var_defs:
                raise ValueError(f"Override for unknown variable: {vname}")
            vdef = var_defs[vname]
            _type_check_var(vname, val, vdef)
            variables[vname] = val

    for vname, val in variables.items():
        vdef = var_defs[vname]
        if isinstance(val, (int, float)):
            if vdef.min is not None and val < vdef.min:
                raise ValueError(f"Variable '{vname}' value {val} below minimum {vdef.min}")
            if vdef.max is not None and val > vdef.max:
                raise ValueError(f"Variable '{vname}' value {val} above maximum {vdef.max}")

    return _walk_resolve(raw_dict, variables)


def _type_check_var(
    name: str,
    value: int | float | bool | str,
    vdef: VarDef,
) -> None:
    expected = {
        "int": int,
        "float": (int, float),
        "bool": bool,
        "str": str,
    }.get(vdef.type)
    if expected is not None and not isinstance(value, expected):
        raise TypeError(f"Variable '{name}' expected type {vdef.type}, got {type(value).__name__}")


def _walk_resolve(
    obj: Any,
    variables: dict[str, int | float | bool | str],
) -> Any:
    if isinstance(obj, str):
        return _resolve_var_string(obj, variables)
    if isinstance(obj, dict):
        return {k: _walk_resolve(v, variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_resolve(item, variables) for item in obj]
    return obj


def load_scenario(
    path: str,
    var_overrides: dict[str, int | float | bool | str] | None = None,
) -> ScenarioConfig:
    scenario_path = Path(path)
    if not scenario_path.is_file():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with open(scenario_path, "r") as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        raw = {}

    return _structure_manual(
        raw,
        var_overrides=var_overrides,
        scenario_dir=scenario_path.parent,
    )


def _structure_manual(
    data: dict[str, Any],
    var_overrides: dict[str, int | float | bool | str] | None = None,
    scenario_dir: Path | None = None,
) -> ScenarioConfig:
    from arena_humansim.agents.loader import load_agent_types

    file_types = load_agent_types(scenario_dir=scenario_dir)

    raw_agent_types = data.get("agent_types", {})
    if raw_agent_types:
        all_var_defs: dict[str, VarDef] = {}
        for _atype_name, atype_fields in raw_agent_types.items():
            if atype_fields and isinstance(atype_fields, dict):
                raw_vars = atype_fields.get("vars")
                if raw_vars and isinstance(raw_vars, dict):
                    for vname, vdef_raw in raw_vars.items():
                        if vname not in all_var_defs:
                            all_var_defs[vname] = converter.structure(vdef_raw, VarDef)

        if all_var_defs:
            resolved_raw = {}
            for atype_name, atype_fields in raw_agent_types.items():
                if atype_fields and isinstance(atype_fields, dict):
                    resolved_raw[atype_name] = resolve_vars(atype_fields, all_var_defs, var_overrides)
                else:
                    resolved_raw[atype_name] = atype_fields
            raw_agent_types = resolved_raw

        inline_types = _parse_agent_types(raw_agent_types)
    else:
        inline_types = {}

    merged_types: dict[str, AgentType] = {**file_types, **inline_types}

    has_extends = any(at.extends is not None for at in merged_types.values())
    if has_extends:
        merged_types = resolve_extends(merged_types, file_types)

    sim = converter.structure(data.get("simulation", {}), SimulationParams)
    modules = converter.structure(data.get("modules", {}), ModuleConfig)
    agents = [converter.structure(a, AgentConfig) for a in data.get("agents", [])]
    scripts = [converter.structure(s, InteractionScript) for s in data.get("interaction_scripts", [])]
    flow_raw = data.get("flow")
    flow = converter.structure(flow_raw, FlowScenarioConfig) if flow_raw else None
    world_objects = [converter.structure(wo, WorldObjectConfig) for wo in data.get("world_objects", [])]
    event_scripts = [converter.structure(es, EventScript) for es in data.get("event_scripts", [])]

    return ScenarioConfig(
        name=data.get("name", "unnamed"),
        description=data.get("description", ""),
        simulation=sim,
        modules=modules,
        agents=agents,
        interaction_scripts=scripts,
        flow=flow,
        agent_types=merged_types,
        world_objects=world_objects,
        event_scripts=event_scripts,
    )
