from __future__ import annotations

__all__ = [
    "BaseAgent",
    "Module",
    "TickPhase",
    "VectorizedModule",
    "ActionDef",
    "AgentType",
    "BUILTIN_AGENTS",
    "NeedCondition",
    "NeedDist",
    "ParamDist",
    "SampledNeed",
    "SampledParams",
    "SequenceDef",
    "StepDef",
    "TransitionDef",
    "VarDef",
    "sample_agent_type",
    "create_agent",
    "create_agent_from_params",
    "resolve_agent_type",
    "load_agent_types",
]

import attrs

from .base import BaseAgent, Module, TickPhase, VectorizedModule  # noqa: F401
from .factory import create_agent, create_agent_from_params  # noqa: F401
from .loader import load_agent_types, load_default_agent_types  # noqa: F401
from .types import ActionDef, AgentType, NeedCondition, NeedDist, ParamDist, SampledNeed, SampledParams, SequenceDef, StepDef, TransitionDef, VarDef, sample_agent_type  # noqa: F401 — re-export

BUILTIN_AGENTS: dict[str, AgentType] = load_default_agent_types()


def resolve_agent_type(
    agent_type: AgentType,
    registry: dict[str, AgentType] | None = None,
) -> AgentType:
    if registry is None:
        registry = BUILTIN_AGENTS

    if agent_type.extends is None:
        return agent_type

    parent_name = agent_type.extends
    if parent_name not in registry:
        raise KeyError(f"Agent type '{agent_type.name}' extends unknown parent '{parent_name}'. Available: {list(registry)}")

    parent = resolve_agent_type(registry[parent_name], registry)

    from .types import LocalPlannerDist, PerceptionDist

    defaults = AgentType(name="")
    merged: dict[str, object] = {}
    _NESTED_FROZEN = {"perception": PerceptionDist, "local_planner_params": LocalPlannerDist}
    for field in attrs.fields(AgentType):
        if field.name in ("name", "extends"):
            continue
        child_val = getattr(agent_type, field.name)
        default_val = getattr(defaults, field.name)
        parent_val = getattr(parent, field.name)
        if field.name in _NESTED_FROZEN:
            cls = _NESTED_FROZEN[field.name]
            nested_defaults = cls()
            nested_merged = {}
            for nf in attrs.fields(cls):
                cv = getattr(child_val, nf.name)
                if cv == getattr(nested_defaults, nf.name):
                    nested_merged[nf.name] = getattr(parent_val, nf.name)
                else:
                    nested_merged[nf.name] = cv
            merged[field.name] = cls(**nested_merged)
        elif child_val == default_val:
            merged[field.name] = parent_val
        else:
            merged[field.name] = child_val

    return AgentType(name=agent_type.name, extends=None, **merged)
