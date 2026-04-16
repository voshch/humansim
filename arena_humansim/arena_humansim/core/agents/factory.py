from __future__ import annotations

__all__ = [
    "create_agent",
    "create_agent_from_params",
]

import numpy as np

from arena_humansim.utils.types import AgentState, NeedsState, NeedState

from .base import BaseAgent, Module
from .types import AgentType, SampledParams, sample_agent_type


def _fallback_bases() -> dict[str, type]:
    from arena_humansim.animation import MotionAnimation
    from arena_humansim.global_planner import GlobalPlanner
    from arena_humansim.local_planner import LocalPlanner
    return {
        "global_planner": GlobalPlanner,
        "local_planner": LocalPlanner,
        "animation": MotionAnimation,
    }


def _pool_lookup(module_pool: dict[str, Module], name: str | None, category: str) -> Module:
    if name is not None and name in module_pool:
        return module_pool[name]
    base = _fallback_bases().get(category)
    if base is not None:
        for v in module_pool.values():
            if isinstance(v, base):
                return v
    raise KeyError(f"{name!r} not found in module_pool and no fallback for {category!r}")


def _resolve_modules(
    perception_stack: tuple[str, ...],
    local_planner: str,
    global_planner: str,
    animation: str,
    module_pool: dict[str, Module],
) -> dict[str, Module | list[Module]]:
    return {
        "perception": [module_pool[name] for name in perception_stack],
        "local_planner": _pool_lookup(module_pool, local_planner, "local_planner"),
        "global_planner": _pool_lookup(module_pool, global_planner, "global_planner"),
        "animation": _pool_lookup(module_pool, animation, "animation"),
    }


def create_agent(
    agent_type: AgentType | str,
    state: AgentState,
    module_pool: dict[str, Module],
    rng: np.random.Generator,
) -> BaseAgent:
    if isinstance(agent_type, str):
        from . import BUILTIN_AGENTS

        agent_type = BUILTIN_AGENTS[agent_type]

    params = sample_agent_type(agent_type, rng)

    needs = None
    if params.needs:
        needs = NeedsState(needs={name: NeedState(value=sn.initial, decay_rate=sn.decay_rate) for name, sn in params.needs.items()})

    modules = _resolve_modules(
        params.perception_stack,
        params.local_planner,
        params.global_planner,
        params.animation,
        module_pool,
    )

    return BaseAgent(state=state, params=params, needs=needs, **modules)


def create_agent_from_params(
    params: SampledParams,
    state: AgentState,
    module_pool: dict[str, Module],
) -> BaseAgent:
    needs = None
    if params.needs:
        needs = NeedsState(needs={name: NeedState(value=sn.initial, decay_rate=sn.decay_rate) for name, sn in params.needs.items()})

    modules = _resolve_modules(
        params.perception_stack,
        params.local_planner,
        params.global_planner,
        params.animation,
        module_pool,
    )

    return BaseAgent(state=state, params=params, needs=needs, **modules)
