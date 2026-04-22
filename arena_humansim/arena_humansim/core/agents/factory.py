from __future__ import annotations

__all__ = [
    "create_agent",
]

import numpy as np

from arena_humansim.utils.types import AgentState, NeedsState, NeedState

from .base import BaseAgent, Module
from .types import AgentType, SampledParams, sample_agent_type


def _resolve_modules(
    perception_stack: tuple[str, ...],
    local_planner: str | None,
    global_planner: str | None,
    animation: str | None,
    module_pool: dict[str, Module],
    defaults: dict[str, str],
) -> dict[str, Module | list[Module]]:
    return {
        "perception": [module_pool[name] for name in perception_stack],
        "local_planner": module_pool[local_planner or defaults["local_planner"]],
        "global_planner": module_pool[global_planner or defaults["global_planner"]],
        "animation": module_pool[animation or defaults["animation"]],
    }


def create_agent(
    agent_type: AgentType | str | SampledParams,
    state: AgentState,
    module_pool: dict[str, Module],
    defaults: dict[str, str],
    rng: np.random.Generator | None = None,
) -> BaseAgent:
    if isinstance(agent_type, str):
        from . import BUILTIN_AGENTS

        agent_type = BUILTIN_AGENTS[agent_type]

    if isinstance(agent_type, AgentType):
        if rng is None:
            raise ValueError("rng is required when agent_type is AgentType")
        params = sample_agent_type(agent_type, rng)
    else:
        params = agent_type

    needs = None
    if params.needs:
        needs = NeedsState(needs={name: NeedState(value=sn.initial, decay_rate=sn.decay_rate) for name, sn in params.needs.items()})

    modules = _resolve_modules(
        params.perception_stack,
        params.local_planner,
        params.global_planner,
        params.animation,
        module_pool,
        defaults,
    )

    return BaseAgent(state=state, params=params, needs=needs, **modules)
