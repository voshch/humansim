from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import py_trees

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.core.interaction_manager import InteractionManager
from arena_humansim.core.pool import AgentPool
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.types import BehaviorTreeMovement, CommandType, HighLevelCommand, InteractionOutcome, SeekSpec, Segments


def make_manager(agents: dict[int, Any]) -> InteractionManager:
    mgr = InteractionManager(RNG(0))
    mgr.set_context(
        agent_lookup=lambda aid: agents.get(aid),  # type: ignore[arg-type]
        visibility_lookup=lambda aid: set(agents) - {aid},
    )
    return mgr


def seed_active(mgr: InteractionManager, participants: list[int], itype: InteractionType) -> int:
    spec = SeekSpec(interaction_type=itype)
    interaction = mgr._create_interaction(creator_id=participants[0], spec=spec)
    for pid in participants[1:]:
        mgr.accept(pid, interaction.id)
    interaction.outcome = InteractionOutcome.ACTIVE
    interaction.contract.formation = None
    return interaction.id


def movement_of(agent: BaseAgent) -> BehaviorTreeMovement:
    return cast(BehaviorTreeMovement, agent.movement)


def with_bt(agent: BaseAgent) -> BaseAgent:
    agent.movement = BehaviorTreeMovement()
    return agent


def tick(node: py_trees.behaviour.Behaviour) -> py_trees.common.Status:
    node.tick_once()
    return node.status


def seek_command(agent_id: int, itype: InteractionType, *, target: str | int | None = None, offer: bool = False) -> HighLevelCommand:
    return HighLevelCommand(
        agent_id=agent_id,
        type=CommandType.SEEK,
        spec=SeekSpec(interaction_type=itype, target=target, offer=offer),
    )


def line_array(*rows: tuple[float, float, float, float]) -> np.ndarray:
    return np.array(rows, dtype=np.float64).reshape(-1, 4)


def random_lines(rng: np.random.Generator, n: int) -> np.ndarray:
    ang = rng.uniform(-math.pi, math.pi, n)
    return np.column_stack([rng.uniform(-2.0, 2.0, (n, 2)), np.cos(ang), np.sin(ang)])


def spec_pool(agent_factory: Callable[..., BaseAgent], specs: list[tuple[tuple[float, float], tuple[float, float] | None, tuple[float, float]]]) -> AgentPool:
    pool = AgentPool(capacity=max(len(specs), 1))
    for i, (pos, goal, vel) in enumerate(specs):
        agent = agent_factory(agent_id=i + 1, x=pos[0], y=pos[1])
        agent.state.velocity = vel
        idx = pool.add_agent(agent)
        if goal is not None:
            pool.goal_pos[idx] = goal
            pool.has_goal[idx] = True
    return pool


def pool_from(agents: list[BaseAgent]) -> AgentPool:
    pool = AgentPool(capacity=max(8, len(agents)))
    for ag in agents:
        pool.add_agent(ag)
    return pool


def wall_clearance(points: np.ndarray, walls: Segments) -> np.ndarray:
    seg = np.asarray(walls, dtype=np.float64).reshape(-1, 4)
    a = seg[:, :2]
    ab = seg[:, 2:] - a
    p = points.reshape(-1, 1, 2)
    t = np.clip(np.sum((p - a) * ab, axis=-1) / np.sum(ab * ab, axis=-1), 0.0, 1.0)
    closest = a + t[..., None] * ab
    return np.linalg.norm(p - closest, axis=-1).min(axis=1).reshape(points.shape[:-1])


def two_rooms(door: float) -> Segments:
    low, high = 3.0 - door / 2, 3.0 + door / 2
    return [
        ((0.0, 0.0), (12.0, 0.0)),
        ((12.0, 0.0), (12.0, 6.0)),
        ((12.0, 6.0), (0.0, 6.0)),
        ((0.0, 6.0), (0.0, 0.0)),
        ((6.0, 0.0), (6.0, low)),
        ((6.0, high), (6.0, 6.0)),
    ]
