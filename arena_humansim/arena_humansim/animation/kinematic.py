from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from arena_humansim.agents import BaseAgent
from arena_humansim.utils.types import InteractionState, Pose2D

from . import MotionAnimation

if TYPE_CHECKING:
    from arena_humansim.pool import AgentPool


class KinematicAnimation(MotionAnimation):
    def compute_batch(
        self,
        agents: Iterable[BaseAgent],
        velocities: dict[int, tuple[float, float]],
        interactions: dict[int, InteractionState],
        dt: float,
    ) -> dict[int, Pose2D]:
        motions = {}
        for agent in agents:
            agent_id = agent.state.agent_id
            if agent_id in velocities:
                vx, vy = velocities[agent_id]
                motions[agent_id] = Pose2D(x=vx * dt, y=vy * dt)
        return motions

    def compute_batch_pool(
        self,
        pool: AgentPool,
        interactions: dict[int, InteractionState],
        dt: float,
    ) -> None:
        pass
