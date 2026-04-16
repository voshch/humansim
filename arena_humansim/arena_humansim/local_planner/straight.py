from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from arena_humansim.core.agents import BaseAgent
from arena_humansim.utils.types import Pose2D

from . import LocalPlanner

if TYPE_CHECKING:
    from arena_humansim.core.pool import AgentPool

_EPS = 1e-6
_ARRIVAL_EPS = 1e-3


class StraightToGoalPlanner(LocalPlanner):
    supports_pool: bool = True
    needs_global_subgoal: bool = False

    def compute_pool(self, pool: AgentPool, store_forces: bool = False, dt: float = 1.0) -> None:
        n = pool.n
        if n == 0:
            return

        pos = pool.pos[:n]
        goal = pool.goal_pos[:n]
        has_goal = pool.has_goal[:n]
        desired_v = pool.desired_vel[:n]
        max_v = pool.max_velocity[:n]

        d = goal - pos
        dist = np.hypot(d[:, 0], d[:, 1])
        dist_safe = np.maximum(dist, _EPS)
        e = d / dist_safe[:, None]

        speed = np.minimum(desired_v, max_v)
        new_vel = e * speed[:, None]

        stopped = (~has_goal) | (dist < _ARRIVAL_EPS)
        new_vel[stopped] = 0.0
        pool.vel[:n] = new_vel

    def compute(
        self,
        agents: Sequence[BaseAgent],
        global_goals: dict[int, Pose2D],
        dt: float = 1.0,
    ) -> dict[int, tuple[float, float]]:
        velocities: dict[int, tuple[float, float]] = {}
        for agent in agents:
            aid = agent.state.agent_id
            goal = global_goals.get(aid)
            if goal is None:
                velocities[aid] = (0.0, 0.0)
                continue

            cx, cy = agent.state.pose.x, agent.state.pose.y
            dx = goal.x - cx
            dy = goal.y - cy
            dist = float(np.hypot(dx, dy))
            if dist < _ARRIVAL_EPS:
                velocities[aid] = (0.0, 0.0)
                continue

            speed = min(agent.params.desired_velocity, agent.params.max_velocity)
            velocities[aid] = (float(dx / dist * speed), float(dy / dist * speed))
        return velocities
