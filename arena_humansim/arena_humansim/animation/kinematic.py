from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

from arena_humansim.core.agents import BaseAgent
from arena_humansim.utils.types import BehaviorTreeMovement, CommandType, InteractionState, Pose2D

from . import MotionAnimation

if TYPE_CHECKING:
    from arena_humansim.core.pool import AgentPool

_GAZE_AMPLITUDE = 0.6  # radians (~35°)
_IDLE_SPEED_THRESHOLD = 0.05  # m/s


class KinematicAnimation(MotionAnimation):
    def __init__(self) -> None:
        # agent_id -> (phi, baseline_theta)
        self._gaze_state: dict[int, tuple[float, float]] = {}
        # agent_id -> idle_gaze_rate_hz
        self._gaze_rates: dict[int, float] = {}

    def _register_agent(self, agent_id: int, rate_hz: float) -> None:
        self._gaze_rates[agent_id] = rate_hz

    def _unregister_agent(self, agent_id: int) -> None:
        self._gaze_rates.pop(agent_id, None)
        self._gaze_state.pop(agent_id, None)

    def _apply_gaze(self, agent_id: int, rate_hz: float, is_idle: bool, current_theta: float, dt: float) -> float:
        if not is_idle or rate_hz <= 0.0:
            self._gaze_state.pop(agent_id, None)
            return current_theta

        if agent_id not in self._gaze_state:
            # snapshot baseline on first idle tick
            self._gaze_state[agent_id] = (0.0, current_theta)

        phi, baseline = self._gaze_state[agent_id]
        phi += 2.0 * math.pi * rate_hz * dt
        self._gaze_state[agent_id] = (phi, baseline)
        return baseline + _GAZE_AMPLITUDE * math.sin(phi)

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
            rate_hz = agent.params.idle_gaze_rate_hz
            self._gaze_rates[agent_id] = rate_hz

            vx, vy = velocities.get(agent_id, (0.0, 0.0))
            speed = math.hypot(vx, vy)

            is_seeking = isinstance(agent.movement, BehaviorTreeMovement) and agent.movement.command is not None and agent.movement.command.type == CommandType.SEEK
            is_idle = is_seeking or speed < _IDLE_SPEED_THRESHOLD

            agent.state.pose.theta = self._apply_gaze(agent_id, rate_hz, is_idle, agent.state.pose.theta, dt)

            if agent_id in velocities:
                motions[agent_id] = Pose2D(x=vx * dt, y=vy * dt)
        return motions

    def compute_batch_pool(
        self,
        pool: AgentPool,
        interactions: dict[int, InteractionState],
        dt: float,
    ) -> None:
        n = pool.n
        if n == 0:
            return
        for i in range(n):
            agent_id = int(pool.agent_ids[i])
            rate_hz = self._gaze_rates.get(agent_id, 0.0)
            if rate_hz <= 0.0:
                self._gaze_state.pop(agent_id, None)
                continue
            vx, vy = float(pool.vel[i, 0]), float(pool.vel[i, 1])
            speed = math.hypot(vx, vy)
            is_idle = speed < _IDLE_SPEED_THRESHOLD
            pool.theta[i] = self._apply_gaze(agent_id, rate_hz, is_idle, float(pool.theta[i]), dt)
