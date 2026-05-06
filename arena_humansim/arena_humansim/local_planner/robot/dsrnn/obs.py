from __future__ import annotations

from collections import deque

import numpy as np

from arena_humansim.core.agents import BaseAgent
from arena_humansim.utils.types import Pose2D


class _PerAgentHistory:
    def __init__(self, max_len: int):
        self.max_len = max_len
        self._buf: dict[int, deque[np.ndarray]] = {}

    def push(self, aid: int, pos: np.ndarray) -> None:
        d = self._buf.get(aid)
        if d is None:
            d = deque(maxlen=self.max_len)
            self._buf[aid] = d
        d.appendleft(pos.copy())

    def evict(self, alive: set[int]) -> None:
        for aid in list(self._buf.keys()):
            if aid not in alive:
                del self._buf[aid]


def build_robot_node(agent: BaseAgent, goal: Pose2D, theta: float) -> np.ndarray:
    # (px, py, radius, gx, gy, v_pref, theta) per upstream get_full_state_list_noV.
    return np.array(
        [
            agent.state.pose.x,
            agent.state.pose.y,
            agent.params.agent_radius,
            goal.x,
            goal.y,
            agent.params.desired_velocity,
            theta,
        ],
        dtype=np.float32,
    )


def build_temporal_edge(agent: BaseAgent) -> np.ndarray:
    vx, vy = agent.state.velocity
    return np.array([vx, vy], dtype=np.float32)


def build_spatial_edges(agent: BaseAgent, max_humans: int) -> np.ndarray:
    # Vectors pointing from robot position to each visible human's position. Pad
    # with the upstream sentinel of (15, 15) — far enough that the SRNN attention
    # weights it negligibly compared to nearby visible humans.
    out = np.full((max_humans, 2), 15.0, dtype=np.float32)
    if agent.belief is None:
        return out
    rx = agent.state.pose.x
    ry = agent.state.pose.y
    self_id = agent.state.agent_id
    pairs: list[tuple[float, float, float]] = []
    for ob in agent.belief.observed_agents:
        if ob.agent_id == self_id:
            continue
        dx = ob.pose.x - rx
        dy = ob.pose.y - ry
        pairs.append((dx * dx + dy * dy, dx, dy))
    pairs.sort(key=lambda t: t[0])
    for i, (_, dx, dy) in enumerate(pairs[:max_humans]):
        out[i, 0] = dx
        out[i, 1] = dy
    return out
