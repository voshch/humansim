from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from arena_humansim.core.agents import BaseAgent
from arena_humansim.utils.types import AgentState, Pose2D


def build_full_state(agent: BaseAgent, goal: Pose2D) -> tuple[float, ...]:
    px = float(agent.state.pose.x)
    py = float(agent.state.pose.y)
    vx, vy = agent.state.velocity
    radius = float(agent.params.agent_radius)
    v_pref = float(agent.params.desired_velocity)
    theta = float(agent.state.pose.theta)
    return (px, py, float(vx), float(vy), radius, float(goal.x), float(goal.y), v_pref, theta)


def build_observable_states(
    agent: BaseAgent,
    default_radius: float,
    max_humans: int = 0,
) -> list[tuple[float, float, float, float, float]]:
    px = float(agent.state.pose.x)
    py = float(agent.state.pose.y)
    observed: Sequence[AgentState] = agent.belief.observed_agents if agent.belief is not None else ()
    cands: list[tuple[float, tuple[float, float, float, float, float]]] = []
    for ob in observed:
        if ob.agent_id == agent.state.agent_id:
            continue
        ox = float(ob.pose.x)
        oy = float(ob.pose.y)
        ovx, ovy = ob.velocity
        d = math.hypot(ox - px, oy - py)
        cands.append((d, (ox, oy, float(ovx), float(ovy), default_radius)))
    cands.sort(key=lambda kv: kv[0])
    if max_humans > 0:
        cands = cands[:max_humans]
    return [c[1] for c in cands]


def rotate(state: np.ndarray) -> np.ndarray:
    # Per upstream cadrl.rotate: transform world-frame (self_state, human_state) row to a
    # 13-dim rotated representation centered on the agent and oriented toward its goal.
    # state shape: (n_humans, 14) with columns
    #   [px, py, vx, vy, radius, gx, gy, v_pref, theta, hx, hy, hvx, hvy, h_radius]
    s = state
    px = s[:, 0]
    py = s[:, 1]
    vx = s[:, 2]
    vy = s[:, 3]
    radius = s[:, 4]
    gx = s[:, 5]
    gy = s[:, 6]
    v_pref = s[:, 7]
    hx = s[:, 9]
    hy = s[:, 10]
    hvx = s[:, 11]
    hvy = s[:, 12]
    hr = s[:, 13]

    dx = gx - px
    dy = gy - py
    rot = np.arctan2(dy, dx)
    cr = np.cos(rot)
    sr = np.sin(rot)
    dg = np.hypot(dx, dy)
    vx_r = vx * cr + vy * sr
    vy_r = vy * cr - vx * sr
    theta = np.zeros_like(v_pref)
    px1 = (hx - px) * cr + (hy - py) * sr
    py1 = (hy - py) * cr - (hx - px) * sr
    vx1 = hvx * cr + hvy * sr
    vy1 = hvy * cr - hvx * sr
    radius_sum = radius + hr
    da = np.hypot(px - hx, py - hy)
    out = np.stack([dg, v_pref, theta, radius, vx_r, vy_r, px1, py1, vx1, vy1, hr, da, radius_sum], axis=1)
    return out.astype(np.float32, copy=False)
