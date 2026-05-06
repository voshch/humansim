from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from arena_humansim.core.agents import BaseAgent
from arena_humansim.utils.types import AgentState, Pose2D

HOST_AVG = np.array([0.0, 0.0, 1.0, 0.5], dtype=np.float32)
HOST_STD = np.array([5.0, 3.14, 1.0, 1.0], dtype=np.float32)
OTHER_AVG = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0], dtype=np.float32)
OTHER_STD = np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0], dtype=np.float32)
SENSING_HORIZON = 8.0


def build_observation(
    agent: BaseAgent,
    goal: Pose2D,
    max_other_agents: int,
    default_other_radius: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, int]:
    px = agent.state.pose.x
    py = agent.state.pose.y
    vx, vy = agent.state.velocity
    radius = agent.params.agent_radius
    pref_speed = agent.params.desired_velocity

    gx = goal.x - px
    gy = goal.y - py
    dist_to_goal = math.hypot(gx, gy)
    if dist_to_goal > 1e-8:
        ref_prll = (gx / dist_to_goal, gy / dist_to_goal)
    else:
        ref_prll = (1.0, 0.0)
    ref_orth = (-ref_prll[1], ref_prll[0])
    heading_global = math.atan2(vy, vx) if (vx * vx + vy * vy) > 1e-9 else math.atan2(ref_prll[1], ref_prll[0])
    ref_prll_angle = math.atan2(ref_prll[1], ref_prll[0])
    heading_ego = _wrap(heading_global - ref_prll_angle)

    host = np.array([dist_to_goal, heading_ego, pref_speed, radius], dtype=np.float32)
    host_norm = (host - HOST_AVG) / HOST_STD

    observed: Sequence[AgentState] = agent.belief.observed_agents if agent.belief is not None else ()
    cands: list[tuple[float, np.ndarray]] = []
    for ob in observed:
        if ob.agent_id == agent.state.agent_id:
            continue
        rel_x = ob.pose.x - px
        rel_y = ob.pose.y - py
        dist_centers = math.hypot(rel_x, rel_y)
        if dist_centers > SENSING_HORIZON:
            continue
        other_r = _other_radius(ob, default_other_radius)
        p_par = rel_x * ref_prll[0] + rel_y * ref_prll[1]
        p_orth = rel_x * ref_orth[0] + rel_y * ref_orth[1]
        ovx, ovy = ob.velocity
        v_par = ovx * ref_prll[0] + ovy * ref_prll[1]
        v_orth = ovx * ref_orth[0] + ovy * ref_orth[1]
        dist_2_other = dist_centers - radius - other_r
        combined_radius = radius + other_r
        feat = np.array(
            [p_par, p_orth, v_par, v_orth, other_r, combined_radius, dist_2_other],
            dtype=np.float32,
        )
        cands.append((dist_2_other, feat))

    cands.sort(key=lambda kv: kv[0])
    cands = cands[:max_other_agents]

    other_seq = np.zeros((max_other_agents, 7), dtype=np.float32)
    for i, (_, feat) in enumerate(cands):
        other_seq[i] = (feat - OTHER_AVG) / OTHER_STD
    n_others = len(cands)
    return host_norm, other_seq, n_others


def _other_radius(ob: AgentState, default: float) -> float:
    return default


def _wrap(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle
