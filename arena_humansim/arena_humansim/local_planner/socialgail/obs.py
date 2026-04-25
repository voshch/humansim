from __future__ import annotations

import numpy as np
import torch

from .history import HistoryBuffer
from .net import GraphData


def _angle_by_x_deg(v: np.ndarray) -> float:
    x, y = float(v[0]), float(v[1])
    theta = np.arctan2(y, x)
    if theta < 0:
        theta += 2.0 * np.pi
    return theta * 180.0 / np.pi


def rotate_to_goal_frame(goal_v: np.ndarray) -> np.ndarray:
    """Matrix R s.t. ``np.dot(v, R)`` rotates v clockwise by angle(goal_v, +x)."""
    ang = _angle_by_x_deg(goal_v) * np.pi / 180.0
    c, s = np.cos(ang), np.sin(ang)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rotate_from_goal_frame(goal_v: np.ndarray) -> np.ndarray:
    """Inverse of rotate_to_goal_frame."""
    ang = _angle_by_x_deg(goal_v) * np.pi / 180.0
    c, s = np.cos(ang), np.sin(ang)
    return np.array([[c, s], [-s, c]], dtype=np.float64)


def _counter_clockwise_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    x1, y1 = float(v1[0]), float(v1[1])
    x2, y2 = float(v2[0]), float(v2[1])
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    theta = np.arctan2(det, dot)
    if theta < 0:
        theta += 2.0 * np.pi
    return theta * 180.0 / np.pi


def build_obs_for_agent(
    aid: int,
    pos: np.ndarray,
    goal: np.ndarray,
    R: np.ndarray,
    neighbor_aids: np.ndarray,
    neighbor_positions: np.ndarray,
    history: HistoryBuffer,
    decision_dt: float,
    radius: float = 6.0,
    past_len: int = 5,
    padd_to_number: int = 60,
    feature_len: int = 5,
) -> GraphData:
    """Build a single-agent SocialGAIL graph observation.

    Mirrors gail_airl_ppo.crowd_env.gym_env._get_ar_relative_graph_observation.
    All positional features are rotated into the goal-aligned frame using ``R``,
    which the caller pins per (agent, current goal) — upstream pins it at
    episode start and never recomputes; recomputing per tick puts the policy
    OOD because it never sees "goal behind me" cases that training relied on.
    The current goal vector itself is still recomputed each tick (goal feature
    shrinks as the agent approaches), only the rotation is pinned. The
    front_flag dot-product is computed in the world frame using the agent's
    own most recent decision-tick velocity.
    """

    goal_v_world = np.asarray(goal, dtype=np.float64) - np.asarray(pos, dtype=np.float64)

    own_last = history.peek(aid, 1)
    if own_last is None:
        last_xy_rot = np.zeros(2, dtype=np.float64)
        last_v_world = goal_v_world
    else:
        delta = own_last - pos
        last_xy_rot = np.dot(delta, R)
        last_v_world = (pos - own_last) / max(decision_dt, 1e-6)

    X: list[list[float]] = [[float(last_xy_rot[0]), float(last_xy_rot[1]), 0.0, 0.0, 0.0]]
    cluster: list[int] = [0]
    edge_src: list[int] = []
    edge_dst: list[int] = []
    sum_ped = 1

    for j in range(len(neighbor_aids)):
        oid = int(neighbor_aids[j])
        if oid == aid:
            continue
        opos_now = neighbor_positions[j]
        if float(np.hypot(opos_now[0] - pos[0], opos_now[1] - pos[1])) >= radius:
            continue

        cur_end_world = np.asarray(opos_now, dtype=np.float64).copy()
        len_nodes = 0
        for k in range(1, past_len + 1):
            start_world = history.peek(oid, k)
            if start_world is None:
                break

            start_rel = start_world - pos
            end_rel = cur_end_world - pos
            start_rot = np.dot(start_rel, R)
            end_rot = np.dot(end_rel, R)

            ang_world = _counter_clockwise_angle_deg(last_v_world, end_rel)
            front_flag = 0.0 if 90.0 <= ang_world <= 270.0 else 1.0

            X.append([float(start_rot[0]), float(start_rot[1]), float(end_rot[0]), float(end_rot[1]), front_flag])
            cluster.append(sum_ped)

            if len_nodes > 0:
                node_idx = len(X) - 1
                edge_src.append(node_idx)
                edge_dst.append(node_idx - 1)

            cur_end_world = np.asarray(start_world, dtype=np.float64).copy()
            len_nodes += 1

        if len_nodes > 0:
            sum_ped += 1

    X_arr = np.array(X, dtype=np.float32)
    cluster_arr = np.array(cluster, dtype=np.int64)

    valid_len = int(cluster_arr.max()) + 1
    pad = padd_to_number - valid_len
    if pad < 0:
        raise ValueError(f"valid_len {valid_len} exceeds padd_to_number {padd_to_number}")
    if pad > 0:
        X_arr = np.vstack([X_arr, np.zeros((pad, feature_len), dtype=X_arr.dtype)])
        cluster_arr = np.hstack([cluster_arr, np.arange(valid_len, padd_to_number, dtype=np.int64)])

    goal_rot = np.dot(goal_v_world, R).astype(np.float32)

    edge_index_t = torch.tensor([edge_src, edge_dst], dtype=torch.int64) if edge_src else torch.zeros((2, 0), dtype=torch.int64)

    return GraphData(
        x=torch.from_numpy(X_arr),
        cluster=torch.from_numpy(cluster_arr),
        edge_index=edge_index_t,
        valid_len=torch.tensor([valid_len], dtype=torch.int64),
        time_step_len=torch.tensor([padd_to_number], dtype=torch.int64),
        goal=torch.from_numpy(goal_rot),
    )
