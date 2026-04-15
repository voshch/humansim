from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from . import Perception

if TYPE_CHECKING:
    from arena_humansim.agents import BaseAgent
    from arena_humansim.pool import AgentPool
    from arena_humansim.utils.types import AgentState, BeliefState, WorldState


class DefaultPerception(Perception):
    supports_pool: bool = True

    def __init__(self) -> None:
        self._shared_tree: cKDTree | None = None
        self._shared_ids: list[int] | None = None
        self._shared_positions: list[tuple[float, float]] | None = None

    def prepare_tick(self, all_agents: dict[int, AgentState]) -> None:
        ids: list[int] = []
        positions: list[tuple[float, float]] = []
        for aid, ag in all_agents.items():
            ids.append(aid)
            positions.append((ag.pose.x, ag.pose.y))
        self._shared_ids = ids
        self._shared_positions = positions
        if len(positions) > 1:
            self._shared_tree = cKDTree(np.array(positions, dtype=np.float64))
        else:
            self._shared_tree = None

    @property
    def shared_tree(self) -> cKDTree | None:
        return self._shared_tree

    @property
    def shared_ids(self) -> list[int] | None:
        return self._shared_ids

    @property
    def shared_positions(self) -> list[tuple[float, float]] | None:
        return self._shared_positions

    _SMALL_N_THRESHOLD = 64

    def compute_pool(self, pool: AgentPool) -> None:
        n = pool.n
        if n <= 1:
            pool.set_neighbor_csr(
                np.zeros(n + 1, dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )
            return

        positions = pool.pos[:n]

        if n <= self._SMALL_N_THRESHOLD:
            self._compute_pool_dense(pool, positions, n)
        else:
            self._compute_pool_kdtree(pool, positions, n)

    def _compute_pool_dense(self, pool: AgentPool, positions: np.ndarray, n: int) -> None:
        diff = positions[None, :, :] - positions[:, None, :]
        dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
        np.fill_diagonal(dists, np.inf)

        vision_range = pool.vision_range[:n]
        mask = dists <= vision_range[:, None]

        vision_fov = pool.vision_fov[:n]
        if not np.all(vision_fov >= 360.0):
            needs_fov = vision_fov < 360.0
            if np.any(needs_fov):
                heading = pool.theta[:n]
                bearing = np.arctan2(diff[:, :, 1], diff[:, :, 0])
                angle_diff = bearing - heading[:, None]
                angle_diff = np.abs(np.arctan2(np.sin(angle_diff), np.cos(angle_diff)))
                half_fov = np.radians(vision_fov * 0.5)
                fov_mask = angle_diff <= half_fov[:, None]
                fov_mask[~needs_fov, :] = True
                mask &= fov_mask

        row, col = np.where(mask)
        if len(row) == 0:
            pool.set_neighbor_csr(
                np.zeros(n + 1, dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )
            return

        counts = mask.sum(axis=1)
        indptr = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(counts, out=indptr[1:])
        pool.set_neighbor_csr(indptr, col.astype(np.int32))

    def _compute_pool_kdtree(self, pool: AgentPool, positions: np.ndarray, n: int) -> None:
        tree = cKDTree(positions)
        self._shared_tree = tree

        max_range = float(pool.vision_range[:n].max())
        sdm = tree.sparse_distance_matrix(tree, max_range, output_type="coo_matrix")

        row = sdm.row
        col = sdm.col
        dist = sdm.data

        mask = row != col
        row = row[mask]
        col = col[mask]
        dist = dist[mask]

        if len(row) == 0:
            pool.set_neighbor_csr(
                np.zeros(n + 1, dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )
            return

        range_ok = dist <= pool.vision_range[row]
        row = row[range_ok]
        col = col[range_ok]

        obs_fov = pool.vision_fov[row]
        omni = obs_fov >= 360.0

        needs_fov = ~omni
        if np.any(needs_fov):
            r_fov = row[needs_fov]
            c_fov = col[needs_fov]

            dx = positions[c_fov, 0] - positions[r_fov, 0]
            dy = positions[c_fov, 1] - positions[r_fov, 1]
            bearing = np.arctan2(dy, dx)
            heading = pool.theta[r_fov]

            angle_diff = np.abs(np.arctan2(np.sin(bearing - heading), np.cos(bearing - heading)))
            half_fov = np.radians(obs_fov[needs_fov] * 0.5)
            fov_ok_sub = angle_diff <= half_fov

            fov_ok = np.ones(len(row), dtype=np.bool_)
            fov_ok[needs_fov] = fov_ok_sub

            row = row[fov_ok]
            col = col[fov_ok]

        counts = np.bincount(row, minlength=n)
        indptr = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(counts, out=indptr[1:])

        order = np.argsort(row, kind="stable")
        indices = col[order].astype(np.int32)

        pool.set_neighbor_csr(indptr, indices)

    def compute(
        self,
        agent: BaseAgent,
        all_agents: dict[int, AgentState],
        world_state: WorldState,
        belief: BeliefState,
    ) -> BeliefState:
        agent_id = agent.state.agent_id
        vision_range: float = agent.params.perception.vision_range
        vision_fov: float = agent.params.perception.vision_fov

        if agent_id not in all_agents:
            return belief

        if self._shared_tree is None or self._shared_ids is None or self._shared_positions is None:
            return belief

        observer = all_agents[agent_id]
        ox, oy = observer.pose.x, observer.pose.y

        if len(all_agents) <= 1:
            return belief

        omnidirectional = vision_fov >= 360.0
        if not omnidirectional:
            half_fov = np.radians(vision_fov * 0.5)
            heading = observer.pose.theta

        neighbor_indices = self._shared_tree.query_ball_point([ox, oy], vision_range)

        for idx in neighbor_indices:
            nid = self._shared_ids[idx]
            if nid == agent_id:
                continue
            if not omnidirectional:
                nx, ny = self._shared_positions[idx]
                bearing = np.arctan2(ny - oy, nx - ox)
                angle_diff = abs(_wrap_angle(bearing - heading))
                if angle_diff > half_fov:
                    continue
            belief.observed_agents.append(all_agents[nid])

        return belief


def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
