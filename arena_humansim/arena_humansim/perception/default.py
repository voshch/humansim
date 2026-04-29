from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from . import Perception

if TYPE_CHECKING:
    from arena_humansim.core.agents import BaseAgent
    from arena_humansim.core.pool import AgentPool
    from arena_humansim.occlusion import Occluder
    from arena_humansim.utils.types import AgentState, BeliefState, WorldState


class DefaultPerception(Perception):
    supports_pool: bool = True

    def __init__(self, occluder: Occluder | None = None) -> None:
        self._occluder = occluder
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

        proximity_sense = pool.proximity_sense[:n]
        if np.any(proximity_sense > 0.0):
            prox_mask = dists <= proximity_sense[:, None]
            mask |= prox_mask

        if self._occluder is not None:
            row, col = np.where(mask)
            if len(row) > 0:
                needs_los = pool.vision_occlusion[:n][row]
                if np.any(needs_los):
                    p_a = positions[row]
                    p_b = positions[col]
                    los = self._occluder.clear(p_a, p_b)
                    # agents with vision_occlusion=False always pass; others require clear LOS
                    los_ok = los | ~needs_los
                    flat = np.ravel_multi_index((row, col), (n, n))
                    flat_blocked = flat[~los_ok]
                    if len(flat_blocked) > 0:
                        mask_flat = mask.ravel()
                        mask_flat[flat_blocked] = False
                        mask = mask_flat.reshape(n, n)

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

        max_range = float(max(pool.vision_range[:n].max(), pool.proximity_sense[:n].max()))
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

        prox_ok = dist <= pool.proximity_sense[row]
        range_ok = dist <= pool.vision_range[row]

        obs_fov = pool.vision_fov[row]
        omni = obs_fov >= 360.0

        fov_ok = np.ones(len(row), dtype=np.bool_)
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
            fov_ok[needs_fov] = angle_diff <= half_fov

        keep = (range_ok & fov_ok) | prox_ok

        if self._occluder is not None and np.any(keep):
            needs_los = pool.vision_occlusion[:n][row[keep]]
            if np.any(needs_los):
                p_a = positions[row[keep]]
                p_b = positions[col[keep]]
                los = self._occluder.clear(p_a, p_b)
                los_ok = los | ~needs_los
                keep_indices = np.where(keep)[0]
                keep[keep_indices[~los_ok]] = False

        row = row[keep]
        col = col[keep]

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
        proximity_sense: float = agent.params.perception.proximity_sense
        vision_occlusion: bool = agent.params.perception.vision_occlusion

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

        query_radius = max(vision_range, proximity_sense)
        neighbor_indices = self._shared_tree.query_ball_point([ox, oy], query_radius)

        candidates: list[int] = []
        for idx in neighbor_indices:
            nid = self._shared_ids[idx]
            if nid == agent_id:
                continue
            nx, ny = self._shared_positions[idx]
            d = float(np.hypot(nx - ox, ny - oy))
            within_prox = d <= proximity_sense
            if not within_prox:
                if d > vision_range:
                    continue
                if not omnidirectional:
                    bearing = np.arctan2(ny - oy, nx - ox)
                    angle_diff = abs(_wrap_angle(bearing - heading))
                    if angle_diff > half_fov:
                        continue
            candidates.append(idx)

        if self._occluder is not None and vision_occlusion and candidates:
            p_a = np.array([[ox, oy]] * len(candidates), dtype=np.float64)
            p_b = np.array([self._shared_positions[idx] for idx in candidates], dtype=np.float64)
            los = self._occluder.clear(p_a, p_b)
            candidates = [idx for idx, clear in zip(candidates, los, strict=True) if clear]

        for idx in candidates:
            nid = self._shared_ids[idx]
            belief.observed_agents.append(all_agents[nid])

        return belief


def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
