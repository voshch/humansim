from __future__ import annotations

import numpy as np

from arena_humansim.collision import CollisionResolver
from arena_humansim.pool import AgentPool


class WallProjectionResolver(CollisionResolver):
    def __init__(self, margin: float = 0.01):
        self._margin = margin
        self._wall_segments_np: np.ndarray = np.empty((0, 2, 2), dtype=np.float64)

    def set_walls(self, segments):
        if segments:
            self._wall_segments_np = np.array(segments, dtype=np.float64).reshape(-1, 2, 2)
        else:
            self._wall_segments_np = np.empty((0, 2, 2), dtype=np.float64)

    def resolve(self, pool: AgentPool) -> None:
        n = pool.n
        W = self._wall_segments_np.shape[0]
        if n == 0 or W == 0:
            return

        pos = pool.pos[:n]
        radii = pool.agent_radius[:n]
        vel = pool.vel[:n]

        A = self._wall_segments_np[:, 0, :]  # (W, 2)
        AB = self._wall_segments_np[:, 1, :] - A  # (W, 2)
        ab_sq = np.einsum("ij,ij->i", AB, AB)  # (W,)

        for _ in range(3):
            AP = pos[:, np.newaxis, :] - A[np.newaxis, :, :]  # (N, W, 2)
            t = np.einsum("nwj,wj->nw", AP, AB) / np.maximum(ab_sq[np.newaxis, :], 1e-12)
            t = np.clip(t, 0.0, 1.0)  # (N, W)
            closest = A[np.newaxis, :, :] + t[:, :, np.newaxis] * AB[np.newaxis, :, :]  # (N, W, 2)
            diff = pos[:, np.newaxis, :] - closest  # (N, W, 2)
            dist = np.linalg.norm(diff, axis=2)  # (N, W)

            threshold = radii[:, np.newaxis] + self._margin
            penetrating = dist < threshold  # (N, W)

            if not penetrating.any():
                break

            safe_dist = np.where(dist > 1e-9, dist, 1e-9)
            normal = diff / safe_dist[:, :, np.newaxis]  # (N, W, 2)
            overlap = (threshold - dist) * penetrating  # (N, W)
            correction = (normal * overlap[:, :, np.newaxis]).sum(axis=1)  # (N, 2)
            pos += correction

            for i in range(n):
                if not penetrating[i].any():
                    continue
                wall_normals = normal[i, penetrating[i]]  # (k, 2)
                v = vel[i]
                for wn in wall_normals:
                    proj = v[0] * wn[0] + v[1] * wn[1]
                    if proj < 0:
                        vel[i] -= proj * wn
