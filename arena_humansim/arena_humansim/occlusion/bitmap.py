from __future__ import annotations

import math

import numpy as np

from arena_humansim.utils.types import Segments

from . import Occluder

_RESOLUTION = 0.05


class BitmapOccluder(Occluder):
    def __init__(self) -> None:
        self._grid: np.ndarray | None = None
        self._origin_x: float = 0.0
        self._origin_y: float = 0.0

    def set_walls(self, segments: Segments) -> None:
        if not segments:
            self._grid = None
            return

        xs = [x for (x, _), (ex, _) in segments for x in (x, ex)]
        ys = [y for (_, y), (_, ey) in segments for y in (y, ey)]
        xmin = min(xs) - 1.0
        ymin = min(ys) - 1.0
        xmax = max(xs) + 1.0
        ymax = max(ys) + 1.0

        self._origin_x = xmin
        self._origin_y = ymin

        width = math.ceil((xmax - xmin) / _RESOLUTION)
        height = math.ceil((ymax - ymin) / _RESOLUTION)
        grid = np.zeros((height, width), dtype=np.bool_)

        for (sx, sy), (ex, ey) in segments:
            c0 = int(math.floor((sx - xmin) / _RESOLUTION))
            r0 = int(math.floor((sy - ymin) / _RESOLUTION))
            c1 = int(math.floor((ex - xmin) / _RESOLUTION))
            r1 = int(math.floor((ey - ymin) / _RESOLUTION))

            dc = abs(c1 - c0)
            dr = abs(r1 - r0)
            sc = 1 if c1 > c0 else -1
            sr = 1 if r1 > r0 else -1
            err = dc - dr

            c, r = c0, r0
            while True:
                cc = max(0, min(c, width - 1))
                rc = max(0, min(r, height - 1))
                grid[rc, cc] = True
                if c == c1 and r == r1:
                    break
                e2 = 2 * err
                if e2 > -dr:
                    err -= dr
                    c += sc
                if e2 < dc:
                    err += dc
                    r += sr

        # 4-neighbor dilation closes the off-diagonal gaps in 8-connected lines that thin walls would otherwise leak rays through.
        dilated = grid.copy()
        dilated[1:, :] |= grid[:-1, :]
        dilated[:-1, :] |= grid[1:, :]
        dilated[:, 1:] |= grid[:, :-1]
        dilated[:, :-1] |= grid[:, 1:]
        self._grid = dilated

    def clear(self, p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
        k = len(p_a)
        if self._grid is None:
            return np.ones(k, dtype=np.bool_)

        grid = self._grid
        height, width = grid.shape
        ox = self._origin_x
        oy = self._origin_y

        dx = p_b[:, 0] - p_a[:, 0]
        dy = p_b[:, 1] - p_a[:, 1]
        lengths = np.hypot(dx, dy)

        result = np.ones(k, dtype=np.bool_)
        nonzero = lengths > 1e-9
        if not np.any(nonzero):
            return result

        # samples per ray: enough that step ≤ resolution/sqrt(2) for the longest ray;
        # all rays use the same count so every ray covers its full [0, 1] range
        step_size = _RESOLUTION / math.sqrt(2.0)
        max_samples = int(np.ceil(lengths[nonzero].max() / step_size))

        ts = np.linspace(0.0, 1.0, max_samples, dtype=np.float64)
        px = p_a[:, 0:1] + ts[np.newaxis, :] * dx[:, np.newaxis]
        py = p_a[:, 1:2] + ts[np.newaxis, :] * dy[:, np.newaxis]

        col = ((px - ox) / _RESOLUTION).astype(np.int32)
        row = ((py - oy) / _RESOLUTION).astype(np.int32)

        in_bounds = (col >= 0) & (col < width) & (row >= 0) & (row < height)
        col_c = np.clip(col, 0, width - 1)
        row_c = np.clip(row, 0, height - 1)

        occupied = grid[row_c, col_c] & in_bounds

        blocked = occupied.any(axis=1)
        result[nonzero] = ~blocked[nonzero]
        return result
