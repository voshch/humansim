from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from arena_humansim.utils.types import AgentState

# Upstream constants from drl_vo_nav (drl_vo/src/cnn_data_pub.py + drl_vo_inference.py).
# Ped grid: 80x80 over x in [0, 20] m forward, y in [-10, 10] m lateral, bin 0.25 m, 2 channels (vx, vy).
# Scan: 720 beams sampled (upstream slices [180:900] from a 1080-beam 270-deg Hokuyo); we synthesize
#   720 beams over `lidar_fov_deg` (default 360 deg). Stack of 10 frames, pooled to 20x80, repmat to
#   1x80x80, flat 6400.
# Goal: (gx, gy) in robot frame.
# Total flat obs: 12800 + 6400 + 2 = 19202.
PED_GRID_CHANNELS = 2
PED_GRID_SIZE = 80
PED_X_MIN = 0.0
PED_X_MAX = 20.0
PED_Y_HALF = 10.0
PED_BIN = 0.25

NUM_BEAMS = 720
NUM_TP = 10
SCAN_POOL_BIN = 9  # 720 / 80
SCAN_REPMAT = 4

PED_FLAT = PED_GRID_CHANNELS * PED_GRID_SIZE * PED_GRID_SIZE  # 12800
SCAN_FLAT = 2 * NUM_TP * PED_GRID_SIZE * SCAN_REPMAT  # 6400
GOAL_FLAT = 2
OBS_FLAT = PED_FLAT + SCAN_FLAT + GOAL_FLAT  # 19202

# Upstream MaxAbsScaler bounds.
PED_VEL_MIN = -2.0
PED_VEL_MAX = 2.0
SCAN_MIN = 0.0
SCAN_MAX = 30.0
GOAL_MIN = -2.0
GOAL_MAX = 2.0


class ScanHistory:
    """Per-agent ten-frame ring buffer of 720-beam range scans, repeated to fill on first read."""

    def __init__(self, num_tp: int = NUM_TP, num_beams: int = NUM_BEAMS):
        self._num_tp = num_tp
        self._num_beams = num_beams
        self._buf: dict[int, deque[np.ndarray]] = {}

    def push(self, agent_id: int, scan: np.ndarray) -> np.ndarray:
        buf = self._buf.get(agent_id)
        if buf is None:
            buf = deque(maxlen=self._num_tp)
            self._buf[agent_id] = buf
        buf.append(scan.astype(np.float32, copy=True))
        while len(buf) < self._num_tp:
            buf.appendleft(scan.astype(np.float32, copy=True))
        return np.stack(list(buf), axis=0)

    def evict(self, keep_ids: set[int]) -> None:
        for aid in list(self._buf.keys()):
            if aid not in keep_ids:
                del self._buf[aid]


def synthesize_lidar(
    px: float,
    py: float,
    yaw: float,
    walls_p1: np.ndarray,
    walls_d: np.ndarray,
    neighbors_xy: np.ndarray,
    neighbors_radius: np.ndarray,
    num_beams: int = NUM_BEAMS,
    max_range: float = 6.0,
    fov_rad: float = 2.0 * np.pi,
) -> np.ndarray:
    """720-beam scan (clockwise from yaw - fov/2 to yaw + fov/2). Returns ranges in [0, max_range]."""
    if num_beams <= 0:
        return np.empty(0, dtype=np.float32)

    half = 0.5 * fov_rad
    if fov_rad >= 2.0 * np.pi - 1e-9:
        angles = yaw + (np.arange(num_beams, dtype=np.float64) / num_beams) * 2.0 * np.pi - np.pi
    else:
        angles = yaw + np.linspace(-half, half, num_beams, dtype=np.float64)

    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    out = np.full(num_beams, max_range, dtype=np.float64)

    n_walls = walls_p1.shape[0]
    if n_walls > 0:
        # Ray-segment intersection. Ray: p + t*d, segment: q + u*s, u in [0, 1], t in [0, max_range].
        ox = px - walls_p1[:, 0]
        oy = py - walls_p1[:, 1]
        sx = walls_d[:, 0]
        sy = walls_d[:, 1]

        # cross of (d, s) per (beam, wall): d.x * s.y - d.y * s.x  -> (B, W)
        denom = cos_a[:, None] * sy[None, :] - sin_a[:, None] * sx[None, :]
        valid = np.abs(denom) > 1e-9
        denom_safe = np.where(valid, denom, 1.0)

        # t = (-ox * s.y + oy * s.x) / -denom  =>  t = (ox * s.y - oy * s.x) / denom (per beam-wall)
        t_num = (-ox[None, :]) * sy[None, :] - (-oy[None, :]) * sx[None, :]
        t = t_num / denom_safe
        u_num = (-ox[None, :]) * sin_a[:, None] - (-oy[None, :]) * cos_a[:, None]
        u = u_num / denom_safe

        hit = valid & (t >= 0.0) & (t <= max_range) & (u >= 0.0) & (u <= 1.0)
        t_masked = np.where(hit, t, max_range)
        wall_min = t_masked.min(axis=1)
        out = np.minimum(out, wall_min)

    n_ped = neighbors_xy.shape[0]
    if n_ped > 0:
        # Ray-circle intersection. Center c, radius r. Solve |p + t*d - c|^2 = r^2.
        # For unit d: t^2 - 2 (c - p).d t + (|c-p|^2 - r^2) = 0.
        rel = neighbors_xy - np.array([px, py], dtype=np.float64)
        proj = cos_a[:, None] * rel[None, :, 0] + sin_a[:, None] * rel[None, :, 1]
        rel_sq = (rel[:, 0] ** 2 + rel[:, 1] ** 2)[None, :]
        r_sq = (neighbors_radius**2)[None, :]
        disc = proj**2 - (rel_sq - r_sq)
        valid = disc >= 0.0
        sqrt_disc = np.sqrt(np.where(valid, disc, 0.0))
        t_near = proj - sqrt_disc
        hit = valid & (t_near >= 0.0) & (t_near <= max_range)
        t_masked = np.where(hit, t_near, max_range)
        ped_min = t_masked.min(axis=1)
        out = np.minimum(out, ped_min)

    return out.astype(np.float32)


def _pool_scan_stack(scan_stack: np.ndarray) -> np.ndarray:
    """Match upstream pooling: per timestep produce (min, mean) of every 9 beams -> (20, 80) -> repmat 4."""
    num_tp, num_beams = scan_stack.shape
    cells = num_beams // SCAN_POOL_BIN
    reshaped = scan_stack.reshape(num_tp, cells, SCAN_POOL_BIN)
    mins = reshaped.min(axis=2)
    means = reshaped.mean(axis=2)
    avg = np.empty((2 * num_tp, cells), dtype=np.float32)
    avg[0::2] = mins
    avg[1::2] = means
    flat = avg.reshape(-1)
    repeated = np.tile(flat, SCAN_REPMAT)
    return repeated.astype(np.float32)


def build_ped_grid(
    px: float,
    py: float,
    yaw: float,
    neighbors_xy: np.ndarray,
    neighbors_vxvy: np.ndarray,
) -> np.ndarray:
    """2 x 80 x 80 cartesian velocity map. Coords transformed to robot frame."""
    grid = np.zeros((PED_GRID_CHANNELS, PED_GRID_SIZE, PED_GRID_SIZE), dtype=np.float32)
    if neighbors_xy.shape[0] == 0:
        return grid

    cy = np.cos(-yaw)
    sy = np.sin(-yaw)
    rel = neighbors_xy - np.array([px, py], dtype=np.float64)
    rx = rel[:, 0] * cy - rel[:, 1] * sy
    ry = rel[:, 0] * sy + rel[:, 1] * cy
    vx_world = neighbors_vxvy[:, 0]
    vy_world = neighbors_vxvy[:, 1]
    vx = vx_world * cy - vy_world * sy
    vy = vx_world * sy + vy_world * cy

    in_box = (rx >= PED_X_MIN) & (rx <= PED_X_MAX) & (np.abs(ry) <= PED_Y_HALF)
    if not np.any(in_box):
        return grid

    rx_in = rx[in_box]
    ry_in = ry[in_box]
    vx_in = vx[in_box]
    vy_in = vy[in_box]

    rows = np.floor(rx_in / PED_BIN).astype(np.int64)
    cols = np.floor(-(ry_in - PED_Y_HALF) / PED_BIN).astype(np.int64)
    rows = np.clip(rows, 0, PED_GRID_SIZE - 1)
    cols = np.clip(cols, 0, PED_GRID_SIZE - 1)

    # Last writer wins, mirroring upstream.
    for i in range(rx_in.shape[0]):
        grid[0, rows[i], cols[i]] = vx_in[i]
        grid[1, rows[i], cols[i]] = vy_in[i]

    return grid


def _scale(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return 2.0 * (arr - lo) / (hi - lo) - 1.0


def assemble_observation(
    px: float,
    py: float,
    yaw: float,
    goal_xy: tuple[float, float],
    walls_p1: np.ndarray,
    walls_d: np.ndarray,
    neighbors: Sequence[AgentState],
    neighbor_radius: float,
    scan_history: ScanHistory,
    agent_id: int,
    num_beams: int = NUM_BEAMS,
    max_range: float = 6.0,
    fov_rad: float = 2.0 * np.pi,
) -> tuple[np.ndarray, float]:
    """Returns (flat 19202 obs, min forward-cone scan distance for stop-gate)."""
    if neighbors:
        nbr_xy = np.empty((len(neighbors), 2), dtype=np.float64)
        nbr_vxvy = np.empty((len(neighbors), 2), dtype=np.float64)
        for i, ag in enumerate(neighbors):
            nbr_xy[i, 0] = ag.pose.x
            nbr_xy[i, 1] = ag.pose.y
            nbr_vxvy[i, 0] = ag.velocity[0]
            nbr_vxvy[i, 1] = ag.velocity[1]
    else:
        nbr_xy = np.zeros((0, 2), dtype=np.float64)
        nbr_vxvy = np.zeros((0, 2), dtype=np.float64)
    nbr_radius = np.full(nbr_xy.shape[0], neighbor_radius, dtype=np.float64)

    scan = synthesize_lidar(
        px=px,
        py=py,
        yaw=yaw,
        walls_p1=walls_p1,
        walls_d=walls_d,
        neighbors_xy=nbr_xy,
        neighbors_radius=nbr_radius,
        num_beams=num_beams,
        max_range=max_range,
        fov_rad=fov_rad,
    )

    scan_stack = scan_history.push(agent_id, scan)
    scan_pooled = _pool_scan_stack(scan_stack)

    # Forward-cone min for the upstream stop-gate. Upstream pulls scan[-540:-180] from a 1080
    # buffer of 720 beams * 10 frames; for our 720-beam stack, take a centered 360-beam slice
    # of the most recent frame.
    last = scan_stack[-1]
    cone_lo = num_beams // 4
    cone_hi = (3 * num_beams) // 4
    cone = last[cone_lo:cone_hi]
    cone = cone[cone > 0.0]
    min_scan = float(cone.min()) if cone.size else float(max_range)

    grid = build_ped_grid(px, py, yaw, nbr_xy, nbr_vxvy)

    cy = np.cos(-yaw)
    sy = np.sin(-yaw)
    gx_w = goal_xy[0] - px
    gy_w = goal_xy[1] - py
    gx = gx_w * cy - gy_w * sy
    gy = gx_w * sy + gy_w * cy
    goal = np.array([gx, gy], dtype=np.float32)

    ped_flat = _scale(grid.reshape(-1).astype(np.float32), PED_VEL_MIN, PED_VEL_MAX)
    scan_scaled = _scale(scan_pooled, SCAN_MIN, SCAN_MAX)
    goal_scaled = _scale(goal, GOAL_MIN, GOAL_MAX)

    obs = np.concatenate([ped_flat, scan_scaled, goal_scaled], axis=0).astype(np.float32)
    return obs, min_scan
