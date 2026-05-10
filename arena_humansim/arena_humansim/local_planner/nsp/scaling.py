"""Meters<->pixels conversion and per-tick batch assembly for the NSP planner.

NSP was trained on Stanford Drone (pixel space, dt=0.4s). We treat the sim's metric world
as a virtual SDD scene by applying a fixed meters_per_pixel scale.
"""

from __future__ import annotations

import numpy as np


def meters_to_pixels(xy_m: np.ndarray, meters_per_pixel: float) -> np.ndarray:
    return xy_m / meters_per_pixel


def pixels_to_meters(xy_px: np.ndarray, meters_per_pixel: float) -> np.ndarray:
    return xy_px * meters_per_pixel


def velocity_from_history(history_px: np.ndarray, nsp_dt: float) -> np.ndarray:
    """Forward-difference velocity at each past frame, in pixels/sec.

    history_px: [N, past_length, 2]; output: [N, past_length, 2]; vel[..., 0, :] = 0.
    """
    vel = np.zeros_like(history_px)
    if history_px.shape[1] >= 2:
        vel[:, 1:, :] = (history_px[:, 1:, :] - history_px[:, :-1, :]) / nsp_dt
    return vel


def assemble_supplement(
    own_pos_px: np.ndarray,
    neighbor_indptr: np.ndarray,
    neighbor_indices: np.ndarray,
    all_pos_px: np.ndarray,
    all_vel_px: np.ndarray,
    max_peds: int,
) -> np.ndarray:
    """Build the [N, max_peds + 1, 5] supplement tensor expected by NSP.forward_coefficient_people.

    Per-row 0..k-1 hold (x, y, vx, vy, flag) for the k visible neighbors;
    row max_peds (the last row) stores (0, k) in slots [0, 1] as a count marker.
    Truncates by closest distance when neighbor count exceeds max_peds.
    """
    n = own_pos_px.shape[0]
    supp = np.zeros((n, max_peds + 1, 5), dtype=np.float64)
    for i in range(n):
        start = int(neighbor_indptr[i])
        end = int(neighbor_indptr[i + 1])
        nbrs = neighbor_indices[start:end]
        if nbrs.size == 0:
            supp[i, -1, 1] = 0.0
            continue
        if nbrs.size > max_peds:
            d = np.linalg.norm(all_pos_px[nbrs] - own_pos_px[i], axis=1)
            keep = np.argpartition(d, max_peds)[:max_peds]
            nbrs = nbrs[keep]
        k = nbrs.size
        supp[i, :k, 0:2] = all_pos_px[nbrs]
        supp[i, :k, 2:4] = all_vel_px[nbrs]
        supp[i, :k, 4] = 1.0
        supp[i, -1, 1] = float(k)
    return supp
