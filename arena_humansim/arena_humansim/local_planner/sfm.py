from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from arena_humansim.agents import BaseAgent
from arena_humansim.utils.types import Pose2D

from . import LocalPlanner

if TYPE_CHECKING:
    from arena_humansim.pool import AgentPool
    from arena_humansim.viz import MarkerPublisher

_EPS = 1e-6

_DEFAULT_WALL_REPULSION_STRENGTH = 3.0
_DEFAULT_WALL_REPULSION_RANGE = 0.1


class SFMPlanner(LocalPlanner):
    def __init__(
        self,
        wall_repulsion_strength: float = _DEFAULT_WALL_REPULSION_STRENGTH,
        wall_repulsion_range: float = _DEFAULT_WALL_REPULSION_RANGE,
    ):
        self.wall_repulsion_strength = wall_repulsion_strength
        self.wall_repulsion_range = wall_repulsion_range
        self._dt: float = 1.0
        self._wall_segments: list = []
        self._wall_segments_np: np.ndarray = np.empty((0, 2, 2), dtype=np.float64)
        self._last_forces: dict[int, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = {}
        self._last_force_arrays: tuple | None = None
        self._last_agents: Sequence[BaseAgent] | None = None

    def set_walls(self, segments: list) -> None:
        self._wall_segments = list(segments)
        if segments:
            self._wall_segments_np = np.array(segments, dtype=np.float64).reshape(-1, 2, 2)
        else:
            self._wall_segments_np = np.empty((0, 2, 2), dtype=np.float64)
        self._wall_p1 = self._wall_segments_np[:, 0, :]
        self._wall_d = self._wall_segments_np[:, 1, :] - self._wall_p1
        self._wall_len_sq = np.sum(self._wall_d ** 2, axis=1)
        self._logger.info(f"Loaded {len(segments)} wall segment(s)")

    def compute_pool(self, pool: AgentPool, store_forces: bool = False, dt: float = 1.0) -> None:
        n = pool.n
        if n == 0:
            self._last_forces = {}
            self._last_force_arrays = None
            return

        pos = pool.pos[:n]
        vel = pool.vel[:n]
        goal = pool.goal_pos[:n]
        has_goal = pool.has_goal[:n]

        desired_v = pool.desired_vel[:n]
        relax = pool.relaxation_time[:n]
        rep_str = pool.repulsion_strength[:n]
        rep_rng = pool.repulsion_range[:n]
        radii = pool.agent_radius[:n]
        max_v = pool.max_velocity[:n]

        d_goal = goal - pos
        dist_goal = np.hypot(d_goal[:, 0], d_goal[:, 1])[:, None]
        dist_goal_safe = np.maximum(dist_goal, _EPS)
        e_goal = d_goal / dist_goal_safe

        f_att = (desired_v[:, None] * e_goal - vel) / relax[:, None]
        at_goal = (dist_goal[:, 0] < _EPS) | ~has_goal
        f_att[at_goal] = 0.0

        f_rep = np.zeros((n, 2), dtype=np.float64)
        indptr = pool.neighbor_indptr
        indices = pool.neighbor_indices

        aniso = pool.anisotropy[:n]

        if len(indices) > 0:
            pair_obs = np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr))
            pair_nbr = indices

            diff = pos[pair_obs] - pos[pair_nbr]
            dists = np.hypot(diff[:, 0], diff[:, 1])
            dists = np.maximum(dists, _EPS)
            normals = diff / dists[:, None]

            r_ij = radii[pair_obs] + radii[pair_nbr]
            magnitudes = rep_str[pair_obs] * np.exp((r_ij - dists) / rep_rng[pair_obs])

            lam = aniso[pair_obs]
            cos_phi = np.sum(-normals * e_goal[pair_obs], axis=1)
            w = lam + (1.0 - lam) * 0.5 * (1.0 + cos_phi)
            magnitudes *= w

            pair_forces = magnitudes[:, None] * normals
            np.add.at(f_rep, pair_obs, pair_forces)

        f_wall = self._compute_wall_forces_vectorized(pos, radii)

        total_f = f_att + f_rep + f_wall
        new_vel = vel + total_f * dt

        speed = np.hypot(new_vel[:, 0], new_vel[:, 1])
        too_fast = speed > max_v
        if np.any(too_fast):
            new_vel[too_fast] *= (max_v[too_fast] / speed[too_fast])[:, None]

        new_vel[at_goal] = 0.0
        pool.vel[:n] = new_vel

        if store_forces:
            self._last_force_arrays = (
                pool.agent_ids[:n].copy(),
                pos.copy(),
                f_att.copy(),
                f_rep.copy(),
                f_wall.copy(),
            )
        else:
            self._last_force_arrays = None

    def _compute_wall_forces_vectorized(
        self,
        agent_pos: np.ndarray,
        agent_radius: np.ndarray,
    ) -> np.ndarray:
        n = agent_pos.shape[0]
        if self._wall_segments_np.shape[0] == 0:
            return np.zeros((n, 2), dtype=np.float64)

        seg_p1 = self._wall_p1
        seg_d = self._wall_d
        seg_len_sq = self._wall_len_sq

        ap = agent_pos[:, None, :]
        diff_to_p1 = ap - seg_p1[None, :, :]

        t = np.sum(diff_to_p1 * seg_d[None, :, :], axis=2) / np.maximum(seg_len_sq[None, :], _EPS)
        t = np.clip(t, 0.0, 1.0)

        cp = seg_p1[None, :, :] + t[:, :, None] * seg_d[None, :, :]
        diff = ap - cp
        dist = np.hypot(diff[:, :, 0], diff[:, :, 1])
        dist = np.maximum(dist, _EPS)
        normals = diff / dist[:, :, None]

        mag = self.wall_repulsion_strength * np.exp((agent_radius[:, None] - dist) / self.wall_repulsion_range)

        forces = mag[:, :, None] * normals
        return forces.sum(axis=1)

    def compute(
        self,
        agents: Sequence[BaseAgent],
        global_goals: dict[int, Any],
    ) -> dict[int, tuple[float, float]]:
        self._last_agents = agents
        if not agents:
            self._last_forces = {}
            return {}

        velocities: dict[int, tuple[float, float]] = {}
        last_forces: dict[int, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = {}

        for agent in agents:
            agent_id = agent.state.agent_id
            params = agent.params
            belief = agent.belief

            goal = global_goals.get(agent_id)
            if goal is None:
                velocities[agent_id] = (0.0, 0.0)
                continue

            desired_velocity: float = params.desired_velocity
            lp = params.local_planner_params
            relaxation_time: float = lp.relaxation_time
            repulsion_strength: float = lp.repulsion_strength
            repulsion_range: float = lp.repulsion_range
            anisotropy: float = lp.anisotropy
            agent_radius: float = params.agent_radius
            max_velocity: float = params.max_velocity

            if isinstance(goal, Pose2D):
                gx, gy = goal.x, goal.y
            elif hasattr(goal, "x") and hasattr(goal, "y"):
                gx, gy = goal.x, goal.y
            else:
                gx, gy = float(goal[0]), float(goal[1])

            cur_x, cur_y = agent.state.pose.x, agent.state.pose.y
            cur_vx, cur_vy = agent.state.velocity

            neighbors = []
            if belief is not None:
                for ag in belief.observed_agents:
                    if ag.agent_id != agent_id:
                        neighbors.append(ag)

            dx_goal = gx - cur_x
            dy_goal = gy - cur_y
            dist_goal = np.hypot(dx_goal, dy_goal)

            if dist_goal < _EPS:
                velocities[agent_id] = (0.0, 0.0)
                continue

            e_goal_x = dx_goal / dist_goal
            e_goal_y = dy_goal / dist_goal

            f_att_x = (desired_velocity * e_goal_x - cur_vx) / relaxation_time
            f_att_y = (desired_velocity * e_goal_y - cur_vy) / relaxation_time

            f_rep_x = 0.0
            f_rep_y = 0.0

            if neighbors:
                n_neighbors = len(neighbors)
                other_pos = np.empty((n_neighbors, 2), dtype=np.float64)
                for i, ag in enumerate(neighbors):
                    other_pos[i, 0] = ag.pose.x
                    other_pos[i, 1] = ag.pose.y

                diff = np.array([cur_x, cur_y]) - other_pos
                dists = np.linalg.norm(diff, axis=1)
                dists = np.maximum(dists, _EPS)

                normals = diff / dists[:, np.newaxis]

                r_ij = 2.0 * agent_radius
                magnitudes = repulsion_strength * np.exp((r_ij - dists) / repulsion_range)

                cos_phi = (-normals[:, 0] * e_goal_x + -normals[:, 1] * e_goal_y)
                w = anisotropy + (1.0 - anisotropy) * 0.5 * (1.0 + cos_phi)
                magnitudes *= w

                forces = magnitudes[:, np.newaxis] * normals
                f_rep_x = float(forces[:, 0].sum())
                f_rep_y = float(forces[:, 1].sum())

            f_wall_x = 0.0
            f_wall_y = 0.0
            if self._wall_segments:
                f_wall_x, f_wall_y = self._compute_wall_forces_scalar(cur_x, cur_y, agent_radius)

            total_fx = f_att_x + f_rep_x + f_wall_x
            total_fy = f_att_y + f_rep_y + f_wall_y

            new_vx = cur_vx + total_fx * self._dt
            new_vy = cur_vy + total_fy * self._dt

            speed = np.hypot(new_vx, new_vy)
            if speed > max_velocity:
                scale = max_velocity / speed
                new_vx *= scale
                new_vy *= scale

            velocities[agent_id] = (float(new_vx), float(new_vy))
            last_forces[agent_id] = (
                (f_att_x, f_att_y),
                (f_rep_x, f_rep_y),
                (f_wall_x, f_wall_y),
            )

        self._last_forces = last_forces
        return velocities

    def publish_markers(self, pub: MarkerPublisher) -> None:
        from arena_humansim.viz import rgba
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker

        c_goal = rgba(0.2, 0.9, 0.2, 0.7)
        c_social = rgba(1.0, 0.2, 0.2, 0.7)
        c_obstacle = rgba(1.0, 0.6, 0.0, 0.7)
        scale = 0.3

        goal_view = pub.view("f_goal", Marker.ARROW)
        social_view = pub.view("f_social", Marker.ARROW)
        obstacle_view = pub.view("f_obstacle", Marker.ARROW)

        if self._last_force_arrays is not None:
            ids, pos, f_att, f_rep, f_wall = self._last_force_arrays
            for i in range(len(ids)):
                aid = int(ids[i])
                x, y = float(pos[i, 0]), float(pos[i, 1])
                fg = (float(f_att[i, 0]), float(f_att[i, 1]))
                fs = (float(f_rep[i, 0]), float(f_rep[i, 1]))
                fo = (float(f_wall[i, 0]), float(f_wall[i, 1]))
                self._emit_force(goal_view, aid, x, y, fg, scale, c_goal)
                self._emit_force(social_view, aid, x, y, fs, scale, c_social)
                self._emit_force(obstacle_view, aid, x, y, fo, scale, c_obstacle)
        elif self._last_forces and self._last_agents is not None:
            for agent in self._last_agents:
                aid = agent.state.agent_id
                forces = self._last_forces.get(aid)
                if forces is None:
                    continue
                x, y = agent.state.pose.x, agent.state.pose.y
                fg, fs, fo = forces
                self._emit_force(goal_view, aid, x, y, fg, scale, c_goal)
                self._emit_force(social_view, aid, x, y, fs, scale, c_social)
                self._emit_force(obstacle_view, aid, x, y, fo, scale, c_obstacle)

    @staticmethod
    def _emit_force(view, aid, x, y, f, scale, color):
        from geometry_msgs.msg import Point

        if abs(f[0]) < 1e-4 and abs(f[1]) < 1e-4:
            return
        m, new = view.get(aid)
        if new:
            m.scale.x, m.scale.y, m.scale.z = 0.02, 0.04, 0.04
            m.color = color
            m.points = [Point(), Point()]
        m.points[0].x, m.points[0].y, m.points[0].z = x, y, 0.15
        m.points[1].x = x + f[0] * scale
        m.points[1].y = y + f[1] * scale
        m.points[1].z = 0.15

    def _compute_wall_forces_scalar(self, px: float, py: float, agent_radius: float) -> tuple[float, float]:
        f_wall_x = 0.0
        f_wall_y = 0.0

        for seg in self._wall_segments:
            (x1, y1), (x2, y2) = seg
            sx, sy = x2 - x1, y2 - y1
            seg_len_sq = sx * sx + sy * sy
            if seg_len_sq < _EPS:
                cp_x, cp_y = x1, y1
            else:
                t = ((px - x1) * sx + (py - y1) * sy) / seg_len_sq
                t = max(0.0, min(1.0, t))
                cp_x = x1 + t * sx
                cp_y = y1 + t * sy

            dx = px - cp_x
            dy = py - cp_y
            dist = np.hypot(dx, dy)
            if dist < _EPS:
                continue

            nx = dx / dist
            ny = dy / dist

            magnitude = self.wall_repulsion_strength * np.exp((agent_radius - dist) / self.wall_repulsion_range)
            f_wall_x += magnitude * nx
            f_wall_y += magnitude * ny

        return f_wall_x, f_wall_y
