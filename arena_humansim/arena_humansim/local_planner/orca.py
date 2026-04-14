from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree

from arena_humansim.agents import BaseAgent
from arena_humansim.utils.types import Pose2D

from . import LocalPlanner

_EPS = 1e-6


class ORCAPlanner(LocalPlanner):
    def __init__(
        self,
        time_horizon: float = 5.0,
        max_neighbors: int = 10,
    ):
        self.time_horizon = time_horizon
        self.max_neighbors = max_neighbors

    def compute(
        self,
        agents: Sequence[BaseAgent],
        global_goals: dict[int, Pose2D],
        dt: float = 1.0,
    ) -> dict[int, tuple[float, float]]:
        if not agents:
            return {}

        velocities: dict[int, tuple[float, float]] = {}

        agent_ids = [a.state.agent_id for a in agents]
        agent_positions = np.array([[a.state.pose.x, a.state.pose.y] for a in agents], dtype=np.float64)
        tree = None
        if len(agents) > 1:
            tree = cKDTree(agent_positions)

        for i, agent in enumerate(agents):
            aid = agent.state.agent_id
            params = agent.params

            goal = global_goals.get(aid)
            if goal is None:
                velocities[aid] = (0.0, 0.0)
                continue

            pos = np.array([agent.state.pose.x, agent.state.pose.y])
            vel = np.array(agent.state.velocity)
            radius_a = params.agent_radius
            max_speed = params.desired_velocity
            desired_vel = params.desired_velocity

            goal_pos = np.array([goal.x, goal.y])

            diff = goal_pos - pos
            dist = np.linalg.norm(diff)
            if dist < _EPS:
                velocities[aid] = (0.0, 0.0)
                continue

            pref_vel = (diff / dist) * desired_vel

            neighbors: list[tuple[np.ndarray, np.ndarray, float]] = []
            if tree is not None:
                k = min(self.max_neighbors + 1, len(agents))
                distances, indices = tree.query(agent_positions[i], k=k)
                if np.ndim(distances) == 0:
                    distances = [distances]
                    indices = [indices]
                for idx in indices:
                    nid = agent_ids[idx]
                    if nid == aid:
                        continue
                    n_agent = agents[idx]
                    n_pos = agent_positions[idx]
                    n_vel = np.array(n_agent.state.velocity)
                    n_radius = n_agent.params.agent_radius
                    neighbors.append((n_pos, n_vel, n_radius))
                    if len(neighbors) >= self.max_neighbors:
                        break

            planes: list[tuple[np.ndarray, np.ndarray]] = []
            inv_tau = 1.0 / self.time_horizon

            for n_pos, n_vel, n_radius in neighbors:
                rel_pos = n_pos - pos
                rel_vel = vel - n_vel
                dist_sq = float(np.dot(rel_pos, rel_pos))
                combined_radius = radius_a + n_radius
                combined_radius_sq = combined_radius * combined_radius

                if dist_sq > combined_radius_sq:
                    w = rel_vel - inv_tau * rel_pos
                    w_len_sq = float(np.dot(w, w))
                    dot_product = float(np.dot(w, rel_pos))

                    if dot_product < 0.0 and (dot_product * dot_product > combined_radius_sq * w_len_sq):
                        w_len = np.sqrt(w_len_sq)
                        if w_len < _EPS:
                            continue
                        unit_w = w / w_len
                        normal = unit_w
                        u = (combined_radius * inv_tau - w_len) * unit_w
                    else:
                        leg = np.sqrt(max(dist_sq - combined_radius_sq, _EPS))

                        if float(np.cross(rel_pos, w)) > 0.0:
                            direction = (
                                np.array(
                                    [
                                        rel_pos[0] * leg - rel_pos[1] * combined_radius,
                                        rel_pos[0] * combined_radius + rel_pos[1] * leg,
                                    ]
                                )
                                / dist_sq
                            )
                        else:
                            direction = (
                                np.array(
                                    [
                                        rel_pos[0] * leg + rel_pos[1] * combined_radius,
                                        -rel_pos[0] * combined_radius + rel_pos[1] * leg,
                                    ]
                                )
                                / dist_sq
                            )

                        dot_rv_dir = float(np.dot(rel_vel, direction))
                        proj = dot_rv_dir * direction
                        u = proj - rel_vel

                        normal = np.array([-direction[1], direction[0]])
                        if float(np.dot(normal, rel_pos)) > 0.0:
                            normal = -normal

                else:
                    dist_val = np.sqrt(dist_sq) if dist_sq > _EPS else _EPS
                    normal = -rel_pos / dist_val
                    u = (combined_radius * inv_tau - float(np.dot(rel_vel, normal))) * normal

                point = vel + 0.5 * u
                planes.append((point, normal))

            new_vel = _solve_linear_program(planes, max_speed, pref_vel)
            velocities[aid] = (float(new_vel[0]), float(new_vel[1]))

        return velocities


def _solve_linear_program(
    planes: Sequence[tuple[np.ndarray, np.ndarray]],
    max_speed: float,
    pref_vel: np.ndarray,
) -> np.ndarray:
    result = pref_vel.copy()

    for i, (point_i, normal_i) in enumerate(planes):
        if np.dot(normal_i, point_i - result) <= 0.0:
            continue

        result = _project_onto_plane(planes[:i], point_i, normal_i, max_speed, pref_vel)

    speed = np.linalg.norm(result)
    if speed > max_speed:
        result = result * (max_speed / speed)

    return result


def _project_onto_plane(
    prev_planes: Iterable[tuple[np.ndarray, np.ndarray]],
    point: np.ndarray,
    normal: np.ndarray,
    max_speed: float,
    pref_vel: np.ndarray,
) -> np.ndarray:
    direction = np.array([-normal[1], normal[0]])
    base_point = point

    dot_bd = float(np.dot(base_point, direction))
    discriminant = dot_bd * dot_bd - (float(np.dot(base_point, base_point)) - max_speed * max_speed)

    if discriminant < 0.0:
        speed = np.linalg.norm(pref_vel)
        if speed < _EPS:
            return np.zeros(2)
        return pref_vel * (max_speed / max(speed, _EPS))

    sqrt_disc = np.sqrt(discriminant)
    t_min = -dot_bd - sqrt_disc
    t_max = -dot_bd + sqrt_disc

    for point_j, normal_j in prev_planes:
        denom = float(np.dot(normal_j, direction))
        numer = float(np.dot(normal_j, point_j - base_point))

        if abs(denom) < _EPS:
            if numer < 0.0:
                t_min = t_max + 1.0
                break
            continue

        t_cross = numer / denom
        if denom > 0.0:
            t_min = max(t_min, t_cross)
        else:
            t_max = min(t_max, t_cross)

        if t_min > t_max:
            break

    if t_min > t_max:
        speed = np.linalg.norm(pref_vel)
        if speed < _EPS:
            return np.zeros(2)
        return pref_vel * (max_speed / max(speed, _EPS))

    t_pref = float(np.dot(pref_vel - base_point, direction))
    t_opt = max(t_min, min(t_max, t_pref))

    return base_point + t_opt * direction
