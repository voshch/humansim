from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.local_planner.force import SOCIAL, WALL
from arena_humansim.local_planner.johansson import JohanssonPlanner
from arena_humansim.utils.types import Pose2D

_TAU, _A, _A_WALL, _B, _STRIDE, _W = range(6)


def _pool(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
    planner: JohanssonPlanner,
    specs: list[tuple[float, float, float, tuple[float, float]]],
    neighbors: bool = True,
) -> AgentPool:
    pool = pool_empty(capacity=max(4, len(specs)))
    planner.attach(pool)
    for k, (x, y, theta, vel) in enumerate(specs):
        agent = agent_factory(agent_id=k + 1, x=x, y=y)
        agent.state.pose.theta = theta
        agent.state.velocity = vel
        pool.add_agent(agent)
    n = len(specs)
    pool.set_goals({k + 1: Pose2D(x=20.0, y=0.0) for k in range(n)})
    if neighbors:
        indices = [j for i in range(n) for j in range(n) if j != i]
        indptr = [i * (n - 1) for i in range(n + 1)]
    else:
        indices = []
        indptr = [0] * (n + 1)
    pool.set_neighbor_csr(np.array(indptr, dtype=np.int32), np.array(indices, dtype=np.int32))
    return pool


def _forces(planner: JohanssonPlanner, pool: AgentPool) -> np.ndarray:
    planner.compute_pool(pool, store_forces=True, dt=0.05)
    assert planner._last_force_arrays is not None
    return planner._last_force_arrays[2]


def test_moving_neighbor_force_bisects_current_and_stride_offset_directions(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = JohanssonPlanner()
    p = planner._defaults
    vel_j = (0.0, 1.0)
    social = _forces(planner, _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, math.pi, (0.0, 0.0)), (2.0, 0.0, 0.0, vel_j)]))[:, SOCIAL]

    rel = np.array([-2.0, 0.0])
    step = np.array(vel_j) * p[_STRIDE]
    off = rel - step
    d = np.linalg.norm(rel)
    od = np.linalg.norm(off)
    term1 = d + od
    b = 0.5 * math.sqrt(term1 * term1 - step @ step)
    expected = p[_A] * p[_W] * term1 / (2.0 * b) * math.exp(-b / p[_B]) * 0.5 * (rel / d + off / od)
    assert social[0, 1] < 0.0
    assert np.allclose(social[0], expected)


def test_wall_force_points_away_and_decays(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = JohanssonPlanner()
    p = planner._defaults
    planner.set_walls([((-5.0, -1.0), (5.0, -1.0))])
    specs = [(0.0, -0.5, 0.0, (0.0, 0.0)), (3.0, -0.2, 0.0, (0.0, 0.0)), (0.0, -1.6, 0.0, (0.0, 0.0)), (5.5, -1.0, math.pi / 2, (0.0, 0.0))]
    wall = _forces(planner, _pool(pool_empty, agent_factory, planner, specs, neighbors=False))[:, WALL]
    side_weight = p[_W] + (1.0 - p[_W]) * 0.5
    assert np.allclose(wall[0], (0.0, p[_A_WALL] * side_weight * math.exp(-0.5 / p[_B])))
    assert 0.0 < wall[1, 1] < wall[0, 1]
    assert np.isclose(wall[1, 0], 0.0)
    assert wall[2, 1] < 0.0
    assert wall[3, 0] > 0.0
    assert np.isclose(wall[3, 1], 0.0)


def test_coincident_and_collinear_stride_agents_stay_finite(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = JohanssonPlanner()
    coincident = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.0, (0.0, 0.0)), (0.0, 0.0, 0.0, (0.5, 0.0))])
    collinear = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.0, (0.0, 0.0)), (1.0, 0.0, 0.0, (-4.0, 0.0))])
    on_stride_end = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.0, (0.0, 0.0)), (1.0, 0.0, 0.0, (-2.0, 0.0))])
    for pool in (coincident, collinear, on_stride_end):
        forces = _forces(planner, pool)
        assert np.all(np.isfinite(forces))
        assert np.all(np.isfinite(pool.vel[:2]))
