from __future__ import annotations

from collections.abc import Callable

import numpy as np

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.local_planner import LocalPlanner
from arena_humansim.local_planner.sfm import SFMPlanner
from arena_humansim.core.pool import AgentPool
from arena_humansim.utils.types import Pose2D


def _build_pool(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], kinds: list[int]) -> AgentPool:
    pool = pool_empty(capacity=8)
    for i, k in enumerate(kinds):
        pool.add_agent(agent_factory(agent_id=i + 1, x=float(i) * 0.5, y=0.0))
        pool.kind[i] = k
        pool.policy_idx[i] = 0
    pool.set_goals({1: Pose2D(x=10.0, y=0.0)})
    indptr = np.array([0, 1, 1], dtype=np.int32)
    indices = np.array([1], dtype=np.int32)
    pool.set_neighbor_csr(indptr, indices)
    return pool


def test_robot_neighbor_produces_larger_repulsion(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    baseline = _build_pool(pool_empty, agent_factory, [0, 0])
    p1 = SFMPlanner()
    p1.compute_pool(baseline, store_forces=False, dt=0.05)
    baseline_vel = baseline.vel[0].copy()

    robot = _build_pool(pool_empty, agent_factory, [0, 1])
    p2 = SFMPlanner()
    p2.compute_pool(robot, store_forces=False, dt=0.05)
    robot_vel = robot.vel[0].copy()

    assert robot_vel[0] < baseline_vel[0]


def test_apply_policy_params_parses_overrides() -> None:
    p = SFMPlanner()
    p.apply_policy_params('{"kind_gains": {"human_robot": {"strength_scale": 3.0, "range_scale": 2.0}}}')
    assert p._gain_strength_scale[0, 1] == 3.0
    assert p._gain_range_scale[0, 1] == 2.0


def test_apply_policy_params_defensive_against_garbage() -> None:
    p = SFMPlanner()
    before_s = p._gain_strength_scale.copy()
    before_r = p._gain_range_scale.copy()
    for bad in ["", "not json", "[1,2,3]", '{"kind_gains": "nope"}', '{"kind_gains": {"bad": {}}}']:
        p.apply_policy_params(bad)
    assert np.array_equal(p._gain_strength_scale, before_s)
    assert np.array_equal(p._gain_range_scale, before_r)


def test_registry_includes_straight() -> None:
    assert "straight" in LocalPlanner.list_available()
