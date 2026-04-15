from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from arena_humansim.agents.base import BaseAgent
from arena_humansim.agents.types import SampledLocalPlanner, SampledParams, SampledPerception
from arena_humansim.perception.default import DefaultPerception
from arena_humansim.pool import AgentPool
from arena_humansim.utils.types import AgentState, Pose2D


def _params(vision_range: float, vision_fov: float) -> SampledParams:
    return SampledParams(
        name="adult",
        desired_velocity=1.1,
        agent_radius=0.25,
        max_velocity=1.5,
        max_acceleration=1.5,
        max_deceleration=2.5,
        min_turning_radius=0.3,
        pivot_angular_velocity=2.0,
        perception=SampledPerception(vision_range=vision_range, vision_fov=vision_fov),
        local_planner_params=SampledLocalPlanner(
            relaxation_time=0.5,
            repulsion_strength=2.1,
            repulsion_range=0.3,
            anisotropy=0.5,
        ),
    )


def _make_agent(
    agent_id: int,
    x: float,
    y: float,
    theta: float = 0.0,
    vision_range: float = 5.0,
    vision_fov: float = 360.0,
) -> BaseAgent:
    state = AgentState(
        agent_id=agent_id,
        pose=Pose2D(x=x, y=y, theta=theta),
        velocity=(0.0, 0.0),
        desired_velocity=1.3,
    )
    return BaseAgent(
        state=state,
        params=_params(vision_range=vision_range, vision_fov=vision_fov),
        global_planner=cast(Any, None),
        local_planner=cast(Any, None),
        animation=cast(Any, None),
    )


def _pool_from(agents: list[BaseAgent]) -> AgentPool:
    pool = AgentPool(capacity=max(8, len(agents)))
    for ag in agents:
        pool.add_agent(ag)
    return pool


def _csr_neighbors(pool: AgentPool, row: int) -> list[int]:
    indptr = pool.neighbor_indptr
    indices = pool.neighbor_indices
    return [int(x) for x in indices[indptr[row]:indptr[row + 1]]]


def test_compute_pool_kdtree_builds_csr_with_range_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DefaultPerception, "_SMALL_N_THRESHOLD", 4)

    agents = [
        _make_agent(1, 0.0, 0.0, vision_range=5.0, vision_fov=360.0),
        _make_agent(2, 1.0, 0.0, vision_range=5.0, vision_fov=360.0),
        _make_agent(3, 0.0, 2.0, vision_range=5.0, vision_fov=360.0),
        _make_agent(4, 100.0, 0.0, vision_range=5.0, vision_fov=360.0),
        _make_agent(5, -1.5, 0.0, vision_range=5.0, vision_fov=360.0),
    ]
    pool = _pool_from(agents)
    perception = DefaultPerception()

    perception.compute_pool(pool)

    indptr = pool.neighbor_indptr
    indices = pool.neighbor_indices

    assert indptr.shape == (pool.n + 1,)
    assert indptr[0] == 0
    assert np.all(np.diff(indptr) >= 0)
    assert indptr[-1] == len(indices)

    for r in range(pool.n):
        neighbors = _csr_neighbors(pool, r)
        assert r not in neighbors, f"self-loop at row {r}: {neighbors}"

    neighbors_of_0 = set(_csr_neighbors(pool, 0))
    assert neighbors_of_0 == {1, 2, 4}

    neighbors_of_3 = _csr_neighbors(pool, 3)
    assert neighbors_of_3 == []


def test_compute_pool_kdtree_empty_after_mask_sets_empty_csr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DefaultPerception, "_SMALL_N_THRESHOLD", 4)

    agents = [
        _make_agent(1, 0.0, 0.0, vision_range=0.1, vision_fov=360.0),
        _make_agent(2, 50.0, 0.0, vision_range=0.1, vision_fov=360.0),
        _make_agent(3, 100.0, 0.0, vision_range=0.1, vision_fov=360.0),
        _make_agent(4, 150.0, 0.0, vision_range=0.1, vision_fov=360.0),
        _make_agent(5, 200.0, 0.0, vision_range=0.1, vision_fov=360.0),
    ]
    pool = _pool_from(agents)
    perception = DefaultPerception()

    perception.compute_pool(pool)

    assert pool.neighbor_indptr.shape == (pool.n + 1,)
    assert np.all(pool.neighbor_indptr == 0)
    assert pool.neighbor_indices.size == 0


def test_compute_pool_kdtree_fov_filters_back_neighbors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DefaultPerception, "_SMALL_N_THRESHOLD", 4)

    agents = [
        _make_agent(1, 0.0, 0.0, theta=0.0, vision_range=5.0, vision_fov=90.0),
        _make_agent(2, 1.0, 0.0, theta=0.0, vision_range=5.0, vision_fov=360.0),
        _make_agent(3, -1.0, 0.0, theta=0.0, vision_range=5.0, vision_fov=360.0),
        _make_agent(4, 0.0, 2.0, theta=0.0, vision_range=5.0, vision_fov=360.0),
        _make_agent(5, 2.0, 0.1, theta=0.0, vision_range=5.0, vision_fov=360.0),
    ]
    pool = _pool_from(agents)
    perception = DefaultPerception()

    perception.compute_pool(pool)

    neighbors_of_observer = set(_csr_neighbors(pool, 0))
    assert 1 in neighbors_of_observer
    assert 4 in neighbors_of_observer
    assert 2 not in neighbors_of_observer
    assert 3 not in neighbors_of_observer
    assert 0 not in neighbors_of_observer


def test_compute_pool_dense_fov_mixed_omni_and_narrow() -> None:
    agents = [
        _make_agent(1, 0.0, 0.0, theta=0.0, vision_range=5.0, vision_fov=60.0),
        _make_agent(2, 0.0, 0.0, theta=0.0, vision_range=5.0, vision_fov=360.0),
        _make_agent(3, 1.0, 0.0, theta=0.0, vision_range=5.0, vision_fov=360.0),
        _make_agent(4, -1.0, 0.0, theta=0.0, vision_range=5.0, vision_fov=360.0),
    ]
    pool = _pool_from(agents)
    pool.pos[0, 0] = 0.0
    pool.pos[0, 1] = 0.0
    pool.pos[1, 0] = 0.0
    pool.pos[1, 1] = 5.0

    perception = DefaultPerception()
    perception.compute_pool(pool)

    indptr = pool.neighbor_indptr
    assert indptr[0] == 0
    assert np.all(np.diff(indptr) >= 0)

    neighbors_of_narrow = set(_csr_neighbors(pool, 0))
    assert 2 in neighbors_of_narrow
    assert 3 not in neighbors_of_narrow
    assert 1 not in neighbors_of_narrow

    neighbors_of_omni = set(_csr_neighbors(pool, 1))
    assert 0 in neighbors_of_omni
    assert 1 not in neighbors_of_omni


def test_prepare_tick_builds_tree_only_when_more_than_one_agent() -> None:
    perception = DefaultPerception()

    perception.prepare_tick({})
    assert perception.shared_tree is None
    assert perception.shared_ids == []
    assert perception.shared_positions == []

    one = _make_agent(1, 0.0, 0.0).state
    perception.prepare_tick({1: one})
    assert perception.shared_tree is None
    assert perception.shared_ids == [1]
    assert perception.shared_positions == [pytest.approx((0.0, 0.0))]

    two = _make_agent(2, 1.0, 0.0).state
    perception.prepare_tick({1: one, 2: two})
    assert perception.shared_tree is not None
    assert perception.shared_ids == [1, 2]
    assert perception.shared_positions is not None
    assert len(perception.shared_positions) == 2
