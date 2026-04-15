from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from arena_humansim.agents.base import BaseAgent
from arena_humansim.local_planner.orca import (
    ORCAPlanner,
    _project_onto_plane,
    _solve_linear_program,
)
from arena_humansim.utils.types import Pose2D


def test_orca_no_agents_returns_empty() -> None:
    planner = ORCAPlanner()
    assert planner.compute([], {}, dt=0.1) == {}


def test_orca_missing_goal_zero_velocity(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner()
    agent = agent_factory(agent_id=1, x=0.0, y=0.0)
    out = planner.compute([agent], {}, dt=0.1)
    assert out[1] == (0.0, 0.0)


def test_orca_at_goal_zero_velocity(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner()
    agent = agent_factory(agent_id=1, x=2.0, y=2.0)
    goals = {1: Pose2D(x=2.0, y=2.0, theta=0.0)}
    out = planner.compute([agent], goals, dt=0.1)
    assert out[1] == (0.0, 0.0)


def test_orca_single_agent_pref_velocity_toward_goal(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner()
    agent = agent_factory(agent_id=1, x=0.0, y=0.0)
    goals = {1: Pose2D(x=10.0, y=0.0, theta=0.0)}
    out = planner.compute([agent], goals, dt=0.1)
    vx, vy = out[1]
    assert vx == pytest.approx(agent.params.desired_velocity, rel=1e-6)
    assert vy == pytest.approx(0.0, abs=1e-9)


def test_orca_head_on_collision_cutoff_circle_branch(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner(time_horizon=5.0)
    a1 = agent_factory(agent_id=1, x=0.0, y=0.0)
    a2 = agent_factory(agent_id=2, x=2.0, y=0.0)
    a1.state.velocity = (1.0, 0.0)
    a2.state.velocity = (-1.0, 0.0)
    goals = {
        1: Pose2D(x=10.0, y=0.0, theta=0.0),
        2: Pose2D(x=-10.0, y=0.0, theta=0.0),
    }
    out = planner.compute([a1, a2], goals, dt=0.1)
    v1 = out[1]
    v2 = out[2]
    assert abs(v1[1]) + abs(v2[1]) > 1e-6 or v1[0] < a1.params.desired_velocity - 1e-6


def test_orca_leg_branch_cross_positive(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner(time_horizon=5.0)
    a1 = agent_factory(agent_id=1, x=0.0, y=0.0)
    a2 = agent_factory(agent_id=2, x=3.0, y=0.5)
    a1.state.velocity = (0.1, 0.0)
    a2.state.velocity = (0.0, 0.0)
    goals = {
        1: Pose2D(x=10.0, y=0.0, theta=0.0),
        2: Pose2D(x=10.0, y=0.5, theta=0.0),
    }
    out = planner.compute([a1, a2], goals, dt=0.1)
    assert 1 in out and 2 in out


def test_orca_leg_branch_cross_negative(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner(time_horizon=5.0)
    a1 = agent_factory(agent_id=1, x=0.0, y=0.0)
    a2 = agent_factory(agent_id=2, x=3.0, y=-0.5)
    a1.state.velocity = (0.1, 0.0)
    a2.state.velocity = (0.0, 0.0)
    goals = {
        1: Pose2D(x=10.0, y=0.0, theta=0.0),
        2: Pose2D(x=10.0, y=-0.5, theta=0.0),
    }
    out = planner.compute([a1, a2], goals, dt=0.1)
    assert 1 in out and 2 in out


def test_orca_already_overlapping_uses_inverse_distance_normal(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ORCAPlanner(time_horizon=5.0)
    a1 = agent_factory(agent_id=1, x=0.0, y=0.0)
    a2 = agent_factory(agent_id=2, x=0.1, y=0.0)
    a1.state.velocity = (0.5, 0.0)
    a2.state.velocity = (-0.5, 0.0)
    goals = {
        1: Pose2D(x=10.0, y=0.0, theta=0.0),
        2: Pose2D(x=-10.0, y=0.0, theta=0.0),
    }
    out = planner.compute([a1, a2], goals, dt=0.1)
    assert 1 in out and 2 in out


def test_solve_linear_program_clamps_max_speed() -> None:
    pref = np.array([10.0, 0.0])
    result = _solve_linear_program([], max_speed=1.5, pref_vel=pref)
    assert np.linalg.norm(result) == pytest.approx(1.5, rel=1e-9)


def test_project_onto_plane_negative_discriminant_returns_scaled_pref() -> None:
    point = np.array([10.0, 0.0])
    normal = np.array([1.0, 0.0])
    pref = np.array([2.0, 0.0])
    result = _project_onto_plane([], point, normal, max_speed=1.0, pref_vel=pref)
    assert np.linalg.norm(result) == pytest.approx(1.0, rel=1e-9)
    assert result[0] == pytest.approx(1.0, rel=1e-9)


def test_project_onto_plane_negative_discriminant_zero_pref_returns_zero() -> None:
    point = np.array([10.0, 0.0])
    normal = np.array([1.0, 0.0])
    pref = np.array([0.0, 0.0])
    result = _project_onto_plane([], point, normal, max_speed=1.0, pref_vel=pref)
    assert result[0] == pytest.approx(0.0, abs=1e-12)
    assert result[1] == pytest.approx(0.0, abs=1e-12)


def test_project_onto_plane_parallel_infeasible_tmin_gt_tmax() -> None:
    point = np.array([0.5, 0.0])
    normal = np.array([1.0, 0.0])
    prev_point = np.array([0.0, 10.0])
    prev_normal = np.array([1.0, 0.0])
    pref = np.array([0.6, 0.0])
    result = _project_onto_plane(
        [(prev_point, prev_normal)],
        point,
        normal,
        max_speed=1.0,
        pref_vel=pref,
    )
    assert np.linalg.norm(result) == pytest.approx(1.0, rel=1e-9)


def test_project_onto_plane_parallel_feasible_continues() -> None:
    point = np.array([0.5, 0.0])
    normal = np.array([1.0, 0.0])
    prev_point = np.array([-1.0, 0.0])
    prev_normal = np.array([-1.0, 0.0])
    pref = np.array([0.5, 0.5])
    result = _project_onto_plane(
        [(prev_point, prev_normal)],
        point,
        normal,
        max_speed=2.0,
        pref_vel=pref,
    )
    assert result[0] == pytest.approx(0.5, rel=1e-6)


def test_project_onto_plane_chooses_tmin_vs_tmax_by_denom() -> None:
    point = np.array([0.0, 0.0])
    normal = np.array([1.0, 0.0])
    prev_a_point = np.array([0.0, 0.2])
    prev_a_normal = np.array([0.0, 1.0])
    prev_b_point = np.array([0.0, 0.8])
    prev_b_normal = np.array([0.0, -1.0])
    pref = np.array([0.0, 0.5])
    result = _project_onto_plane(
        [(prev_a_point, prev_a_normal), (prev_b_point, prev_b_normal)],
        point,
        normal,
        max_speed=1.0,
        pref_vel=pref,
    )
    assert result[0] == pytest.approx(0.0, abs=1e-9)
    assert 0.2 - 1e-6 <= result[1] <= 0.8 + 1e-6


def test_project_onto_plane_empty_prev_planes_base_plus_tpref() -> None:
    point = np.array([0.0, 0.0])
    normal = np.array([1.0, 0.0])
    pref = np.array([0.3, 0.7])
    result = _project_onto_plane([], point, normal, max_speed=2.0, pref_vel=pref)
    assert result[0] == pytest.approx(0.0, abs=1e-9)
    assert result[1] == pytest.approx(0.7, rel=1e-6)
