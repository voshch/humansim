from __future__ import annotations

from collections.abc import Callable

import numpy as np

from arena_humansim.core.agent_manager import arrival_damp_step, arrival_latch_step
from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.utils.types import Pose2D

R_ENTER = 0.15
R_EXIT = 0.30
TAU_BRAKE = 0.15
DT = 0.05


def _set_goal(pool: AgentPool, aid: int, x: float, y: float) -> None:
    pool.set_goals({aid: Pose2D(x=x, y=y, theta=0.0)})


def test_latch_enters_when_inside_r_enter(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=1)
    pool.pos[0] = [2.0, 2.0]
    pool.vel[0] = [0.01, 0.0]
    _set_goal(pool, 1, 2.05, 2.0)

    arrival_latch_step(pool, R_ENTER, R_EXIT)

    assert bool(pool.latched[0]) is True
    assert pool.goal_pos[0, 0] == 2.0
    assert pool.goal_pos[0, 1] == 2.0
    assert bool(pool.has_goal[0]) is False


def test_latch_enters_regardless_of_speed(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=1)
    pool.pos[0] = [2.0, 2.0]
    pool.vel[0] = [1.0, 0.0]
    _set_goal(pool, 1, 2.05, 2.0)

    arrival_latch_step(pool, R_ENTER, R_EXIT)

    assert bool(pool.latched[0]) is True


def test_latch_does_not_enter_when_far(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=1)
    pool.pos[0] = [0.0, 0.0]
    pool.vel[0] = [0.0, 0.0]
    _set_goal(pool, 1, 1.0, 0.0)

    arrival_latch_step(pool, R_ENTER, R_EXIT)

    assert bool(pool.latched[0]) is False


def test_latch_stays_across_sub_exit_goal_drift(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=1)
    pool.pos[0] = [2.0, 2.0]
    pool.vel[0] = [0.0, 0.0]
    _set_goal(pool, 1, 2.0, 2.0)
    arrival_latch_step(pool, R_ENTER, R_EXIT)
    assert bool(pool.latched[0]) is True

    _set_goal(pool, 1, 2.05, 2.03)
    arrival_latch_step(pool, R_ENTER, R_EXIT)

    assert bool(pool.latched[0]) is True
    assert pool.goal_pos[0, 0] == 2.0
    assert pool.goal_pos[0, 1] == 2.0


def test_latch_releases_on_goal_jump_past_r_exit(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=1)
    pool.pos[0] = [2.0, 2.0]
    pool.vel[0] = [0.0, 0.0]
    _set_goal(pool, 1, 2.0, 2.0)
    arrival_latch_step(pool, R_ENTER, R_EXIT)
    assert bool(pool.latched[0]) is True

    _set_goal(pool, 1, 5.0, 2.0)
    arrival_latch_step(pool, R_ENTER, R_EXIT)

    assert bool(pool.latched[0]) is False
    assert pool.goal_pos[0, 0] == 5.0
    assert pool.goal_pos[0, 1] == 2.0
    assert bool(pool.has_goal[0]) is True


def test_latch_releases_on_pose_shove_past_r_exit(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=1)
    pool.pos[0] = [2.0, 2.0]
    pool.vel[0] = [0.0, 0.0]
    _set_goal(pool, 1, 2.0, 2.0)
    arrival_latch_step(pool, R_ENTER, R_EXIT)
    assert bool(pool.latched[0]) is True

    pool.pos[0] = [2.5, 2.0]
    _set_goal(pool, 1, 2.0, 2.0)
    arrival_latch_step(pool, R_ENTER, R_EXIT)

    assert bool(pool.latched[0]) is False


def test_latch_absorbs_natural_offset_within_disc(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=1)
    pool.pos[0] = [2.08, 1.97]
    pool.vel[0] = [0.02, -0.01]
    _set_goal(pool, 1, 2.0, 2.0)

    arrival_latch_step(pool, R_ENTER, R_EXIT)

    assert bool(pool.latched[0]) is True
    assert pool.goal_pos[0, 0] == 2.08
    assert pool.goal_pos[0, 1] == 1.97


def test_latch_ignores_agent_without_goal(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=1)
    pool.pos[0] = [2.0, 2.0]
    pool.vel[0] = [0.0, 0.0]
    pool.has_goal[0] = False

    arrival_latch_step(pool, R_ENTER, R_EXIT)

    assert bool(pool.latched[0]) is False


def test_damp_decays_latched_velocity(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=1)
    pool.latched[0] = True
    pool.vel[0] = [0.4, 0.3]

    arrival_damp_step(pool, DT, TAU_BRAKE)

    decay = float(np.exp(-DT / TAU_BRAKE))
    assert pool.vel[0, 0] == 0.4 * decay
    assert pool.vel[0, 1] == 0.3 * decay


def test_damp_leaves_unlatched_velocity(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=1)
    pool.latched[0] = False
    pool.vel[0] = [0.4, 0.3]

    arrival_damp_step(pool, DT, TAU_BRAKE)

    assert pool.vel[0, 0] == 0.4
    assert pool.vel[0, 1] == 0.3


def test_damp_mixed_latched_and_moving(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=3)
    pool.latched[0] = True
    pool.latched[1] = False
    pool.latched[2] = True
    pool.vel[0] = [1.0, 0.0]
    pool.vel[1] = [1.0, 0.0]
    pool.vel[2] = [0.0, 1.0]

    arrival_damp_step(pool, DT, TAU_BRAKE)

    decay = float(np.exp(-DT / TAU_BRAKE))
    assert pool.vel[0, 0] == 1.0 * decay
    assert pool.vel[1, 0] == 1.0
    assert pool.vel[2, 1] == 1.0 * decay


def test_empty_pool_is_noop(pool_empty: Callable[..., AgentPool]) -> None:
    pool = pool_empty(capacity=8)
    arrival_latch_step(pool, R_ENTER, R_EXIT)
    arrival_damp_step(pool, DT, TAU_BRAKE)
    assert pool.n == 0
