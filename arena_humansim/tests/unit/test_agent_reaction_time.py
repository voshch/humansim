from __future__ import annotations

import pytest

pytest.importorskip("rclpy")

import numpy as np
from arena_humansim.core.agents.types import (
    AgentType,
    ParamDist,
    _sample_lognormal_dist,
    sample_agent_type,
)


def test_sample_lognormal_median_near_configured_mean() -> None:
    rng = np.random.default_rng(42)
    dist = ParamDist(mean=0.4, std=0.3, clip_low=0.05, clip_high=1.5)
    samples = [_sample_lognormal_dist(dist, rng) for _ in range(5000)]
    median = float(np.median(samples))
    assert median == pytest.approx(0.4, abs=0.03)
    assert min(samples) >= dist.clip_low
    assert max(samples) <= dist.clip_high


def test_sample_lognormal_zero_std_returns_mean() -> None:
    rng = np.random.default_rng(0)
    dist = ParamDist(mean=0.5, std=0.0, clip_low=0.01, clip_high=10.0)
    assert _sample_lognormal_dist(dist, rng) == pytest.approx(0.5)


def test_sample_agent_type_populates_reaction_time_and_personal_space() -> None:
    rng = np.random.default_rng(123)
    at = AgentType(name="tester")
    sp = sample_agent_type(at, rng)
    assert 0.05 <= sp.reaction_time <= 1.5
    assert 0.2 <= sp.personal_space_min <= 2.0


def test_sample_agent_type_deterministic_under_seed() -> None:
    at = AgentType(name="tester")
    sp1 = sample_agent_type(at, np.random.default_rng(7))
    sp2 = sample_agent_type(at, np.random.default_rng(7))
    assert sp1.reaction_time == sp2.reaction_time
    assert sp1.personal_space_min == sp2.personal_space_min


def test_reaction_time_varies_across_agents() -> None:
    at = AgentType(name="tester", reaction_time=ParamDist(mean=0.4, std=0.5, clip_low=0.05, clip_high=1.5))
    rng = np.random.default_rng(99)
    samples = {sample_agent_type(at, rng).reaction_time for _ in range(30)}
    assert len(samples) > 5  # not all collapsed to clip or identical
