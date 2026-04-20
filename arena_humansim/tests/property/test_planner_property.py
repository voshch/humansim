from __future__ import annotations

"""Global planner output invariants."""

import math
from collections.abc import Callable
from typing import Any

import pytest

pytest.importorskip("arena_humansim_msgs.msg")

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.global_planner import GlobalPlanner, _registry
from arena_humansim.utils.types import Pose2D, Segments, WallAware

_IMPL_IDS = sorted(_registry._registry.keys())

_coord = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
_agent_positions = st.lists(st.tuples(_coord, _coord), min_size=0, max_size=6)
_wall_segment = st.tuples(st.tuples(_coord, _coord), st.tuples(_coord, _coord))
_walls = st.lists(_wall_segment, min_size=0, max_size=4)


def _is_wall_aware(obj: object) -> bool:
    return isinstance(obj, WallAware) and type(obj).set_walls is not WallAware.set_walls


@pytest.mark.parametrize("impl", _IMPL_IDS)
@given(positions=_agent_positions, walls=_walls)
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_planner_outputs_finite_and_keyed_on_inputs(
    impl: str,
    positions: list[tuple[float, float]],
    walls: list[tuple[tuple[float, float], tuple[float, float]]],
    agent_factory: Callable[..., BaseAgent],
    commands_factory: Callable[..., dict[int, Any]],
) -> None:
    """Planner returns finite Pose2D goals keyed on a subset of input agent ids."""
    planner = GlobalPlanner.create(impl)
    segments: Segments = list(walls) if _is_wall_aware(planner) else []
    planner.set_walls(segments)

    agents = [agent_factory(agent_id=i + 1, x=float(x), y=float(y)) for i, (x, y) in enumerate(positions)]
    ids = [a.state.agent_id for a in agents]
    commands = commands_factory(agent_ids=ids, target=(5.0, 0.0))

    goals = planner.compute(agents, commands)

    assert set(goals.keys()).issubset(set(ids))
    for pose in goals.values():
        assert isinstance(pose, Pose2D)
        assert math.isfinite(pose.x)
        assert math.isfinite(pose.y)
        assert math.isfinite(pose.theta)
