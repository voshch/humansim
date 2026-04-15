from __future__ import annotations

from collections.abc import Callable

import pytest

from arena_humansim.agents.base import BaseAgent
from arena_humansim.utils.types import Segments


@pytest.fixture
def single_agent(agent_factory: Callable[..., BaseAgent]) -> list[BaseAgent]:
    return [agent_factory(agent_id=1, x=0.0, y=0.0)]


@pytest.fixture
def head_on_pair(agent_factory: Callable[..., BaseAgent]) -> list[BaseAgent]:
    return [
        agent_factory(agent_id=1, x=0.0, y=0.0),
        agent_factory(agent_id=2, x=2.0, y=0.0),
    ]


@pytest.fixture
def vertical_wall() -> Segments:
    return [((5.0, -5.0), (5.0, 5.0))]


@pytest.fixture
def corridor_walls() -> Segments:
    return [
        ((-5.0, -1.0), (5.0, -1.0)),
        ((-5.0, 1.0), (5.0, 1.0)),
    ]
