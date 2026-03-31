from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from arena_humansim.agents import BaseAgent
from arena_humansim.utils.types import InteractionState

from . import MotionAnimation


class NoopAnimation(MotionAnimation):
    def compute_batch(
        self,
        agents: Iterable[BaseAgent],
        velocities: dict[int, tuple[float, float]],
        interactions: dict[int, InteractionState],
        dt: float,
    ) -> dict[int, Any]:
        return {}
