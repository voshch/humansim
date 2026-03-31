from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from arena_humansim.agents import BaseAgent
from arena_humansim.utils import ModuleRegistry
from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.types import InteractionState

if TYPE_CHECKING:
    from arena_humansim.pool import AgentPool

_registry = ModuleRegistry()


class MotionAnimation(Loggable, ABC):
    @abstractmethod
    def compute_batch(
        self,
        agents: Iterable[BaseAgent],
        velocities: dict[int, tuple[float, float]],
        interactions: dict[int, InteractionState],
        dt: float,
    ) -> dict[int, Any]: ...

    def compute_batch_pool(
        self,
        pool: AgentPool,
        interactions: dict[int, InteractionState],
        dt: float,
    ) -> None:
        pass

    @classmethod
    def register(cls, name: str):
        return _registry.register(name)

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> MotionAnimation:
        return _registry.get(name)(*args, **kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        return _registry.list_available()


def _load_noop():
    from .noop import NoopAnimation

    return NoopAnimation


def _load_kinematic():
    from .kinematic import KinematicAnimation

    return KinematicAnimation


_registry.register("noop")(_load_noop)
_registry.register("kinematic")(_load_kinematic)
