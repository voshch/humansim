from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from arena_humansim.agents import BaseAgent
from arena_humansim.utils import ModuleRegistry
from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.types import Pose2D, WallAware

if TYPE_CHECKING:
    from arena_humansim.viz import MarkerPublisher

_registry: ModuleRegistry[LocalPlanner] = ModuleRegistry()


class LocalPlanner(WallAware, Loggable, ABC):
    supports_pool: bool = False
    needs_global_subgoal: bool = True

    @abstractmethod
    def compute(
        self,
        agents: Sequence[BaseAgent],
        global_goals: dict[int, Pose2D],
        dt: float = 1.0,
    ) -> dict[int, tuple[float, float]]: ...

    def publish_markers(self, pub: MarkerPublisher) -> None:
        pass

    @classmethod
    def register(cls, name: str) -> Callable[[Callable[[], type[LocalPlanner]]], Callable[[], type[LocalPlanner]]]:
        return _registry.register(name)

    @classmethod
    def create(cls, name: str, *args: Any, **kwargs: Any) -> LocalPlanner:
        return _registry.get(name)(*args, **kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        return _registry.list_available()


def _load_sfm() -> type[LocalPlanner]:
    from .sfm import SFMPlanner

    return SFMPlanner


def _load_orca() -> type[LocalPlanner]:
    from .orca import ORCAPlanner

    return ORCAPlanner


def _load_straight() -> type[LocalPlanner]:
    from .straight import StraightToGoalPlanner

    return StraightToGoalPlanner


_registry.register("sfm")(_load_sfm)
_registry.register("orca")(_load_orca)
_registry.register("straight")(_load_straight)
