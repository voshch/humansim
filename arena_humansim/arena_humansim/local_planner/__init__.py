from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any

from arena_humansim.agents import BaseAgent
from arena_humansim.utils import ModuleRegistry
from arena_humansim.utils.loggable import Loggable

_registry = ModuleRegistry()


class LocalPlanner(Loggable, ABC):
    @abstractmethod
    def compute(
        self,
        agents: Sequence[BaseAgent],
        global_goals: dict[int, Any],
    ) -> dict[int, tuple[float, float]]: ...

    def get_markers(self, agents: Iterable[BaseAgent], stamp) -> list:
        return []

    @classmethod
    def register(cls, name: str):
        return _registry.register(name)

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> LocalPlanner:
        return _registry.get(name)(*args, **kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        return _registry.list_available()


def _load_sfm():
    from .sfm import SFMPlanner

    return SFMPlanner


def _load_orca():
    from .orca import ORCAPlanner

    return ORCAPlanner


_registry.register("sfm")(_load_sfm)
_registry.register("orca")(_load_orca)
