from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

from arena_humansim.utils import ModuleRegistry
from arena_humansim.utils.loggable import Loggable

if TYPE_CHECKING:
    from arena_humansim.agents import BaseAgent
    from arena_humansim.utils.types import AgentState, BeliefState

_registry = ModuleRegistry()


class Perception(Loggable, ABC):
    @abstractmethod
    def compute(
        self,
        agent: BaseAgent,
        all_agents: dict[int, AgentState],
        world_state: dict,
        belief: BeliefState,
    ) -> BeliefState: ...

    def get_markers(self, agents: Iterable[BaseAgent], stamp) -> list:
        return []

    @classmethod
    def get(cls, name: str) -> type:
        return _registry.get(name)

    @classmethod
    def list_available(cls) -> list[str]:
        return _registry.list_available()

    @classmethod
    def register(cls, name: str):
        return _registry.register(name)


def _load_default():
    from .default import DefaultPerception

    return DefaultPerception


_registry.register("default")(_load_default)
