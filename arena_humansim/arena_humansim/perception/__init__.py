from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from arena_humansim.utils import ModuleRegistry
from arena_humansim.utils.loggable import Loggable

if TYPE_CHECKING:
    from arena_humansim.core.agents import BaseAgent
    from arena_humansim.core.viz import MarkerPublisher
    from arena_humansim.utils.types import AgentState, BeliefState, WorldState

_registry: ModuleRegistry[Perception] = ModuleRegistry()


class Perception(Loggable, ABC):
    supports_pool: bool = False

    @abstractmethod
    def compute(
        self,
        agent: BaseAgent,
        all_agents: dict[int, AgentState],
        world_state: WorldState,
        belief: BeliefState,
    ) -> BeliefState: ...

    def publish_markers(self, pub: MarkerPublisher) -> None:
        pass

    @classmethod
    def get(cls, name: str) -> type[Perception]:
        return _registry.get(name)

    @classmethod
    def list_available(cls) -> list[str]:
        return _registry.list_available()

    @classmethod
    def register(cls, name: str) -> Callable[[Callable[[], type[Perception]]], Callable[[], type[Perception]]]:
        return _registry.register(name)


def _load_default() -> type[Perception]:
    from .default import DefaultPerception

    return DefaultPerception


_registry.register("default")(_load_default)
