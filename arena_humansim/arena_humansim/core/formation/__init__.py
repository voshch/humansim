from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from arena_humansim.utils import ModuleRegistry
from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.types import Pose2D

from .anchor import AgentAnchor, Anchor, CentroidAnchor, ObjectAnchor, PoseAnchor

if TYPE_CHECKING:
    from arena_humansim.core.agents import BaseAgent


AgentLookup = Callable[[int], "BaseAgent | None"]

_registry: ModuleRegistry[Formation] = ModuleRegistry()


class Formation(Loggable, ABC):
    @abstractmethod
    def on_join(self, agent_id: int) -> None: ...

    @abstractmethod
    def on_leave(self, agent_id: int) -> None: ...

    @abstractmethod
    def tick(self, dt: float) -> dict[int, Pose2D]: ...

    def arrived(self, agent_id: int) -> bool:
        return True

    @classmethod
    def register(cls, name: str) -> Callable[[Callable[[], type[Formation]]], Callable[[], type[Formation]]]:
        return _registry.register(name)

    @classmethod
    def create(cls, name: str, *args: Any, **kwargs: Any) -> Formation:
        return _registry.get(name)(*args, **kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        return _registry.list_available()


def _load_line() -> type[Formation]:
    from .line import LineFormation

    return LineFormation


def _load_cluster() -> type[Formation]:
    from .cluster import ClusterFormation

    return ClusterFormation


def _load_f_formation() -> type[Formation]:
    from .f_formation import FFormation

    return FFormation


def _load_dyad() -> type[Formation]:
    from .dyad import DyadFormation

    return DyadFormation


_registry.register("line")(_load_line)
_registry.register("cluster")(_load_cluster)
_registry.register("f_formation")(_load_f_formation)
_registry.register("dyad")(_load_dyad)


__all__ = [
    "AgentAnchor",
    "AgentLookup",
    "Anchor",
    "CentroidAnchor",
    "Formation",
    "ObjectAnchor",
    "PoseAnchor",
]
