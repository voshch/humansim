from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np

from arena_humansim.pool import AgentPool
from arena_humansim.utils import ModuleRegistry
from arena_humansim.utils.loggable import Loggable

_registry = ModuleRegistry()


class CollisionResolver(Loggable, ABC):
    @abstractmethod
    def set_walls(self, segments: Iterable[tuple[tuple[float, float], tuple[float, float]]]) -> None: ...

    @abstractmethod
    def resolve(self, pool: AgentPool) -> None: ...

    @classmethod
    def register(cls, name: str):
        return _registry.register(name)

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> CollisionResolver:
        return _registry.get(name)(*args, **kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        return _registry.list_available()


class NoopCollisionResolver(CollisionResolver):
    def set_walls(self, segments):
        pass

    def resolve(self, pool):
        pass


def _load_noop():
    return NoopCollisionResolver


def _load_wall_projection():
    from .wall_projection import WallProjectionResolver

    return WallProjectionResolver


_registry.register("noop")(_load_noop)
_registry.register("wall_projection")(_load_wall_projection)
