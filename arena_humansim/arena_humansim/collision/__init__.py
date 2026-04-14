from __future__ import annotations

from abc import ABC, abstractmethod

from arena_humansim.pool import AgentPool
from arena_humansim.utils import ModuleRegistry
from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.types import WallAware

_registry = ModuleRegistry()


class CollisionResolver(WallAware, Loggable, ABC):
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
    def resolve(self, pool: AgentPool) -> None:
        pass


def _load_noop():
    return NoopCollisionResolver


def _load_wall_projection():
    from .wall_projection import WallProjectionResolver

    return WallProjectionResolver


_registry.register("noop")(_load_noop)
_registry.register("wall_projection")(_load_wall_projection)
