from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from arena_humansim.core.pool import AgentPool
from arena_humansim.utils import ModuleRegistry
from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.types import WallAware

_registry: ModuleRegistry[CollisionResolver] = ModuleRegistry()


class CollisionResolver(WallAware, Loggable, ABC):
    @abstractmethod
    def resolve(self, pool: AgentPool) -> set[int]: ...

    @classmethod
    def register(cls, name: str) -> Callable[[Callable[[], type[CollisionResolver]]], Callable[[], type[CollisionResolver]]]:
        return _registry.register(name)

    @classmethod
    def create(cls, name: str, *args: Any, **kwargs: Any) -> CollisionResolver:
        return _registry.get(name)(*args, **kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        return _registry.list_available()


class NoopCollisionResolver(CollisionResolver):
    def resolve(self, pool: AgentPool) -> set[int]:
        return set()


def _load_noop() -> type[CollisionResolver]:
    return NoopCollisionResolver


def _load_wall_projection() -> type[CollisionResolver]:
    from .wall_projection import WallProjectionResolver

    return WallProjectionResolver


_registry.register("noop")(_load_noop)
_registry.register("wall_projection")(_load_wall_projection)
