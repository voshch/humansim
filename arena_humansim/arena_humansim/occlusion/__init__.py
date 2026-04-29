from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from arena_humansim.core.pool import PoolAware
from arena_humansim.utils import ModuleRegistry
from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.types import Segments, WallAware

_registry: ModuleRegistry[Occluder] = ModuleRegistry()


class Occluder(PoolAware, WallAware, Loggable, ABC):
    @abstractmethod
    def clear(self, p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def set_walls(self, segments: Segments) -> None: ...

    @classmethod
    def get(cls, name: str) -> type[Occluder]:
        return _registry.get(name)

    @classmethod
    def list_available(cls) -> list[str]:
        return _registry.list_available()

    @classmethod
    def register(cls, name: str) -> Callable[[Callable[[], type[Occluder]]], Callable[[], type[Occluder]]]:
        return _registry.register(name)


def _load_bitmap() -> type[Occluder]:
    from .bitmap import BitmapOccluder

    return BitmapOccluder


def _load_noop() -> type[Occluder]:
    from .noop import NoopOccluder

    return NoopOccluder


_registry.register("bitmap")(_load_bitmap)
_registry.register("noop")(_load_noop)
