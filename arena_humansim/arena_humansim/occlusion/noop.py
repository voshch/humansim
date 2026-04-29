from __future__ import annotations

import numpy as np

from arena_humansim.utils.types import Segments

from . import Occluder


class NoopOccluder(Occluder):
    def set_walls(self, segments: Segments) -> None:
        pass

    def clear(self, p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
        return np.ones(len(p_a), dtype=np.bool_)
