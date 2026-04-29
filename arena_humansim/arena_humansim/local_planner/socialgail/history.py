from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import numpy as np


class HistoryBuffer:
    """Per-agent past-position ring buffer at decision-tick spacing.

    Stores strictly past positions: index 0 (peek(aid, 1)) is the most recent
    decision-tick snapshot, index past_len-1 is the oldest. The agent's current
    position is read from the pool, never from here.
    """

    def __init__(self, past_len: int = 5):
        self.past_len = past_len
        self._buf: dict[int, deque[np.ndarray]] = {}

    def update_many(self, agent_ids: np.ndarray, positions: np.ndarray) -> None:
        for i in range(len(agent_ids)):
            aid = int(agent_ids[i])
            d = self._buf.get(aid)
            if d is None:
                d = deque(maxlen=self.past_len)
                self._buf[aid] = d
            d.appendleft(positions[i].copy())

    def peek(self, aid: int, k: int) -> np.ndarray | None:
        d = self._buf.get(aid)
        if d is None or k < 1 or k > len(d):
            return None
        return d[k - 1]

    def evict(self, alive_ids: Iterable[int]) -> None:
        alive = set(int(a) for a in alive_ids)
        for aid in list(self._buf.keys()):
            if aid not in alive:
                del self._buf[aid]

    def __contains__(self, aid: int) -> bool:
        return aid in self._buf

    def __len__(self) -> int:
        return len(self._buf)
