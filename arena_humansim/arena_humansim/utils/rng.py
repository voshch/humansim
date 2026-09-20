import hashlib

import numpy as np


def derive_seed(base_seed: int, name: str) -> int:
    digest = hashlib.blake2b(f"{base_seed}/{name}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


class RNG:
    def __init__(self, seed: int = 0):
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._substreams: dict[str, np.random.Generator] = {}

    def get_substream(self, name: str) -> np.random.Generator:
        if name not in self._substreams:
            self._substreams[name] = np.random.default_rng(derive_seed(self._seed, name))
        return self._substreams[name]

    def get_agent_substream(self, agent_id: int, module_name: str) -> np.random.Generator:
        return self.get_substream(f"agent_{agent_id}_{module_name}")

    def remove_agent_substreams(self, agent_id: int) -> None:
        prefix = f"agent_{agent_id}_"
        keys = [k for k in self._substreams if k.startswith(prefix)]
        for k in keys:
            del self._substreams[k]

    def reset(self, seed: int | None = None):
        self._seed = seed if seed is not None else self._seed
        self._rng = np.random.default_rng(self._seed)
        self._substreams.clear()

    @property
    def seed(self) -> int:
        return self._seed
