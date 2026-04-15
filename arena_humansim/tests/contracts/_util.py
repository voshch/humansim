from __future__ import annotations

from arena_humansim.utils.registry import ModuleRegistry


def registry_ids(registry: ModuleRegistry) -> list[str]:
    return sorted(registry._registry.keys())
