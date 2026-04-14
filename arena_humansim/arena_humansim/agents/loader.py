from __future__ import annotations

from pathlib import Path

import attrs
import yaml

from .types import AgentType


def _get_share_agent_types_dir() -> Path | None:
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("arena_humansim")) / "config" / "agent_types"
    except Exception:
        pass
    candidates = [
        Path(__file__).resolve().parents[2] / "config" / "agent_types",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def load_agent_type_from_file(path: str | Path) -> AgentType:
    from arena_humansim.utils.types import converter

    p = Path(path)
    with open(p) as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raw = {}
    if "name" not in raw:
        raw["name"] = p.stem
    at = converter.structure(raw, AgentType)
    return attrs.evolve(at, source_path=p.resolve())


def load_agent_types_from_dir(directory: Path) -> dict[str, AgentType]:
    result: dict[str, AgentType] = {}
    if not directory.is_dir():
        return result
    for path in sorted(directory.glob("*.yaml")):
        try:
            agent_type = load_agent_type_from_file(path)
            result[agent_type.name] = agent_type
        except Exception:
            pass
    return result


def load_default_agent_types() -> dict[str, AgentType]:
    share_dir = _get_share_agent_types_dir()
    if share_dir is None:
        return {}
    return load_agent_types_from_dir(share_dir)


def is_path_agent_type(name: str) -> bool:
    return "/" in name or name.endswith(".yaml")


def resolve_agent_type_name(
    name: str,
    registry: dict[str, AgentType],
) -> AgentType | None:
    if is_path_agent_type(name):
        p = Path(name)
        if p.is_file():
            return load_agent_type_from_file(p)
        return None
    return registry.get(name)


def load_agent_types(scenario_dir: Path | None = None) -> dict[str, AgentType]:
    from arena_humansim.agents import resolve_agent_type

    defaults = load_default_agent_types()
    if scenario_dir is not None:
        scenario_local = load_agent_types_from_dir(scenario_dir)
    else:
        scenario_local = {}

    merged = {**defaults, **scenario_local}

    has_extends = any(at.extends is not None for at in merged.values())
    if has_extends:
        from arena_humansim.utils.scenario import resolve_extends

        merged = resolve_extends(merged, defaults)

    return merged
