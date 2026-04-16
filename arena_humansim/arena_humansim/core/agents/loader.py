from __future__ import annotations

from pathlib import Path
from typing import Any

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


def _structure_raw(raw: dict[str, Any], source_path: Path | None) -> AgentType:
    from arena_humansim.utils.types import converter

    at = converter.structure(raw, AgentType)
    if source_path is not None:
        at = attrs.evolve(at, source_path=source_path)
    return at


def load_agent_type_raw_from_file(path: str | Path) -> tuple[str, dict[str, Any], Path]:
    p = Path(path).resolve()
    with open(p) as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raw = {}
    if "name" not in raw:
        raw["name"] = p.stem
    return raw["name"], raw, p


def load_agent_type_from_file(path: str | Path) -> AgentType:
    _name, raw, src = load_agent_type_raw_from_file(path)
    return _structure_raw(raw, src)


def load_agent_types_raw_from_dir(directory: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    result: dict[str, tuple[dict[str, Any], Path]] = {}
    if not directory.is_dir():
        return result
    for path in sorted(directory.glob("*.yaml")):
        try:
            name, raw, src = load_agent_type_raw_from_file(path)
            _structure_raw(dict(raw), src)
            result[name] = (raw, src)
        except Exception:
            pass
    return result


def load_agent_types_from_dir(directory: Path) -> dict[str, AgentType]:
    return {name: _structure_raw(raw, src) for name, (raw, src) in load_agent_types_raw_from_dir(directory).items()}


def load_default_agent_types() -> dict[str, AgentType]:
    share_dir = _get_share_agent_types_dir()
    if share_dir is None:
        return {}
    return load_agent_types_from_dir(share_dir)


def _load_default_agent_types_raw() -> dict[str, tuple[dict[str, Any], Path]]:
    share_dir = _get_share_agent_types_dir()
    if share_dir is None:
        return {}
    return load_agent_types_raw_from_dir(share_dir)


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
    defaults_raw = _load_default_agent_types_raw()
    scenario_local_raw = load_agent_types_raw_from_dir(scenario_dir) if scenario_dir is not None else {}
    merged_raw: dict[str, tuple[dict[str, Any], Path]] = {**defaults_raw, **scenario_local_raw}

    has_extends = any(raw.get("extends") is not None for raw, _ in merged_raw.values())
    if has_extends:
        from arena_humansim.utils.scenario import resolve_extends

        raw_dicts = {name: raw for name, (raw, _) in merged_raw.items()}
        defaults_dicts = {name: raw for name, (raw, _) in defaults_raw.items()}
        resolved = resolve_extends(raw_dicts, defaults_dicts)
        return {name: _structure_raw(resolved[name], merged_raw[name][1]) for name in resolved}

    return {name: _structure_raw(raw, src) for name, (raw, src) in merged_raw.items()}
