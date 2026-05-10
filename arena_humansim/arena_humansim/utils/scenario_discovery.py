from __future__ import annotations

from pathlib import Path

# Search order: evaluation/ holds the structured benchmark scenarios
# (bucket/density/modality), scenarios/ is the catch-all for showcase / ad-hoc.
SCENARIO_ROOT_NAMES: tuple[str, ...] = ("evaluation", "scenarios")


def _share_root() -> Path | None:
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("arena_humansim"))
    except Exception:
        return None


def scenario_roots() -> list[Path]:
    """Absolute paths of every scenario root that exists on this install."""
    share = _share_root()
    if share is None:
        return []
    return [share / "config" / name for name in SCENARIO_ROOT_NAMES if (share / "config" / name).is_dir()]


def find_scenario_path(name_or_path: str) -> Path:
    """Resolve a scenario name or relative path to its YAML file.

    Resolution order:
      1. Treat input as an absolute path; return if it's a file.
      2. For each scenario root, try the input as a relative path under that root.
      3. Fall back to a leaf-name glob across all roots - succeeds only if the
         leaf is unique across all roots.
    Raises FileNotFoundError if no match, or if the leaf glob is ambiguous.
    """
    p = Path(name_or_path)
    if p.is_file():
        return p
    roots = scenario_roots()
    if not roots:
        raise FileNotFoundError(f"scenario not found: {name_or_path}")

    rel = p if p.suffix else p.with_suffix(".yaml")
    target = p.name if p.suffix else f"{p.name}.yaml"

    for root in roots:
        relative_candidate = root / rel
        if relative_candidate.is_file():
            return relative_candidate

    hits: list[Path] = []
    for root in roots:
        hits.extend(sorted(h for h in root.glob(f"**/{target}") if h.is_file()))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        share = _share_root()
        rels = [str(h.relative_to(share / "config")) if share else str(h) for h in hits]
        raise FileNotFoundError(f"scenario name {name_or_path!r} is ambiguous; matches {rels}.")
    raise FileNotFoundError(f"scenario not found: {name_or_path}")


def parse_scenario_id(sid: str) -> tuple[str, str, str, bool]:
    """Decompose a canonical scenario id into (bucket, density, modality, is_robot_variant).

    Convention: ``<bucket>_<density>_<modality>``, optionally prefixed with ``robot_`` for the
    robot variant. Het scenarios are robot-constitutive - names like
    ``het_dense_robot_in_group`` are NOT robot variants; the leading token is the bucket,
    not the variant flag.
    """
    is_robot = sid.startswith("robot_")
    base = sid.removeprefix("robot_") if is_robot else sid
    parts = base.split("_", 2)
    bucket = parts[0] if parts else ""
    density = parts[1] if len(parts) > 1 else ""
    modality = parts[2] if len(parts) > 2 else ""
    return bucket, density, modality, is_robot


def scenario_has_robot(name_or_path: str) -> bool | None:
    """True if scenario YAML declares a kind=1 agent. None if YAML can't be located/parsed."""
    try:
        path = find_scenario_path(name_or_path)
    except FileNotFoundError:
        return None
    try:
        import yaml

        data = yaml.safe_load(path.read_text())
    except Exception:
        return None
    agents = data.get("agents", []) if isinstance(data, dict) else []
    return any(isinstance(a, dict) and a.get("kind") == 1 for a in agents)
