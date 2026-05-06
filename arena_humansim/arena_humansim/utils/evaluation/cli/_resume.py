from __future__ import annotations

import json
import os
from pathlib import Path


def peds_root() -> Path | None:
    data_dir = os.environ.get("ARENA_DATA_DIR")
    if not data_dir:
        return None
    return Path(data_dir) / "peds"


def expected_trials(manifest: dict) -> int:
    stored_robot = [rp for rp in (manifest.get("robot_policies") or []) if rp]
    n_robot = max(1, len(stored_robot))
    return len(manifest["scenarios"]) * len(manifest["planners"]) * n_robot * manifest["seeds"]


def latest_sweep_dir(*, incomplete_only: bool = False) -> tuple[Path, dict, int, int] | None:
    """Newest <ARENA_DATA_DIR>/peds/<ts>_sweep with a manifest.

    Returns (run_dir, manifest, done_count, expected_count) or None if no match.
    With incomplete_only=True, skips dirs whose .done count meets the expected trial count.
    """
    root = peds_root()
    if root is None:
        return None
    for cand in sorted(root.glob("*_sweep"), reverse=True):
        manifest_path = cand / "run_args.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        expected = expected_trials(manifest)
        done = sum(1 for _ in cand.rglob(".done"))
        if incomplete_only and done >= expected:
            continue
        return cand, manifest, done, expected
    return None
