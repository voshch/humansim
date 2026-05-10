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
    """Newest run dir under <ARENA_DATA_DIR>/peds/ that contains a `run_args.json` manifest.

    Returns (run_dir, manifest, done_count, expected_count) or None if no match.
    With incomplete_only=True, skips dirs whose .done count meets the expected trial count.
    The directory name is irrelevant - the manifest is the signal.
    """
    root = peds_root()
    if root is None:
        return None
    candidates = sorted(
        (p for p in root.iterdir() if p.is_dir() and (p / "run_args.json").is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for cand in candidates:
        manifest = json.loads((cand / "run_args.json").read_text())
        expected = expected_trials(manifest)
        done = sum(1 for _ in cand.rglob(".done"))
        if incomplete_only and done >= expected:
            continue
        return cand, manifest, done, expected
    return None
