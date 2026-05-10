import argparse
import sys
from pathlib import Path

from arena_humansim.utils.scenario import dump_resolved_scenario


def _scenario_name_from_trial_dir(name: str) -> str | None:
    parts = name.split("__")
    if len(parts) < 3:
        return None
    return parts[0]


def _resolve_scenario_path(scenario_name: str, scenarios_dir: Path) -> Path | None:
    candidate = scenarios_dir / f"{scenario_name}.yaml"
    return candidate if candidate.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill scenario.yaml snapshots into existing trial dirs. Uses CURRENT scenario YAMLs from the source tree, not the recording-time YAMLs (those weren't preserved). Treat the resulting snapshots as best-effort.")
    parser.add_argument("--root", type=str, required=True, help="recordings root (e.g. /home/arena/arena_ws/data/peds)")
    parser.add_argument("--scenarios_dir", type=str, required=True, help="path to config/scenarios/ source-of-truth dir")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing scenario.yaml snapshots")
    parser.add_argument("--dry_run", action="store_true", help="print what would be done; write nothing")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    scenarios_dir = Path(args.scenarios_dir).resolve()
    if not root.is_dir():
        sys.exit(f"--root not a directory: {root}")
    if not scenarios_dir.is_dir():
        sys.exit(f"--scenarios_dir not a directory: {scenarios_dir}")

    written = 0
    skipped_existing = 0
    skipped_not_trial = 0
    skipped_no_scenario = 0
    failed = 0

    sweep_dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    print(f"Scanning {len(sweep_dirs)} sweep dirs under {root}")
    for sweep_idx, sweep_dir in enumerate(sweep_dirs, start=1):
        trial_dirs = [t for t in sorted(sweep_dir.iterdir()) if t.is_dir() and (t / "bag").exists()]
        print(f"[{sweep_idx}/{len(sweep_dirs)}] {sweep_dir.name}: {len(trial_dirs)} trials", flush=True)
        sweep_written = 0
        for trial_idx, trial_dir in enumerate(trial_dirs, start=1):
            scenario_name = _scenario_name_from_trial_dir(trial_dir.name)
            if scenario_name is None:
                skipped_not_trial += 1
                continue
            target = trial_dir / "scenario.yaml"
            if target.exists() and not args.overwrite:
                skipped_existing += 1
                continue
            scenario_path = _resolve_scenario_path(scenario_name, scenarios_dir)
            if scenario_path is None:
                print(f"  no source for {scenario_name!r}; skipping {trial_dir.name}", flush=True)
                skipped_no_scenario += 1
                continue
            if args.dry_run:
                written += 1
                continue
            try:
                dump_resolved_scenario(scenario_path, target)
                written += 1
                sweep_written += 1
            except Exception as exc:
                print(f"  FAILED {trial_dir.name}: {exc}", flush=True)
                failed += 1
            if not args.dry_run and trial_idx % 50 == 0:
                print(f"    ... {trial_idx}/{len(trial_dirs)} ({sweep_written} written this sweep, {written} total)", flush=True)
        if not args.dry_run:
            print(f"  -> {sweep_written} snapshots written for {sweep_dir.name}", flush=True)

    print()
    print(f"written: {written}{' (dry-run)' if args.dry_run else ''}")
    print(f"skipped (existing snapshot, no --overwrite): {skipped_existing}")
    print(f"skipped (no scenario source found): {skipped_no_scenario}")
    print(f"skipped (not a trial dir): {skipped_not_trial}")
    print(f"failed: {failed}")


if __name__ == "__main__":
    main()
