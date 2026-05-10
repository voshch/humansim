import argparse
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from arena_humansim.utils import renderer as _renderer
from arena_humansim.utils.evaluation.analyze import parse_trial_dir
from arena_humansim.utils.evaluation.cli._resume import latest_sweep_dir, peds_root

Trial = tuple[Path, Path, str, str, str, int]


def _resolve_dirs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[Path]:
    if args.all:
        meta = Path(args.recordings_dir).resolve() if args.recordings_dir else peds_root()
        if meta is None:
            parser.error("--all requires either --recordings_dir or $ARENA_DATA_DIR")
        if not meta.is_dir():
            parser.error(f"--all: meta dir {meta} does not exist")
        sweeps = sorted(p for p in meta.iterdir() if p.is_dir() and (p / "run_args.json").is_file())
        if not sweeps:
            parser.error(f"--all: no sweep dirs (with run_args.json) found under {meta}")
        return sweeps
    if args.recordings_dir:
        return [Path(args.recordings_dir).resolve()]
    found = latest_sweep_dir()
    if found is None:
        root = peds_root()
        parser.error("--recordings_dir required when $ARENA_DATA_DIR is unset" if root is None else f"no sweep runs found under {root}")
    sweep_dir, _, done, expected = found
    partial = "" if done >= expected else f" (partial: {done}/{expected} trials done)"
    print(f"Auto-resolved recordings_dir={sweep_dir}{partial}")
    return [sweep_dir]


def _trials(recordings_dirs: list[Path]) -> list[Trial]:
    out: list[Trial] = []
    for rd in recordings_dirs:
        for trial in sorted(rd.iterdir()):
            if not trial.is_dir() or not (trial / "bag").is_dir():
                continue
            try:
                scenario, planner, robot_policy, seed = parse_trial_dir(trial.name)
            except ValueError:
                continue
            out.append((rd, trial, scenario, planner, robot_policy, seed))
    return out


def _select_diagonal(trials: list[Trial], offset_seed: int) -> list[Trial]:
    by_combo: dict[tuple[str, str], dict[str, list[Trial]]] = {}
    for t in trials:
        _, _, scenario, planner, robot_policy, _ = t
        by_combo.setdefault((planner, robot_policy), {}).setdefault(scenario, []).append(t)
    combos = sorted(by_combo)
    scenarios = sorted({s for combo in by_combo.values() for s in combo})
    if not combos or not scenarios:
        return []
    offset = random.Random(offset_seed).randrange(len(scenarios))
    selected: list[Trial] = []
    for i, combo in enumerate(combos):
        for k in range(len(scenarios)):
            scenario = scenarios[(i + offset + k) % len(scenarios)]
            options = by_combo[combo].get(scenario)
            if options:
                selected.append(min(options, key=lambda t: t[5]))
                break
    return selected


def _select_scenario(trials: list[Trial], name: str) -> list[Trial]:
    return [t for t in trials if t[2] == name]


def _render_one(trial_dir: Path, fmt: str, out_dir: Path | None) -> int:
    bag_dir = trial_dir / "bag"
    if out_dir is not None:
        output = out_dir / f"{trial_dir.name}.{fmt}"
        log = out_dir / f"{trial_dir.name}.log"
    else:
        output = trial_dir / f"scenario.{fmt}"
        log = trial_dir / "render.log"
    print(f"  rendering {trial_dir.name} -> {output}")
    try:
        return _renderer.main([str(bag_dir), "--output", str(output), "--format", fmt, "--log-file", str(log)])
    except Exception as exc:
        print(f"  failed: {type(exc).__name__}: {exc}")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-render selected trial bags as mp4/gif using arena_humansim_render. Default mode 'diagonal' picks one trial per (planner, robot_policy) combo, walking scenarios with a seeded offset.")
    parser.add_argument("--recordings_dir", type=str, default=None, help="Sweep dir (or meta-dir with --all). Defaults to newest sweep under $ARENA_DATA_DIR/peds.")
    parser.add_argument("--all", action="store_true", help="Treat --recordings_dir as a meta-dir; pool trials across child sweeps.")
    parser.add_argument("--mode", choices=("diagonal", "all", "scenario"), default="diagonal", help="diagonal (default): one trial per (planner,robot_policy), Latin walk over scenarios. all: every trial. scenario: every trial of one scenario (requires --scenario).")
    parser.add_argument("--scenario", type=str, default=None, help="scenario name (required with --mode scenario).")
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument("--seed", type=int, default=0, help="seed for diagonal-mode scenario offset (reproducibility).")
    parser.add_argument("--workers", type=int, default=1, help="parallel render processes. 0 = one process per selected trial (all concurrent). 1 = sequential.")
    parser.add_argument("--out_dir", type=str, default=None, help="if set, write all videos flat to this dir as <trial_name>.<fmt> (and <trial_name>.log). Default: in-place under each trial dir.")
    parser.add_argument("--dry_run", action="store_true", help="Print selection but don't render.")
    args = parser.parse_args()

    if args.mode == "scenario" and not args.scenario:
        parser.error("--mode scenario requires --scenario <name>")

    recordings_dirs = _resolve_dirs(args, parser)
    trials = _trials(recordings_dirs)
    if not trials:
        parser.error(f"no trials with bags under {[str(d) for d in recordings_dirs]}")

    if args.mode == "diagonal":
        selected = _select_diagonal(trials, offset_seed=args.seed)
    elif args.mode == "all":
        selected = list(trials)
    else:
        selected = _select_scenario(trials, args.scenario)
        if not selected:
            available = sorted({t[2] for t in trials})
            parser.error(f"no trials match scenario={args.scenario!r}; available: {available}")

    print(f"Selected {len(selected)} trial(s) (mode={args.mode}, format={args.format}):")
    for _, trial, scenario, planner, robot_policy, seed in selected:
        rp = robot_policy or "-"
        print(f"  {trial.name}  [scenario={scenario} planner={planner} robot_policy={rp} seed={seed}]")

    if args.dry_run:
        return

    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Writing flat to {out_dir}")

    failures: list[str] = []
    if args.workers == 1:
        for _, trial, *_meta in selected:
            rc = _render_one(trial, args.format, out_dir)
            if rc != 0:
                failures.append(trial.name)
    else:
        max_workers = len(selected) if args.workers == 0 else args.workers
        print(f"Running {max_workers} concurrent render process(es).")
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_render_one, trial, args.format, out_dir): trial.name for _, trial, *_ in selected}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    rc = fut.result()
                except Exception as exc:
                    print(f"  {name}: {type(exc).__name__}: {exc}")
                    rc = 1
                if rc != 0:
                    failures.append(name)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for name in failures:
            print(f"  {name}")
        sys.exit(1)
    print(f"Done - rendered {len(selected)} trial(s).")


if __name__ == "__main__":
    main()
