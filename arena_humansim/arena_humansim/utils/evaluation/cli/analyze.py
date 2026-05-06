import argparse
import json
from pathlib import Path

from arena_humansim.utils.evaluation.analyze import run_analysis
from arena_humansim.utils.evaluation.cli._resume import latest_sweep_dir, peds_root
from arena_humansim.utils.evaluation.robots import run_robots_analysis
from arena_humansim.utils.evaluation.robots_failures import run_failure_analysis


def _read_manifest(sweep_dir: Path) -> dict:
    manifest_path = sweep_dir / "run_args.json"
    return json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}


def _union_manifest(manifests: list[dict]) -> dict:
    mode = "robots" if any(m.get("mode") == "robots" for m in manifests) else "divergence"
    cop = any(bool(m.get("cop", False)) for m in manifests)
    return {"mode": mode, "cop": cop}


def _resolve_dirs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[list[Path], dict]:
    if args.all:
        meta = Path(args.recordings_dir).resolve() if args.recordings_dir else peds_root()
        if meta is None:
            parser.error("--all requires either --recordings_dir or $ARENA_DATA_DIR")
        if not meta.is_dir():
            parser.error(f"--all: meta dir {meta} does not exist")
        sweeps = sorted(p for p in meta.iterdir() if p.is_dir() and (p / "run_args.json").is_file())
        if not sweeps:
            parser.error(f"--all: no sweep dirs (with run_args.json) found under {meta}")
        manifest = _union_manifest([_read_manifest(s) for s in sweeps])
        print(f"Meta-analysis: {len(sweeps)} sweep dir(s) under {meta}")
        for s in sweeps:
            print(f"  - {s.name}")
        return sweeps, manifest

    if args.recordings_dir:
        recordings_dir = Path(args.recordings_dir).resolve()
        return [recordings_dir], _read_manifest(recordings_dir)

    found = latest_sweep_dir()
    if found is None:
        root = peds_root()
        if root is None:
            parser.error("--recordings_dir required when $ARENA_DATA_DIR is unset")
        parser.error(f"no sweep runs found under {root}")
    sweep_dir, manifest, done, expected = found
    partial = "" if done >= expected else f" (partial: {done}/{expected} trials done)"
    print(f"Auto-resolved recordings_dir={sweep_dir}{partial}")
    return [sweep_dir], manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a sweep run dir. Always runs pairwise driver-divergence; if the manifest mode is `robots`, also runs robot-policy metrics + failure classification. Pass --all to pool every sweep under a meta-dir.")
    parser.add_argument("--recordings_dir", type=str, default=None, help="Sweep dir to analyze (or meta-dir with --all). If omitted, picks the newest sweep under $ARENA_DATA_DIR/peds.")
    parser.add_argument("--all", action="store_true", help="Treat --recordings_dir (or $ARENA_DATA_DIR/peds) as a meta-dir; pool trials across every child sweep. Trial-key collisions are deduped, keeping the first occurrence.")
    parser.add_argument("--out_dir", type=str, default=None, help="CSV output dir (default: first input dir for single-sweep mode, or meta-dir for --all)")
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="bootstrap iterations for K confidence interval")
    parser.add_argument("--ci_seed", type=int, default=0, help="RNG seed for bootstrap (reproducibility)")
    parser.add_argument("--framing_threshold", type=float, default=1.2, help="multiplier above K_nav for a stressor bucket to 'clear'")
    args = parser.parse_args()

    recordings_dirs, manifest = _resolve_dirs(args, parser)

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    elif args.all:
        meta = Path(args.recordings_dir).resolve() if args.recordings_dir else peds_root()
        out_dir = meta / "_meta_analysis" if meta is not None else recordings_dirs[0]
    else:
        out_dir = recordings_dirs[0]

    mode = manifest.get("mode", "divergence")
    cop = bool(manifest.get("cop", False))
    print(f"Manifest mode={mode}{' (COP)' if cop else ''} | out_dir={out_dir}")

    print()
    print("=== Pairwise divergence analysis ===")
    run_analysis(
        recordings_dirs=recordings_dirs,
        out_dir=out_dir,
        n_bootstrap=args.n_bootstrap,
        ci_seed=args.ci_seed,
        framing_threshold=args.framing_threshold,
    )

    if mode == "robots":
        print()
        print("=== Robot-policy analysis ===")
        run_robots_analysis(recordings_dirs, out_dir, n_bootstrap=args.n_bootstrap, ci_seed=args.ci_seed)
        print()
        print("=== Robot failure classification ===")
        run_failure_analysis(recordings_dirs, out_dir)
        if cop:
            print()
            print("=== COP analysis: not yet implemented (manifest is correctly cop=true) ===")


if __name__ == "__main__":
    main()
