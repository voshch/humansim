import argparse
from pathlib import Path

from arena_humansim.utils.evaluation.cli._resume import latest_sweep_dir, peds_root
from arena_humansim.utils.evaluation.export import run_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a sweep (or pooled meta-sweep) to a Croissant-annotated parquet tree ready for HuggingFace.")
    parser.add_argument("--recordings_dir", type=str, default=None, help="Sweep dir to export (or meta-dir with --all). If omitted, picks the newest sweep under $ARENA_DATA_DIR/peds.")
    parser.add_argument("--all", action="store_true", help="Treat --recordings_dir (or $ARENA_DATA_DIR/peds) as a meta-dir; pool trials across every child sweep.")
    parser.add_argument("--out_dir", type=str, required=True, help="Destination dir for the dataset tree.")
    parser.add_argument("--name", type=str, required=True, help="Dataset name (used in README + Croissant `name`).")
    parser.add_argument("--mode", choices=("auto", "robots", "divergence"), default="auto", help="Force robots-mode metrics on/off; auto-detects from non-empty robot_policy column.")
    args = parser.parse_args()

    if args.all:
        meta = Path(args.recordings_dir).resolve() if args.recordings_dir else peds_root()
        if meta is None:
            parser.error("--all requires either --recordings_dir or $ARENA_DATA_DIR")
        if not meta.is_dir():
            parser.error(f"--all: meta dir {meta} does not exist")
        sweeps = sorted(p for p in meta.iterdir() if p.is_dir() and (p / "run_args.json").is_file())
        if not sweeps:
            parser.error(f"--all: no sweep dirs (with run_args.json) found under {meta}")
        recordings_dirs = sweeps
        print(f"Exporting {len(sweeps)} sweep dir(s) under {meta}")
    elif args.recordings_dir:
        recordings_dirs = [Path(args.recordings_dir).resolve()]
    else:
        found = latest_sweep_dir()
        if found is None:
            root = peds_root()
            parser.error(f"--recordings_dir required when $ARENA_DATA_DIR is unset" if root is None else f"no sweep runs found under {root}")
        sweep_dir, _, done, expected = found
        partial = "" if done >= expected else f" (partial: {done}/{expected} trials done)"
        print(f"Auto-resolved recordings_dir={sweep_dir}{partial}")
        recordings_dirs = [sweep_dir]

    out_dir = Path(args.out_dir).resolve()
    summary = run_export(
        recordings_dirs=recordings_dirs,
        out_dir=out_dir,
        dataset_name=args.name,
        mode=args.mode,
    )
    print()
    print(f"Exported {summary['n_trials']} trials → {summary['n_parquet_shards']} parquet shards + {len(summary['metrics_tables'])} metric table(s) at {summary['out_dir']}")
    print("Next steps:")
    print("  1. Edit README.md (license, citation, links) and croissant.json (TODO fields).")
    print("  2. `huggingface-cli upload <repo_id> <out_dir> .` (or use the `datasets` Python API).")


if __name__ == "__main__":
    main()
