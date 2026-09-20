import argparse
from pathlib import Path

from arena_humansim.utils.evaluation.plots import run_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the paper figures from analysis CSVs.")
    parser.add_argument("--divergence", type=str, default=None, help="analyzed divergence sweep dir (pairwise_distance.csv, headline.csv, ...)")
    parser.add_argument("--robots", type=str, default=None, help="analyzed closed-loop sweep dir (robots_cell.csv, failures_by_policy_bucket.csv)")
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    if args.divergence is None and args.robots is None:
        parser.error("pass --divergence and/or --robots")
    run_plots(Path(args.divergence).resolve() if args.divergence else None, Path(args.robots).resolve() if args.robots else None, Path(args.out_dir).resolve())


if __name__ == "__main__":
    main()
